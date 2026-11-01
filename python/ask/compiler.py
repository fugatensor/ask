"""In-process compiler bindings, with an explicit ask-opt compatibility path."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ir import Module


class CompilerError(RuntimeError):
    def __init__(self, message: str, *, diagnostics: str = "", returncode: int | None = None) -> None:
        self.diagnostics = diagnostics
        self.returncode = returncode
        super().__init__(message + ("\n" + diagnostics.strip() if diagnostics.strip() else ""))


@dataclass(frozen=True)
class PreparedModule:
    """Verified/prepared or optimized IR and diagnostics, not an executed kernel."""

    ir: str
    report: dict[str, Any]


class Compiler:
    """Use in-process bindings by default; an executable selects CLI compatibility.

    ``timeout`` applies to CLI/tool subprocesses. Native preparation owns its
    context for the duration of the call and does not offer asynchronous cancellation.
    """
    def __init__(self, executable: str | os.PathLike[str] | None = None, *, timeout: float = 60.0) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout
        self._native = None
        self._cli_checked = False
        if executable is None:
            try:
                from . import _native
            except ImportError as exc:
                raise CompilerError("ASK native bindings are not installed; build/install ASK or add "
                                    "build/<mode>/python to PYTHONPATH. For the explicit CLI compatibility "
                                    "path, pass Compiler(executable='/path/to/ask-opt').") from exc
            self._native = _native
            package = Path(_native.__file__).resolve().parent
            candidates = [package / "_bin" / "ask-opt", package.parent.parent / "bin" / "ask-opt"]
            candidates.extend(parent / "bin" / "ask-opt" for parent in package.parents)
            self.executable = str(next((path for path in candidates if path.is_file()), Path("ask-opt")))
        else:
            self.executable = os.fspath(executable)
            if not self.executable:
                raise ValueError("executable must not be empty")

    def _check_companion(self) -> None:
        if self._native is None or self._cli_checked:
            return
        result = self._run(["--capabilities"])
        try:
            cli_caps = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise CompilerError("companion ask-opt returned invalid capabilities") from exc
        if cli_caps != self.capabilities():
            raise CompilerError("ask-opt and the loaded ASK native extension belong to different builds")
        self._cli_checked = True

    def _run(self, arguments: list[str]) -> subprocess.CompletedProcess[str]:
        if arguments != ["--capabilities"]:
            self._check_companion()
        try:
            result = subprocess.run([self.executable, *arguments], capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=self.timeout,
                                    check=False)
        except FileNotFoundError as exc:
            raise CompilerError(f"native ASK executable not found: {self.executable}") from exc
        except subprocess.TimeoutExpired as exc:
            diagnostics = exc.stderr or ""
            if isinstance(diagnostics, bytes):
                diagnostics = diagnostics.decode("utf-8", errors="replace")
            raise CompilerError("native ASK compiler timed out", diagnostics=diagnostics) from exc
        except OSError as exc:
            raise CompilerError(f"cannot start native ASK compiler: {exc}") from exc
        if result.returncode < 0:
            raise CompilerError(f"native ASK compiler terminated by signal {-result.returncode}",
                                diagnostics=result.stderr or result.stdout,
                                returncode=result.returncode)
        if result.returncode:
            raise CompilerError("native ASK compiler rejected the request",
                                diagnostics=result.stderr or result.stdout,
                                returncode=result.returncode)
        return result

    def capabilities(self) -> dict[str, Any]:
        if self._native is not None:
            return json.loads(self._native.capabilities())
        result = self._run(["--capabilities"])
        try:
            capabilities = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise CompilerError("native ASK compiler returned invalid capability JSON",
                                diagnostics=result.stdout) from exc
        if not isinstance(capabilities, dict):
            raise CompilerError("native ASK capability report must be a JSON object")
        return capabilities

    def prepare(self, module: Module | str) -> PreparedModule:
        return self._process(module, ["--prepare"])

    def compile_cuda(self, module: Module | str, *, cache_dir, output_dir,
                     triton_python, triton_worker, llc, ptxas):
        """Fuse the supported ASK→Triton→ASK graph, reusing verified IR and executable caches.

        Explicit paths keep producer dependencies out of SDK import and builds
        with no external compilers. See docs/12-GPU融合与IR缓存.md for the subset.
        """
        from .gpu import compile_graph
        # Check before cache lookup too: an IR hit must not mix the loaded
        # extension with another build's driver/toolchain fingerprint.
        if self._native is not None and self.capabilities().get("triton_llir_fusion"):
            self._check_companion()
        return compile_graph(self, module, cache_dir=cache_dir, output_dir=output_dir,
                             triton_python=triton_python, triton_worker=triton_worker,
                             llc=llc, ptxas=ptxas)

    def optimize(self, module: Module | str, *, fold_constants: bool = True,
                 eliminate_dead_code: bool = True,
                 fuse_elementwise: bool = True) -> PreparedModule:
        """Prepare and optimize ASK IR using the native compiler's explicit passes."""
        options = ["--optimize"]
        for enabled, flag in ((fold_constants, "--no-constant-fold"),
                              (eliminate_dead_code, "--no-dce"),
                              (fuse_elementwise, "--no-fusion")):
            if type(enabled) is not bool:
                raise TypeError("optimization options must be bool values")
            if not enabled:
                options.append(flag)
        return self._process(module, options)

    def _process(self, module: Module | str, options: list[str]) -> PreparedModule:
        if isinstance(module, Module):
            source = module.to_mlir()
        elif isinstance(module, str):
            source = module
        else:
            raise TypeError("prepare/optimize expects a Module or MLIR string")
        if self._native is not None:
            try:
                ir, report = self._native.prepare(
                    source, optimize="--optimize" in options,
                    fold_constants="--no-constant-fold" not in options,
                    eliminate_dead_code="--no-dce" not in options,
                    fuse_elementwise="--no-fusion" not in options)
            except RuntimeError as exc:
                raise CompilerError("native ASK compiler rejected the request", diagnostics=str(exc)) from exc
            return PreparedModule(ir=ir, report=json.loads(report))
        with tempfile.TemporaryDirectory(prefix="ask-prepare-") as directory:
            path = Path(directory)
            input_path = path / "input.mlir"
            output_path = path / "prepared.mlir"
            report_path = path / "report.json"
            input_path.write_text(source, encoding="utf-8")
            result = self._run([str(input_path), "-o", str(output_path), *options,
                                "--report-json", str(report_path)])
            try:
                ir = output_path.read_text(encoding="utf-8")
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise CompilerError("native ASK compiler did not produce valid preparation output",
                                    diagnostics=result.stderr) from exc
            if not isinstance(report, dict):
                raise CompilerError("native ASK preparation report must be a JSON object")
            return PreparedModule(ir=ir, report=report)
