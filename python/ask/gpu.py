"""End-to-end native LLVM fusion, persistent IR caching, and explicit CUDA codegen."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .cache import IRCache
from .compiler import CompilerError
from .cuda import compile_cuda, execute
from .ir import Module


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _numerics(request):
    """Bind cache identity to the dtype and ASK's arithmetic semantics."""
    signature = request.get("signature", {})
    if not isinstance(signature, dict) or len(signature) != 2:
        raise CompilerError("CUDA fusion requires two explicitly typed data pointers")
    types = list(signature.values())
    dtypes = {"*i1": "bool", "*i32": "i32", "*i64": "i64", "*fp32": "f32", "*fp16": "f16", "*bf16": "bf16"}
    if any(value not in dtypes for value in types) or types[0] != types[1]:
        raise CompilerError("CUDA fusion requires matching bool/i32/i64/f32/f16/bf16 data-pointer types")
    dtype = dtypes[types[0]]
    cast = {"cast_semantics": "rne-saturating-fptosi-nan-zero-v1", "compute_types": "explicit-per-SSA-value"}
    if dtype == "bool":
        return {"element_type": dtype, "arithmetic": "boolean", "storage_bytes": 1, **cast}
    if dtype in ("i32", "i64"):
        return {"element_type": dtype, "arithmetic": "wrap-modulo-2^" + dtype[1:], **cast}
    return {"element_type": dtype, "arithmetic": {"f32": "ieee754-binary32",
            "f16": "ieee754-binary16", "bf16": "bfloat16"}[dtype],
            "rounding": "roundTiesToEven", "fast_math": False, "contraction": False,
            "operation_result_type": "explicit-per-SSA-value", **cast}


def _validate_environment():
    allowed = {"TRITON_CACHE_DIR", "TRITON_HOME", "TRITON_ALWAYS_COMPILE",
               "TRITON_KERNEL_DUMP", "TRITON_DUMP_DIR", "TRITON_DISABLE_LINE_INFO"}
    unsupported = sorted(name for name, value in os.environ.items() if value and
                         ((name.startswith("TRITON_") and name not in allowed)
                          or name == "LLVM_PASS_PLUGIN_PATH"))
    if unsupported:
        raise CompilerError("fusion/cache does not support compiler environment overrides: "
                            + ", ".join(unsupported))


def _tool(path):
    resolved = shutil.which(os.fspath(path))
    if not resolved:
        raise CompilerError(f"compiler tool not found: {path}")
    # Preserve virtualenv executable paths: resolving the Python symlink before
    # launching it can select the base environment instead of the chosen venv.
    resolved = Path(resolved).absolute()
    hasher = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return {"path": str(resolved), "sha256": hasher.hexdigest()}


# A package version alone misses modified wheels, libtriton, libdevice and ptxas.
# Run in the selected producer environment, after the build allowlist gate.
_FINGERPRINT = r'''
import hashlib, json, pathlib, platform, triton
root = pathlib.Path(triton.__file__).resolve().parent
files = []
for p in sorted(root.rglob('*')):
    if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo'):
        digest = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(block)
        files.append([str(p.relative_to(root)), digest.hexdigest()])
print(json.dumps({'name':'triton', 'version':triton.__version__, 'python':platform.python_version(),
                 'package':str(root), 'files_sha256':hashlib.sha256(
                     json.dumps(files, separators=(',', ':')).encode()).hexdigest()}))
'''


@dataclass(frozen=True)
class CUDAKernel:
    artifact: dict
    codegen: dict
    report: dict
    cache: dict

    def run(self, values, *, device_ordinal=0, driver_library=None):
        """Execute the fused cubin using the artifact's dtype for both buffers."""
        return execute(self.artifact, self.codegen, values,
                       device_ordinal=device_ordinal, driver_library=driver_library)


def compile_graph(compiler, module, *, cache_dir, output_dir, triton_python,
                  triton_worker, llc, ptxas):
    """Cache fused LLVM IR; generate a fresh PTX/cubin for each explicit request.

    Cache hits bypass Triton export and ASK fusion, not graph validation or
    GPU code generation. Tensor values are runtime inputs and never cache keys.
    """
    caps = compiler.capabilities()
    if "triton" not in caps.get("third_party_compilers", []) or not caps.get("triton_llir_fusion"):
        raise CompilerError("this ASK build does not enable Triton LLIR fusion")
    _validate_environment()
    source = module.to_mlir() if isinstance(module, Module) else module
    if not isinstance(source, str):
        raise TypeError("compile_cuda expects a Module or MLIR string")
    prepared = compiler.prepare(source)
    requests = prepared.report.get("external_requests", [])
    if len(requests) != 1 or requests[0].get("compiler") != "triton":
        raise CompilerError("CUDA fusion requires exactly one Triton external operation")
    tools = {name: _tool(path) for name, path in {
        "ask-opt": compiler.executable, "python": triton_python, "llc": llc, "ptxas": ptxas}.items()}
    worker = Path(triton_worker).resolve()
    adapter = {name: _sha((worker.parent / name).read_bytes())
               for name in (worker.name, "fusion.py")}
    adapter["orchestrator_sha256"] = _sha(Path(__file__).read_bytes())
    try:
        probe = subprocess.run([tools["python"]["path"], "-c", _FINGERPRINT],
                               capture_output=True, text=True, timeout=compiler.timeout, check=True)
        producer = json.loads(probe.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise CompilerError(f"cannot fingerprint the selected Triton installation: {exc}") from exc
    request = requests[0]
    key = {
        "schema_version": 1, "pipeline_version": "ask.triton.typed-boundary.v4",
        "graph": _sha(source.encode()), "dsl": _sha(request["source"].encode()),
        "config": {"request": {k: v for k, v in request.items() if k != "source"},
                   "environment_sha256": _sha(_json(dict(os.environ)))},
        "llvm_revision": caps["llvm_revision"], "producer": producer, "adapter": adapter,
        "tools": tools, "target": request["target"],
        "shape": prepared.report["shape_snapshot"], "numerics": _numerics(request),
    }
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise CompilerError("output_dir must be empty; use a fresh destination")
    destination.mkdir(parents=True, exist_ok=True)

    def build():
        with tempfile.TemporaryDirectory(prefix="ask-fused-ir-") as directory:
            work = Path(directory)
            graph = work / "graph.mlir"
            graph.write_text(source)
            exports = work / "export"
            compiler._run([str(graph), "--fuse-external", "--export-dir", str(exports),
                           "--triton-python", tools["python"]["path"], "--triton-worker", str(worker),
                           "-o", str(work / "fused.ll"), "--report-json", str(work / "report.json")])
            report = json.loads((work / "report.json").read_text())
            # Temporary filesystem locations are not reusable artifact identity.
            report["external_artifacts"] = [{"artifact": "external.artifact.json",
                                              "llvm_ir": "external.ll", "stage": "exported"}]
            return {"fused.ll": (work / "fused.ll").read_bytes(),
                    "fused.artifact.json": (exports / "fused.artifact.json").read_bytes(),
                    "ask.helpers.ll": (exports / "ask.helpers.ll").read_bytes(),
                    "external.ll": (exports / "external-0.ll").read_bytes(),
                    "external.artifact.json": (exports / "external-0.artifact.json").read_bytes(),
                    "report.json": _json(report)}

    artifacts, stats = IRCache(cache_dir).get_or_create(key, build)
    for name, contents in artifacts.items():
        (destination / name).write_bytes(contents)
    fused = json.loads(artifacts["fused.artifact.json"])
    report = json.loads(artifacts["report.json"])
    report["ir_cache"] = stats
    report["ir_cache_key_inputs"] = key
    (destination / "report.json").write_bytes(_json(report) + b"\n")
    codegen = compile_cuda(destination / "fused.ll", fused, destination / "codegen",
                           llc=tools["llc"]["path"], ptxas=tools["ptxas"]["path"])
    return CUDAKernel(fused, codegen, report, stats)
