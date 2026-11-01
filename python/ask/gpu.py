"""ASK GPU lowering, optional external LLVM fusion, and separate IR/code caches."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .adapters import AdapterRegistry, CompilerEnvironment
from .cache import IRCache
from .compiler import CompilerError
from .cuda import compile_cuda, execute
from .ir import Module


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def _sha(data):
    return hashlib.sha256(data).hexdigest()

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

@dataclass(frozen=True)
class CUDAKernel:
    artifact: dict
    codegen: dict
    report: dict
    cache: dict

    def prepare(self, session):
        """Validate and load once for repeated asynchronous device execution."""
        from .executable import PreparedCUDAKernel
        return PreparedCUDAKernel(self, session)

    def run(self, values, *, device_ordinal=0, driver_library=None):
        """Execute the fused cubin using the artifact's dtype for both buffers."""
        return execute(self.artifact, self.codegen, values,
                       device_ordinal=device_ordinal, driver_library=driver_library)


def compile_graph(compiler, module, *, cache_dir, output_dir, llc, ptxas,
                  compilers=None, target=None, block_size=128, ptx_version=80,
                  **legacy_environments):
    """Compile a pure ASK graph or fuse a supported external graph.

    Graph validation precedes cache access. Executable hits also skip llc and
    ptxas. Tensor values are runtime inputs and never cache keys.
    """
    caps = compiler.capabilities()
    registry = AdapterRegistry(caps)
    # Explicit compiler configurations must still obey the build allowlist,
    # even on cache hits. Empty configurations do not require any adapters.
    environments = (registry.environments(compilers, legacy_environments)
                    if compilers or legacy_environments else {})
    source = module.to_mlir() if isinstance(module, Module) else module
    if not isinstance(source, str):
        raise TypeError("compile_cuda expects a Module or MLIR string")
    prepared = compiler.prepare(source)
    requests = prepared.report.get("external_requests", [])
    if not requests:
        return _compile_owned_graph(compiler, source, prepared, caps, target=target,
                                    block_size=block_size, ptx_version=ptx_version,
                                    cache_dir=cache_dir, output_dir=output_dir,
                                    llc=llc, ptxas=ptxas)
    if len(requests) != 1 or not isinstance(requests[0], dict):
        raise CompilerError("CUDA fusion currently requires exactly one external operation")
    request = requests[0]
    compiler_id = request.get("compiler")
    registry.require_enabled(compiler_id, fusion=True)
    if compiler_id not in environments:
        raise CompilerError(f"provide CompilerEnvironment for compiler {compiler_id!r}")
    environment = environments[compiler_id]
    tool_paths = {"ask-opt": compiler.executable, "llc": llc, "ptxas": ptxas}
    if environment.python is not None:
        tool_paths["python"] = environment.python
    else:
        tool_paths["worker"] = environment.worker
    tools = {name: _tool(path) for name, path in tool_paths.items()}
    environment = CompilerEnvironment(python=tools["python"]["path"] if "python" in tools else None,
                                      worker=tools["worker"]["path"] if "worker" in tools
                                      else Path(environment.worker).resolve())
    adapter = registry.load(compiler_id)
    identity = adapter.cache_identity(request, environment, timeout=compiler.timeout)
    fields = {"producer", "adapter", "dsl", "target", "numerics", "config"}
    if not isinstance(identity, dict) or not fields <= identity.keys():
        raise CompilerError(f"compiler adapter {compiler_id!r} returned incomplete cache identity")
    if target is not None and target != identity["target"]:
        raise CompilerError("external graph target must match its compiler's fixed target")
    if block_size != 128 or ptx_version != 80:
        raise CompilerError("block_size and ptx_version overrides apply only to pure ASK graphs; "
                            "an external graph retains its compiler's execution policy")
    adapter_identity = {"policy": identity["adapter"], "registration": registry.registration(compiler_id),
                        "orchestrator_sha256": _sha(Path(__file__).read_bytes())}
    key = {
        "schema_version": 1, "pipeline_version": "ask.external.typed-boundary.v1",
        "graph": _sha(source.encode()), "dsl": identity["dsl"],
        "config": {"compiler_id": compiler_id, "request": identity["config"],
                   "environment_sha256": _sha(_json(dict(os.environ)))},
        "llvm_revision": caps["llvm_revision"], "producer": identity["producer"], "adapter": adapter_identity,
        "tools": tools, "target": identity["target"],
        "shape": prepared.report["shape_snapshot"], "numerics": identity["numerics"],
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
            environment_flags = ["--compiler-worker", compiler_id + "=" + environment.worker]
            if environment.python is not None:
                environment_flags.extend(["--compiler-python", compiler_id + "=" + environment.python])
            compiler._run([str(graph), "--fuse-external", "--export-dir", str(exports), *environment_flags,
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
    codegen = compile_cuda(destination / "fused.ll", fused, destination / "codegen",
                           llc=tools["llc"]["path"], ptxas=tools["ptxas"]["path"],
                           cache_dir=Path(cache_dir) / "executables")
    report["executable_cache"] = codegen["executable_cache"]
    (destination / "report.json").write_bytes(_json(report) + b"\n")
    return CUDAKernel(fused, codegen, report, stats)


def _compile_owned_graph(compiler, source, prepared, caps, *, target, block_size,
                         ptx_version, cache_dir, output_dir, llc, ptxas):
    if caps.get("cuda_graph_codegen") is not True:
        raise CompilerError("this ASK build does not support native ASK GPU lowering")
    if (not isinstance(target, dict) or set(target) != {"backend", "arch"}
            or target["backend"] != "cuda" or type(target["arch"]) is not int
            or not 10 <= target["arch"] <= 999):
        raise CompilerError("pure ASK CUDA compilation requires target={'backend': 'cuda', 'arch': SM_NUMBER}")
    if type(block_size) is not int or not 1 <= block_size <= 1024:
        raise CompilerError("block_size must be an integer between 1 and 1024")
    if type(ptx_version) is not int or not 10 <= ptx_version <= 999:
        raise CompilerError("ptx_version must be an ordinary integer between 10 and 999")
    target = dict(target)
    tools = {name: _tool(path) for name, path in
             (("ask-opt", compiler.executable), ("llc", llc), ("ptxas", ptxas))}
    key = {
        "schema_version": 1, "pipeline_version": "ask.owned.cuda.v1",
        "graph": _sha(source.encode()), "dsl": None,
        "config": {"block_size": block_size, "ptx_version": ptx_version},
        "llvm_revision": caps["llvm_revision"], "producer": {"name": "ask"},
        "adapter": None, "tools": tools, "target": target,
        "shape": prepared.report["shape_snapshot"],
        "numerics": {"fp_contract": "off"},
        "orchestrator_sha256": _sha(Path(__file__).read_bytes()),
    }
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise CompilerError("output_dir must be empty; use a fresh destination")
    destination.mkdir(parents=True, exist_ok=True)

    def build():
        with tempfile.TemporaryDirectory(prefix="ask-owned-gpu-") as directory:
            work = Path(directory)
            graph = work / "graph.mlir"
            graph.write_text(source, encoding="utf-8")
            compiler._run([str(graph), "--optimize", "--emit-cuda-llvm", "--cuda-arch", str(target["arch"]),
                           "--cuda-block-size", str(block_size), "--cuda-ptx-version", str(ptx_version),
                           "-o", str(work / "kernel.ll"), "--report-json", str(work / "report.json")])
            report = json.loads((work / "report.json").read_text())
            artifact = report.pop("cuda_artifact")
            return {"kernel.ll": (work / "kernel.ll").read_bytes(),
                    "kernel.artifact.json": _json(artifact), "report.json": _json(report)}

    artifacts, stats = IRCache(cache_dir).get_or_create(key, build)
    for name, contents in artifacts.items():
        (destination / name).write_bytes(contents)
    artifact = json.loads(artifacts["kernel.artifact.json"])
    report = json.loads(artifacts["report.json"])
    report["ir_cache"], report["ir_cache_key_inputs"] = stats, key
    codegen = compile_cuda(destination / "kernel.ll", artifact, destination / "codegen",
                           llc=tools["llc"]["path"], ptxas=tools["ptxas"]["path"],
                           cache_dir=Path(cache_dir) / "executables")
    report["executable_cache"] = codegen["executable_cache"]
    (destination / "report.json").write_bytes(_json(report) + b"\n")
    return CUDAKernel(artifact, codegen, report, stats)
