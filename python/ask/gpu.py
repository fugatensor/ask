"""ASK-owned CUDA compilation and opaque external executable scheduling."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .adapters import CompilerEnvironment, worker_command, worker_identity, export_kernel
from .cache import IRCache
from .compiler import CompilerError
from .cuda import compile_cuda, execute, install_external
from .ir import Module


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _tool(path):
    if path is None:
        raise CompilerError("ASK-owned CUDA steps require explicit llc and ptxas paths")
    resolved = shutil.which(os.fspath(path))
    if not resolved:
        raise CompilerError(f"compiler tool not found: {path}")
    resolved = Path(resolved).absolute()
    return {"path": str(resolved), "sha256": _sha(resolved.read_bytes())}


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
        """Run a single-input/output kernel with host values."""
        return execute(self.artifact, self.codegen, values,
                       device_ordinal=device_ordinal, driver_library=driver_library)


def compile_graph(compiler, module, *, cache_dir, output_dir, llc=None, ptxas=None,
                  compilers=None, target=None, block_size=128, ptx_version=80):
    """Compile owned regions; externally compiled kernels stay separate steps."""
    if compilers is not None and (not isinstance(compilers, Mapping) or any(
            not isinstance(key, str) or not key or not isinstance(value, CompilerEnvironment)
            for key, value in compilers.items())):
        raise TypeError("compilers must map nonempty compiler IDs to CompilerEnvironment values")
    environments = dict(compilers or {})
    caps = compiler.capabilities()
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
    from .plan import CUDAProgram, validate_plan
    for request in requests:
        if request.get("compiler") not in environments:
            raise CompilerError(f"provide CompilerEnvironment for compiler {request.get('compiler')!r}")
    if target is None:
        requested = requests[0].get("target", {})
        target = {"backend": requested.get("backend"), "arch": requested.get("arch")}
    if (not isinstance(target, dict) or set(target) != {"backend", "arch"}
            or target["backend"] != "cuda" or type(target["arch"]) is not int):
        raise CompilerError("CUDA compilation requires target={'backend': 'cuda', 'arch': SM_NUMBER}")
    if any(any(request["target"].get(key) != value for key, value in target.items()) for request in requests):
        raise CompilerError("external graph target must match its compiler's fixed target")
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise CompilerError("output_dir must be empty; use a fresh destination")
    partitioned = compiler.partition(source)
    plan = validate_plan(partitioned.report.get("execution_plan"))
    sources = partitioned.report.get("step_sources")
    if not isinstance(sources, dict) or set(sources) != {s["id"] for s in plan["steps"]}:
        raise CompilerError("partition report must provide exactly one source per step")
    # Validate local tool requirements before starting external work.
    if any(step["kind"] == "owned" for step in plan["steps"]) and (llc is None or ptxas is None):
        raise CompilerError("ASK-owned CUDA steps require explicit llc and ptxas paths")
    destination.mkdir(parents=True, exist_ok=True)
    kernels = {}
    for step in plan["steps"]:
        step_source = sources[step["id"]]
        if not isinstance(step_source, str):
            raise CompilerError("partition step source must be MLIR text")
        step_prepared = compiler.prepare(step_source)
        step_dir = destination if len(plan["steps"]) == 1 else destination / step["id"]
        if step["kind"] == "owned":
            kernel = _compile_owned_graph(
                compiler, step_source, step_prepared, caps, target=target,
                block_size=block_size, ptx_version=ptx_version, cache_dir=cache_dir,
                output_dir=step_dir, llc=llc, ptxas=ptxas)
        else:
            kernel = _compile_external_graph(compiler, step_prepared, environments,
                                             cache_dir=cache_dir, output_dir=step_dir)
        kernels[step["id"]] = kernel
    if len(plan["steps"]) == 1:
        step = plan["steps"][0]
        if step["inputs"] == plan["inputs"] and step["outputs"] == plan["outputs"]:
            kernel = kernels[step["id"]]
            kernel.report["execution_plan"] = plan
            (destination / "execution_plan.json").write_bytes(_json(plan) + b"\n")
            (destination / "report.json").write_bytes(_json(kernel.report) + b"\n")
            return kernel
    report = {"phase": "compiled_plan", "execution_plan": plan,
              "steps": {name: kernel.report for name, kernel in kernels.items()}}
    program = CUDAProgram(plan, kernels, report, {"steps": {name: kernel.cache for name, kernel in kernels.items()}})
    program.save(destination)
    (destination / "report.json").write_bytes(_json(report) + b"\n")
    return program


def _compile_external_graph(compiler, prepared, environments, *, cache_dir, output_dir):
    requests = prepared.report.get("external_requests", [])
    if len(requests) != 1 or not isinstance(requests[0], dict):
        raise CompilerError("each external partition must contain exactly one external operation")
    original = requests[0]
    fields = ("protocol_version", "source", "entry", "signature", "source_bindings", "constants",
              "fixed_options", "target", "grid", "port_shapes")
    request = {key: original[key] for key in fields}
    command = worker_command(environments[original["compiler"]])
    identity = worker_identity(command, timeout=compiler.timeout, request=request)
    key = {"schema_version": 1, "pipeline_version": "ask.external.executable.v2",
           "request": request, "worker": identity,
           "environment_sha256": _sha(_json(dict(os.environ))),
           "orchestrator_sha256": _sha(Path(__file__).read_bytes()),
           "protocol_sha256": _sha(Path(__file__).with_name("adapters.py").read_bytes())}
    def build():
        artifact = export_kernel(command, request, timeout=compiler.timeout)
        if artifact["producer"] != identity["probe"]["producer"]:
            raise CompilerError("external producer differs from its cache probe")
        if worker_identity(command, timeout=compiler.timeout, request=request) != identity:
            raise CompilerError("external compiler dependencies changed during compilation; retry with a stable environment")
        return {"artifact.json": _json(artifact)}
    artifacts, stats = IRCache(Path(cache_dir) / "external").get_or_create(key, build)
    artifact = json.loads(artifacts["artifact.json"])
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise CompilerError("output_dir must be empty; use a fresh destination")
    # install_external validates the same executable contract on cache hits.
    codegen = install_external(artifact, destination / "codegen")
    report = {"phase": "external_executable", "compiler": original["compiler"],
              "external_cache": stats, "external_cache_key_inputs": key}
    (destination / "external.artifact.json").write_bytes(_json(artifact) + b"\n")
    (destination / "report.json").write_bytes(_json(report) + b"\n")
    return CUDAKernel(artifact, codegen, report, stats)


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
