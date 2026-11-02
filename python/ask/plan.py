"""Versioned, compiler-independent CUDA execution plans and portable packages."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

from .cuda import CUDAError, _kernel_contract, _require

_WIDTH = {"bool": 1, "i32": 4, "i64": 8, "f32": 4, "f16": 2, "bf16": 2}
_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")


def _identifier(value):
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def validate_plan(plan):
    """Return an owned snapshot after structural, SSA and dependency checks.

    Buffer IDs are logical values, never paths. Shapes imply dense row-major
    storage. The ordered input/output lists are the caller ABI; output aliases
    and repeated step inputs are permitted. Steps are already topologically
    sorted and dependencies name exactly their input producers.
    """
    _require(isinstance(plan, dict) and set(plan) == {
        "schema_version", "kind", "inputs", "outputs", "buffers", "steps", "scheduling"},
        "execution plan has missing or unknown fields")
    _require(type(plan["schema_version"]) is int and plan["schema_version"] == 1
             and plan["kind"] == "ask.cuda.execution_plan", "unsupported execution plan version or kind")
    _require(plan["scheduling"] == {"policy": "sequential", "stream": "session"},
             "execution plan requires sequential scheduling on the session stream")
    _require(isinstance(plan["buffers"], list) and isinstance(plan["steps"], list),
             "plan buffers and steps must be arrays")
    buffers = {}
    for buffer in plan["buffers"]:
        _require(isinstance(buffer, dict) and set(buffer) == {"id", "shape", "element_type"},
                 "invalid execution plan buffer fields")
        name, shape, dtype = buffer["id"], buffer["shape"], buffer["element_type"]
        _require(isinstance(name, str) and bool(name) and name not in buffers, "invalid or duplicate plan buffer ID")
        _require(isinstance(dtype, str) and dtype in _WIDTH, "unsupported plan buffer element type")
        _require(isinstance(shape, list) and all(type(n) is int and n >= 0 for n in shape)
                 and math.prod(shape) <= (2**63 - 1) // _WIDTH[dtype],
                 "plan buffer shape must be static and its byte count must fit int64")
        buffers[name] = buffer
    for kind in ("inputs", "outputs"):
        _require(isinstance(plan[kind], list) and all(isinstance(n, str) and n in buffers for n in plan[kind]),
                 f"plan {kind} must reference declared buffers")
    _require(len(set(plan["inputs"])) == len(plan["inputs"]), "plan input IDs must be unique")
    _require(bool(plan["outputs"]), "plan requires at least one output")
    producers = dict.fromkeys(plan["inputs"])
    steps = set()
    for step in plan["steps"]:
        _require(isinstance(step, dict) and set(step) in (
            {"id", "kind", "function", "inputs", "outputs", "depends_on"},
            {"id", "kind", "function", "inputs", "outputs", "depends_on", "external_id"}),
            "invalid execution plan step fields")
        name, kind = step["id"], step["kind"]
        _require(_identifier(name) and name not in steps, "invalid or duplicate plan step ID")
        _require(kind in ("owned", "external") and _identifier(step["function"]),
                 "invalid plan step kind or function symbol")
        _require((kind == "owned" and "external_id" not in step)
                 or (kind != "owned" and isinstance(step.get("external_id"), str) and bool(step["external_id"])),
                 "external steps require an external_id; owned steps must omit it")
        for port in ("inputs", "outputs", "depends_on"):
            _require(isinstance(step[port], list) and all(isinstance(n, str) for n in step[port]),
                     f"step {port} must be an array of IDs")
        _require(all(n in producers for n in step["inputs"]), "step reads an undefined or future buffer")
        _require(bool(step["outputs"]) and len(set(step["outputs"])) == len(step["outputs"])
                 and all(n in buffers and n not in producers for n in step["outputs"]),
                 "step outputs must define fresh, declared SSA buffers")
        dependencies = {producers[n] for n in step["inputs"]} - {None}
        _require(len(set(step["depends_on"])) == len(step["depends_on"])
                 and set(step["depends_on"]) == dependencies,
                 "step dependencies must match its input producers")
        producers.update(dict.fromkeys(step["outputs"], name))
        steps.add(name)
    _require(set(producers) == set(buffers), "plan contains buffers without definitions")
    _require(all(n in producers for n in plan["outputs"]), "plan output is undefined")
    return copy.deepcopy(plan)


def validate_kernels(plan, kernels):
    _require(isinstance(kernels, dict) and set(kernels) == {s["id"] for s in plan["steps"]},
             "compiled kernels must cover exactly the execution plan steps")
    buffers = {b["id"]: b for b in plan["buffers"]}
    architectures = set()
    for step in plan["steps"]:
        artifact = kernels[step["id"]].artifact
        _, _, inputs, outputs = _kernel_contract(artifact)
        architectures.add(artifact["target"]["arch"])
        for kind, ports in (("inputs", inputs), ("outputs", outputs)):
            _require(len(ports) == len(step[kind]), f"step {step['id']} {kind} ABI port count mismatch")
            for name, port in zip(step[kind], ports):
                buffer = buffers[name]
                _require(port["shape"] == buffer["shape"] and port["element_type"] == buffer["element_type"],
                         f"step {step['id']} buffer {name} disagrees with the compiled kernel ABI")
    _require(len(architectures) <= 1, "all plan kernels must use the same CUDA architecture")


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read_reference(root, reference):
    _require(isinstance(reference, dict) and set(reference) == {"path", "sha256"}, "invalid package file reference")
    relative = reference["path"]
    _require(isinstance(relative, str) and relative and "\\" not in relative,
             "package paths must be relative POSIX paths")
    path = PurePosixPath(relative)
    _require(not path.is_absolute() and all(part not in (".", "..") for part in relative.split("/")),
             "package path escapes its directory")
    resolved = (root / path).resolve()
    _require(resolved.is_relative_to(root), "package symlink escapes its directory")
    data = resolved.read_bytes()
    _require(isinstance(reference["sha256"], str) and _sha(data) == reference["sha256"],
             "package file digest mismatch")
    return data, resolved


class CUDAProgram:
    """A checked logical plan plus one reusable CUDAKernel per step.

    ``save`` writes a relocatable package and ``load`` verifies its checksums.
    Checksums detect corruption; they do not authenticate untrusted GPU code.
    """
    def __init__(self, plan, kernels, report=None, cache=None):
        self.plan = validate_plan(plan)
        validate_kernels(self.plan, kernels)
        self.kernels = dict(kernels)
        self.report = copy.deepcopy(report or {})
        self.cache = copy.deepcopy(cache or {})

    def prepare(self, session):
        from .executable import PreparedCUDAProgram
        return PreparedCUDAProgram(self, session)

    def save(self, directory):
        root = Path(directory).resolve()
        root.mkdir(parents=True, exist_ok=True)
        _require(not (root / "program.json").exists(), "program package already exists")
        plan = validate_plan(self.plan)
        validate_kernels(plan, self.kernels)
        from .executable import validate_kernel_package
        for kernel in self.kernels.values():
            validate_kernel_package(kernel)
        def write(name, data):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
            return {"path": name, "sha256": _sha(data)}
        receipt = {"schema_version": 1, "kind": "ask.cuda.program",
                   "plan": write("execution_plan.json", _json(plan)), "steps": {}}
        for step in plan["steps"]:
            name = step["id"]
            kernel = self.kernels[name]
            codegen = copy.deepcopy(kernel.codegen)
            prefix = f"package/{name}"
            files = {}
            for key, filename in (("llvm_ir", "kernel.ll"), ("cubin", "kernel.cubin"), ("ptx", "kernel.ptx")):
                if key not in codegen:
                    continue
                data = Path(codegen[key]).read_bytes()
                digest_key = "llvm_sha256" if key == "llvm_ir" else key + "_sha256"
                _require(_sha(data) == codegen.get(digest_key), f"{key} digest mismatch")
                files[key] = write(f"{prefix}/{filename}", data)
                codegen[key] = files[key]["path"]
            receipt["steps"][name] = {
                "artifact": write(f"{prefix}/artifact.json", _json(kernel.artifact)),
                "codegen": write(f"{prefix}/codegen.json", _json(codegen)), "files": files}
        receipt["report"] = write("program-report.json", _json({"report": self.report, "cache": self.cache}))
        write("program.json", _json(receipt))
        return root / "program.json"

    @classmethod
    def load(cls, directory):
        from .gpu import CUDAKernel
        root = Path(directory).resolve()
        try:
            receipt = json.loads((root / "program.json").read_bytes())
            _require(isinstance(receipt, dict) and set(receipt) == {"schema_version", "kind", "plan", "steps", "report"}
                     and type(receipt["schema_version"]) is int and receipt["schema_version"] == 1
                     and receipt["kind"] == "ask.cuda.program", "unsupported CUDA program receipt")
            plan = validate_plan(json.loads(_read_reference(root, receipt["plan"])[0]))
            _require(isinstance(receipt["steps"], dict) and set(receipt["steps"]) == {s["id"] for s in plan["steps"]},
                     "program receipt step set differs from its plan")
            kernels = {}
            for name, item in receipt["steps"].items():
                _require(isinstance(item, dict) and set(item) == {"artifact", "codegen", "files"},
                         "invalid compiled step receipt")
                artifact = json.loads(_read_reference(root, item["artifact"])[0])
                _require(isinstance(artifact, dict), "compiled artifact must be an object")
                codegen = json.loads(_read_reference(root, item["codegen"])[0])
                _require(isinstance(codegen, dict) and isinstance(item["files"], dict), "invalid compiled step file set")
                file_keys = set(item["files"])
                if artifact.get("format") == "llvm-ir":
                    _require({"llvm_ir", "cubin"} <= file_keys <= {"llvm_ir", "cubin", "ptx"},
                             "invalid compiled step file set")
                else:
                    _require(artifact.get("format") in ("cubin", "ptx") and file_keys == {artifact["format"]},
                             "invalid external step file set")
                for key, reference in item["files"].items():
                    _, path = _read_reference(root, reference)
                    _require(codegen.get(key) == reference["path"], "codegen path disagrees with package receipt")
                    codegen[key] = str(path)
                kernels[name] = CUDAKernel(artifact, codegen, {}, {})
            from .executable import validate_kernel_package
            for kernel in kernels.values():
                validate_kernel_package(kernel)
            metadata = json.loads(_read_reference(root, receipt["report"])[0])
            return cls(plan, kernels, metadata["report"], metadata["cache"])
        except (OSError, UnicodeError, ValueError, KeyError, TypeError) as error:
            raise CUDAError(f"invalid CUDA program package: {error}") from error
