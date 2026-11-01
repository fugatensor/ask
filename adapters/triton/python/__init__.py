"""Triton-specific Python policy; imported only after the build registration gate."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from ask.adapters import CompilerAdapter, CompilerEnvironment
from ask.compiler import CompilerError

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


class TritonAdapter(CompilerAdapter):
    compiler_id = "triton"

    def cache_identity(self, request, environment, *, timeout):
        if environment.python is None:
            raise CompilerError("Triton requires an explicit Python interpreter environment")
        _validate_environment()
        numerics = _numerics(request)
        worker = Path(environment.worker).resolve()
        adapter = {name: _sha((worker.parent / name).read_bytes())
                   for name in (worker.name, "fusion.py")}
        adapter["python_policy_sha256"] = _sha(Path(__file__).read_bytes())
        try:
            probe = subprocess.run([environment.python, "-c", _FINGERPRINT], capture_output=True,
                                   text=True, timeout=timeout, check=True)
            producer = json.loads(probe.stdout)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            raise CompilerError(f"cannot fingerprint the selected Triton installation: {error}") from error
        return {"producer": producer, "adapter": adapter, "dsl": _sha(request["source"].encode()),
                "target": request["target"], "numerics": numerics,
                "config": {name: value for name, value in request.items() if name != "source"}}


def create_adapter():
    return TritonAdapter()


def _require(condition, message):
    if not condition:
        raise CompilerError(message)


def normalize_legacy_artifact(artifact, *, shape=None):
    """Explicit migration for old Triton artifacts; generic CUDA code never calls it.

    The migrated metadata changes the artifact digest. Recompile it to obtain a
    matching codegen receipt; an old cubin receipt cannot be silently reused.
    """
    result = copy.deepcopy(artifact)
    _require(result.get("producer", {}).get("name") == "triton", "legacy artifact is not produced by Triton")
    invocation = result.get("invocation", {})
    _require(invocation.get("version") == 1, "legacy conversion requires invocation version 1")
    _legacy_contract(result)
    constants = result.get("specializations", {}).get("constants", {})
    count, block = constants.get("N"), constants.get("BLOCK")
    _require(type(count) is int and count >= 0 and type(block) is int and block > 0,
             "legacy Triton artifact lacks valid N/BLOCK specializations")
    _require(invocation.get("grid") == [max(1, (count + block - 1) // block), 1, 1],
             "legacy launch grid disagrees with N/BLOCK")
    if shape is None:
        shape = result.get("fusion", {}).get("shape", [count])
    _require(isinstance(shape, (list, tuple)) and all(type(n) is int and n >= 0 for n in shape)
             and math.prod(shape) == count, "legacy shape does not match the element count")
    widths = {"bool": 1, "i32": 4, "i64": 8, "f32": 4, "f16": 2, "bf16": 2}
    ports = result.get("port_map", [])
    _require(len(ports) == 2 and all(port.get("element_type") in widths for port in ports),
             "legacy typed ports are incomplete")
    dtype = ports[0]["element_type"]
    _require(all(port["element_type"] == dtype for port in ports), "legacy data pointer types differ")
    for port in ports:
        port.update(shape=list(shape), layout="contiguous_row_major", storage_bytes=widths[dtype])
    required = set(result.get("required_element_types", [])) | {dtype}
    for name in ("fusion", "fusion_contract", "source_fusion_contract"):
        required.update(result.get(name, {}).get("required_element_types", []))
    result["required_element_types"] = sorted(required)
    result["codegen"] = {"fp_contract": _legacy_fp_contract(result, dtype), "ptx_version": 80}
    result["producer_metadata"] = {name: invocation.pop(name) for name in
                                   ("compiler_metadata", "num_warps", "num_ctas", "resources")}
    invocation["version"] = 2
    return result


def _legacy_fp_contract(artifact, dtype):
    """Keep F's backend contraction setting without applying it to ASK ops."""
    metadata = artifact["invocation"].get("compiler_metadata")
    _require(isinstance(metadata, dict)
             and type(metadata.get("enable_fp_fusion")) is bool
             and type(metadata.get("enable_reflect_ftz")) is bool,
             "floating code generation requires the producer's explicit numerical settings")
    enabled = metadata["enable_fp_fusion"]
    if "fusion" not in artifact and "source_fusion_contract" not in artifact:
        # Triton's raw LLIR leaves this setting to its LLVM backend. There are
        # no ASK operations in this artifact, so the original global policy is
        # appropriate. After fusion that permission must be local to F instead.
        return "fast" if enabled else "off"
    fusion = artifact.get("fusion")
    validation = artifact.get("validation")
    _require(isinstance(fusion, dict) and isinstance(validation, dict)
             and validation.get("stage") == "fused_llvm"
             and validation.get("llvm_module_verified") is True,
             "floating fused code generation requires a verified fusion receipt")
    normalized = fusion.get("producer_numerics")
    provenance = artifact.get("source_fusion_contract", {}).get("numerics")
    expected = ("bool-exact" if dtype == "bool" else dtype + "-wrap" if dtype in ("i32", "i64")
                else f"{dtype}-rne-no-fast-math-no-contract")
    _require(isinstance(normalized, dict) and isinstance(provenance, dict)
             and normalized.get("representation") == "original-F-instruction-contract-flags"
             and normalized.get("producer_enable_fp_fusion") is enabled
             and normalized.get("producer_enable_reflect_ftz") is metadata["enable_reflect_ftz"]
             and provenance.get("enable_fp_fusion") is enabled
             and provenance.get("enable_reflect_ftz") is metadata["enable_reflect_ftz"]
             and fusion.get("ask_numerics") == expected,
             "floating fusion numerical normalization does not match the producer settings")
    return "on"


_STORAGE_TYPES = {"bool", "i32", "i64", "f32", "f16", "bf16"}

def _legacy_contract(artifact):
    """Check metadata, without trying to infer memory semantics from LLVM/PTX."""
    _require(isinstance(artifact, dict), "artifact must be an object")
    _require(artifact.get("format") == "llvm-ir", "expected native LLVM IR artifact")
    _require(isinstance(artifact.get("payload"), str) and bool(artifact["payload"]),
             "artifact must carry the LLVM IR that will actually be compiled")
    entry = artifact.get("entry")
    _require(isinstance(entry, str) and entry.isascii() and entry.isidentifier(),
             "unsupported entry name")
    target = artifact.get("target", {})
    _require(isinstance(target, dict) and target.get("backend") == "cuda"
             and type(target.get("arch")) is int and target["arch"] >= 70
             and target.get("warp_size") == 32,
             "CUDA runtime requires an explicit ordinary sm architecture >= 70 and warp32")
    invocation = artifact.get("invocation", {})
    _require(isinstance(invocation, dict) and invocation.get("kind") == "cuda-kernel" and invocation.get("version") == 1,
             "unsupported invocation kind/version")
    warps = invocation.get("num_warps")
    _require(type(warps) is int and 0 < warps <= 32 and warps & (warps - 1) == 0,
             "num_warps must be a positive power of two within CUDA's thread-block limit")
    block = invocation.get("block")
    _require(isinstance(block, list) and len(block) == 3
             and all(type(n) is int for n in block) and block == [warps * target["warp_size"], 1, 1],
             "launch block must match the producer's declared num_warps and warp size")
    _require(invocation.get("num_ctas") == 1 and invocation.get("shared_bytes") == 0,
             "current CUDA runner requires one CTA per program and no shared memory")
    grid = invocation.get("grid")
    _require(isinstance(grid, list) and len(grid) == 3
             and all(type(n) is int and 0 < n <= 2**31 - 1 for n in grid), "invalid launch grid")
    arguments = invocation.get("arguments")
    _require(isinstance(arguments, list) and len(arguments) == 4,
             "initial CUDA ABI requires two data pointers and two unused scratch pointers")
    data_bindings = []
    for index, argument in enumerate(arguments):
        _require(isinstance(argument, dict) and argument.get("argument") == index
                 and argument.get("llvm_type") == "ptr addrspace(1)", "invalid pointer ABI mapping")
        if index < 2:
            binding = argument.get("binding")
            _require(isinstance(binding, str), "data pointer binding must name a logical port")
            data_bindings.append(binding)
        else:
            resource = ("global_scratch", "profile_scratch")[index - 2]
            resources = invocation.get("resources", {})
            _require(isinstance(resources, dict), "resources must describe the unused scratch buffers")
            description = resources.get(resource, {})
            _require(argument.get("binding") == "null" and argument.get("resource") == resource
                     and isinstance(description, dict) and description.get("size_bytes") == 0,
                     "scratch must be explicitly unused")
    _require(set(data_bindings) == {"input:0", "output:0"}, "expected input:0 and output:0")
    ports = artifact.get("port_map")
    _require(isinstance(ports, list) and len(ports) == 2,
             "typed input/output port_map is required")
    port_types = {}
    for port in ports:
        _require(isinstance(port, dict), "logical ports must be objects")
        index, binding, dtype = port.get("argument"), port.get("logical_port"), port.get("element_type")
        _require(type(index) is int and 0 <= index < 2 and binding == data_bindings[index]
                 and binding not in port_types, "logical port and invocation argument mappings disagree")
        _require(isinstance(dtype, str) and dtype in _STORAGE_TYPES,
                 "CUDA buffers support bool, i32, i64, f32, f16 and bf16 element types")
        port_types[binding] = dtype
    dtype = port_types["input:0"]
    _require(port_types["output:0"] == dtype,
             "current CUDA runner requires matching input/output element types")
    # A fused artifact keeps the removed markers' contract only as provenance.
    # The active storage description is port_map; cross-check any provenance
    # rather than interpreting opaque LLVM pointers as typed data pointers.
    for name in ("fusion_contract", "source_fusion_contract"):
        if name in artifact:
            contract = artifact[name]
            _require(isinstance(contract, dict) and contract.get("dtype") == dtype,
                     "fusion provenance and logical port element types disagree")
    specializations = artifact.get("specializations", {})
    _require(isinstance(specializations, dict) and isinstance(specializations.get("constants"), dict),
             "explicit constants specialization is required")
    constants = specializations["constants"]
    size = constants.get("N")
    _require(type(size) is int and 0 <= size <= 2**31 - 1,
             "initial CUDA runner requires explicit nonnegative constexpr N")
    elements_per_program = constants.get("BLOCK")
    _require(type(elements_per_program) is int and 0 < elements_per_program <= 2**30
             and elements_per_program & (elements_per_program - 1) == 0,
             "constexpr BLOCK must be a positive power of two within the signed-index range")
    _require(grid == [max(1, (size + elements_per_program - 1) // elements_per_program), 1, 1],
             "launch grid must match the producer's declared N/BLOCK element partition")
    return entry, invocation, size, dtype
