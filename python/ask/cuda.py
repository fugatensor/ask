"""Explicit CUDA execution with host buffers typed by the artifact's logical ports.

Importing this module needs only the standard library. The driver is loaded only
by execute; neither a Triton module nor a Triton compiled kernel is imported.
Code generation consumes the supplied LLVM file, including ASK's transformations.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
from pathlib import Path
import subprocess

from .numerics import decode_float, encode_float, round_float


class CUDAError(RuntimeError):
    pass


_STORAGE_TYPES = {"i32": ctypes.c_int32, "f32": ctypes.c_float,
                  "f16": ctypes.c_uint16, "bf16": ctypes.c_uint16}


def _require(condition, message):
    if not condition:
        raise CUDAError(message)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _artifact_digest(artifact):
    return _digest(json.dumps(artifact, sort_keys=True, separators=(",", ":"),
                              allow_nan=False).encode("utf-8"))


def _contract(artifact):
    """Check metadata, without trying to infer memory semantics from LLVM/PTX."""
    _require(isinstance(artifact, dict), "artifact must be an object")
    _require(artifact.get("format") == "llvm-ir", "expected native LLVM IR artifact")
    _require(isinstance(artifact.get("payload"), str) and bool(artifact["payload"]),
             "artifact must carry the LLVM IR that will actually be compiled")
    entry = artifact.get("entry")
    _require(isinstance(entry, str) and entry.isascii() and entry.isidentifier(),
             "unsupported entry name")
    target = artifact.get("target", {})
    _require(isinstance(target, dict) and target.get("backend") == "cuda" and target.get("arch") == 80
             and target.get("warp_size") == 32, "initial CUDA runtime requires cuda/sm_80/warp32")
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
                 "CUDA buffers support i32, f32, f16 and bf16 element types")
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


def _host_buffers(values, size, dtype):
    """Convert Python scalars once at the boundary, using the IR element type.

    Floating inputs round to their declared width before graph operations. Explicit
    infinities, NaNs and signed zero are accepted. A finite value outside the
    representable dtype range is rejected instead of silently becoming infinity.
    """
    _require(isinstance(values, (list, tuple)) and len(values) == size,
             "input length must equal the artifact's explicit N specialization")
    if dtype == "i32":
        _require(all(type(value) is int and -(2**31) <= value < 2**31 for value in values),
                 "input values must be signed i32 integers")
    else:
        _require(all(type(value) in (int, float) for value in values),
                 f"{dtype} input values must be Python int or float scalars")
        rounded = []
        for value in values:
            try:
                rounded.append(round_float(value, dtype) if dtype == "f32" else
                               int.from_bytes(encode_float(value, dtype), "little"))
            except (ValueError, TypeError, OverflowError) as exc:
                raise CUDAError(f"invalid {dtype} input value: {exc}") from exc
        values = rounded
    storage = _STORAGE_TYPES[dtype] * max(size, 1)
    return storage(*values), storage()


def _decode_output(storage, size, dtype):
    if dtype in ("f16", "bf16"):
        return [decode_float(int(bits).to_bytes(2, "little"), dtype)
                for bits in storage[:size]]
    return list(storage)[:size]


def _check_device_dtype(dtype, major, minor, name):
    # NVIDIA CUDA Programming Guide, Supported Floating-Point Types:
    # BF16 requires compute capability 8.0 or later. This is checked against
    # the selected physical device before creating a context or loading code.
    # https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html
    if dtype == "bf16":
        _require(major >= 8,
                 f"bf16 requires hardware support (CUDA compute capability >= 8.0); "
                 f"selected device {name!r} has {major}.{minor}. No emulation fallback is enabled")


def _fp_contract(artifact, dtype):
    """Keep F's backend contraction setting without applying it to ASK ops."""
    if dtype == "i32":
        return "on"
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
    _require(isinstance(normalized, dict) and isinstance(provenance, dict)
             and normalized.get("representation") == "original-F-instruction-contract-flags"
             and normalized.get("producer_enable_fp_fusion") is enabled
             and normalized.get("producer_enable_reflect_ftz") is metadata["enable_reflect_ftz"]
             and provenance.get("enable_fp_fusion") is enabled
             and provenance.get("enable_reflect_ftz") is metadata["enable_reflect_ftz"]
             and fusion.get("ask_numerics") == f"{dtype}-rne-no-fast-math-no-contract",
             "floating fusion numerical normalization does not match the producer settings")
    return "on"


def compile_cuda(llvm_ir, artifact, output_directory, *, llc, ptxas):
    """Compile the exact supplied LLVM bytes to a new cubin; never reuse Triton binaries.

    llc and ptxas are explicit executable paths. No target or launch policy is
    selected here. The current target is sm_80; launch dimensions come from the
    producer artifact rather than a runtime-selected scheduling policy.
    """
    entry, invocation, _, dtype = _contract(artifact)
    fp_contract = _fp_contract(artifact, dtype)
    llvm_ir, output_directory = Path(llvm_ir).resolve(), Path(output_directory).resolve()
    llc, ptxas = Path(llc).resolve(), Path(ptxas).resolve()
    source = llvm_ir.read_bytes()
    _require(source == artifact["payload"].encode("utf-8"),
             "LLVM file differs from artifact payload; bind the transformed IR explicitly")
    output_directory.mkdir(parents=True, exist_ok=True)
    ptx = output_directory / "kernel.ptx"
    cubin = output_directory / "kernel.cubin"
    receipt = output_directory / "codegen.json"
    _require(not any(path.exists() for path in (ptx, cubin, receipt)),
             "codegen destination already contains artifacts; use a fresh directory")
    commands = [
        [str(llc), "-march=nvptx64", "-mcpu=sm_80", "-mattr=+ptx80", "--fp-contract=" + fp_contract,
         str(llvm_ir), "-o", str(ptx)],
        [str(ptxas), "--gpu-name=sm_80", "-v", str(ptx), "-o", str(cubin)],
    ]
    logs = []
    for command in commands:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CUDAError(f"code generation failed: {exc}") from exc
        if result.returncode:
            raise CUDAError(f"code generation failed ({result.returncode}): {' '.join(command)}\n"
                            + result.stdout + result.stderr)
        logs.append(result.stdout + result.stderr)
    _require(llvm_ir.read_bytes() == source, "LLVM input changed during compilation")
    result = {"schema_version": 1, "entry": entry, "arch": 80, "element_type": dtype,
              "fp_contract": fp_contract,
              "artifact_sha256": _artifact_digest(artifact),
              "llvm_ir": str(llvm_ir), "llvm_sha256": _digest(source),
              "ptx": str(ptx), "ptx_sha256": _digest(ptx.read_bytes()),
              "cubin": str(cubin), "cubin_sha256": _digest(cubin.read_bytes()),
              "commands": commands, "compiler_logs": logs,
              "invocation": invocation, "original_triton_binary_used": False}
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


class _Driver:
    def __init__(self, library=None):
        candidates = [library] if library else [ctypes.util.find_library("cuda"),
                                                "/usr/lib/wsl/lib/libcuda.so.1", "libcuda.so.1"]
        errors = []
        for candidate in candidates:
            if not candidate:
                continue
            try:
                self.library = ctypes.CDLL(str(candidate))
                break
            except OSError as exc:
                errors.append(str(exc))
        else:
            raise CUDAError("CUDA driver unavailable: " + "; ".join(errors))
        pointer = ctypes.c_void_p
        integer = ctypes.c_int
        uint = ctypes.c_uint
        deviceptr = ctypes.c_uint64
        size = ctypes.c_size_t
        self.bind("cuGetErrorName", [integer, ctypes.POINTER(ctypes.c_char_p)])
        self.bind("cuGetErrorString", [integer, ctypes.POINTER(ctypes.c_char_p)])
        self.bind("cuInit", [uint])
        self.bind("cuDriverGetVersion", [ctypes.POINTER(integer)])
        self.bind("cuDeviceGet", [ctypes.POINTER(integer), integer])
        self.bind("cuDeviceGetName", [pointer, integer, integer])
        self.bind("cuDeviceGetAttribute", [ctypes.POINTER(integer), integer, integer])
        self.bind("cuCtxCreate_v2", [ctypes.POINTER(pointer), uint, integer])
        self.bind("cuCtxDestroy_v2", [pointer])
        self.bind("cuCtxSynchronize", [])
        self.bind("cuModuleLoadDataEx", [ctypes.POINTER(pointer), pointer, uint, pointer, pointer])
        self.bind("cuModuleUnload", [pointer])
        self.bind("cuModuleGetFunction", [ctypes.POINTER(pointer), pointer, ctypes.c_char_p])
        self.bind("cuFuncGetAttribute", [ctypes.POINTER(integer), integer, pointer])
        self.bind("cuFuncGetParamInfo", [pointer, size, ctypes.POINTER(size), ctypes.POINTER(size)])
        self.bind("cuMemAlloc_v2", [ctypes.POINTER(deviceptr), size])
        self.bind("cuMemFree_v2", [deviceptr])
        self.bind("cuMemcpyHtoD_v2", [deviceptr, pointer, size])
        self.bind("cuMemcpyDtoH_v2", [pointer, deviceptr, size])
        self.bind("cuLaunchKernel", [pointer, uint, uint, uint, uint, uint, uint,
                                     uint, pointer, ctypes.POINTER(pointer), pointer])

    def bind(self, name, argtypes):
        function = getattr(self.library, name)
        function.argtypes = argtypes
        function.restype = ctypes.c_int

    def call(self, name, *arguments):
        status = getattr(self.library, name)(*arguments)
        if status:
            error_name, detail = ctypes.c_char_p(), ctypes.c_char_p()
            self.library.cuGetErrorName(status, ctypes.byref(error_name))
            self.library.cuGetErrorString(status, ctypes.byref(detail))
            raise CUDAError(f"{name}: {(error_name.value or b'CUDA_ERROR').decode()} ({status}): "
                            + (detail.value or b'').decode())

    def attribute(self, device, attribute):
        value = ctypes.c_int()
        self.call("cuDeviceGetAttribute", ctypes.byref(value), attribute, device)
        return value.value

    def function_attribute(self, function, attribute):
        value = ctypes.c_int()
        self.call("cuFuncGetAttribute", ctypes.byref(value), attribute, function)
        return value.value


def execute(artifact, codegen, values, *, device_ordinal=0, driver_library=None):
    """Synchronously execute one explicit invocation, then release its resources.

    This is an experimental runner for a trusted, verified N-element kernel, not
    a GPU memory-safety sandbox. The caller owns the kernel's logical contract.
    Failed launches are never replayed or silently replaced by another kernel.
    """
    entry, invocation, size, dtype = _contract(artifact)
    _require(isinstance(codegen, dict), "codegen receipt must be an object")
    host_input, host_output = _host_buffers(values, size, dtype)
    _require(codegen.get("entry") == entry and codegen.get("arch") == 80
             and codegen.get("invocation") == invocation and codegen.get("element_type") == dtype
             and codegen.get("fp_contract") == _fp_contract(artifact, dtype),
             "codegen and execution contracts disagree")
    _require(codegen.get("artifact_sha256") == _artifact_digest(artifact),
             "artifact digest differs from the compiled artifact")
    llvm_digest = _digest(artifact["payload"].encode("utf-8"))
    _require(llvm_digest == codegen.get("llvm_sha256")
             and _digest(Path(codegen["llvm_ir"]).read_bytes()) == llvm_digest,
             "LLVM input digest differs from the compiled IR")
    binary = Path(codegen["cubin"]).read_bytes()
    _require(_digest(binary) == codegen.get("cubin_sha256"), "cubin digest mismatch")
    driver = _Driver(driver_library)
    driver.call("cuInit", 0)
    device, driver_version = ctypes.c_int(), ctypes.c_int()
    driver.call("cuDeviceGet", ctypes.byref(device), device_ordinal)
    driver.call("cuDriverGetVersion", ctypes.byref(driver_version))
    name = ctypes.create_string_buffer(256)
    driver.call("cuDeviceGetName", name, len(name), device)
    major, minor = driver.attribute(device, 75), driver.attribute(device, 76)
    _check_device_dtype(dtype, major, minor, name.value.decode())
    # Cubin compatibility for ordinary sm_80 targets is within the same major
    # family with an equal or newer minor revision; do not assume across-major.
    _require(major == 8 and minor >= 0, "sm_80 cubin requires a compatible compute capability 8.x device")
    _require(driver.attribute(device, 10) == 32, "device warp size disagrees with contract")
    grid = invocation["grid"]
    _require(all(value <= driver.attribute(device, attr) for value, attr in zip(grid, (5, 6, 7))),
             "launch grid exceeds device limits")
    block = invocation["block"]
    _require(driver.attribute(device, 1) >= block[0]
             and all(value <= driver.attribute(device, attr) for value, attr in zip(block, (2, 3, 4))),
             "device cannot launch the producer's block dimensions")
    context, module = ctypes.c_void_p(), ctypes.c_void_p()
    allocations = []
    # Use push/pop context semantics so a caller's existing current context is
    # restored when this new context is destroyed.
    driver.call("cuCtxCreate_v2", ctypes.byref(context), 0, device)
    try:
        image = ctypes.create_string_buffer(binary)
        driver.call("cuModuleLoadDataEx", ctypes.byref(module), image, 0, None, None)
        function = ctypes.c_void_p()
        driver.call("cuModuleGetFunction", ctypes.byref(function), module, entry.encode("ascii"))
        parameter_layout = []
        for index in range(4):
            offset, width = ctypes.c_size_t(), ctypes.c_size_t()
            driver.call("cuFuncGetParamInfo", function, index, ctypes.byref(offset), ctypes.byref(width))
            _require(offset.value == index * 8 and width.value == 8,
                     "compiled kernel parameter layout differs from the four-pointer ABI")
            parameter_layout.append({"offset": offset.value, "size": width.value})
        extra_offset, extra_width = ctypes.c_size_t(), ctypes.c_size_t()
        status = driver.library.cuFuncGetParamInfo(function, 4, ctypes.byref(extra_offset),
                                                  ctypes.byref(extra_width))
        _require(status == 1, "compiled kernel must have exactly four parameters")
        _require(driver.function_attribute(function, 1) == 0,
                 "compiled kernel unexpectedly requires static shared memory")
        _require(driver.function_attribute(function, 0) >= block[0],
                 "compiled kernel resource use cannot support the producer's block dimensions")
        registers = driver.function_attribute(function, 4)
        for _ in range(2):
            allocation = ctypes.c_uint64()
            driver.call("cuMemAlloc_v2", ctypes.byref(allocation), ctypes.sizeof(host_input))
            allocations.append(allocation)
        driver.call("cuMemcpyHtoD_v2", allocations[0], host_input, ctypes.sizeof(host_input))
        driver.call("cuMemcpyHtoD_v2", allocations[1], host_output, ctypes.sizeof(host_output))
        bindings = {"input:0": allocations[0], "output:0": allocations[1]}
        parameters = [bindings[argument["binding"]] if index < 2 else ctypes.c_uint64(0)
                      for index, argument in enumerate(invocation["arguments"])]
        addresses = (ctypes.c_void_p * 4)(*[ctypes.cast(ctypes.byref(value), ctypes.c_void_p)
                                         for value in parameters])
        driver.call("cuLaunchKernel", function, *grid, *invocation["block"], 0, None, addresses, None)
        driver.call("cuCtxSynchronize")
        driver.call("cuMemcpyDtoH_v2", host_output, allocations[1], ctypes.sizeof(host_output))
        return {"output": _decode_output(host_output, size, dtype), "entry": entry, "element_type": dtype,
                "device": {"ordinal": device_ordinal, "name": name.value.decode(),
                           "compute_capability": [major, minor], "driver_version": driver_version.value},
                "grid": grid, "block": invocation["block"],
                "parameter_layout": parameter_layout, "registers_per_thread": registers,
                "llvm_sha256": codegen["llvm_sha256"], "cubin_sha256": codegen["cubin_sha256"],
                "device_execution_performed": True, "original_triton_binary_used": False}
    finally:
        # Best-effort cleanup must not hide the original compiler/launch error.
        for allocation in reversed(allocations):
            driver.library.cuMemFree_v2(allocation)
        if module.value:
            driver.library.cuModuleUnload(module)
        driver.library.cuCtxDestroy_v2(context)


def execute_i32(artifact, codegen, values, *, device_ordinal=0, driver_library=None):
    """Compatibility entry for callers that explicitly require an i32 kernel."""
    _require(_contract(artifact)[3] == "i32", "execute_i32 requires an i32 artifact; use execute")
    return execute(artifact, codegen, values, device_ordinal=device_ordinal,
                   driver_library=driver_library)
