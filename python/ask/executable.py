"""Validate a compiled package once, then submit resident device tensors."""
from __future__ import annotations

import copy
import ctypes
import math
from pathlib import Path

from .cuda import (_artifact_digest, _contract, _digest, _fp_contract,
                   _required_types, _require, _STORAGE_TYPES)


class PreparedCUDAKernel:
    """A checked executable bound to one execution session.

    Preparation reads and hashes the on-disk package. ``submit`` only checks
    tensor metadata and enqueues work; it neither reads files nor copies tensor
    values through Python. The session owns the loaded module and CUDA stream.
    """

    def __init__(self, kernel, session):
        artifact, codegen = copy.deepcopy(kernel.artifact), copy.deepcopy(kernel.codegen)
        entry, invocation, size, dtype = _contract(artifact)
        _require(isinstance(codegen, dict), "codegen receipt must be an object")
        arch = artifact["target"]["arch"]
        _require(codegen.get("entry") == entry and codegen.get("arch") == arch
                 and codegen.get("invocation") == invocation
                 and codegen.get("element_type") == dtype
                 and codegen.get("fp_contract") == _fp_contract(artifact, dtype),
                 "codegen and execution contracts disagree")
        _require(codegen.get("artifact_sha256") == _artifact_digest(artifact),
                 "artifact digest differs from the compiled artifact")
        source_digest = _digest(artifact["payload"].encode("utf-8"))
        _require(source_digest == codegen.get("llvm_sha256")
                 and _digest(Path(codegen["llvm_ir"]).read_bytes()) == source_digest,
                 "LLVM input digest differs from the compiled IR")
        binary = Path(codegen["cubin"]).read_bytes()
        _require(_digest(binary) == codegen.get("cubin_sha256"), "cubin digest mismatch")

        # The current graph bridge compiles shape-preserving pointwise regions.
        # Preserve the graph's rank instead of treating equal element counts as
        # interchangeable tensor shapes. Raw packages without a graph report
        # retain the explicit one-dimensional N contract.
        snapshot = kernel.report.get("shape_snapshot", {})
        bindings = snapshot.get("shape_bindings", [])
        shapes = [item["shape"] for item in bindings
                  if isinstance(item, dict) and item.get("type", "").startswith("tensor<")]
        shape = shapes[0] if shapes else [size]
        _require(isinstance(shape, list) and all(type(n) is int and n >= 0 for n in shape)
                 and math.prod(shape) == size and all(other == shape for other in shapes),
                 "compiled pointwise graph has inconsistent logical tensor shapes")

        self._session = session
        self._shape, self._dtype = tuple(shape), dtype
        self._nbytes = size * ctypes.sizeof(_STORAGE_TYPES[dtype])
        self._invocation = invocation
        self._kernel = session.load(binary, entry, arch=arch,
                                    required_types=sorted(_required_types(artifact, dtype)))

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    def submit(self, input, output):
        """Enqueue into the session stream; return an event with wait/query."""
        for name, tensor in (("input", input), ("output", output)):
            _require(tuple(tensor.shape) == self._shape,
                     f"{name} shape must equal compiled shape {self._shape}")
            _require(tensor.dtype == self._dtype,
                     f"{name} dtype must equal compiled dtype {self._dtype}")
            _require(tensor.nbytes == self._nbytes,
                     f"{name} buffer size disagrees with the compiled tensor type")
        _require(not output.readonly, "output tensor is read-only")
        # The graph ABI assumes distinct logical input/output storage. Merely
        # checking Python object identity would miss overlapping external views.
        _require(input.nbytes == 0 or output.nbytes == 0
                 or input.data_ptr + input.nbytes <= output.data_ptr
                 or output.data_ptr + output.nbytes <= input.data_ptr,
                 "compiled graph requires non-overlapping input and output buffers")
        bindings = {"input:0": input, "output:0": output}
        parameters = [bindings[argument["binding"]] if index < 2 else None
                      for index, argument in enumerate(self._invocation["arguments"])]
        return self._session.submit(self._kernel, parameters,
                                    grid=self._invocation["grid"],
                                    block=self._invocation["block"],
                                    shared_bytes=self._invocation["shared_bytes"])
