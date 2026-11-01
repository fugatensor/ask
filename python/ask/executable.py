"""Validate a compiled package once, then submit resident device tensors."""
from __future__ import annotations

import copy
import ctypes
import math
from pathlib import Path

from .cuda import (_artifact_digest, _kernel_contract, _digest, _fp_contract,
                   _required_types, _require, _storage_dtype, _STORAGE_TYPES)


class PreparedCUDAKernel:
    """A checked executable bound to one execution session.

    Preparation reads and hashes the on-disk package. ``submit`` only checks
    tensor metadata and enqueues work; it neither reads files nor copies tensor
    values through Python. The session owns the loaded module and CUDA stream.
    """

    def __init__(self, kernel, session):
        artifact, codegen = copy.deepcopy(kernel.artifact), copy.deepcopy(kernel.codegen)
        entry, invocation, inputs, outputs = _kernel_contract(artifact)
        dtype = _storage_dtype(inputs, outputs)
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

        # Shape is part of the normalized execution ABI, independent of the
        # producer's graph report or its original specialization vocabulary.
        self._session = session
        self._inputs, self._outputs = inputs, outputs
        self._invocation = invocation
        self._kernel = session.load(binary, entry, arch=arch,
                                    required_types=sorted(_required_types(artifact, dtype)),
                                    expected_parameters=len(invocation["arguments"]))

    @property
    def shape(self):
        _require(len(self._outputs) == 1, "multiple outputs have separate shapes; use outputs")
        return tuple(self._outputs[0]["shape"])

    @property
    def dtype(self):
        _require(len(self._outputs) == 1, "multiple outputs have separate dtypes; use outputs")
        return self._outputs[0]["element_type"]

    @property
    def inputs(self):
        return copy.deepcopy(self._inputs)

    @property
    def outputs(self):
        return copy.deepcopy(self._outputs)

    def submit(self, input, output):
        """One-input, one-output convenience wrapper, including casts."""
        return self.submit_many([input], [output])

    def submit_many(self, inputs, outputs):
        """Check ordered ports and enqueue once; return an event with wait/query.

        Inputs may share storage. Each output must be disjoint from every
        input and other output, as required by the graph's pointer ABI.
        """
        _require(isinstance(inputs, (tuple, list)) and len(inputs) == len(self._inputs),
                 f"kernel expects {len(self._inputs)} input tensors")
        _require(isinstance(outputs, (tuple, list)) and len(outputs) == len(self._outputs),
                 f"kernel expects {len(self._outputs)} output tensors")
        bindings = {}
        for port, tensor in zip((*self._inputs, *self._outputs), (*inputs, *outputs)):
            name, shape, dtype = port["logical_port"], tuple(port["shape"]), port["element_type"]
            _require(tuple(tensor.shape) == shape,
                     f"{name} shape must equal compiled shape {shape}")
            _require(tensor.dtype == dtype,
                     f"{name} dtype must equal compiled dtype {dtype}")
            _require(tensor.nbytes == math.prod(shape) * ctypes.sizeof(_STORAGE_TYPES[dtype]),
                     f"{name} buffer size disagrees with the compiled tensor type")
            if name.startswith("output:"):
                _require(not tensor.readonly, "output tensor is read-only")
            bindings[name] = tensor
        for index, output in enumerate(outputs):
            for other in (*inputs, *outputs[:index]):
                _require(other.nbytes == 0 or output.nbytes == 0
                         or other.data_ptr + other.nbytes <= output.data_ptr
                         or output.data_ptr + output.nbytes <= other.data_ptr,
                         "compiled graph requires non-overlapping input/output and output/output buffers")
        parameters = [bindings[argument["binding"]] if argument["binding"] != "null" else None
                      for argument in self._invocation["arguments"]]
        return self._session.submit(self._kernel, parameters,
                                    grid=self._invocation["grid"],
                                    block=self._invocation["block"],
                                    shared_bytes=self._invocation["shared_bytes"])
