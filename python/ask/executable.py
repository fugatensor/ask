"""Validate a compiled package once, then submit resident device tensors."""
from __future__ import annotations

import copy
import ctypes
import math

from .cuda import (_kernel_contract, _validated_image,
                   _required_types, _require, _storage_dtype, _STORAGE_TYPES)


def validate_kernel_package(kernel):
    artifact, codegen = copy.deepcopy(kernel.artifact), copy.deepcopy(kernel.codegen)
    entry, invocation, inputs, outputs = _kernel_contract(artifact)
    binary = _validated_image(artifact, codegen)

    return artifact, codegen, binary, entry, invocation, inputs, outputs


class PreparedCUDAKernel:
    """A checked executable bound to one execution session.

    Preparation reads and hashes the on-disk package. ``submit`` only checks
    tensor metadata and enqueues work; it neither reads files nor copies tensor
    values through Python. The session owns the loaded module and CUDA stream.
    """

    def __init__(self, kernel, session):
        artifact, codegen, binary, entry, invocation, inputs, outputs = validate_kernel_package(kernel)
        dtype, arch = _storage_dtype(inputs, outputs), artifact["target"]["arch"]

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
        parameters = self._parameters(inputs, outputs)
        return self._submit(parameters)

    def _parameters(self, inputs, outputs):
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
        return [bindings[argument["binding"]] if argument["binding"] != "null" else None
                for argument in self._invocation["arguments"]]

    def _submit(self, parameters):
        return self._session.submit(self._kernel, parameters,
                                    grid=self._invocation["grid"],
                                    block=self._invocation["block"],
                                    shared_bytes=self._invocation["shared_bytes"])


class _CompletedEvent:
    def query(self):
        return True

    def wait(self):
        return self


class PreparedCUDAProgram:
    """A plan on one stream, with all tensor ports checked before any launch."""
    def __init__(self, program, session):
        from .plan import validate_plan, validate_kernels
        self._plan = validate_plan(program.plan)
        validate_kernels(self._plan, program.kernels)
        self._session = session
        self._buffers = {b["id"]: b for b in self._plan["buffers"]}
        self._kernels = {s["id"]: program.kernels[s["id"]].prepare(session) for s in self._plan["steps"]}

    @property
    def plan(self):
        return copy.deepcopy(self._plan)

    def _ports(self, kind):
        return [{"logical_port": f"{kind[:-1]}:{index}", "shape": list(self._buffers[name]["shape"]),
                 "element_type": self._buffers[name]["element_type"], "layout": "contiguous_row_major"}
                for index, name in enumerate(self._plan[kind])]

    @property
    def inputs(self):
        return self._ports("inputs")

    @property
    def outputs(self):
        return self._ports("outputs")

    @property
    def shape(self):
        _require(len(self.outputs) == 1, "multiple outputs have separate shapes; use outputs")
        return tuple(self.outputs[0]["shape"])

    @property
    def dtype(self):
        _require(len(self.outputs) == 1, "multiple outputs have separate dtypes; use outputs")
        return self.outputs[0]["element_type"]

    def submit(self, input, output):
        return self.submit_many([input], [output])

    def submit_many(self, inputs, outputs):
        from .runtime import DeviceTensor
        session = self._session
        with session._lock:
            session._check_open()
            bound = {}
            for kind, tensors in (("inputs", inputs), ("outputs", outputs)):
                _require(isinstance(tensors, (list, tuple)) and len(tensors) == len(self._plan[kind]),
                         f"program expects {len(self._plan[kind])} {kind}")
                for name, tensor in zip(self._plan[kind], tensors):
                    _require(isinstance(tensor, DeviceTensor), "program ports require DeviceTensor values")
                    buffer = self._buffers[name]
                    _require(tuple(tensor.shape) == tuple(buffer["shape"]) and tensor.dtype == buffer["element_type"],
                             f"program buffer {name} shape/dtype differs from its compiled type")
                    _require(tensor.nbytes == math.prod(buffer["shape"]) * ctypes.sizeof(_STORAGE_TYPES[tensor.dtype]),
                             f"program buffer {name} byte count differs from its compiled type")
                    _require(kind != "outputs" or name in self._plan["inputs"] or not tensor.readonly,
                             "program output tensor is read-only")
                    if name in bound:
                        _require(tensor.data_ptr == bound[name].data_ptr and tensor.nbytes == bound[name].nbytes,
                                 "repeated logical buffers must use the same device storage")
                    else:
                        bound[name] = tensor
                    tensor._prepare_access(session)
            # Distinct logical outputs cannot overwrite any other user port,
            # including an input whose last consumer would run in a later step.
            for name in set(self._plan["outputs"]) - set(self._plan["inputs"]):
                output = bound[name]
                for other_name, other in bound.items():
                    if name == other_name:
                        continue
                    _require(output.nbytes == 0 or other.nbytes == 0
                             or output.data_ptr + output.nbytes <= other.data_ptr
                             or other.data_ptr + other.nbytes <= output.data_ptr,
                             "distinct program output buffers must not overlap other ports")
            for name, buffer in self._buffers.items():
                if name not in bound:
                    bound[name] = session.tensor(buffer["shape"], buffer["element_type"])
            prepared = []
            for step in self._plan["steps"]:
                kernel = self._kernels[step["id"]]
                parameters = kernel._parameters([bound[n] for n in step["inputs"]],
                                                [bound[n] for n in step["outputs"]])
                prepared.append((kernel, parameters))
            if not prepared:
                # Identity-only graphs have no kernel on which to record an
                # event. Complete producer-stream waits before reporting done.
                session.synchronize()
                return _CompletedEvent()
            event = None
            for kernel, parameters in prepared:
                event = kernel._submit(parameters)
            # Earlier submissions also keep their own bindings in the session
            # queue. This final event additionally covers every temporary and
            # remains safe if the caller discards both program and event.
            event._references = (*event._references, *bound.values(), *self._kernels.values())
            return event
