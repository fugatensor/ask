"""Resident CUDA sessions and contiguous device tensors.

Only host convenience copies wait. submit() queues work on a retained primary
context and an owned nonblocking stream, then returns an event. External tensors
use the CUDA Array Interface; the producer owns their memory and ASK retains
that producer until all submitted accesses have completed.
"""
from __future__ import annotations

import math
import functools
import struct
import sys
import threading
import weakref

from .cuda import CUDAError
from .numerics import decode_float, encode_float
from .types import ScalarType

_WIDTH = {"bool": 1, "i32": 4, "i64": 8, "f32": 4, "f16": 2, "bf16": 2}
_CAI_TYPES = {"|b1": "bool", "<i4": "i32", "<i8": "i64", "<f4": "f32", "<f2": "f16"}
_TYPESTR = {dtype: typestr for typestr, dtype in _CAI_TYPES.items()}
_DESCRIPTOR_STREAM = object()


def _require(condition, message):
    if not condition:
        raise CUDAError(message)


def _shape_dtype(shape, dtype):
    dtype = ScalarType(dtype).dtype
    _require(isinstance(shape, (tuple, list)) and
             all(type(n) is int and n >= 0 for n in shape),
             "device tensor shape must contain concrete nonnegative integers")
    shape = tuple(shape)
    _require(math.prod(shape) <= sys.maxsize // _WIDTH[dtype], "device tensor byte count overflows")
    return shape, dtype


def _cuda_array_description(owner, *, producer_stream=_DESCRIPTOR_STREAM):
    """Validate one CAI snapshot without treating uint16 as bfloat16."""
    description = getattr(owner, "__cuda_array_interface__", None)
    _require(isinstance(description, dict), "object does not provide a CUDA Array Interface dictionary")
    description = dict(description)
    _require(type(description.get("version")) is int and description["version"] in (2, 3),
             "CUDA Array Interface version must be 2 or 3")
    typestr = description.get("typestr")
    if isinstance(typestr, str) and typestr.startswith("=") and sys.byteorder == "little":
        typestr = "<" + typestr[1:]
    _require(isinstance(typestr, str) and typestr in _CAI_TYPES,
             "unsupported CUDA Array Interface typestr; use bool, i32, i64, f32 or f16. "
             "BF16 has no standard CAI typestr; uint16/void storage is not implicitly BF16")
    shape, dtype = _shape_dtype(description.get("shape"), _CAI_TYPES[typestr])
    _require(description.get("mask") is None, "masked CUDA arrays are unsupported")
    data = description.get("data")
    _require(isinstance(data, (tuple, list)) and len(data) == 2 and
             type(data[0]) is int and 0 <= data[0] < 2**64 and type(data[1]) is bool,
             "CUDA Array Interface data must be (device pointer, readonly bool)")
    pointer, readonly = data
    nbytes = math.prod(shape) * _WIDTH[dtype]
    _require((pointer != 0 or nbytes == 0) and pointer % _WIDTH[dtype] == 0,
             "external tensor has a null or misaligned device pointer")
    _require(pointer + nbytes < 2**64, "external tensor pointer range overflows")
    strides = description.get("strides")
    if strides is not None:
        _require(isinstance(strides, (tuple, list)) and len(strides) == len(shape)
                 and all(type(n) is int and n >= 0 for n in strides),
                 "external tensor strides must be nonnegative byte strides")
        expected = _WIDTH[dtype]
        for dimension, stride in zip(reversed(shape), reversed(strides)):
            _require(dimension <= 1 or stride == expected,
                     "external tensor must be contiguous in row-major order")
            expected *= dimension
    if producer_stream is _DESCRIPTOR_STREAM:
        _require(description["version"] != 2 or "stream" in description,
                 "CUDA Array Interface version 2 has no synchronization guarantee; "
                 "provide producer_stream=1/2/a CUDA stream handle, or explicitly "
                 "producer_stream=None if all production is complete")
        stream = description.get("stream")
    else:
        stream = producer_stream
    _require(stream is None or (type(stream) is int and 0 < stream < 2**64),
             "CUDA Array Interface stream must be None, 1, 2 or a nonzero stream pointer")
    return shape, dtype, pointer, readonly, stream


def _encode(values, dtype, count):
    _require(isinstance(values, (list, tuple)) and len(values) == count,
             "host values must be a flat list/tuple matching the device tensor element count")
    parts = []
    for value in values:
        if dtype == "bool":
            _require(type(value) is bool, "bool host values must be Python bool scalars")
            parts.append(bytes((int(value),)))
        elif dtype in ("i32", "i64"):
            width = 8 * _WIDTH[dtype]
            _require(type(value) is int and -(2**(width-1)) <= value < 2**(width-1),
                     f"host values must be signed {dtype} integers")
            parts.append(value.to_bytes(_WIDTH[dtype], "little", signed=True))
        else:
            try:
                parts.append(encode_float(value, dtype))
            except (TypeError, ValueError, OverflowError) as error:
                raise CUDAError(f"invalid {dtype} host value: {error}") from error
    return b"".join(parts)


def _decode(data, dtype):
    if dtype == "bool":
        return [bool(value) for value in data]
    if dtype in ("i32", "i64"):
        return [value[0] for value in struct.iter_unpack("<i" if dtype == "i32" else "<q", data)]
    width = _WIDTH[dtype]
    return [decode_float(data[index:index+width], dtype) for index in range(0, len(data), width)]


class DeviceTensor:
    """Contiguous storage plus shape/dtype; construction goes through a session.

    The native allocation keeps its context alive. A tensor cannot submit/copy
    after its session is closed or deleted; a retained external owner still
    keeps the underlying storage alive until pending GPU work has completed.
    """
    def __init__(self, session, buffer, shape, dtype, *, owner=None, snapshot=None,
                 producer_stream=_DESCRIPTOR_STREAM):
        self._session_ref = weakref.ref(session)
        self._buffer = buffer
        self._shape, self._dtype = shape, dtype
        self._owner, self._snapshot = owner, snapshot
        self._producer_stream = producer_stream

    def _session(self):
        session = self._session_ref()
        _require(session is not None and not session.closed, "device tensor's execution session is closed")
        return session

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def nbytes(self):
        return self._buffer.nbytes

    @property
    def data_ptr(self):
        return self._buffer.data_ptr

    @property
    def readonly(self):
        return self._buffer.readonly

    @property
    def __cuda_array_interface__(self):
        session = self._session()
        _require(self.dtype != "bf16", "BF16 has no standard CUDA Array Interface typestr")
        return {"version": 3, "shape": self.shape, "typestr": _TYPESTR[self.dtype],
                "strides": None, "data": (self.data_ptr if self.nbytes else 0, self.readonly),
                "stream": session.stream}

    def _prepare_access(self, session):
        _require(self._session() is session, "device tensor belongs to a different execution session; "
                 "import it explicitly with from_cuda_array for cross-stream access")
        if self._owner is not None:
            snapshot = _cuda_array_description(self._owner, producer_stream=self._producer_stream)
            _require(snapshot[:4] == self._snapshot[:4],
                     "external tensor pointer, shape, dtype or readonly state changed after import")
            if snapshot[4] is not None:
                session._native.wait_for_external_stream(snapshot[4])

    def copy_from_host(self, values):
        session = self._session()
        _require(not self.readonly, "device tensor is read-only")
        data = _encode(values, self.dtype, math.prod(self.shape))
        with session._lock:
            self._prepare_access(session)
            self._buffer.copy_from_host(data)
            session.reap()
        return self

    def to_host(self):
        session = self._session()
        with session._lock:
            self._prepare_access(session)
            data = self._buffer.to_host_bytes()
            session.reap()
        return _decode(data, self.dtype)


class Event:
    """A completion event; query never waits and wait covers this submission."""
    def __init__(self, session, native, references):
        self._session_ref = weakref.ref(session)
        self._native = native
        # Retain external producers even when the caller immediately discards
        # this Event. Session._pending owns it until completion is observed.
        self._references = references

    def query(self):
        complete = self._native.query()
        if complete:
            self._references = ()
        return complete

    def wait(self):
        self._native.wait()
        self._references = ()
        session = self._session_ref()
        if session is not None:
            session.reap()
        return self


def _serialized(method):
    @functools.wraps(method)
    def call(self, *args, **kwargs):
        # Native calls release the GIL. Protect the Python owner queue across
        # the entire enqueue-and-retain transaction, not only list updates.
        with self._lock:
            return method(self, *args, **kwargs)
    return call


class ExecutionSession:
    """One retained CUDA primary context, stream, module cache and pending work.

    close() waits for this session's stream. It never resets a primary context
    shared with another framework. DeviceBuffer/Event native objects retain
    their necessary CUDA resources independently of the Python session object.
    """
    def __init__(self, device=0):
        _require(type(device) is int and device >= 0, "device ordinal must be a nonnegative integer")
        try:
            from . import _native
        except ImportError as error:
            raise CUDAError("ASK native runtime is not installed; add build/<mode>/python to "
                            "PYTHONPATH or install the native ASK package") from error
        self._lock = threading.RLock()
        self._pending = []
        self._native = _native.CUDASession.create(device)

    @property
    def closed(self):
        return self._native.closed

    @property
    def stream(self):
        return self._native.stream

    @property
    def device(self):
        return self._native.device

    @property
    def capability(self):
        return tuple(self._native.capability)

    @property
    def stats(self):
        return dict(self._native.stats)

    def _check_open(self):
        _require(not self.closed, "CUDA execution session is closed")

    @_serialized
    def tensor(self, shape, dtype="f32"):
        self._check_open()
        shape, dtype = _shape_dtype(shape, dtype)
        self._native.check_types([dtype])
        buffer = self._native.allocate(math.prod(shape) * _WIDTH[dtype])
        return DeviceTensor(self, buffer, shape, dtype)

    @_serialized
    def from_host(self, values, *, shape=None, dtype="f32"):
        if shape is None:
            _require(isinstance(values, (list, tuple)), "host values must be a flat list or tuple")
            shape = (len(values),)
        shape, dtype = _shape_dtype(shape, dtype)
        data = _encode(values, dtype, math.prod(shape))
        tensor = self.tensor(shape, dtype)
        tensor._buffer.copy_from_host(data)
        self.reap()
        return tensor

    @_serialized
    def from_cuda_array(self, owner, *, producer_stream=_DESCRIPTOR_STREAM):
        """Borrow external storage and establish its producer dependency.

        CAI v3 supplies a stream/readiness contract. A v2 producer without a
        stream must be given an explicit override; None asserts that production
        is complete. Overrides persist for every subsequent access, so callers
        must reimport when their declared producer stream/readiness changes.
        Before the original producer reads or writes this storage again, wait
        for the returned submission Event or establish an equivalent dependency.
        """
        self._check_open()
        snapshot = _cuda_array_description(owner, producer_stream=producer_stream)
        shape, dtype, pointer, readonly, stream = snapshot
        self._native.check_types([dtype])
        buffer = self._native.import_buffer(pointer, math.prod(shape) * _WIDTH[dtype], readonly)
        if stream is not None:
            self._native.wait_for_external_stream(stream)
        return DeviceTensor(self, buffer, shape, dtype, owner=owner, snapshot=snapshot,
                            producer_stream=producer_stream)

    @_serialized
    def load(self, cubin, entry, *, required_types, arch, expected_parameters=4):
        self._check_open()
        _require(isinstance(cubin, bytes), "load expects cubin bytes")
        _require(isinstance(entry, str) and entry.isascii() and entry.isidentifier(), "invalid CUDA kernel entry")
        _require(type(arch) is int and arch >= 70, "CUDA target architecture must be an integer >= 70")
        _require(isinstance(required_types, (list, tuple)) and
                 all(isinstance(dtype, str) and dtype in _WIDTH for dtype in required_types),
                 "required_types must name all kernel element types")
        _require(type(expected_parameters) is int and 0 < expected_parameters <= 64,
                 "expected_parameters must be an integer between 1 and 64")
        return self._native.load(cubin, entry, required_types, arch, expected_parameters)

    @_serialized
    def submit(self, kernel, bindings, *, grid, block, shared_bytes=0, writable_indices=()):
        self._check_open()
        _require(isinstance(bindings, (list, tuple)), "kernel bindings must be a list/tuple")
        for dimensions in (grid, block):
            _require(isinstance(dimensions, (list, tuple)) and len(dimensions) == 3 and
                     all(type(value) is int and 0 < value < 2**32 for value in dimensions),
                     "CUDA grid/block must contain three positive 32-bit dimensions")
        _require(type(shared_bytes) is int and 0 <= shared_bytes < 2**32, "invalid dynamic shared byte count")
        _require(isinstance(writable_indices, (tuple, list)) and
                 all(type(index) is int and 0 <= index < len(bindings) for index in writable_indices),
                 "invalid writable parameter indices")
        buffers = []
        for index, tensor in enumerate(bindings):
            _require(tensor is None or isinstance(tensor, DeviceTensor), "bindings must be DeviceTensor or None")
            if tensor is not None:
                _require(index not in writable_indices or not tensor.readonly, "output tensor is read-only")
                tensor._prepare_access(self)
            buffers.append(None if tensor is None else tensor._buffer)
        self.reap()
        native = self._native.submit(kernel, buffers, grid, block, shared_bytes)
        event = Event(self, native, tuple(bindings))
        self._pending.append(event)
        return event

    @_serialized
    def reap(self):
        # A native event holds native resources; its Python peer additionally
        # owns external framework tensors. Drop both only after GPU completion.
        self._pending[:] = [event for event in self._pending if not event.query()]
        self._native.reap()

    @_serialized
    def synchronize(self):
        self._check_open()
        self._native.synchronize()
        self.reap()

    def close(self):
        if not hasattr(self, "_native"):
            return
        with self._lock:
            self._native.close()
            # A failed close keeps owners retained for an explicit retry. After
            # success the owned stream is complete, even if callers kept Events.
            for event in self._pending:
                event._references = ()
            self._pending.clear()

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
