"""Scalar encoding and natural layout for CUDA's by-value parameter ABI."""
from __future__ import annotations

import math
import struct

from .numerics import encode_float


SCALAR_WIDTHS = {"i1": 1, "i8": 1, "i16": 2, "i32": 4, "i64": 8,
                 "u8": 1, "u16": 2, "u32": 4, "u64": 8,
                 "fp16": 2, "bf16": 2, "fp32": 4, "fp64": 8}


def scalar_bytes(dtype, value):
    """Encode a finite JSON scalar, checking range without implicit promotion."""
    if not isinstance(dtype, str) or dtype not in SCALAR_WIDTHS:
        raise ValueError(f"unsupported CUDA scalar type: {dtype!r}")
    width = SCALAR_WIDTHS[dtype]
    if dtype == "i1":
        if type(value) is not bool:
            raise ValueError("i1 scalar requires a Python bool")
        return bytes((value,))
    if dtype[0] in ("i", "u"):
        signed = dtype[0] == "i"
        lower, upper = (-(1 << (8 * width - 1)), 1 << (8 * width - 1)) if signed else (0, 1 << (8 * width))
        if type(value) is not int or not lower <= value < upper:
            raise ValueError(f"{dtype} scalar is outside its integer range")
        return value.to_bytes(width, "little", signed=signed)
    if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
        raise ValueError(f"{dtype} scalar requires a finite Python int or float")
    if dtype == "fp64":
        try:
            result = struct.pack("<d", value)
        except (OverflowError, struct.error) as error:
            raise ValueError("fp64 scalar is outside its finite range") from error
        if not math.isfinite(struct.unpack("<d", result)[0]):
            raise ValueError("fp64 scalar is outside its finite range")
        return result
    return encode_float(value, {"fp16": "f16", "fp32": "f32", "bf16": "bf16"}[dtype])


def parameter_types(invocation):
    return [argument["type"] if argument.get("kind") == "scalar" else "pointer"
            for argument in invocation["arguments"]]


def parameter_widths(types):
    """Native ABI uses zero for a device pointer, a positive width for a scalar."""
    if not isinstance(types, (list, tuple)) or not 0 < len(types) <= 64:
        raise ValueError("CUDA ABI requires between 1 and 64 parameter types")
    result = []
    for dtype in types:
        if dtype == "pointer":
            result.append(0)
        elif isinstance(dtype, str) and dtype in SCALAR_WIDTHS:
            result.append(SCALAR_WIDTHS[dtype])
        else:
            raise ValueError(f"unsupported CUDA parameter type: {dtype!r}")
    return result


def parameter_layout(types):
    offset, layout = 0, []
    for width in parameter_widths(types):
        width = width or 8
        offset = (offset + width - 1) // width * width
        layout.append({"offset": offset, "size": width})
        offset += width
    return layout
