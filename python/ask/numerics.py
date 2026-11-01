"""Exact numeric boundary conversions shared by the IR builder and runtime."""

from __future__ import annotations

import math
import struct

# Exponent bits, stored fraction bits. ASK uses MLIR's f16 spelling; fp16 is
# accepted at Python boundaries for callers using the customary dtype name.
_FORMATS = {"f16": (5, 10), "bf16": (8, 7), "f32": (8, 23)}


def _format(dtype: str) -> tuple[int, int]:
    dtype = "f16" if dtype == "fp16" else dtype
    if not isinstance(dtype, str) or dtype not in _FORMATS:
        raise ValueError(f"unsupported floating dtype: {dtype!r}")
    return _FORMATS[dtype]


def _round_shift(significand: int, shift: int) -> int:
    if shift <= 0:
        return significand << -shift
    quotient, remainder = divmod(significand, 1 << shift)
    halfway = 1 << (shift - 1)
    return quotient + (remainder > halfway or (remainder == halfway and quotient & 1))


def encode_float(value: int | float, dtype: str) -> bytes:
    """Round once to dtype, returning its little-endian IEEE encoding.

    Integer and binary64 inputs are rounded directly to the requested format,
    including subnormals; there is no intermediate f32 conversion. Explicit
    infinities and NaNs are accepted. Finite values rounding to infinity raise
    ValueError. NaN payloads are not part of the ASK numerical contract.
    """
    exponent_bits, fraction_bits = _format(dtype)
    width = 1 + exponent_bits + fraction_bits
    exponent_mask = (1 << exponent_bits) - 1
    bias = (1 << (exponent_bits - 1)) - 1
    if type(value) is int:
        sign, significand, power = int(value < 0), abs(value), 0
    elif type(value) is float:
        bits = int.from_bytes(struct.pack("<d", value), "little")
        sign, exponent = bits >> 63, (bits >> 52) & 0x7FF
        significand = bits & ((1 << 52) - 1)
        if exponent == 0x7FF:
            fraction = 1 << (fraction_bits - 1) if significand else 0
            result = (sign << (width - 1)) | (exponent_mask << fraction_bits) | fraction
            return result.to_bytes(width // 8, "little")
        if exponent:
            significand |= 1 << 52
            power = exponent - 1023 - 52
        else:
            power = -1074
    else:
        raise ValueError(f"{dtype} requires a Python int or float scalar")
    if not significand:
        return (sign << (width - 1)).to_bytes(width // 8, "little")
    exponent = significand.bit_length() - 1 + power
    if exponent > bias:
        raise ValueError(f"finite value is outside the {dtype} range")
    minimum_exponent = 1 - bias
    # Quantize either to the normal precision or the fixed subnormal spacing.
    scale = max(exponent, minimum_exponent) - fraction_bits
    rounded = _round_shift(significand, scale - power)
    if rounded >= 1 << (fraction_bits + 1):
        rounded >>= 1
        exponent += 1
    if exponent > bias:
        raise ValueError(f"finite value is outside the {dtype} range")
    if exponent < minimum_exponent:
        # Rounding the largest subnormal can produce the smallest normal.
        encoded_exponent = 1 if rounded >= 1 << fraction_bits else 0
    else:
        encoded_exponent = exponent + bias
    result = ((sign << (width - 1)) | (encoded_exponent << fraction_bits)
              | (rounded & ((1 << fraction_bits) - 1)))
    return result.to_bytes(width // 8, "little")


def decode_float(data: bytes, dtype: str) -> float:
    """Decode a little-endian floating value exactly into Python binary64."""
    exponent_bits, fraction_bits = _format(dtype)
    width = 1 + exponent_bits + fraction_bits
    if not isinstance(data, (bytes, bytearray)) or len(data) != width // 8:
        raise ValueError(f"{dtype} encoding must contain exactly {width // 8} bytes")
    bits = int.from_bytes(data, "little")
    sign = -1.0 if bits >> (width - 1) else 1.0
    exponent = (bits >> fraction_bits) & ((1 << exponent_bits) - 1)
    fraction = bits & ((1 << fraction_bits) - 1)
    bias = (1 << (exponent_bits - 1)) - 1
    if exponent == (1 << exponent_bits) - 1:
        return math.copysign(float("nan") if fraction else float("inf"), sign)
    if exponent == 0:
        return sign * math.ldexp(fraction, 1 - bias - fraction_bits)
    return sign * math.ldexp((1 << fraction_bits) | fraction, exponent - bias - fraction_bits)


def round_float(value: int | float, dtype: str) -> float:
    return decode_float(encode_float(value, dtype), dtype)


def encode_f32(value: int | float) -> bytes:
    """Compatibility wrapper for the original f32 conversion API."""
    return encode_float(value, "f32")


def round_f32(value: int | float) -> float:
    return round_float(value, "f32")
