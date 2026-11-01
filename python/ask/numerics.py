"""Numeric boundary conversions shared by the IR builder and runtime."""

from __future__ import annotations

import struct


def encode_f32(value: int | float) -> bytes:
    """Round once to IEEE binary32, returning four little-endian bytes.

    Python integers are rounded directly, without an intermediate binary64
    conversion. Python floats already represent binary64 values. Both paths
    round to nearest with ties to even. Explicit infinities and NaNs are valid;
    finite values that would round to infinity raise ValueError.
    """
    if type(value) is float:
        try:
            return struct.pack("<f", value)
        except (OverflowError, struct.error) as exc:
            raise ValueError("finite value is outside the f32 range") from exc
    if type(value) is not int:
        raise ValueError("f32 requires a Python int or float scalar")
    if value == 0:
        return bytes(4)
    magnitude = abs(value)
    exponent = magnitude.bit_length() - 1
    if exponent > 127:
        raise ValueError("finite value is outside the f32 range")
    shift = exponent - 23
    if shift <= 0:
        significand = magnitude << -shift
    else:
        significand, remainder = divmod(magnitude, 1 << shift)
        halfway = 1 << (shift - 1)
        if remainder > halfway or (remainder == halfway and significand & 1):
            significand += 1
        if significand == 1 << 24:
            significand >>= 1
            exponent += 1
            if exponent > 127:
                raise ValueError("finite value is outside the f32 range")
    bits = ((int(value < 0) << 31) | ((exponent + 127) << 23)
            | (significand & 0x7FFFFF))
    return bits.to_bytes(4, "little")


def round_f32(value: int | float) -> float:
    """Return the encoded binary32 value as an exactly representable float."""
    return struct.unpack("<f", encode_f32(value))[0]
