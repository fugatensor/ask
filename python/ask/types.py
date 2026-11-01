"""Small type descriptions used to print ASK's real MLIR input format."""

from __future__ import annotations

from dataclasses import dataclass

_DTYPES = frozenset({"bool", "i32", "i64", "f32", "f16", "bf16"})


def _dtype(value: str) -> str:
    value = "f16" if value == "fp16" else value
    if not isinstance(value, str) or value not in _DTYPES:
        raise ValueError(f"ASK supports bool, i32, i64, f32, f16 and bf16, got {value!r}")
    return value


@dataclass(frozen=True)
class ScalarType:
    dtype: str = "i32"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", _dtype(self.dtype))

    def __str__(self) -> str:
        return "i1" if self.dtype == "bool" else self.dtype


@dataclass(frozen=True)
class TensorType:
    shape: tuple[int | None, ...]
    dtype: str = "i32"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", _dtype(self.dtype))
        try:
            shape = tuple(self.shape)
        except TypeError as exc:
            raise TypeError("tensor shape must be an iterable of dimensions") from exc
        for dim in shape:
            if dim is not None and (type(dim) is not int or not 0 <= dim < 2**63):
                raise ValueError(f"invalid tensor dimension: {dim!r}")
        object.__setattr__(self, "shape", shape)

    def __str__(self) -> str:
        dims = "".join(("?" if dim is None else str(dim)) + "x" for dim in self.shape)
        element = "i1" if self.dtype == "bool" else self.dtype
        return f"tensor<{dims}{element}>"


Type = ScalarType | TensorType


def as_type(value: Type | str) -> Type:
    if isinstance(value, (ScalarType, TensorType)):
        return value
    if isinstance(value, str):
        return ScalarType(value)
    raise TypeError(f"expected ScalarType or TensorType, got {type(value).__name__}")


def compatible_shapes(actual: Type, declared: Type) -> bool:
    if type(actual) is not type(declared):
        return False
    if isinstance(actual, TensorType) and isinstance(declared, TensorType):
        return len(actual.shape) == len(declared.shape) and all(
            left is None or right is None or left == right
            for left, right in zip(actual.shape, declared.shape)
        )
    return True


def compatible(actual: Type, declared: Type) -> bool:
    return actual.dtype == declared.dtype and compatible_shapes(actual, declared)
