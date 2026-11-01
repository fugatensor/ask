"""One typed pointwise graph and an independent arithmetic oracle.

Tests and the GPU example share this builder. A new dtype selects types and
numeric data; it does not duplicate graph topology, operators or fusion steps.
The reference uses exact rational arithmetic, not ASK's codecs or LLVM output.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from fractions import Fraction
import math
import random
from typing import Any

from ask import Module, TensorType

DTYPES = ("bool", "i32", "i64", "f32", "f16", "bf16")
# fraction bits, minimum normal exponent, maximum normal exponent
FLOAT_FORMATS = {"f32": (23, -126, 127), "f16": (10, -14, 15), "bf16": (7, -126, 127)}

def build_module(size: int | None = None, *, shape: tuple[int, ...] | None = None,
                 block: int = 128, num_warps: int = 4, bias: int | float = 1,
                 factor: int | float = 2, pre_factor: int | float = 1,
                 post_factor: int | float = 1, aliased_dsl: bool = False,
                 dtype: str = "i32", offset: int | float = 3,
                 tail_offset: int | float | None = None,
                 external_offset: int | float | None = None,
                 enable_fp_fusion: bool | None = None, external: bool = True,
                 compute_dtype: str | None = None, roundtrip_dtype: str | None = None,
                 arch: int = 80) -> Module:
    """Build pointwise arithmetic over a concrete shape and declared F strategy.

    Runtime values are flat contiguous lists; shape remains in ASK's logical
    types and ShapeSnapshot. BLOCK counts elements, not CUDA threads.
    """
    if shape is None:
        if type(size) is not int or size < 0 or (external and size == 0):
            raise ValueError("provide a concrete size; external graphs require positive dimensions")
        shape = (size,)
    if any(type(dimension) is not int or dimension < 0 or (external and dimension == 0)
           for dimension in shape):
        raise ValueError("shape dimensions must be nonnegative integers; external graphs require positive dimensions")
    elements = math.prod(shape)
    if size is not None and size != elements:
        raise ValueError("size must equal the product of shape dimensions")
    size = elements
    dtype = TensorType(shape, dtype).dtype
    compute_dtype = compute_dtype or ("i32" if dtype == "bool" else dtype)
    compute_dtype = TensorType(shape, compute_dtype).dtype
    triton_types = {"bool": "int1", "i32": "int32", "i64": "int64", "f32": "float32",
                    "f16": "float16", "bf16": "bfloat16"}
    # F has the same declared storage ABI as the surrounding graph. Its own
    # widening/rounding is explicit in the DSL and remains the producer's policy.
    working_value = (f"value.to(tl.{triton_types[compute_dtype]})"
                     if compute_dtype != dtype else "value")
    expression = f"{working_value} * {factor}"
    if external_offset is not None:
        expression += f" + {external_offset}"
    if compute_dtype != dtype:
        expression = f"({expression}).to(value.dtype)"
    source = f'''import triton
import triton.language as tl

@triton.jit
def scale(t, u, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(t + i, mask=i < N, other=0)
    tl.store(u + i, {expression}, mask=i < N)
'''
    if aliased_dsl:
        source = f'''import triton
import triton.language as tl

@triton.jit
def scale(t, u, N: tl.constexpr, BLOCK: tl.constexpr):
    tile = BLOCK * tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    offsets = lane + tile
    active = offsets < N
    input_address = t + offsets
    output_address = offsets + u
    value = tl.load(input_address, active, other=0)
    scaled = {expression}
    tl.store(output_address, scaled, active)
'''
    module = Module()
    tensor = TensorType(shape, dtype)
    if external:
        dynamic = TensorType((None,) * len(shape), dtype)
        semantics = module.semantics(f"scale_{dtype}_v1", inputs=[dynamic], results=[dynamic])
        signature = {"bool": "*i1", "i32": "*i32", "i64": "*i64", "f32": "*fp32",
                     "f16": "*fp16", "bf16": "*bf16"}[dtype]
        options = {"num_warps": num_warps}
        if enable_fp_fusion is not None:
            options["enable_fp_fusion"] = enable_fp_fusion
        implementation = module.implementation(
            "triton_scale", compiler="triton", source=source, entry="scale",
            signature={"t": signature, "u": signature},
            source_bindings={"t": "input:0", "u": "output:0"},
            constants={"N": size, "BLOCK": block}, options=options,
            target={"backend": "cuda", "arch": arch, "warp_size": 32},
            grid=(max(1, (size + block - 1) // block), 1, 1))
    graph = module.graph("pipeline", inputs={"x": tensor}, results=[tensor])
    incoming = graph.inputs["x"]
    if roundtrip_dtype:
        incoming = graph.cast(graph.cast(incoming, dtype=roundtrip_dtype, id="cast_out"),
                              dtype=dtype, id="cast_back")
    incoming = graph.cast(incoming, dtype=compute_dtype, id="P_compute")
    increment = graph.splat(graph.constant(bias, dtype=compute_dtype, id="bias"), shape, id="bias_tensor")
    before = graph.add(incoming, increment, id="P" if pre_factor == 1 else "P_add",
                       value_id="v_t" if pre_factor == 1 else "v_pre_add")
    if pre_factor != 1:
        multiplier = graph.splat(graph.constant(pre_factor, dtype=compute_dtype, id="pre_factor"), shape,
                                  id="pre_factor_tensor")
        before = graph.mul(before, multiplier, id="P", value_id="v_t")
    before = graph.cast(before, dtype=dtype, id="P_storage")
    if external:
        before = graph.mark(before, mark_id="before_F", role="input", port_id="F.in")
        external_value = graph.external(before, semantics=semantics, implementation=implementation,
                                  id="F", value_id="v_u")
        external_value = graph.mark(external_value, mark_id="after_F", role="output", port_id="F.out")
    else:
        before = graph.cast(before, dtype=compute_dtype, id="F_compute")
        scale = graph.splat(graph.constant(factor, dtype=compute_dtype, id="factor"), shape)
        external_value = graph.mul(before, scale, id="F")
        if external_offset is not None:
            shift = graph.splat(graph.constant(external_offset, dtype=compute_dtype, id="F_offset"), shape)
            external_value = graph.add(external_value, shift, id="F_add")
        external_value = graph.cast(external_value, dtype=dtype, id="F_storage")
    external_value = graph.cast(external_value, dtype=compute_dtype, id="Q_compute")
    offset_value = graph.splat(graph.constant(offset, dtype=compute_dtype, id="offset"), shape, id="offset_tensor")
    result = graph.add(external_value, offset_value, id="Q" if post_factor == 1 else "Q_add",
                       value_id="v_y" if post_factor == 1 else "v_post_add")
    if post_factor != 1:
        multiplier = graph.splat(graph.constant(post_factor, dtype=compute_dtype, id="post_factor"), shape,
                                  id="post_factor_tensor")
        result = graph.mul(result, multiplier, id="Q", value_id="v_y")
    if tail_offset is not None:
        tail = graph.splat(graph.constant(tail_offset, dtype=compute_dtype, id="tail"), shape, id="tail_tensor")
        result = graph.add(result, tail, id="Q_final", value_id="v_final")
    result = graph.cast(result, dtype=dtype, id="Q_storage")
    graph.return_(result)
    return module


def round_reference(value: int | float | Fraction, dtype: str, *, negative_zero: bool = False):
    """Round an exact rational directly to a format, allowing arithmetic overflow.

    This oracle quantizes at the exponent-dependent spacing with integer division.
    It does not share production encode/decode code, host half arithmetic or LLVM.
    """
    if dtype == "bool":
        return bool(value)
    if dtype in ("i32", "i64"):
        width = int(dtype[1:])
        integer = int(value) % (1 << width)
        return integer - (1 << width) if integer >= 1 << (width - 1) else integer
    if isinstance(value, float):
        if not math.isfinite(value):
            return value
        if value == 0:
            negative_zero = math.copysign(1.0, value) < 0
    exact = Fraction(value)
    if exact == 0:
        return -0.0 if negative_zero else 0.0
    negative = exact < 0
    exact = abs(exact)
    numerator, denominator = exact.numerator, exact.denominator
    fraction_bits, minimum, maximum = FLOAT_FORMATS[dtype]
    exponent = numerator.bit_length() - denominator.bit_length()
    unit = Fraction(2)**exponent
    if exact < unit:
        exponent -= 1
    spacing_exponent = max(exponent, minimum) - fraction_bits
    units = exact / (Fraction(2)**spacing_exponent)
    significand, remainder = divmod(units.numerator, units.denominator)
    twice = remainder * 2
    if twice > units.denominator or (twice == units.denominator and significand & 1):
        significand += 1
    rounded = Fraction(significand) * (Fraction(2)**spacing_exponent)
    largest = (Fraction(2) - Fraction(2)**-fraction_bits) * (Fraction(2)**maximum)
    if rounded > largest:
        result = float("inf")
    else:
        result = float(rounded)
    return -result if negative else result


def arithmetic_reference(kind: str, left, right, dtype: str):
    if dtype in ("i32", "i64"):
        return round_reference(left + right if kind == "add" else left * right, dtype)
    if not math.isfinite(left) or not math.isfinite(right):
        return left + right if kind == "add" else left * right
    a, b = Fraction(left), Fraction(right)
    if kind == "add":
        negative_zero = left == right == 0 and math.copysign(1, left) < 0 and math.copysign(1, right) < 0
        return round_reference(a + b, dtype, negative_zero=negative_zero)
    negative_zero = math.copysign(1, left) != math.copysign(1, right)
    return round_reference(a * b, dtype, negative_zero=negative_zero)


def fma_reference(left, right, addend, dtype):
    """Single-rounding reference for F, including subnormal and overflow cases."""
    if any(math.isnan(v) for v in (left, right, addend)):
        return float("nan")
    if not all(math.isfinite(v) for v in (left, right, addend)):
        return left * right + addend
    product_negative = math.copysign(1, left) != math.copysign(1, right)
    negative_zero = (left == 0 or right == 0) and addend == 0 and product_negative and math.copysign(1, addend) < 0
    return round_reference(Fraction(left) * Fraction(right) + Fraction(addend), dtype,
                           negative_zero=negative_zero)


def input_values(size: int, dtype: str = "i32") -> list[int | float]:
    randomizer = random.Random(7357 + size)
    if dtype == "bool":
        return [bool(index % 2) for index in range(size)]
    if dtype in ("i32", "i64"):
        width = int(dtype[1:])
        edges = [2**(width-1)-1, -(2**(width-1)), -1, 0, 1, 2**(width-2), -(2**(width-2))]
        if dtype == "i64":
            # Low bits below f32/binary64 precision decide which side of a
            # bf16 midpoint the integer occupies. Keep them as Python integers.
            edges += [2**40 + 2**32 + 1, -(2**40 + 2**32 + 1),
                      2**60 + 2**52 + 1, -(2**60 + 2**52 + 1)]
        return edges[:size] + [randomizer.randint(-(2**(width-1)), 2**(width-1) - 1)
                              for _ in range(max(0, size - len(edges)))]
    fraction, minimum, maximum = FLOAT_FORMATS[dtype]
    smallest = 2.0**(minimum - fraction)
    largest = (2.0 - 2.0**-fraction) * 2.0**maximum
    edges = [1.0 + 2.0**-fraction, -1.25, 0.1, 0.0, -0.0, smallest, -smallest,
             float("inf"), float("-inf"), float("nan"), largest, -largest,
             1.0 + 2.0**(-fraction - 1), 1.0 + 3 * 2.0**(-fraction - 1)]
    return edges[:size] + [randomizer.uniform(-1000, 1000)
                          for _ in range(max(0, size - len(edges)))]


def cast_reference(value, source, target):
    """Reference ASK casts, distinct from raw input encoding and F's policies."""
    if target == "bool":
        return bool(value)  # NaN is nonzero; either sign of zero is false.
    if target in ("i32", "i64") and source in FLOAT_FORMATS:
        width = int(target[1:])
        lower, upper = -(1 << (width - 1)), (1 << (width - 1)) - 1
        if math.isnan(value):
            return 0
        if value <= lower:
            return lower
        if value >= upper:
            return upper
        return math.trunc(value)
    return round_reference(value, target)


def expected_values(values, bias, factor, *, pre_factor=1, post_factor=1, dtype="i32",
                    offset=3, tail_offset=None, external_offset=None,
                    enable_fp_fusion=None, external=True, compute_dtype=None, roundtrip_dtype=None):
    compute_dtype = compute_dtype or ("i32" if dtype == "bool" else dtype)
    convert = lambda value: round_reference(value, compute_dtype)
    add = lambda left, right: arithmetic_reference("add", left, right, compute_dtype)
    mul = lambda left, right: arithmetic_reference("mul", left, right, compute_dtype)
    boundary = lambda value: cast_reference(cast_reference(value, compute_dtype, dtype), dtype, compute_dtype)
    result = []
    for value in values:
        value = round_reference(value, dtype)
        if roundtrip_dtype:
            value = cast_reference(cast_reference(value, dtype, roundtrip_dtype), roundtrip_dtype, dtype)
        before = add(cast_reference(value, dtype, compute_dtype), convert(bias))
        if pre_factor != 1:
            before = mul(before, convert(pre_factor))
        before = boundary(before)
        if compute_dtype in FLOAT_FORMATS and external and external_offset is not None and enable_fp_fusion is not False:
            middle = fma_reference(before, convert(factor), convert(external_offset), compute_dtype)
        else:
            middle = mul(before, convert(factor))
            if external_offset is not None:
                middle = add(middle, convert(external_offset))
        after = add(boundary(middle), convert(offset))
        if post_factor != 1:
            after = mul(after, convert(post_factor))
        if tail_offset is not None:
            after = add(after, convert(tail_offset))
        result.append(cast_reference(after, compute_dtype, dtype))
    return result


def numerically_equal(actual, expected, dtype):
    if len(actual) != len(expected):
        return False
    if dtype in ("bool", "i32", "i64"):
        return actual == expected
    # Every representable f16/bf16/f32 finite value is exact in Python binary64.
    # Equality plus zero sign therefore checks the full format's value bits.
    return all((math.isnan(left) and math.isnan(right)) or
               (left == right and (left != 0 or math.copysign(1, left) == math.copysign(1, right)))
               for left, right in zip(actual, expected))


def json_values(values):
    return [value if not isinstance(value, float) or math.isfinite(value)
            else "NaN" if math.isnan(value) else "+Inf" if value > 0 else "-Inf" for value in values]


@dataclass(frozen=True)
class GraphCase:
    name: str
    options: dict[str, Any] = field(default_factory=dict)
    floating: bool = False
    extended: bool = False
    numeric_variant: str | None = None
    storage_types: tuple[str, ...] | None = None

    def for_dtype(self, dtype: str):
        if self.storage_types is not None and dtype not in self.storage_types:
            return None
        if self.floating and dtype not in FLOAT_FORMATS:
            return None
        options = dict(self.options)
        if self.numeric_variant:
            precision = FLOAT_FORMATS[dtype][0]
            step = 2.0**-precision
            if self.numeric_variant == "mixed_accumulate":
                # (x + 2**precision) - 2**precision loses x in low precision,
                # but preserves these values when evaluated as f32 first.
                options.update(offset=2**precision, tail_offset=-(2**precision))
            elif self.numeric_variant == "ask_rounding":
                options.update(post_factor=1 + step, tail_offset=-(1 + 2 * step))
            else:
                options.update(bias=-2 * step, factor=1 + step)
        return options


# Each topology/configuration is declared once. Dtype selects types and the
# precision-specific ULP used for rounding tests; no graph is copied per format.
GRAPH_CASES = (
    GraphCase("single", {"size": 1}),
    GraphCase("same-shape", {"size": 10}),
    GraphCase("tail", {"size": 129}),
    GraphCase("shape-3x5", {"shape": (3, 5), "block": 64}),
    GraphCase("multiple-sites", {"shape": (3, 171), "block": 256, "aliased_dsl": True}),
    GraphCase("expression-chain", {"size": 17, "pre_factor": -3, "post_factor": 5}),
    GraphCase("overflow", {"size": 17, "bias": 0, "factor": 1, "pre_factor": 2, "offset": 0}),
    GraphCase("fractions", {"size": 129, "bias": 0.1, "factor": 0.75,
                           "pre_factor": -1.25, "post_factor": 0.1, "offset": -0.3}, floating=True),
    GraphCase("special", {"size": 17, "bias": -0.0, "factor": 1.0, "offset": -0.0}, floating=True),
    GraphCase("ask-rounding", {"size": 1, "bias": 0.0, "factor": 1.0, "offset": 0.0},
              floating=True, numeric_variant="ask_rounding"),
    GraphCase("F-contract-enabled", {"size": 1, "external_offset": -1.0, "offset": -0.0,
                                    "enable_fp_fusion": True}, floating=True, numeric_variant="F_rounding"),
    GraphCase("F-contract-disabled", {"size": 1, "external_offset": -1.0, "offset": -0.0,
                                     "enable_fp_fusion": False}, floating=True, numeric_variant="F_rounding"),
    GraphCase("mixed-accumulate", {"size": 129, "bias": 0.0, "factor": 1.0, "compute_dtype": "f32"},
              floating=True, numeric_variant="mixed_accumulate"),
    GraphCase("mixed-special", {"size": 17, "bias": -0.0, "factor": 1.0, "offset": -0.0,
                               "compute_dtype": "f32"}, floating=True),
    GraphCase("mixed-F-contract-enabled", {"size": 17, "external_offset": -1.0, "offset": -0.0,
              "compute_dtype": "f32", "enable_fp_fusion": True}, floating=True, numeric_variant="F_rounding"),
    GraphCase("mixed-F-contract-disabled", {"size": 17, "external_offset": -1.0, "offset": -0.0,
              "compute_dtype": "f32", "enable_fp_fusion": False}, floating=True, numeric_variant="F_rounding"),
    *(GraphCase("cast-" + destination, {"size": 129, "bias": 0, "factor": 1, "offset": 0,
                                       "roundtrip_dtype": destination}) for destination in DTYPES),
    GraphCase("block-end", {"size": 127}, extended=True),
    GraphCase("whole-block", {"size": 128}, extended=True),
    GraphCase("multi-block", {"size": 4097}, extended=True),
    GraphCase("shape-2x3x17", {"shape": (2, 3, 17), "num_warps": 8}, extended=True),
    GraphCase("wide-block", {"size": 513, "block": 512, "num_warps": 8}, extended=True),
    # Compare each against same-shape: a single changed input to compilation
    # must produce a cold miss, followed by a hit for that exact new request.
    GraphCase("changed-ask-constant", {"size": 10, "bias": 2}, extended=True),
    GraphCase("changed-F-DSL", {"size": 10, "factor": 3}, extended=True),
    GraphCase("changed-shape", {"size": 11}, extended=True),
    GraphCase("changed-compute-type", {"size": 10, "compute_dtype": "f32"}, extended=True,
              storage_types=("f16", "bf16")),
    GraphCase("mixed-underflow", {"size": 17, "bias": -0.0, "factor": 2.0,
                                 "pre_factor": 2.0, "post_factor": 0.5, "offset": -0.0,
                                 "compute_dtype": "f32"}, extended=True, floating=True),
)
