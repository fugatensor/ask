"""Execute compiler-independent LLVM fusion output through the host backend."""
import ctypes
import math
from pathlib import Path
import subprocess
import sys
import tempfile

test_binary, llvm_tools = sys.argv[1:]
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "tests")]
from ask.numerics import encode_float, decode_float
from support.pointwise import DTYPES, FLOAT_FORMATS, input_values, cast_reference, round_reference, numerically_equal

STORAGE = {"bool": ctypes.c_uint8, "i32": ctypes.c_int32, "i64": ctypes.c_int64,
           "f32": ctypes.c_float, "f16": ctypes.c_uint16, "bf16": ctypes.c_uint16}


def encode(value, dtype):
    if dtype in ("f16", "bf16"):
        return int.from_bytes(encode_float(value, dtype), "little")
    return value


def decode(value, dtype):
    if dtype in ("f16", "bf16"):
        return decode_float(value.to_bytes(2, "little"), dtype)
    return value


def checked(arguments):
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(f"{arguments}: {result.stdout}\n{result.stderr}")
    return result


with tempfile.TemporaryDirectory(prefix="ask-generic-inline-") as directory:
    work = Path(directory)
    checked([test_binary, str(work / "fused.ll")])
    checked([str(Path(llvm_tools) / "llc"), "-relocation-model=pic", "-filetype=obj",
             str(work / "fused.ll"), "-o", str(work / "fused.o")])
    checked(["cc", "-shared", str(work / "fused.o"), "-o", str(work / "fused.so")])
    library = ctypes.CDLL(str(work / "fused.so"))
    library.twice_i64.argtypes = [ctypes.c_int64]
    library.twice_i64.restype = ctypes.c_int64
    for value in (0, -123, 2**63 - 1, -(2**63), 99999999999):
        assert library.twice_i64(value) == ctypes.c_int64(value + 14).value
    library.multiple_arguments.argtypes = [ctypes.c_int32, ctypes.c_int32]
    library.multiple_arguments.restype = ctypes.c_int32
    for x, y in ((1, 2), (-9, 7), (2**31 - 1, 2**30)):
        assert library.multiple_arguments(x, y) == ctypes.c_int32(x + 2*y).value
    for name, ctype in (("scalar_f32", ctypes.c_float), ("scalar_f64", ctypes.c_double)):
        function = getattr(library, name)
        function.argtypes = [ctype]
        function.restype = ctype
        for value in (-4.25, 0.0, 1.5, 32.0, math.inf, -math.inf, math.nan):
            result = function(value)
            expected = ctype(ctype(value * 2).value + 3).value
            assert math.isnan(result) if math.isnan(expected) else result == expected
    library.intrinsic_abs.argtypes = [ctypes.c_float]
    library.intrinsic_abs.restype = ctypes.c_float
    for value in (-4.25, -math.inf, 1.5, math.nan):
        result = library.intrinsic_abs(value)
        assert math.isnan(result) if math.isnan(value) else result == abs(value)
    library.guarded_i64.argtypes = [ctypes.c_int64, ctypes.c_int32]
    library.guarded_i64.restype = ctypes.c_int64
    for value in (-123, 0, 5):
        assert library.guarded_i64(value, 1) == abs(value + 1)
        assert library.guarded_i64(value, 0) == value
    # The inactive helper contains nsw arithmetic feeding a branch. Its poison
    # must never be evaluated as a branch condition when the boundary is masked.
    assert library.guarded_i64(2**63 - 1, 0) == 2**63 - 1
    vector = ctypes.c_float * 4
    library.masked_vector.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    library.masked_vector.restype = None
    values, output = vector(-3.0, 4.0, 1.5, -9.0), vector()
    library.masked_vector(values, output)
    assert list(output) == [2.0, 4.0, 6.5, -9.0]
    cast_checks = 0
    for source in DTYPES:
        values = input_values(24, source)
        if source == "i64":
            # Integer -> bf16 halfway cases whose low bits disappear in f32.
            values += [(2**40 + 2**32 + delta) * sign for delta in (-1, 0, 1)
                       for sign in (-1, 1)]
        if source == "f32":
            values += [2**-134, 3 * 2**-134, 2**-133, 2**-24, 2**-25,
                       65504.0, 65520.0, 2**31, -(2**31), 2**63, -(2**63)]
        values = [round_reference(value, source) for value in values]
        for target in DTYPES:
            function = getattr(library, f"cast_{source}_{target}")
            function.argtypes = [ctypes.POINTER(STORAGE[source]), ctypes.POINTER(STORAGE[target])]
            function.restype = None
            for value in values:
                input_buffer = STORAGE[source](encode(value, source))
                output_buffer = STORAGE[target]()
                function(ctypes.byref(input_buffer), ctypes.byref(output_buffer))
                actual = decode(output_buffer.value, target)
                expected = cast_reference(value, source, target)
                assert numerically_equal([actual], [expected], target), (source, target, value, actual, expected)
                cast_checks += 1
    print(f"Executed ordinary LLVM scalar/vector fusion and {cast_checks} values across all 36 cast pairs")
