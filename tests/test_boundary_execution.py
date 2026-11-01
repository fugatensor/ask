"""Execute compiler-independent LLVM fusion output through the host backend."""
import ctypes
import math
from pathlib import Path
import subprocess
import sys
import tempfile

test_binary, llvm_tools = sys.argv[1:]


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
    print("Executed ordinary LLVM i64/i32/f32/f64/vector fusion with multiple sites, scalar and lane masks")
