"""Exercise the native verifier, lowering and resulting C ABI as a client."""
import ctypes
import json
import math
from pathlib import Path
import shutil
import subprocess
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = str(Path(sys.argv.pop(1)).resolve())
sys.path.insert(0, str(ROOT / "python"))
from ask import Module, ScalarType, TensorType


def run(*args, source=None):
    return subprocess.run([DRIVER, *map(str, args)], input=source, text=True,
                          capture_output=True, timeout=60)


class NativeTests(unittest.TestCase):
    def test_positive_fixtures(self):
        for name in ("pipeline", "refine-and-erase"):
            with self.subTest(name=name):
                result = run(ROOT / "tests/ir" / (name + ".mlir"), "--prepare")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn('"ask.mark"', result.stdout)
                repeated = run("-", "--prepare", source=result.stdout)
                self.assertEqual(repeated.returncode, 0, repeated.stderr)
                self.assertEqual(repeated.stdout, result.stdout)

    def test_negative_fixtures_have_diagnostics(self):
        for path in sorted((ROOT / "tests/ir").glob("*.mlir")):
            if path.stem in ("pipeline", "refine-and-erase"):
                continue
            with self.subTest(name=path.name):
                result = run(path, "--prepare")
                self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("error", result.stderr.lower())

    def test_native_llvm_import_preserves_inline_ptx(self):
        source = '''target triple = "nvptx64-nvidia-cuda"
define void @f() {
  call void asm sideeffect "bar.sync 0;", ""()
  ret void
}
'''
        result = run("-", "--check-llir", source=source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('asm sideeffect "bar.sync 0;"', result.stdout)
        invalid = run("-", "--check-llir", source="define void @f() { ret i32 1 }")
        self.assertGreater(invalid.returncode, 0)
        self.assertIn("reader", invalid.stderr)

    def test_external_request_preserves_boolean_options(self):
        source = (ROOT / "tests/ir/pipeline.mlir").read_text().replace(
            'options = {num_warps = 4 : i64}',
            'options = {num_warps = 4 : i64, enable_fp_fusion = true}')
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            result = run("-", "--prepare", "--report-json", report, source=source)
            self.assertEqual(result.returncode, 0, result.stderr)
            options = json.loads(report.read_text())["external_requests"][0]["fixed_options"]
            self.assertIs(options["enable_fp_fusion"], True)

    def test_external_is_not_silently_lowered_as_semantic_identity(self):
        result = run(ROOT / "tests/ir/pipeline.mlir", "--emit-llvm")
        self.assertGreater(result.returncode, 0)
        self.assertIn("external", result.stderr)

    def test_disabled_exporter_rejected_before_artifacts(self):
        caps = json.loads(run("--capabilities").stdout)
        if "triton" in caps["third_party_compilers"]:
            self.skipTest("this build explicitly enables Triton")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "exports"
            result = run(ROOT / "tests/ir/pipeline.mlir", "--export-external",
                         "--export-dir", destination)
            self.assertGreater(result.returncode, 0)
            self.assertIn("not enabled", result.stderr)
            self.assertFalse(destination.exists())

    def test_host_objects_execute_correctly(self):
        if not json.loads(run("--capabilities").stdout)["host_codegen"]:
            self.skipTest("host backend disabled")
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "C compiler is required to link the emitted object")
        module = Module()
        shapes = [(10,), (2, 3), (), (0,)]
        for index, shape in enumerate(shapes):
            tensor = TensorType(shape)
            graph = module.graph(f"calculate{index}", inputs={"x": tensor, "s": ScalarType()},
                                 results=[tensor, ScalarType(), tensor])
            x, scalar = graph.inputs["x"], graph.inputs["s"]
            biased = graph.add(x, graph.splat(scalar, shape))
            marked = graph.mark(biased, mark_id=f"trace{index}")
            doubled = graph.mul(marked, graph.splat(graph.constant(2), shape))
            scalar_result = graph.add(scalar, graph.constant(1))
            graph.return_(doubled, scalar_result, x)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            obj, shared, report = path / "graph.o", path / "graph.so", path / "abi.json"
            result = run("-", "--emit-object", "-o", obj, "--report-json", report,
                         source=module.to_mlir())
            self.assertEqual(result.returncode, 0, result.stderr)
            linked = subprocess.run([cc, "-shared", str(obj), "-o", str(shared)],
                                    text=True, capture_output=True)
            self.assertEqual(linked.returncode, 0, linked.stderr)
            metadata = json.loads(report.read_text())
            self.assertEqual(metadata["phase"], "host_object")
            self.assertEqual(len(metadata["abi"]["functions"]), len(shapes))
            library = ctypes.CDLL(str(shared))
            pointer = ctypes.POINTER(ctypes.c_int32)
            for index, shape in enumerate(shapes):
                count = math.prod(shape)
                data = [2147483647, -2147483648, -7, 0, 1, 7, 42, 1024, -1024, 9][:count]
                x = (ctypes.c_int32 * max(count, 1))(*data)
                out = (ctypes.c_int32 * max(count, 1))(*([123] * max(count, 1)))
                copy = (ctypes.c_int32 * max(count, 1))(*([456] * max(count, 1)))
                scalar = ctypes.c_int32()
                function = getattr(library, f"calculate{index}")
                function.argtypes = [pointer, ctypes.c_int32, pointer, pointer, pointer]
                function.restype = None
                function(x, 2147483647, out, ctypes.byref(scalar), copy)
                self.assertEqual(list(out)[:count], [ctypes.c_int32((v + 2147483647) * 2).value for v in data])
                self.assertEqual(list(copy)[:count], data)
                self.assertEqual(scalar.value, -2147483648)
                if count == 0:
                    self.assertEqual(out[0], 123)
                    self.assertEqual(copy[0], 456)

    def test_float_host_abi_preserves_rounding_and_special_values(self):
        if not json.loads(run("--capabilities").stdout)["host_codegen"]:
            self.skipTest("host backend disabled")
        cc = shutil.which("cc")
        self.assertIsNotNone(cc)
        module = Module()
        tensor, scalar_type = TensorType((2, 4), "f32"), ScalarType("f32")
        graph = module.graph("calculate", inputs={"x": tensor, "a": scalar_type},
                             results=[tensor, tensor, scalar_type])
        product = graph.mul(graph.inputs["x"], graph.splat(graph.inputs["a"], (2, 4)))
        adjusted = graph.add(product, graph.splat(graph.constant(-1.0, dtype="f32"), (2, 4)))
        scalar = graph.add(graph.inputs["a"], graph.constant(-1.0, dtype="f32"))
        graph.return_(product, adjusted, scalar)
        source = module.to_mlir()
        llvm_result = run("-", "--emit-llvm", source=source)
        self.assertEqual(llvm_result.returncode, 0, llvm_result.stderr)
        self.assertIn("fadd float", llvm_result.stdout)
        self.assertIn("fmul float", llvm_result.stdout)
        self.assertNotIn("fadd fast", llvm_result.stdout)
        self.assertNotIn("fmul contract", llvm_result.stdout)
        data = [1.0 + 2 ** -23, -0.0, float("inf"), float("-inf"),
                float("nan"), 2 ** -149, -2 ** -149, 0.0]
        coefficient = 1.0 - 2 ** -23
        expected_products = [ctypes.c_float(value * coefficient).value for value in data]
        expected_outputs = [ctypes.c_float(value - 1.0).value for value in expected_products]
        self.assertEqual(expected_outputs[0], 0.0)  # Contracted FMA would differ.
        with tempfile.TemporaryDirectory() as directory:
            for optimized in (False, True):
                path = Path(directory)
                obj = path / f"float{optimized}.o"
                shared = path / f"float{optimized}.so"
                report = path / f"float{optimized}.json"
                args = ["-", "--emit-object", "-o", obj, "--report-json", report]
                if optimized:
                    args.append("--optimize")
                result = run(*args, source=source)
                self.assertEqual(result.returncode, 0, result.stderr)
                metadata = json.loads(report.read_text())
                abi = metadata["abi"]["functions"][0]
                self.assertTrue(all(item["dtype"] == "f32" for item in abi["inputs"] + abi["outputs"]))
                self.assertEqual(abi["lowering"]["loop_count"], 2 if optimized else 3)
                linked = subprocess.run([cc, "-shared", str(obj), "-o", str(shared)],
                                        text=True, capture_output=True)
                self.assertEqual(linked.returncode, 0, linked.stderr)
                function = ctypes.CDLL(str(shared)).calculate
                pointer = ctypes.POINTER(ctypes.c_float)
                function.argtypes = [pointer, ctypes.c_float, pointer, pointer, pointer]
                function.restype = None
                storage = ctypes.c_float * 8
                x, product_out, result_out, scalar_out = storage(*data), storage(), storage(), ctypes.c_float()
                function(x, coefficient, product_out, result_out, ctypes.byref(scalar_out))
                for actual, expected in zip(list(product_out) + list(result_out),
                                            expected_products + expected_outputs):
                    if math.isnan(expected):
                        self.assertTrue(math.isnan(actual))
                    else:
                        self.assertEqual(struct.pack("f", actual), struct.pack("f", expected))
                self.assertEqual(scalar_out.value, -2 ** -23)

    def test_float_type_mismatch_and_integer_only_options_are_rejected(self):
        source = '''module {
  func.func @f(%x: f32, %y: f32) -> f32 {
    %z = "ask.add"(%x, %y) : (f32, f32) -> f32
    return %z : f32
  }
}'''
        for attribute in ('overflow = "wrap"', 'fastmath = "fast"', 'fastmathFlags = "contract"'):
            invalid = source.replace('"ask.add"(%x, %y)', f'"ask.add"(%x, %y) {{{attribute}}}')
            with self.subTest(attribute=attribute):
                result = run("-", "--prepare", source=invalid)
                self.assertGreater(result.returncode, 0)
        splat = '''module {
  func.func @f(%x: f32) -> tensor<2xi32> {
    %z = "ask.splat"(%x) {shape = array<i64: 2>} : (f32) -> tensor<2xi32>
    return %z : tensor<2xi32>
  }
}'''
        result = run("-", "--prepare", source=splat)
        self.assertGreater(result.returncode, 0)
        self.assertIn("element types must match", result.stderr)


if __name__ == "__main__":
    unittest.main()
