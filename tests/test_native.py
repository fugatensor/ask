"""Exercise the native verifier, lowering and resulting C ABI as a client."""
import ctypes
import json
import math
from pathlib import Path
import shutil
import subprocess
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


if __name__ == "__main__":
    unittest.main()
