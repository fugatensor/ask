"""Verify native CUDA graph lowering without requiring a CUDA device or Triton."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = str(Path(sys.argv.pop(1)).resolve())
sys.path.insert(0, str(ROOT / "python"))
from ask import Module, ScalarType, TensorType
from support.pointwise import build_module, DTYPES


def run(source, *flags):
    return subprocess.run([DRIVER, "-", *map(str, flags)], input=source,
                          text=True, capture_output=True, timeout=60)


class CUDALoweringTests(unittest.TestCase):
    def lower(self, module, *, arch=86, optimize=True):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            flags = ["--emit-cuda-llvm", f"--cuda-arch={arch}", "--report-json", report]
            if optimize:
                flags.append("--optimize")
            result = run(module.to_mlir(), *flags)
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads(report.read_text())
        artifact = metadata["cuda_artifact"]
        self.assertEqual(metadata["phase"], "cuda_llvm")
        self.assertEqual(artifact["payload"], result.stdout)
        self.assertEqual(artifact["invocation"]["version"], 2)
        self.assertEqual(artifact["producer"]["name"], "ASK")
        self.assertEqual(metadata["external_requests"], [])
        self.assertIn("!ask.shape_snapshot", result.stdout)
        verified = run(result.stdout, "--check-llir")
        self.assertEqual(verified.returncode, 0, verified.stderr)
        return artifact, metadata

    def test_all_precisions_use_same_graph_and_native_kernel(self):
        for dtype in DTYPES:
            with self.subTest(dtype=dtype):
                artifact, metadata = self.lower(build_module(129, dtype=dtype, external=False))
                self.assertIn("define ptx_kernel void @pipeline", artifact["payload"])
                self.assertIn("llvm.nvvm.read.ptx.sreg.tid.x", artifact["payload"])
                self.assertEqual(artifact["invocation"]["grid"], [2, 1, 1])
                self.assertEqual(artifact["invocation"]["block"], [128, 1, 1])
                self.assertIn(dtype, artifact["required_element_types"])
                self.assertEqual(metadata["cuda_lowering"]["temporary_buffer_count"], 0)

    def test_multiple_ports_scalars_and_independent_output_shapes(self):
        module = Module()
        vector, other = TensorType((129,), "f32"), TensorType((3,), "i64")
        graph = module.graph("calculate", inputs={"x": vector, "s": ScalarType("f32"), "y": other},
                             results=[vector, TensorType((129,), "i64"), other, ScalarType("f32")])
        value = graph.add(graph.inputs["x"], graph.splat(graph.inputs["s"], (129,)))
        graph.return_(value, graph.cast(value, dtype="i64"), graph.inputs["y"], graph.inputs["s"])
        artifact, report = self.lower(module)
        self.assertEqual([port["logical_port"] for port in artifact["port_map"]],
                         ["input:0", "input:1", "input:2", "output:0", "output:1", "output:2", "output:3"])
        self.assertEqual(artifact["port_map"][1]["shape"], [])
        self.assertEqual(report["cuda_lowering"]["output_loop_count"], 3)
        self.assertEqual(report["cuda_lowering"]["groups"][0]["outputs"], [0, 1])
        self.assertTrue(artifact["requires_disjoint_outputs"])
        self.assertTrue(artifact["outputs_must_not_alias_inputs"])

    def test_empty_graph_has_no_buffer_access(self):
        artifact, report = self.lower(build_module(shape=(0, 3), dtype="f16", external=False))
        self.assertNotIn(" load ", artifact["payload"])
        self.assertNotIn(" store ", artifact["payload"])
        self.assertNotIn("getelementptr", artifact["payload"])
        self.assertEqual(artifact["invocation"]["grid"], [1, 1, 1])
        self.assertEqual(report["cuda_lowering"]["zero_element_outputs"], 1)

    def test_no_input_constant_graph_and_rank_zero(self):
        module = Module()
        graph = module.graph("constant", inputs={}, results=[TensorType((), "f32")])
        graph.return_(graph.splat(graph.constant(2.0, dtype="f32"), ()))
        artifact, _ = self.lower(module)
        self.assertEqual(len(artifact["port_map"]), 1)
        self.assertEqual(artifact["port_map"][0]["logical_port"], "output:0")
        self.assertEqual(artifact["port_map"][0]["shape"], [])

    def test_bf16_intermediates_require_supported_target(self):
        module = Module()
        tensor = TensorType((3,), "f32")
        graph = module.graph("convert", inputs={"x": tensor}, results=[tensor])
        graph.return_(graph.cast(graph.cast(graph.inputs["x"], dtype="bf16"), dtype="f32"))
        result = run(module.to_mlir(), "--emit-cuda-llvm", "--cuda-arch=75")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("bf16 compilation requires target sm80", result.stderr)
        artifact, _ = self.lower(module)
        self.assertIn("bf16", artifact["required_element_types"])

    def test_invalid_launch_options_and_external_graph_rejected(self):
        source = build_module(3, external=False).to_mlir()
        for flags, diagnostic in [
            (["--emit-cuda-llvm"], "explicit --cuda-arch"),
            (["--emit-cuda-llvm", "--cuda-arch=86", "--cuda-block-size=0"], "block size"),
            (["--emit-cuda-llvm", "--cuda-arch=86", "--cuda-block-size=1025"], "block size"),
            (["--emit-cuda-llvm", "--cuda-arch=86", "--cuda-ptx-version=0"], "PTX version"),
            (["--emit-llvm", "--cuda-arch=86"], "require --emit-cuda-llvm"),
            (["--emit-llvm", "--emit-cuda-llvm"], "select only one"),
        ]:
            with self.subTest(flags=flags):
                result = run(source, *flags)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stderr)
        result = run(build_module(3).to_mlir(), "--emit-cuda-llvm", "--cuda-arch=86")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot compile ask.external", result.stderr)

    def test_pointer_limit_and_dynamic_shape_rejected(self):
        module = Module()
        tensor = TensorType((1,), "i32")
        graph = module.graph("too_many", inputs={f"x{i}": tensor for i in range(64)}, results=[tensor])
        graph.return_(graph.inputs["x0"])
        result = run(module.to_mlir(), "--emit-cuda-llvm", "--cuda-arch=86")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("at most 64", result.stderr)
        module = Module()
        tensor = TensorType((None,), "f32")
        graph = module.graph("dynamic", inputs={"x": tensor}, results=[tensor])
        graph.return_(graph.inputs["x"])
        result = run(module.to_mlir(), "--emit-cuda-llvm", "--cuda-arch=86")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("concrete", result.stderr)

    def test_large_static_shape_keeps_64_bit_indexing_and_bounded_grid(self):
        module = Module()
        tensor = TensorType((2**31 + 17,), "f32")
        graph = module.graph("large", inputs={"x": tensor}, results=[tensor])
        graph.return_(graph.inputs["x"])
        artifact, _ = self.lower(module)
        self.assertEqual(artifact["invocation"]["grid"], [65535, 1, 1])
        self.assertIn("icmp ult i64", artifact["payload"])
        self.assertIn("2147483665", artifact["payload"])
        self.assertIn("8388480", artifact["payload"])  # 65535 * 128 stride.

    def test_overflow_and_multiple_entry_points_rejected(self):
        module = Module()
        tensor = TensorType((2**62,), "f32")
        graph = module.graph("overflow", inputs={"x": tensor}, results=[tensor])
        graph.return_(graph.inputs["x"])
        result = run(module.to_mlir(), "--emit-cuda-llvm", "--cuda-arch=86")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("signed 64-bit size range", result.stderr)
        module = Module()
        tensor = TensorType((3,), "i32")
        for name in ("first", "second"):
            graph = module.graph(name, inputs={"x": tensor}, results=[tensor])
            graph.return_(graph.inputs["x"])
        result = run(module.to_mlir(), "--emit-cuda-llvm", "--cuda-arch=86")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly one graph", result.stderr)


if __name__ == "__main__":
    unittest.main()
