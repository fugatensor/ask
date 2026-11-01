"""High-level optimization contracts, including identities and frozen shapes."""
import ctypes
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = str(Path(sys.argv.pop(1)).resolve())
sys.path.insert(0, str(ROOT / "python"))
from ask import Compiler, CompilerError, Module, ScalarType, TensorType


def shared_graph():
    module = Module()
    tensor = TensorType((10,))
    graph = module.graph("main", inputs={"x": tensor}, results=[tensor, tensor])
    one = graph.splat(graph.constant(1), (10,))
    t = graph.add(graph.inputs["x"], one, id="P", value_id="v_t")
    y = graph.mul(t, graph.splat(graph.constant(2), (10,)), id="Q", value_id="v_y")
    graph.return_(t, y)
    return module


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        self.compiler = Compiler(DRIVER)

    def test_scalar_constant_folding_preserves_output_identity(self):
        module = Module()
        graph = module.graph("main", inputs={}, results=[ScalarType()])
        value = graph.add(graph.constant(2147483647), graph.constant(1),
                          id="sum", value_id="wrapped")
        marked = graph.mark(value, mark_id="sum_out", role="output", port_id="sum.out")
        graph.return_(marked)
        result = self.compiler.optimize(module)
        self.assertNotIn('"ask.add"', result.ir)
        self.assertIn("value = -2147483648 : i32", result.ir)
        self.assertIn('ask.value_id = "wrapped"', result.ir)
        self.assertEqual(result.report["optimization"]["constant_folds"], 1)
        self.assertEqual(result.report["markers"][0]["status"], "rewritten")
        self.assertEqual(result.report["shape_snapshot"]["semantic_revision"],
                         result.report["input_shape_snapshot"]["semantic_revision"] + 1)

    def test_float_folding_rounds_each_operation_without_algebraic_identities(self):
        module = Module()
        graph = module.graph("rounding", inputs={}, results=[ScalarType("f32")])
        product = graph.mul(graph.constant(1.0 + 2 ** -23, dtype="f32"),
                            graph.constant(1.0 - 2 ** -23, dtype="f32"))
        graph.return_(graph.add(product, graph.constant(-1.0, dtype="f32")))
        optimized = self.compiler.optimize(module)
        self.assertEqual(optimized.report["optimization"]["constant_folds"], 2)
        self.assertIn("value = 0.000000e+00 : f32", optimized.ir)
        self.assertNotIn('"ask.add"', optimized.ir)
        self.assertNotIn('"ask.mul"', optimized.ir)
        repeated = self.compiler.optimize(optimized.ir)
        self.assertEqual(repeated.ir, optimized.ir)

        dynamic = Module()
        graph = dynamic.graph("identity", inputs={"x": ScalarType("f32")},
                              results=[ScalarType("f32"), ScalarType("f32")])
        zero = graph.constant(0.0, dtype="f32")
        graph.return_(graph.add(graph.inputs["x"], zero), graph.mul(graph.inputs["x"], zero))
        result = self.compiler.optimize(dynamic)
        self.assertIn('"ask.add"', result.ir)  # x + 0 changes the sign of -0.
        self.assertIn('"ask.mul"', result.ir)  # NaN/Inf * 0 cannot become 0.
        self.assertEqual(result.report["optimization"]["constant_folds"], 0)

    def test_markers_do_not_keep_dead_computations_alive(self):
        module = Module()
        tensor = TensorType((10,))
        graph = module.graph("main", inputs={"x": tensor}, results=[tensor])
        dead = graph.add(graph.inputs["x"], graph.splat(graph.constant(1), (10,)),
                         id="dead", value_id="dead_value")
        graph.mark(dead, mark_id="dead_trace")
        graph.return_(graph.inputs["x"])
        result = self.compiler.optimize(module, fold_constants=False)
        self.assertNotIn('"ask.add"', result.ir)
        self.assertNotIn('"ask.splat"', result.ir)
        self.assertEqual(result.report["markers"][0]["status"], "eliminated")
        before = {x["value_id"] for x in result.report["input_shape_snapshot"]["shape_bindings"]}
        after = {x["value_id"] for x in result.report["shape_snapshot"]["shape_bindings"]}
        self.assertIn("dead_value", before)
        self.assertNotIn("dead_value", after)
        self.assertIn("main.arg0", after)
        retained = self.compiler.optimize(module, fold_constants=False, eliminate_dead_code=False,
                                          fuse_elementwise=False)
        self.assertIn('"ask.add"', retained.ir)
        self.assertEqual(retained.report["optimization"]["dead_ops"], 0)

    def test_external_compiler_boundary_and_configuration_are_preserved(self):
        source = (ROOT / "tests/ir/pipeline.mlir").read_text()
        before = self.compiler.prepare(source)
        result = self.compiler.optimize(source)
        self.assertEqual(result.report["external_requests"], before.report["external_requests"])
        groups = result.report["optimization"]["fusion_groups"]
        self.assertTrue(groups)
        for group in groups:
            self.assertNotIn("F", group["operations"])
            self.assertFalse({"P", "Q"}.issubset(group["operations"]))
        # An unused external result does not license deletion of its invocation.
        unused = source.replace("return %y : tensor<10xi32>", "return %x : tensor<10xi32>")
        optimized = self.compiler.optimize(unused)
        self.assertIn('"ask.external"', optimized.ir)
        self.assertEqual(len(optimized.report["external_requests"]), 1)
        self.assertNotIn('ask.id = "Q"', optimized.ir)

    def test_repeated_optimization_and_fusion_only_preserve_shape_snapshot(self):
        baseline = self.compiler.prepare(shared_graph())
        optimized = self.compiler.optimize(shared_graph())
        self.assertTrue(optimized.report["optimization"]["fusion_groups"])
        self.assertEqual(optimized.report["shape_snapshot"], baseline.report["shape_snapshot"])
        self.assertGreater(optimized.report["ir_revision"], baseline.report["ir_revision"])
        repeated = self.compiler.optimize(optimized.ir)
        self.assertEqual(repeated.ir, optimized.ir)
        self.assertEqual(repeated.report["shape_snapshot"], optimized.report["shape_snapshot"])
        self.assertEqual(repeated.report["ir_revision"], optimized.report["ir_revision"])
        self.assertEqual(repeated.report["optimization"]["constant_folds"], 0)
        self.assertEqual(repeated.report["optimization"]["dead_ops"], 0)
        disabled = self.compiler.optimize(optimized.ir, fuse_elementwise=False)
        self.assertNotIn("ask.fusion_group", disabled.ir)
        self.assertEqual(disabled.report["optimization"]["fusion_groups"], [])
        self.assertEqual(disabled.report["shape_snapshot"], optimized.report["shape_snapshot"])

    def test_folded_scalar_ids_avoid_operation_and_value_ids_in_other_functions(self):
        module = Module()
        tensor = TensorType((5,))
        graph = module.graph("main", inputs={}, results=[tensor])
        left = graph.splat(graph.constant(1), (5,))
        right = graph.splat(graph.constant(2), (5,))
        graph.return_(graph.add(left, right, id="fold", value_id="folded_tensor"))
        # Reserve the first candidate operation ID, then the next candidate's
        # logical ValueId, in a function that occurs later in the module.
        reserve = module.graph("reserve", inputs={}, results=[ScalarType(), ScalarType()])
        first = reserve.constant(17, id="fold.folded_scalar", value_id="reserved_value")
        second = reserve.constant(19, id="reserved_op", value_id="fold.folded_scalar.1.result0")
        reserve.return_(first, second)

        result = self.compiler.optimize(module)
        self.assertEqual(result.report["optimization"]["constant_folds"], 1)
        operation_ids = re.findall(r'ask\.id = "([^"]+)"', result.ir)
        value_ids = re.findall(r'ask\.value_id = "([^"]+)"', result.ir)
        self.assertEqual(len(operation_ids), len(set(operation_ids)))
        self.assertEqual(len(value_ids), len(set(value_ids)))
        self.assertIn("fold.folded_scalar", operation_ids)
        self.assertIn("fold.folded_scalar.1.result0", value_ids)
        self.assertIn("folded_tensor", value_ids)
        folded_scalar = next(line for line in result.ir.splitlines()
                             if '"ask.constant"' in line and "value = 3 : i32" in line)
        self.assertNotIn('ask.id = "fold.folded_scalar"', folded_scalar)
        self.assertNotIn('ask.value_id = "fold.folded_scalar.1.result0"', folded_scalar)
        self.assertIn("value = 17 : i32", result.ir)
        self.assertIn("value = 19 : i32", result.ir)

    def test_repeated_constant_folding_without_dce_keeps_ir_and_revisions_stable(self):
        module = Module()
        tensor = TensorType((5,))
        graph = module.graph("main", inputs={}, results=[tensor])
        scalar = graph.add(graph.constant(2147483647), graph.constant(1), id="scalar_sum")
        left = graph.splat(scalar, (5,), id="left")
        right = graph.splat(graph.constant(-1), (5,), id="right")
        product = graph.mul(left, right, id="product")
        graph.return_(graph.add(product, graph.splat(graph.constant(2), (5,)), id="sum"))

        first = self.compiler.optimize(module, eliminate_dead_code=False)
        self.assertEqual(first.report["optimization"]["constant_folds"], 3)
        self.assertEqual(first.report["optimization"]["dead_ops"], 0)
        self.assertIn('ask.id = "left"', first.ir)
        self.assertIn('ask.id = "right"', first.ir)
        self.assertNotIn('"ask.add"', first.ir)
        self.assertNotIn('"ask.mul"', first.ir)
        repeated = self.compiler.optimize(first.ir, eliminate_dead_code=False)
        self.assertEqual(repeated.ir, first.ir)
        self.assertEqual(repeated.report["ir_revision"], first.report["ir_revision"])
        self.assertEqual(repeated.report["shape_snapshot"], first.report["shape_snapshot"])
        self.assertEqual(repeated.report["optimization"]["constant_folds"], 0)
        self.assertEqual(repeated.report["optimization"]["dead_ops"], 0)

    def test_removed_consumer_port_is_eliminated_even_when_its_value_survives(self):
        module = Module()
        tensor = TensorType((10,))
        graph = module.graph("main", inputs={"x": tensor}, results=[tensor])
        x = graph.inputs["x"]
        marked = graph.mark(x, mark_id="dead_input", role="input", port_id="dead.in0")
        graph.add(marked, graph.splat(graph.constant(1), (10,)),
                  id="dead", value_id="dead_value")
        graph.mark(x, mark_id="live_value")
        graph.return_(x)

        result = self.compiler.optimize(module, fold_constants=False)
        markers = {item["mark_id"]: item for item in result.report["markers"]}
        self.assertEqual(markers["dead_input"]["status"], "eliminated")
        self.assertIn("port", markers["dead_input"]["reason"])
        self.assertEqual(markers["live_value"]["status"], "exact")
        self.assertEqual(markers["dead_input"]["value_id"], x.value_id)
        surviving = {item["value_id"]
                     for item in result.report["shape_snapshot"]["shape_bindings"]}
        self.assertIn(x.value_id, surviving)
        self.assertNotIn("dead_value", surviving)
        self.assertEqual(len(result.report["input_markers"]), 2)
        self.assertTrue(all("status" not in item for item in result.report["input_markers"]))

    def test_invalid_ir_is_rejected_before_dce(self):
        with self.assertRaises(CompilerError) as caught:
            self.compiler.optimize((ROOT / "tests/ir/shape-conflict.mlir").read_text())
        self.assertGreater(caught.exception.returncode, 0)
        self.assertIn("error", caught.exception.diagnostics)

    def test_option_validation(self):
        with self.assertRaises(TypeError):
            self.compiler.optimize(shared_graph(), fold_constants="yes")
        for args in (["--no-fusion"], ["--optimize", "--optimize-llir"],
                     ["--optimize-llir", "--emit-llvm"]):
            result = subprocess.run([DRIVER, *args], text=True, capture_output=True, input="")
            self.assertGreater(result.returncode, 0, result.stderr)
            self.assertNotIn("PLEASE submit a bug", result.stderr)

    def test_folded_tensor_constants_execute_with_wrapping_arithmetic(self):
        if not self.compiler.capabilities()["host_codegen"]:
            self.skipTest("host backend disabled")
        cc = shutil.which("cc")
        self.assertIsNotNone(cc)
        module = Module()
        cases = []
        for shape in [(7,), (), (0,)]:
            for opcode, lhs, rhs in [("add", 2147483647, 1), ("mul", -2147483648, -1),
                                     ("mul", 2147483647, 2147483647), ("add", -19, 7)]:
                name = f"case{len(cases)}"
                tensor = TensorType(shape)
                graph = module.graph(name, inputs={}, results=[tensor])
                left = graph.splat(graph.constant(lhs), shape)
                right = graph.splat(graph.constant(rhs), shape)
                graph.return_(getattr(graph, opcode)(left, right))
                expected = ctypes.c_int32(lhs + rhs if opcode == "add" else lhs * rhs).value
                cases.append((name, shape, expected))
        optimized = self.compiler.optimize(module)
        self.assertNotIn('"ask.add"', optimized.ir)
        self.assertNotIn('"ask.mul"', optimized.ir)
        self.assertEqual(optimized.report["optimization"]["constant_folds"], len(cases))
        repeated = self.compiler.optimize(optimized.ir)
        self.assertEqual(repeated.ir, optimized.ir)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for label, source in [("before", module.to_mlir()), ("after", optimized.ir)]:
                obj, shared = path / (label + ".o"), path / (label + ".so")
                compiled = subprocess.run([DRIVER, "-", "--emit-object", "-o", str(obj)],
                                          input=source, text=True, capture_output=True)
                self.assertEqual(compiled.returncode, 0, compiled.stderr)
                linked = subprocess.run([cc, "-shared", str(obj), "-o", str(shared)],
                                        text=True, capture_output=True)
                self.assertEqual(linked.returncode, 0, linked.stderr)
                library = ctypes.CDLL(str(shared))
                for name, shape, expected in cases:
                    count = shape[0] if shape else 1
                    output = (ctypes.c_int32 * max(count, 1))(*([42] * max(count, 1)))
                    function = getattr(library, name)
                    function.argtypes = [ctypes.POINTER(ctypes.c_int32)]
                    function.restype = None
                    function(output)
                    self.assertEqual(list(output), [expected] * count if count else [42])

    def test_folded_float_constants_match_executed_ieee_results(self):
        if not self.compiler.capabilities()["host_codegen"]:
            self.skipTest("host backend disabled")
        cc = shutil.which("cc")
        self.assertIsNotNone(cc)
        module = Module()
        cases = [
            ("add", 16777216.0, 1.0, 16777216.0),
            ("add", -0.0, -0.0, -0.0),
            ("add", -0.0, 0.0, 0.0),
            ("mul", float("inf"), 0.0, float("nan")),
            ("mul", 1.0, 2 ** -149, 2 ** -149),
            ("mul", 3.4028234663852886e38, 2.0, float("inf")),
        ]
        for index, (opcode, left, right, _) in enumerate(cases):
            tensor = TensorType((2, 3), "f32")
            graph = module.graph(f"case{index}", inputs={}, results=[tensor])
            lhs = graph.splat(graph.constant(left, dtype="f32"), (2, 3))
            rhs = graph.splat(graph.constant(right, dtype="f32"), (2, 3))
            graph.return_(getattr(graph, opcode)(lhs, rhs))
        optimized = self.compiler.optimize(module)
        self.assertEqual(optimized.report["optimization"]["constant_folds"], len(cases))
        self.assertTrue(all(item["element_type"] == "f32"
                            for item in optimized.report["shape_snapshot"]["shape_bindings"]))
        self.assertNotIn('"ask.add"', optimized.ir)
        self.assertNotIn('"ask.mul"', optimized.ir)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for label, source in [("before", module.to_mlir()), ("after", optimized.ir)]:
                obj, shared = path / (label + ".o"), path / (label + ".so")
                compiled = subprocess.run([DRIVER, "-", "--emit-object", "-o", str(obj)],
                                          input=source, text=True, capture_output=True)
                self.assertEqual(compiled.returncode, 0, compiled.stderr)
                linked = subprocess.run([cc, "-shared", str(obj), "-o", str(shared)],
                                        text=True, capture_output=True)
                self.assertEqual(linked.returncode, 0, linked.stderr)
                library = ctypes.CDLL(str(shared))
                for index, (_, _, _, expected) in enumerate(cases):
                    function = getattr(library, f"case{index}")
                    function.argtypes = [ctypes.POINTER(ctypes.c_float)]
                    function.restype = None
                    output = (ctypes.c_float * 6)()
                    function(output)
                    for actual in output:
                        if math.isnan(expected):
                            self.assertTrue(math.isnan(actual))
                        else:
                            self.assertEqual(struct.pack("f", actual), struct.pack("f", expected))


if __name__ == "__main__":
    unittest.main()
