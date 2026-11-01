"""Check observable outputs and actual loop sharing of host tensor fusion."""
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


def i32(value):
    return ctypes.c_int32(value).value


def shared_graph(shape, *, scalar_output=False, single_output=False,
                 repeat_output=False):
    module = Module()
    tensor = TensorType(shape)
    results = [tensor] if single_output else [tensor, tensor]
    if scalar_output:
        results.insert(1, ScalarType())
    if repeat_output:
        results.append(tensor)
    graph = module.graph("calculate", inputs={"x": tensor, "s": ScalarType()},
                         results=results)
    scalar = graph.inputs["s"]
    biased = graph.add(graph.inputs["x"], graph.splat(scalar, shape))
    # A scalar result between tensor operations must not become a tensor root.
    scalar_result = graph.add(scalar, graph.constant(1)) if scalar_output else None
    doubled = graph.mul(biased, graph.splat(graph.constant(2), shape))
    final = graph.add(doubled, graph.splat(graph.constant(3), shape))
    outputs = [final] if single_output else [biased, final]
    if scalar_output:
        outputs.insert(1, scalar_result)
    if repeat_output:
        outputs.append(biased)
    graph.return_(*outputs)
    return module.to_mlir()


class FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        capabilities = run("--capabilities")
        if capabilities.returncode:
            raise RuntimeError(capabilities.stderr)
        if not json.loads(capabilities.stdout)["host_codegen"]:
            raise unittest.SkipTest("host backend disabled")
        cls.cc = shutil.which("cc")
        if not cls.cc:
            raise RuntimeError("C compiler is required to link the emitted object")

    def compile(self, source, directory, name, *, optimize, object_file=False,
                no_fusion=False):
        output = directory / (name + (".o" if object_file else ".ll"))
        report = directory / (name + ".json")
        args = ["-", "--emit-object" if object_file else "--emit-llvm",
                "-o", output, "--report-json", report]
        if optimize:
            args.append("--optimize")
        if no_fusion:
            args.append("--no-fusion")
        result = run(*args, source=source)
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads(report.read_text())["abi"]
        function = metadata["functions"][0]
        self.assertTrue(function["requires_disjoint_outputs"])
        self.assertTrue(function["outputs_must_not_alias_inputs"])
        self.assertEqual(metadata["lowering"]["statistics_stage"], "host_lowering")
        self.assertEqual(function["lowering"]["statistics_stage"], "host_lowering")
        self.assertEqual(metadata["lowering"]["loop_count"],
                         function["lowering"]["loop_count"])
        if not object_file:
            return function["lowering"], output.read_text()
        library = directory / (name + ".so")
        linked = subprocess.run([self.cc, "-shared", str(output), "-o", str(library)],
                                text=True, capture_output=True, timeout=60)
        self.assertEqual(linked.returncode, 0, linked.stderr)
        return function["lowering"], ctypes.CDLL(str(library))

    def execute_shared(self, library, shape, *, scalar_output=False,
                       single_output=False, repeat_output=False):
        count = math.prod(shape)
        data = [2147483647, -2147483648, -7, 0, 1, 7, 42, 1024, -1024, 9][:count]
        storage = ctypes.c_int32 * max(count, 1)
        x = storage(*data)
        biased, final = storage(*([123] * max(count, 1))), storage(*([456] * max(count, 1)))
        repeated = storage(*([789] * max(count, 1)))
        scalar = ctypes.c_int32(987)
        outputs = [final] if single_output else [biased, final]
        if scalar_output:
            outputs.insert(1, ctypes.byref(scalar))
        if repeat_output:
            outputs.append(repeated)
        pointer = ctypes.POINTER(ctypes.c_int32)
        function = library.calculate
        function.argtypes = [pointer, ctypes.c_int32] + [pointer] * len(outputs)
        function.restype = None
        function(x, 2147483647, *outputs)
        expected_biased = [i32(value + 2147483647) for value in data]
        expected_final = [i32(value * 2 + 3) for value in expected_biased]
        self.assertEqual(list(final)[:count], expected_final)
        if not single_output:
            self.assertEqual(list(biased)[:count], expected_biased)
        if repeat_output:
            self.assertEqual(list(repeated)[:count], expected_biased)
        if scalar_output:
            self.assertEqual(scalar.value, -2147483648)
        self.assertEqual(list(x)[:count], data)
        if count == 0:
            self.assertEqual(biased[0], 123)
            self.assertEqual(final[0], 456)
            # An empty tensor must not cause even a speculative load/store.
            function(None, 2147483647, *([None] * len(outputs)))
        return list(biased), list(final), scalar.value, list(repeated)

    def test_multiple_outputs_share_loop_and_producer(self):
        source = shared_graph((10,))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            baseline, baseline_ir = self.compile(source, directory, "baseline", optimize=False)
            fused, fused_ir = self.compile(source, directory, "fused", optimize=True)
            self.assertEqual(baseline["loop_count"], 2)
            self.assertEqual(fused["loop_count"], 1)
            self.assertEqual(fused["unfused_output_loop_count"], 2)
            self.assertEqual(fused["fused_output_groups"], 1)
            self.assertEqual(fused["groups"][0]["outputs"], [0, 1])
            self.assertEqual(baseline["input_load_count"], 2)
            self.assertEqual(fused["input_load_count"], 1)
            self.assertEqual(baseline_ir.count("load i32"), 2)
            self.assertEqual(fused_ir.count("load i32"), 1)
            self.assertEqual(fused_ir.count("store i32"), 2)
            # ASK i32 arithmetic wraps; the lowering cannot add overflow promises.
            self.assertNotIn("add nsw", fused_ir)
            self.assertNotIn("mul nsw", fused_ir)
            results = []
            for optimize in (False, True):
                _, library = self.compile(source, directory, f"object{optimize}",
                                          optimize=optimize, object_file=True)
                results.append(self.execute_shared(library, (10,)))
            self.assertEqual(results[0], results[1])
            disabled, _ = self.compile(source, directory, "disabled", optimize=True,
                                       no_fusion=True)
            self.assertEqual(disabled["loop_count"], 2)
            self.assertEqual(disabled["fused_output_groups"], 0)

    def test_single_output_was_already_inlined(self):
        source = shared_graph((10,), single_output=True)
        with tempfile.TemporaryDirectory() as temporary:
            for optimize in (False, True):
                report, library = self.compile(source, Path(temporary), str(optimize),
                                               optimize=optimize, object_file=True)
                self.assertEqual(report["loop_count"], 1)
                self.assertEqual(report["input_load_count"], 1)
                self.assertEqual(report["fused_output_groups"], 0)
                self.execute_shared(library, (10,), single_output=True)

    def test_rank_zero_tensors_share_loop_but_scalar_stays_separate(self):
        source = shared_graph((), scalar_output=True, repeat_output=True)
        with tempfile.TemporaryDirectory() as temporary:
            for optimize, loops in ((False, 4), (True, 2)):
                report, library = self.compile(source, Path(temporary), str(optimize),
                                               optimize=optimize, object_file=True)
                self.assertEqual(report["loop_count"], loops)
                self.assertEqual(report["fused_output_groups"], int(optimize))
                if optimize:
                    self.assertEqual(report["groups"][0]["outputs"], [0, 2, 3])
                    self.assertEqual(report["input_load_count"], 1)
                self.execute_shared(library, (), scalar_output=True, repeat_output=True)

    def test_empty_outputs_do_not_dereference_or_claim_loop_savings(self):
        source = shared_graph((0,))
        with tempfile.TemporaryDirectory() as temporary:
            for optimize in (False, True):
                report, library = self.compile(source, Path(temporary), str(optimize),
                                               optimize=optimize, object_file=True)
                self.assertEqual(report["loop_count"], 0)
                self.assertEqual(report["input_load_count"], 0)
                self.assertEqual(report["fused_output_groups"], 0)
                self.assertEqual(report["zero_element_outputs"], 2)
                self.execute_shared(library, (0,))

    def test_equal_element_count_different_shapes_stay_in_separate_loops(self):
        module = Module()
        flat, matrix = TensorType((10,)), TensorType((2, 5))
        graph = module.graph("calculate", inputs={"x": flat, "z": matrix, "s": ScalarType()},
                             results=[flat, matrix])
        graph.return_(graph.add(graph.inputs["x"], graph.splat(graph.inputs["s"], (10,))),
                      graph.add(graph.inputs["z"], graph.splat(graph.inputs["s"], (2, 5))))
        with tempfile.TemporaryDirectory() as temporary:
            for optimize in (False, True):
                report, library = self.compile(module.to_mlir(), Path(temporary), str(optimize),
                                               optimize=optimize, object_file=True)
                self.assertEqual(report["loop_count"], 2)
                self.assertEqual(report["fused_output_groups"], 0)
                storage = ctypes.c_int32 * 10
                x, z, out0, out1 = storage(*range(10)), storage(*range(10, 20)), storage(), storage()
                pointer = ctypes.POINTER(ctypes.c_int32)
                library.calculate.argtypes = [pointer, pointer, ctypes.c_int32, pointer, pointer]
                library.calculate.restype = None
                library.calculate(x, z, 2147483647, out0, out1)
                self.assertEqual(list(out0), [i32(i + 2147483647) for i in range(10)])
                self.assertEqual(list(out1), [i32(i + 2147483647) for i in range(10, 20)])

    def test_input_and_scalar_returns_and_cross_group_producer(self):
        module = Module()
        flat, matrix = TensorType((10,)), TensorType((2, 5))
        graph = module.graph("calculate", inputs={"x": flat, "z": matrix, "s": ScalarType()},
                             results=[flat, flat, matrix, flat, ScalarType(), flat])
        scalar = graph.inputs["s"]
        biased = graph.add(graph.inputs["x"], graph.splat(scalar, (10,)))
        doubled = graph.mul(biased, graph.splat(graph.constant(2), (10,)))
        # This intervening shape splits the groups. The final group still uses
        # biased, but cannot reuse a loop-local LLVM value from the first group.
        matrix_result = graph.add(graph.inputs["z"], graph.splat(scalar, (2, 5)))
        final = graph.add(biased, graph.splat(graph.constant(3), (10,)))
        scalar_result = graph.add(scalar, graph.constant(1))
        graph.return_(doubled, graph.inputs["x"], matrix_result, final,
                      scalar_result, biased)
        with tempfile.TemporaryDirectory() as temporary:
            for optimize, loops, loads in ((False, 6, 5), (True, 5, 4)):
                report, library = self.compile(module.to_mlir(), Path(temporary), str(optimize),
                                               optimize=optimize, object_file=True)
                self.assertEqual(report["loop_count"], loops)
                self.assertEqual(report["input_load_count"], loads)
                if optimize:
                    self.assertEqual(report["groups"][0]["outputs"], [0, 5])
                storage = ctypes.c_int32 * 10
                x, z = storage(*range(10)), storage(*range(10, 20))
                out0, out1, out2, out3, out5 = (storage() for _ in range(5))
                out4 = ctypes.c_int32()
                pointer = ctypes.POINTER(ctypes.c_int32)
                library.calculate.argtypes = [pointer, pointer, ctypes.c_int32] + [pointer] * 6
                library.calculate.restype = None
                library.calculate(x, z, 2147483647, out0, out1, out2, out3,
                                  ctypes.byref(out4), out5)
                expected = [i32(i + 2147483647) for i in range(10)]
                self.assertEqual(list(out0), [i32(i * 2) for i in expected])
                self.assertEqual(list(out1), list(x))
                self.assertEqual(list(out2), [i32(i + 2147483647) for i in range(10, 20)])
                self.assertEqual(list(out3), [i32(i + 3) for i in expected])
                self.assertEqual(out4.value, -2147483648)
                self.assertEqual(list(out5), expected)

    def test_invalid_group_attributes_are_rejected(self):
        # Do not use --optimize here: a supplied lowering plan must be checked,
        # whereas the optimizer deliberately discards old plans and replans.
        source = '''module {
  func.func @calculate(%x: tensor<10xi32>) -> tensor<10xi32> {
    %one = "ask.constant"() {value = 1 : i32} : () -> i32
    %ones = "ask.splat"(%one) {shape = array<i64: 10>, ask.fusion_group = "g"} : (i32) -> tensor<10xi32>
    %result = "ask.add"(%x, %ones) {overflow = "wrap", ask.fusion_group = "g"} : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
    return %result : tensor<10xi32>
  }
}
'''
        cases = {
            "scalar": source.replace("value = 1 : i32", 'value = 1 : i32, ask.fusion_group = "g"'),
            "not_string": source.replace('ask.fusion_group = "g"', "ask.fusion_group = 7 : i64"),
            "empty": source.replace('ask.fusion_group = "g"', 'ask.fusion_group = ""'),
            "singleton": source.replace(', ask.fusion_group = "g"', "", 1),
            "resumed": source.replace('    %result =', '    %extra = "ask.splat"(%one) {shape = array<i64: 10>} : (i32) -> tensor<10xi32>\n    %result ='),
            "shape_mismatch": source.replace('    %result =', '    %extra = "ask.splat"(%one) {shape = array<i64: 2, 5>, ask.fusion_group = "g"} : (i32) -> tensor<2x5xi32>\n    %result ='),
        }
        valid = run("-", "--emit-llvm", source=source)
        self.assertEqual(valid.returncode, 0, valid.stderr)
        for name, invalid in cases.items():
            with self.subTest(name=name):
                result = run("-", "--emit-llvm", source=invalid)
                self.assertGreater(result.returncode, 0, result.stdout)
                self.assertIn("fusion_group", result.stderr)

    def test_external_remains_an_external_compiler_boundary(self):
        result = run(ROOT / "tests/ir/pipeline.mlir", "--optimize", "--emit-llvm")
        self.assertGreater(result.returncode, 0)
        self.assertIn("external", result.stderr)


if __name__ == "__main__":
    unittest.main()
