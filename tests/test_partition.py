"""Native static graph partitioning and execution-plan SSA contracts, without a GPU."""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

from ask import Compiler, CompilerError, Module, TensorType
from support.host_runner import ask_opt_path

ROOT = Path(__file__).resolve().parents[1]


class PartitionTests(unittest.TestCase):
    def partition(self, source, **options):
        return Compiler().partition(source, **options)

    def assert_plan(self, result):
        plan = result.report["execution_plan"]
        self.assertEqual((plan["schema_version"], plan["kind"]), (1, "ask.cuda.execution_plan"))
        self.assertEqual(plan["scheduling"], {"policy": "sequential", "stream": "session"})
        buffers = {buffer["id"]: buffer for buffer in plan["buffers"]}
        available, completed = set(plan["inputs"]), set()
        for step in plan["steps"]:
            self.assertTrue(set(step["inputs"]) <= available)
            self.assertTrue(set(step["depends_on"]) <= completed)
            self.assertTrue(set(step["outputs"]) <= buffers.keys())
            self.assertFalse(set(step["outputs"]) & available)
            available.update(step["outputs"])
            completed.add(step["id"])
            source = result.report["step_sources"][step["id"]]
            self.assertIn(f'ask.kernel_id = "{step["id"]}"', source)
            self.assertIn(f'ask.kernel_kind = "{step["kind"]}"', source)
            Compiler().prepare(source)
        self.assertTrue(set(plan["outputs"]) <= available)
        self.assertIn("ask.execution_plan", result.ir)
        return plan

    def test_owned_graph_and_identity_outputs_preserve_aliases(self):
        module = Module()
        type_ = TensorType((3,), "i32")
        graph = module.graph("main", inputs={"x": type_}, results=[type_, type_, type_])
        result = graph.add(graph.inputs["x"], graph.splat(graph.constant(2), (3,)))
        graph.return_(graph.inputs["x"], result, result)
        plan = self.assert_plan(self.partition(module))
        self.assertEqual([step["kind"] for step in plan["steps"]], ["owned"])
        self.assertEqual(plan["outputs"][0], plan["inputs"][0])
        self.assertEqual(plan["outputs"][1], plan["outputs"][2])
        identity = self.partition("module { func.func @id(%x: tensor<i32>) -> tensor<i32> { return %x : tensor<i32> } }")
        plan = self.assert_plan(identity)
        self.assertEqual(plan["steps"], [])
        self.assertEqual(plan["inputs"], plan["outputs"])

    def test_external_boundaries_preserve_owned_regions_and_provenance(self):
        source = (ROOT / "tests/ir/pipeline.mlir").read_text()
        result = self.partition(source)
        plan = self.assert_plan(result)
        self.assertEqual([step["kind"] for step in plan["steps"]], ["owned", "external", "owned"])
        self.assertEqual(plan["steps"][1]["external_id"], "F")
        self.assertEqual(len(result.report["markers"]), 2)
        self.assertIn("ask.source_markers", result.ir)
        self.assertEqual(result.report["external_requests"][0]["protocol_version"], 2)
        self.assertEqual(result.report["external_requests"][0]["port_shapes"],
                         {"input:0": [10], "output:0": [10]})

    def test_fanout_preserves_public_outputs(self):
        source = (ROOT / "tests/ir/pipeline.mlir").read_text()
        source = source.replace('-> tensor<10xi32> {\n    %one', '-> (tensor<10xi32>, tensor<10xi32>) {\n    %one')
        source = source.replace('return %y : tensor<10xi32>', 'return %t, %y : tensor<10xi32>, tensor<10xi32>')
        plan = self.assert_plan(self.partition(source))
        self.assertGreaterEqual(len(plan["steps"]), 2)
        self.assertEqual(len(plan["outputs"]), 2)

    def test_multiport_shape_change_and_duplicate_external_inputs(self):
        source = '''module {
          "ask.semantics"() {sym_name = "ports", function_type = (tensor<4xf32>, tensor<4xf32>) -> (tensor<2xf32>, tensor<1xi32>), shape_rule = "fixed", effects = "none"} : () -> ()
          "ask.implementation"() {sym_name = "impl", compiler = "fixture", source = "opaque source", entry = "ports", signature = {a = "*fp32", b = "*fp32", c = "*fp32", d = "*i32"}, source_bindings = {a = "input:0", b = "input:1", c = "output:0", d = "output:1"}, constants = {}, options = {}, target = {backend = "cuda", arch = 80 : i64}, grid = array<i64: 1, 1, 1>} : () -> ()
          func.func @main(%x: tensor<4xf32>) -> (tensor<2xf32>, tensor<1xi32>) {
            %a, %b = "ask.external"(%x, %x) {ask.id = "multi", semantics = @ports, implementation = @impl} : (tensor<4xf32>, tensor<4xf32>) -> (tensor<2xf32>, tensor<1xi32>)
            return %a, %b : tensor<2xf32>, tensor<1xi32>
          }
        }'''
        result = self.partition(source)
        plan = self.assert_plan(result)
        self.assertEqual(len(plan["steps"]), 1)
        step = plan["steps"][0]
        self.assertEqual(step["kind"], "external")
        self.assertEqual(step["inputs"], [plan["inputs"][0]] * 2)
        self.assertEqual(len(step["outputs"]), 2)
        self.assertEqual(result.report["external_requests"][0]["port_shapes"],
                         {"input:0": [4], "input:1": [4], "output:0": [2], "output:1": [1]})

    def test_sdk_multioutput_scalar_abi_and_json_constants_roundtrip(self):
        incoming = TensorType((4,), "f32")
        outputs = (TensorType((2,), "f32"), TensorType((1,), "i32"))
        scalar_types = ("i1", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64",
                        "fp16", "bf16", "fp32", "fp64")
        scalar_values = (True, -128, -32768, -(2**31), -(2**63), 255, 65535,
                         2**32 - 1, 2**64 - 1, 1.5, 2, -0.0, 0.1)
        signature = {"x": "*fp32", "a": "*fp32", "b": "*i32", **dict(zip(scalar_types, scalar_types))}
        bindings = {"x": "input:0", "a": "output:0", "b": "output:1", **dict(zip(scalar_types, scalar_values))}
        constants = {"dtype": {"__compiler_dtype__": "float32"}, "flags": [True, False, None],
                     "one": 1.0, "minus_one": -1.0, "integer": -1,
                     "nested": {"key with spaces": ["text", 0.1, -0.0, -(2**63), 2**64-1]}}
        options = {"limit": 0.25, "features": ["a", None, {"enabled": True}]}
        module = Module()
        semantics = module.semantics("sem", inputs=[incoming], results=outputs, shape_rule="fixed")
        implementation = module.implementation("impl", compiler="fixture", source="opaque", entry="kernel",
            signature=signature, source_bindings=bindings, constants=constants, options=options,
            target={"backend": "cuda", "arch": 80})
        graph = module.graph("main", inputs={"x": incoming}, results=outputs)
        results = graph.external(graph.inputs["x"], semantics=semantics, implementation=implementation,
                                 result_types=outputs, id="multi")
        self.assertIsInstance(results, tuple)
        self.assertEqual([value.value_id for value in results], ["multi.result0", "multi.result1"])
        graph.return_(*results)
        compiled = self.partition(module)
        plan = self.assert_plan(compiled)
        self.assertEqual(len(plan["steps"][0]["outputs"]), 2)
        request = compiled.report["external_requests"][0]
        self.assertEqual(request["source_bindings"], bindings)
        self.assertEqual(request["constants"], constants)
        self.assertEqual(request["fixed_options"], options)
        self.assertIs(type(request["constants"]["one"]), float)
        self.assertIs(type(request["constants"]["minus_one"]), float)
        self.assertIs(type(request["constants"]["integer"]), int)
        self.assertEqual(math.copysign(1, request["source_bindings"]["fp32"]), -1)
        self.assertEqual(math.copysign(1, request["constants"]["nested"]["key with spaces"][2]), -1)
        for compiler in (Compiler(), Compiler(executable=ask_opt_path())):
            for method in (compiler.prepare, compiler.partition):
                with self.subTest(compiler=compiler.executable, method=method.__name__):
                    restored = method(module).report["external_requests"][0]
                    self.assertEqual(restored, request)
                    self.assertIs(type(restored["constants"]["one"]), float)
                    self.assertIs(type(restored["constants"]["minus_one"]), float)
                    self.assertIs(type(restored["constants"]["integer"]), int)
                    self.assertEqual(math.copysign(1, restored["source_bindings"]["fp32"]), -1)

        for parameter, replacement in (("i1", 1), ("i8", 128), ("u8", -1), ("fp32", True),
                                       ("i64", 2**63), ("u64", -1)):
            with self.subTest(parameter=parameter):
                bad = Module()
                sem = bad.semantics("sem", inputs=[incoming], results=outputs, shape_rule="fixed")
                impl = bad.implementation("impl", compiler="fixture", source="opaque", entry="kernel",
                    signature=signature, source_bindings={**bindings, parameter: replacement},
                    target={"backend": "cuda", "arch": 80})
                body = bad.graph("main", inputs={"x": incoming}, results=outputs)
                body.return_(*body.external(body.inputs["x"], semantics=sem, implementation=impl))
                with self.assertRaisesRegex(CompilerError, "invalid scalar source binding"):
                    self.partition(bad)

    def test_cli_and_binding_have_the_same_contract_and_reject_dynamic_inputs(self):
        source = (ROOT / "tests/ir/pipeline.mlir").read_text()
        native = self.partition(source, optimize=False)
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            output = Path(directory) / "partition.mlir"
            result = subprocess.run([ask_opt_path(), "--partition", "--no-optimize",
                                     "--report-json", str(report), "-o", str(output)],
                                    input=source, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(report.read_text()), native.report)
            self.assertEqual(output.read_text(), native.ir)
        with self.assertRaisesRegex(CompilerError, "concrete"):
            self.partition('module { func.func @main(%x: tensor<?xi32>) -> tensor<?xi32> { return %x : tensor<?xi32> } }')


if __name__ == "__main__":
    unittest.main()
