"""Native static graph partitioning and execution-plan SSA contracts, without a GPU."""
from __future__ import annotations

import json
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

    def test_fusion_partition_and_selective_fallback_preserve_provenance(self):
        source = (ROOT / "tests/ir/pipeline.mlir").read_text()
        fused = self.partition(source)
        plan = self.assert_plan(fused)
        self.assertEqual([step["kind"] for step in plan["steps"]], ["fused"])
        self.assertEqual(plan["steps"][0]["external_id"], "F")
        self.assertEqual(len(fused.report["markers"]), 2)
        self.assertIn("ask.source_markers", fused.ir)
        split = self.partition(source, unfused_externals=["F"])
        plan = self.assert_plan(split)
        self.assertEqual([step["kind"] for step in plan["steps"]], ["owned", "external", "owned"])
        self.assertEqual(plan, self.partition(source, fuse_external=False).report["execution_plan"])
        self.assertEqual(plan["inputs"], fused.report["execution_plan"]["inputs"])
        self.assertEqual(plan["outputs"], fused.report["execution_plan"]["outputs"])
        self.assertEqual(split.report["external_requests"][0]["port_shapes"],
                         {"input:0": [10], "output:0": [10]})
        with self.assertRaisesRegex(CompilerError, "unknown unfused external"):
            self.partition(source, unfused_externals=["missing"])

    def test_fanout_cannot_be_hidden_inside_a_single_output_fusion(self):
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

    def test_cli_and_binding_have_the_same_contract_and_reject_dynamic_inputs(self):
        source = (ROOT / "tests/ir/pipeline.mlir").read_text()
        native = self.partition(source, optimize=False, fuse_external=False)
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            output = Path(directory) / "partition.mlir"
            result = subprocess.run([ask_opt_path(), "--partition", "--no-optimize", "--no-external-fusion",
                                     "--report-json", str(report), "-o", str(output)],
                                    input=source, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(report.read_text()), native.report)
            self.assertEqual(output.read_text(), native.ir)
        with self.assertRaisesRegex(CompilerError, "concrete"):
            self.partition('module { func.func @main(%x: tensor<?xi32>) -> tensor<?xi32> { return %x : tensor<?xi32> } }')


if __name__ == "__main__":
    unittest.main()
