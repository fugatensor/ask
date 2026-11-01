"""Validate matrix coverage and CLI selection without tools or a GPU."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "python")]
from support.gpu_validation import plan_matrix
from support.pointwise import DTYPES, build_module


class PrecisionMatrixTests(unittest.TestCase):
    def test_plan_accounts_for_eligible_and_skipped_combinations_before_execution(self):
        plan = plan_matrix(DTYPES)
        self.assertEqual(plan.gpu_executions, 208)
        self.assertEqual(len(plan.eligible), 104)
        self.assertEqual(len(plan.skipped), 28)
        self.assertEqual(len(plan.eligible) + len(plan.skipped), len(plan.cases) * len(DTYPES))
        self.assertTrue(all(item["reason"] for item in plan.skipped))
        self.assertEqual(plan.cache_counters,
                         {"requests": 208, "hits": 104, "misses": 104, "invalid": 0, "builds": 104})

    def test_identical_typed_graphs_are_accounted_for_before_cache_access(self):
        plan = plan_matrix(["f32", "f16", "bf16"], ["special", "mixed-special"])
        self.assertEqual(plan.gpu_executions, 10)
        self.assertEqual(plan.skipped, ({"case": "mixed-special", "dtype": "f32",
                                        "reason": "identical typed graph already covered by special"},))

    def test_alias_selection_runs_one_case_and_unknown_or_empty_selections_fail(self):
        plan = plan_matrix(["fp16", "f16"], ["same-shape"])
        self.assertEqual(plan.dtypes, ("f16",))
        self.assertEqual(plan.gpu_executions, 2)
        self.assertEqual(len(plan.skipped), 0)
        for dtypes, cases in ((["i32"], ["fractions"]), (["f16"], []),
                              ([], ["same-shape"]), (["f64"], ["same-shape"]),
                              (["bf16"], ["misspelled-graph"])):
            with self.subTest(dtypes=dtypes, cases=cases), self.assertRaises(ValueError):
                plan_matrix(dtypes, cases)

    def test_same_graph_topology_is_reused_for_every_dtype(self):
        plan = plan_matrix(DTYPES, ["expression-chain"])
        topologies = []
        for item in plan.eligible:
            module = build_module(dtype=item.dtype, **item.options)
            text = module.to_mlir()
            topologies.append(re.findall(r'"(ask\.[a-z_]+)"\(', text))
            self.assertIn(f"tensor<17x{'i1' if item.dtype == 'bool' else item.dtype}>", text)
            self.assertIn('ask.id = "P"', text)
            self.assertIn('ask.id = "F"', text)
            self.assertIn('ask.id = "Q"', text)
        self.assertTrue(topologies[0])
        self.assertTrue(all(topology == topologies[0] for topology in topologies))

    def test_cache_invalidation_cases_change_only_one_compile_input(self):
        names = ["same-shape", "changed-ask-constant", "changed-F-DSL", "changed-shape"]
        plan = plan_matrix(["bf16"], names)
        options = {item.graph.name: item.options for item in plan.eligible}
        self.assertEqual(options["same-shape"], {"size": 10})
        self.assertEqual(options["changed-ask-constant"], {"size": 10, "bias": 2})
        self.assertEqual(options["changed-F-DSL"], {"size": 10, "factor": 3})
        self.assertEqual(options["changed-shape"], {"size": 11})
        self.assertEqual(plan.gpu_executions, 8)
        modules = [build_module(dtype="bf16", **item.options).to_mlir() for item in plan.eligible]
        self.assertEqual(len(set(modules)), 4)
        default = plan_matrix(["bf16"])
        self.assertFalse(any(item.graph.name.startswith("changed-") for item in default.eligible))
        extended = plan_matrix(["bf16"], extended=True)
        self.assertTrue(set(names) <= {item.graph.name for item in extended.eligible})

    def test_compute_precision_is_distinct_from_storage_dtype(self):
        plan = plan_matrix(["f16", "bf16", "f32"], ["same-shape", "changed-compute-type"])
        self.assertEqual(plan.gpu_executions, 10)
        self.assertEqual(len(plan.skipped), 1)
        self.assertEqual(plan.skipped[0]["dtype"], "f32")
        for dtype in ("f16", "bf16"):
            low = build_module(size=10, dtype=dtype).to_mlir()
            mixed = build_module(size=10, dtype=dtype, compute_dtype="f32").to_mlir()
            self.assertNotEqual(low, mixed)
            self.assertIn("tensor<10xf32>", mixed)
            self.assertIn(f"tensor<10x{dtype}>", mixed)
            self.assertIn('"ask.cast"', mixed)

    def test_cli_rejects_zero_eligible_matrix_before_tool_or_output_access(self):
        with tempfile.TemporaryDirectory(prefix="ask-empty-matrix-") as temporary:
            directory = Path(temporary)
            output = directory / "should-not-exist"
            command = [sys.executable, str(ROOT / "examples/triton_fused_gpu.py")]
            for option in ("ask-opt", "triton-python", "triton-worker", "llc", "ptxas"):
                command.extend(["--" + option, str(directory / "missing-tool")])
            command.extend(["--cache-dir", str(directory / "cache"), "--output-dir", str(output),
                            "--dtypes", "i32", "--cases", "fractions"])
            result = subprocess.run(command, text=True, capture_output=True, timeout=30,
                                    env={**os.environ, "PYTHONPATH": str(ROOT / "python")})
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("no eligible dtype/graph combinations", result.stderr)
            self.assertNotIn("Verified", result.stdout)
            self.assertFalse(output.exists())
            self.assertFalse((directory / "cache").exists())


if __name__ == "__main__":
    unittest.main()
