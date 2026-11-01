"""Exercise the required extension without launching an ask-opt subprocess."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from ask import Compiler, CompilerError, Module, TensorType

AVAILABLE = importlib.util.find_spec("ask._native") is not None
if os.environ.get("ASK_NATIVE_REQUIRED") and not AVAILABLE:
    raise RuntimeError("the required ASK native extension was not built or is missing from PYTHONPATH")


def graph_source():
    module = Module()
    tensor = TensorType((2, 5), "f16")
    graph = module.graph("mixed", inputs={"x": tensor}, results=[tensor])
    wide = graph.cast(graph.inputs["x"], dtype="f32")
    bias = graph.splat(graph.constant(0.1, dtype="f32"), (2, 5))
    result = graph.cast(graph.add(wide, bias), dtype="f16")
    graph.return_(graph.mark(result, mark_id="output"))
    return module


@unittest.skipUnless(AVAILABLE, "build/install the required native extension for binding tests")
class BindingTests(unittest.TestCase):
    def test_prepare_and_capabilities_run_inside_current_process(self):
        with patch("ask.compiler.subprocess.run", side_effect=AssertionError("unexpected subprocess")):
            compiler = Compiler()
            caps = compiler.capabilities()
            result = compiler.prepare(graph_source())
        self.assertEqual(caps["python_bindings"], "nanobind")
        self.assertTrue(caps["python_bindings_enabled"])
        self.assertTrue(caps["native_prepare"])
        self.assertEqual(result.report["phase"], "prepared")
        self.assertNotIn('"ask.mark"', result.ir)
        self.assertEqual(result.report["markers"][0]["mark_id"], "output")
        self.assertTrue(all(item["shape"] == [2, 5] for item in result.report["shape_snapshot"]["shape_bindings"]))

    def test_native_optimization_and_diagnostics(self):
        with patch("ask.compiler.subprocess.run", side_effect=AssertionError("unexpected subprocess")):
            compiler = Compiler()
            result = compiler.optimize(graph_source())
            self.assertEqual(result.report["phase"], "optimized")
            with self.assertRaises(CompilerError) as caught:
                compiler.prepare("module { unknown }")
        self.assertIn("error:", caught.exception.diagnostics)
        self.assertIn("<ask-python>", caught.exception.diagnostics)
        self.assertIsNone(caught.exception.returncode)

    def test_calls_own_independent_contexts_and_reports(self):
        source = graph_source().to_mlir()
        compiler = Compiler()
        with ThreadPoolExecutor(max_workers=4) as executor:
            prepared = list(executor.map(compiler.prepare, [source] * 12))
        self.assertTrue(all(result.ir == prepared[0].ir for result in prepared))
        self.assertTrue(all(result.report == prepared[0].report for result in prepared))

    @unittest.skipUnless(os.environ.get("ASK_OPT"), "requires matching CLI for compatibility comparison")
    def test_cli_and_extension_share_capabilities_and_preparation(self):
        native = Compiler()
        cli = Compiler(os.environ["ASK_OPT"])
        self.assertEqual(native.capabilities(), cli.capabilities())
        self.assertEqual(native.prepare(graph_source()), cli.prepare(graph_source()))
        self.assertEqual(Path(native.executable), Path(os.environ["ASK_OPT"]))
        native._check_companion()

    def test_different_cli_build_cannot_be_used_for_external_compilation(self):
        compiler = Compiler()
        capabilities = compiler.capabilities()
        capabilities["build_id"] = "another-build"
        with patch("ask.compiler.subprocess.run", return_value=subprocess.CompletedProcess(
                [], 0, json.dumps(capabilities), "")):
            with self.assertRaisesRegex(CompilerError, "different builds"):
                compiler._check_companion()


if __name__ == "__main__":
    unittest.main()
