"""ASK IR verification, ASK IR graph optimization, LLVM IR cleanup."""
from __future__ import annotations

import ctypes
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT / "python"), str(ROOT / "tests")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from ask import Compiler, CompilerError, Module, ScalarType, TensorType
from support.host_runner import (
    STORAGE,
    ask_opt_path,
    compile_llvm_to_cdll,
    decode_scalar,
    encode_scalar,
    link_object_to_cdll,
    run_ask_opt,
)
from support.pointwise import DTYPES, cast_reference, input_values, numerically_equal, round_reference

NEGATIVE_IR_CASES = {
    "duplicate-external-binding": (
        """module {
  "ask.semantics"() {sym_name = "copy", function_type = (tensor<?xi32>) -> tensor<?xi32>, shape_rule = "same_as", effects = "none"} : () -> ()
  "ask.implementation"() {sym_name = "impl", compiler = "triton", entry = "copy", source = "x",
    signature = {x = "*i32", y = "*i32"}, source_bindings = {x = "input:0", y = "input:0"},
    constants = {}, options = {}, target = {backend = "cuda"}, grid = array<i64: 1, 1, 1>} : () -> ()
  func.func @bad(%x: tensor<10xi32>) -> tensor<10xi32> {
    %y = "ask.external"(%x) {semantics = @copy, implementation = @impl} : (tensor<10xi32>) -> tensor<10xi32>
    return %y : tensor<10xi32>
  }
}""",
        "exactly one valid source binding",
    ),
    "duplicate-operation-id": (
        """module {
  func.func @first() -> i32 {
    %x = "ask.constant"() {ask.id = "same", value = 1 : i32} : () -> i32
    return %x : i32
  }
  func.func @second() -> i32 {
    %x = "ask.constant"() {ask.id = "same", value = 2 : i32} : () -> i32
    return %x : i32
  }
}""",
        "duplicate",
    ),
    "external-shape-conflict": (
        """module {
  "ask.semantics"() {sym_name = "copy", function_type = (tensor<?xi32>) -> tensor<?xi32>, shape_rule = "same_as", effects = "none"} : () -> ()
  "ask.implementation"() {sym_name = "impl", compiler = "triton", entry = "copy", source = "x",
    signature = {x = "*i32", y = "*i32"}, source_bindings = {x = "input:0", y = "output:0"},
    constants = {}, options = {}, target = {backend = "cuda"}, grid = array<i64: 1, 1, 1>} : () -> ()
  func.func @bad(%x: tensor<10xi32>) -> tensor<9xi32> {
    %y = "ask.external"(%x) {semantics = @copy, implementation = @impl} : (tensor<10xi32>) -> tensor<9xi32>
    return %y : tensor<9xi32>
  }
}""",
        "error",
    ),
    "invalid-revision": (
        'module attributes {ask.ir_revision = "old"} { func.func @main(%x: tensor<10xi32>) -> tensor<10xi32> { return %x : tensor<10xi32> } }',
        "ir_revision",
    ),
    "malformed-shape-operand": (
        'module { "ask.semantics"() {sym_name = "fixed", function_type = () -> tensor<10xi32>, shape_rule = "fixed", shape_operand = "first", effects = "none"} : () -> () func.func @main() { return } }',
        "shape_operand",
    ),
    "marker-identity-error": (
        'module { func.func @main(%x: tensor<10xi32> {ask.value_id = "v_x"}) -> tensor<10xi32> { %m = "ask.mark"(%x) {mark_id = "bad", value_id = "unbound", role = "value"} : (tensor<10xi32>) -> tensor<10xi32> return %m : tensor<10xi32> } }',
        "value_id",
    ),
    "marker-port-error": (
        'module { func.func @main(%x: tensor<10xi32> {ask.value_id = "v_x"}) -> tensor<10xi32> { %m = "ask.mark"(%x) {mark_id = "bad", value_id = "v_x", role = "input", port_id = "Ghost.in"} : (tensor<10xi32>) -> tensor<10xi32> return %m : tensor<10xi32> } }',
        "port_id must name an existing data port",
    ),
    "shape-conflict": (
        'module { func.func @main(%x: tensor<10xi32>) -> tensor<10xi32> { %one = "ask.constant"() {value = 1 : i32} : () -> i32 %ones = "ask.splat"(%one) {shape = array<i64: 9>} : (i32) -> tensor<9xi32> %t = "ask.add"(%x, %ones) : (tensor<10xi32>, tensor<9xi32>) -> tensor<10xi32> return %t : tensor<10xi32> } }',
        "error",
    ),
    "tensor-size-overflow": (
        "module { func.func @bad(%x: tensor<2305843009213693952xi32>) -> tensor<2305843009213693952xi32> { return %x : tensor<2305843009213693952xi32> } }",
        "tensor byte count exceeds",
    ),
    "unknown-input-shape": (
        "module { func.func @main(%x: tensor<?xi32>) -> tensor<?xi32> { return %x : tensor<?xi32> } }",
        "concrete",
    ),
    "unresolved-symbol": (
        'module { func.func @main(%x: tensor<10xi32>) -> tensor<10xi32> { %y = "ask.external"(%x) {semantics = @missing, implementation = @missing_impl} : (tensor<10xi32>) -> tensor<10xi32> return %y : tensor<10xi32> } }',
        "semantics must resolve to an ask.semantics symbol",
    ),
    "unsupported-graph-op": (
        "module { func.func @other(%x: tensor<10xi32>) -> tensor<10xi32> { return %x : tensor<10xi32> } func.func @main(%x: tensor<10xi32>) -> tensor<10xi32> { %y = func.call @other(%x) : (tensor<10xi32>) -> tensor<10xi32> return %y : tensor<10xi32> } }",
        "func.call",
    ),
    "unsupported-scalar": (
        'module { func.func @main() -> f64 { %x = "ask.constant"() {value = 1.0 : f64} : () -> f64 return %x : f64 } }',
        "requires a scalar bool, i32, i64, f32, f16 or bf16 result",
    ),
    "unsupported-schema": (
        "module attributes {ask.schema_version = 2 : i64} { func.func @main(%x: tensor<10xi32>) -> tensor<10xi32> { return %x : tensor<10xi32> } }",
        "schema_version",
    ),
}


def shared_graph() -> Module:
    module = Module()
    tensor = TensorType((10,))
    graph = module.graph("main", inputs={"x": tensor}, results=[tensor, tensor])
    t = graph.add(graph.inputs["x"], graph.splat(graph.constant(1), (10,)), id="P", value_id="v_t")
    y = graph.mul(t, graph.splat(graph.constant(2), (10,)), id="Q", value_id="v_y")
    graph.return_(t, y)
    return module


class IRVerificationTests(unittest.TestCase):
    def test_positive_fixtures_are_idempotent(self):
        for name in ("pipeline", "refine-and-erase"):
            with self.subTest(name=name):
                result = run_ask_opt(ROOT / "tests/ir" / f"{name}.mlir", "--prepare")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn('"ask.mark"', result.stdout)
                again = run_ask_opt("-", "--prepare", source=result.stdout)
                self.assertEqual(again.returncode, 0, again.stderr)
                self.assertEqual(again.stdout, result.stdout)

    def test_negative_ir_cases_emit_expected_diagnostics(self):
        for name, (mlir, needle) in NEGATIVE_IR_CASES.items():
            with self.subTest(name=name):
                result = run_ask_opt("-", "--prepare", source=mlir)
                self.assertGreater(result.returncode, 0, result.stdout)
                self.assertIn(needle.lower(), result.stderr.lower(), result.stderr)

    def test_llir_check_and_external_codegen_boundary(self):
        ptx_ll = 'target triple = "nvptx64-nvidia-cuda"\ndefine void @f() { call void asm sideeffect "bar.sync 0;", ""() ret void }\n'
        ok = run_ask_opt("-", "--check-llir", source=ptx_ll)
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertIn('asm sideeffect "bar.sync 0;"', ok.stdout)
        bad = run_ask_opt("-", "--check-llir", source="define void @f() { ret i32 1 }")
        self.assertGreater(bad.returncode, 0)

        ext_lower = run_ask_opt(ROOT / "tests/ir/pipeline.mlir", "--emit-llvm")
        self.assertGreater(ext_lower.returncode, 0)
        self.assertIn("external", ext_lower.stderr)

        caps = json.loads(run_ask_opt("--capabilities").stdout)
        self.assertTrue(caps["external_execution"])
        self.assertEqual(caps["external_formats"], ["cubin", "ptx"])


class GraphOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.compiler = Compiler(ask_opt_path())

    def test_constant_and_cast_folding_across_all_types(self):
        module, reference = Module(), Module()
        for source in DTYPES:
            values = input_values(12, source)
            for target in DTYPES:
                name = f"{source}_to_{target}"
                g = module.graph(name, inputs={}, results=[ScalarType(target)] * len(values))
                ref = reference.graph(name, inputs={}, results=[ScalarType(target)] * len(values))
                g.return_(*(g.cast(g.constant(v, dtype=source), dtype=target) for v in values))
                ref.return_(*(ref.constant(cast_reference(round_reference(v, source), source, target),
                                           dtype=target) for v in values))
        optimized = self.compiler.optimize(module)
        prepared_ref = self.compiler.prepare(reference)
        self.assertNotIn('"ask.cast"', optimized.ir)
        self.assertEqual(optimized.report["optimization"]["constant_folds"], len(DTYPES) ** 2 * 12)
        extract = lambda ir: re.findall(r"\bvalue = ([^},]+)", ir)
        self.assertEqual(extract(optimized.ir), extract(prepared_ref.ir))

    def test_float_non_algebraic_rounding_and_dce_markers(self):
        for dtype, bits in (("f32", 23), ("f16", 10), ("bf16", 7)):
            mod = Module()
            g = mod.graph("rounding", inputs={}, results=[ScalarType(dtype)])
            prod = g.mul(g.constant(1.0 + 2**-bits, dtype=dtype), g.constant(1.0 - 2**-bits, dtype=dtype))
            g.return_(g.add(prod, g.constant(-1.0, dtype=dtype)))
            opt = self.compiler.optimize(mod)
            self.assertIn(f"value = 0.000000e+00 : {dtype}", opt.ir)

            dyn = Module()
            dg = dyn.graph("id", inputs={"x": ScalarType(dtype)}, results=[ScalarType(dtype), ScalarType(dtype)])
            zero = dg.constant(0.0, dtype=dtype)
            dg.return_(dg.add(dg.inputs["x"], zero), dg.mul(dg.inputs["x"], zero))
            res = self.compiler.optimize(dyn)
            self.assertIn('"ask.add"', res.ir)
            self.assertIn('"ask.mul"', res.ir)

        dce_mod = Module()
        tensor = TensorType((10,))
        g = dce_mod.graph("main", inputs={"x": tensor}, results=[tensor])
        x = g.inputs["x"]
        dead_in = g.mark(x, mark_id="dead_input", role="input", port_id="dead.in0")
        g.add(dead_in, g.splat(g.constant(1), (10,)), id="dead", value_id="dead_value")
        g.mark(x, mark_id="live_value")
        g.return_(x)
        result = self.compiler.optimize(dce_mod, fold_constants=False)
        markers = {m["mark_id"]: m for m in result.report["markers"]}
        self.assertEqual(markers["dead_input"]["status"], "eliminated")
        self.assertEqual(markers["live_value"]["status"], "exact")

        pipeline_src = (ROOT / "tests/ir/pipeline.mlir").read_text(encoding="utf-8")
        ext_opt = self.compiler.optimize(pipeline_src.replace("return %y : tensor<10xi32>", "return %x : tensor<10xi32>"))
        self.assertIn('"ask.external"', ext_opt.ir)
        self.assertNotIn('ask.id = "Q"', ext_opt.ir)

    def test_folded_constants_match_host_execution(self):
        if not self.compiler.capabilities()["host_codegen"]:
            self.skipTest("host backend disabled")
        module = Module()
        float_cases = [
            ("add", 16777216.0, 1.0, 16777216.0),
            ("add", -0.0, -0.0, -0.0),
            ("add", -0.0, 0.0, 0.0),
            ("mul", float("inf"), 0.0, float("nan")),
            ("mul", 3.4028234663852886e38, 2.0, float("inf")),
        ]
        for idx, (op, lhs, rhs, _) in enumerate(float_cases):
            t = TensorType((2, 2), "f32")
            g = module.graph(f"fcase{idx}", inputs={}, results=[t])
            g.return_(getattr(g, op)(g.splat(g.constant(lhs, dtype="f32"), (2, 2)),
                                     g.splat(g.constant(rhs, dtype="f32"), (2, 2))))
        optimized = self.compiler.optimize(module)
        self.assertNotIn('"ask.add"', optimized.ir)
        self.assertNotIn('"ask.mul"', optimized.ir)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            for label, src in (("before", module.to_mlir()), ("after", optimized.ir)):
                obj = work / f"{label}.o"
                res = run_ask_opt("-", "--emit-object", "-o", obj, source=src)
                self.assertEqual(res.returncode, 0, res.stderr)
                lib = link_object_to_cdll(obj, work / f"{label}.so")
                for idx, (_, _, _, expected) in enumerate(float_cases):
                    fn = getattr(lib, f"fcase{idx}")
                    fn.argtypes = [ctypes.POINTER(ctypes.c_float)]
                    out = (ctypes.c_float * 4)()
                    fn(out)
                    for actual in out:
                        if math.isnan(expected):
                            self.assertTrue(math.isnan(actual))
                        else:
                            self.assertEqual(struct.pack("f", actual), struct.pack("f", expected))


class LLIROptimizationAndBoundaryExecutionTests(unittest.TestCase):
    def optimize_llir(self, source: str):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            result = run_ask_opt("-", "--optimize-llir", "--report-json", report, source=source)
            self.assertEqual(result.returncode, 0, result.stderr)
            return result.stdout, json.loads(report.read_text(encoding="utf-8"))["llir_optimization"]

    def test_llir_simplification_ptx_survival_and_metadata_scrubbing(self):
        ir, report = self.optimize_llir(
            "define i32 @calc() {\nentry:\n  %d = mul i32 9, 7\n  %s = add i32 2, 3\n"
            "  %c = icmp eq i32 %s, 5\n  br i1 %c, label %y, label %n\n"
            "y:\n  %r = mul i32 %s, 4\n  ret i32 %r\nn:\n  ret i32 -1\n}\n"
        )
        self.assertIn("ret i32 20", ir)
        self.assertEqual([p["name"] for p in report["passes"]], ["instcombine", "simplifycfg", "adce"])

        nvptx, _ = self.optimize_llir(
            'target triple = "nvptx64-nvidia-cuda"\n'
            'define ptx_kernel void @k(ptr addrspace(1) %in, ptr addrspace(1) %out) {\n'
            '  %v = load volatile i32, ptr addrspace(1) %in, align 4\n'
            '  call void asm sideeffect "bar.sync 0;", ""()\n'
            '  store volatile i32 %v, ptr addrspace(1) %out, align 4\n  ret void\n}\n'
            '!nvvm.annotations = !{!0}\n!0 = !{ptr @k, !"kernel", i32 1}\n'
        )
        self.assertIn('asm sideeffect "bar.sync 0;"', nvptx)
        self.assertIn("!nvvm.annotations", nvptx)

        meta_ir, meta_rep = self.optimize_llir(
            'define i32 @api(i32 %x) {\n  %r = add i32 %x, 1, !ask.trace !1, !ask.value_id !1\n  ret i32 %r\n}\n'
            '!ask.shape_snapshot = !{!0}\n!ask.trace = !{!1}\n!0 = !{!"s"}\n!1 = !{!"t"}\n'
        )
        self.assertIn("!ask.shape_snapshot", meta_ir)
        self.assertNotIn("!ask.trace", meta_ir)
        self.assertIn("trace", meta_rep["invalidated_analyses"])



if __name__ == "__main__":
    unittest.main()
