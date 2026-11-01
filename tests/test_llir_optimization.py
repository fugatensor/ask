"""Check the explicitly selected native LLVM simplification pipeline."""

import ctypes
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = str(Path(sys.argv.pop(1)).resolve())


def run(*arguments, source=None):
    return subprocess.run([DRIVER, *map(str, arguments)], input=source, text=True,
                          capture_output=True, timeout=60)


class LLIROptimizationTests(unittest.TestCase):
    def optimize(self, source):
        with tempfile.TemporaryDirectory(prefix="ask-llir-opt-") as directory:
            report = Path(directory) / "report.json"
            result = run("-", "--optimize-llir", "--report-json", report, source=source)
            self.assertEqual(result.returncode, 0, result.stderr)
            return result.stdout, json.loads(report.read_text())["llir_optimization"]

    def test_constant_control_flow_and_dead_instructions(self):
        source = '''define i32 @calculate() {
entry:
  %dead = mul i32 99, 7
  %sum = add i32 2, 3
  %test = icmp eq i32 %sum, 5
  br i1 %test, label %yes, label %no
yes:
  %result = mul i32 %sum, 4
  ret i32 %result
no:
  ret i32 -1
}
'''
        ir, report = self.optimize(source)
        self.assertIn("ret i32 20", ir)
        self.assertNotIn("br i1", ir)
        self.assertNotIn("mul i32", ir)
        self.assertEqual(report["status"], "applied")
        self.assertEqual([entry["name"] for entry in report["passes"]],
                         ["instcombine", "simplifycfg", "adce"])
        self.assertLess(report["after"]["instructions"], report["before"]["instructions"])
        self.assertEqual(report["after"]["basic_blocks"], 1)

    def test_volatile_inline_ptx_and_target_annotations_survive(self):
        source = '''target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-p:64:64-i64:64-n16:32:64"
define ptx_kernel void @kernel(ptr addrspace(1) %input, ptr addrspace(1) %output) {
  %value = load volatile i32, ptr addrspace(1) %input, align 4
  %unused = load volatile i32, ptr addrspace(1) %input, align 4
  call void asm sideeffect "bar.sync 0;", ""()
  %identity = add i32 %value, 0
  store volatile i32 %identity, ptr addrspace(1) %output, align 4
  ret void
}
!nvvm.annotations = !{!0}
!0 = !{ptr @kernel, !"kernel", i32 1}
'''
        ir, _ = self.optimize(source)
        self.assertIn('target triple = "nvptx64-nvidia-cuda"', ir)
        self.assertIn('target datalayout = "e-p:64:64-i64:64-n16:32:64"', ir)
        self.assertEqual(ir.count("load volatile i32"), 2)
        self.assertIn("store volatile i32", ir)
        self.assertIn('asm sideeffect "bar.sync 0;"', ir)
        self.assertIn("define ptx_kernel void @kernel", ir)
        self.assertIn("!nvvm.annotations", ir)

    def test_semantic_records_survive_and_ssa_maps_are_invalidated(self):
        source = '''define protected fastcc i32 @api(i32 signext %x) noinline {
  %result = add i32 %x, 1, !ask.trace !2, !ask.value_id !3
  ret i32 %result
}
!ask.abi = !{!0}
!ask.shape_snapshot = !{!1}
!ask.boundary_map = !{!2}
!ask.custom_analysis = !{!3}
!llvm.ident = !{!4}
!0 = !{!"abi:api(i32)->i32"}
!1 = !{!"shape:v_x=[10]"}
!2 = !{!"trace:result-is-exact"}
!3 = !{!"logical-v_y"}
!4 = !{!"example-producer"}
'''
        ir, report = self.optimize(source)
        self.assertIn("define protected fastcc i32 @api(i32 signext %x)", ir)
        self.assertIn("noinline", ir)
        self.assertIn("!ask.abi", ir)
        self.assertIn('!"abi:api(i32)->i32"', ir)
        self.assertIn("!ask.shape_snapshot", ir)
        self.assertIn('!"shape:v_x=[10]"', ir)
        self.assertNotIn("!ask.trace", ir)
        self.assertNotIn("!ask.value_id", ir)
        self.assertNotIn("!ask.boundary_map", ir)
        self.assertIn("!llvm.ident", ir)
        self.assertIn("ask.custom_analysis", report["unvalidated_named_metadata"])
        self.assertEqual(report["removed_instruction_attachments"], 2)
        self.assertIn("BoundaryMap", report["invalidated_analyses"])
        self.assertTrue(report["metadata_changed"])

    def test_function_calls_are_not_automatically_inlined(self):
        source = '''declare i32 @external(i32)
define internal i32 @helper(i32 %x) {
  %doubled = mul i32 %x, 2
  ret i32 %doubled
}
define i32 @entry(i32 %x) {
  %a = call i32 @helper(i32 %x)
  %b = call i32 @external(i32 %a)
  ret i32 %b
}
'''
        ir, _ = self.optimize(source)
        self.assertIn("call i32 @helper", ir)
        self.assertIn("call i32 @external", ir)

    def test_function_and_global_trace_attachments_are_removed(self):
        source = '''@buffer = global i32 0, !ask.boundary_map !0, !ask.semantic_tag !1, !ask.boundary_contract !1
define i32 @identity(i32 %x) !ask.trace !0 !ask.semantic_tag !1 {
  ret i32 %x
}
!0 = !{!"old SSA map"}
!1 = !{!"semantic record"}
'''
        ir, report = self.optimize(source)
        self.assertNotIn("!ask.trace", ir)
        self.assertNotIn("!ask.boundary_map", ir)
        self.assertEqual(ir.count("!ask.semantic_tag"), 2)
        self.assertIn("!ask.boundary_contract", ir)
        self.assertEqual(report["removed_object_attachments"], 2)
        self.assertTrue(report["metadata_changed"])
        self.assertEqual(report["status"], "skipped")

    def test_libc_names_are_not_assumed_but_explicit_intrinsics_work(self):
        source = '''target triple = "unknown-unknown-unknown"
@text = private constant [4 x i8] c"abc\\00"
declare i64 @strlen(ptr)
declare i32 @llvm.ctpop.i32(i32)
define i64 @unknown_library() {
  %result = call i64 @strlen(ptr @text)
  ret i64 %result
}
define i32 @known_intrinsic() {
  %result = call i32 @llvm.ctpop.i32(i32 7)
  ret i32 %result
}
'''
        ir, report = self.optimize(source)
        self.assertIn("call i64 @strlen", ir)
        self.assertNotIn("ret i64 3", ir)
        self.assertIn("ret i32 3", ir)
        self.assertFalse(report["library_function_inference"])

    def test_optnone_function_is_skipped(self):
        ir, report = self.optimize('''define i32 @fixed(i32 %x) noinline optnone {
  %identity = add i32 %x, 0
  ret i32 %identity
}
''')
        self.assertIn("add i32 %x, 0", ir)
        self.assertEqual(report["status"], "skipped")
        self.assertEqual(report["functions_skipped_optnone"], 1)

    def test_unchanged_ir_reports_skipped_passes(self):
        ir, report = self.optimize("define i32 @identity(i32 %x) { ret i32 %x }\n")
        self.assertIn("ret i32 %x", ir)
        self.assertEqual(report["status"], "skipped")
        self.assertEqual(report["functions_processed"], 1)
        self.assertEqual(report["functions_changed"], 0)
        self.assertTrue(all(entry["status"] == "skipped" for entry in report["passes"]))
        self.assertFalse(report["metadata_changed"])

    def test_invalid_ir_does_not_replace_existing_outputs(self):
        source = '''define i32 @invalid(i1 %condition) {
entry:
  br i1 %condition, label %left, label %right
left:
  %value = add i32 1, 2
  br label %right
right:
  ret i32 %value
}
'''
        with tempfile.TemporaryDirectory() as directory:
            output, report = Path(directory) / "output.ll", Path(directory) / "report.json"
            output.write_text("previous valid artifact")
            report.write_text("previous valid report")
            result = run("-", "--optimize-llir", "-o", output, "--report-json", report, source=source)
            self.assertGreater(result.returncode, 0, result.stderr)
            self.assertIn("dominate", result.stderr)
            self.assertEqual(output.read_text(), "previous valid artifact")
            self.assertEqual(report.read_text(), "previous valid report")

    def test_numeric_results_match_before_and_after_optimization(self):
        source = '''define i32 @calculate(i32 %x) {
entry:
  %dead = mul i32 %x, 13
  %biased = add i32 %x, 1
  br i1 true, label %selected, label %unused
selected:
  %doubled = mul i32 %biased, 2
  %result = add i32 %doubled, 3
  ret i32 %result
unused:
  ret i32 99
}
define double @floating(double %x) {
  %result = fadd double %x, 0.0
  ret double %result
}
'''
        optimized, _ = self.optimize(source)
        lock = json.loads((ROOT / "cmake/llvm.lock.json").read_text())
        tools = Path(os.environ.get("ASK_LLVM_TOOLS_DIR", ROOT / ".deps" / lock["directory"] / "bin"))
        llc, cc = tools / "llc", shutil.which("cc")
        self.assertTrue(llc.is_file(), "set ASK_LLVM_TOOLS_DIR to the pinned LLVM tools directory")
        self.assertIsNotNone(cc, "a C compiler is required to link the LLVM objects")
        with tempfile.TemporaryDirectory(prefix="ask-llir-numeric-") as directory:
            libraries = []
            for stem, text in (("before", source), ("after", optimized)):
                path = Path(directory)
                llvm_ir, obj, shared = path / (stem + ".ll"), path / (stem + ".o"), path / (stem + ".so")
                llvm_ir.write_text(text)
                compiled = subprocess.run([str(llc), "-filetype=obj", "-relocation-model=pic",
                                           str(llvm_ir), "-o", str(obj)], capture_output=True, text=True, timeout=60)
                self.assertEqual(compiled.returncode, 0, compiled.stderr)
                linked = subprocess.run([cc, "-shared", str(obj), "-o", str(shared)],
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(linked.returncode, 0, linked.stderr)
                library = ctypes.CDLL(str(shared))
                library.calculate.argtypes = [ctypes.c_int32]
                library.calculate.restype = ctypes.c_int32
                library.floating.argtypes = [ctypes.c_double]
                library.floating.restype = ctypes.c_double
                libraries.append(library)
            for value in [-2**31, -2**31 + 1, -1234567, -1, 0, 1, 7, 1234567, 2**31 - 1]:
                with self.subTest(value=value):
                    expected = ctypes.c_int32((value + 1) * 2 + 3).value
                    self.assertEqual(libraries[0].calculate(value), expected)
                    self.assertEqual(libraries[1].calculate(value), expected)
            # Dropping strict fadd +0 would incorrectly preserve negative zero.
            for value in [-0.0, 0.0, -1.5, 1.5, float("inf"), float("nan")]:
                with self.subTest(floating=value):
                    expected = value + 0.0
                    for library in libraries:
                        actual = library.floating(value)
                        if math.isnan(expected):
                            self.assertTrue(math.isnan(actual))
                        else:
                            self.assertEqual(actual, expected)
                            if expected == 0.0:
                                self.assertEqual(math.copysign(1.0, actual), math.copysign(1.0, expected))


if __name__ == "__main__":
    unittest.main()
