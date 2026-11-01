"""Strict boundary proofs and opt-in real Triton identity-trace export."""

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/scale2.request.json"
PYTHON = os.environ.get("ASK_TEST_TRITON_PYTHON")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fusion = load_module("ask_boundary_fusion_tests", ROOT / "fusion.py")
worker = load_module("ask_boundary_worker_tests", ROOT / "worker.py")


def request():
    value = json.loads(FIXTURE.read_text())
    value["fusion"] = {"enabled": True}
    return value


class BoundarySourceTests(unittest.TestCase):
    def test_environment_cannot_override_code_after_boundary_proof(self):
        rejected = ("TRITON_KERNEL_OVERRIDE", "TRITON_OVERRIDE_DIR", "TRITON_LIBDEVICE_PATH",
                    "TRITON_PTXAS_PATH", "TRITON_CACHE_MANAGER", "TRITON_NEW_UNKNOWN_SWITCH",
                    "LLVM_PASS_PLUGIN_PATH")
        for name in rejected:
            with self.subTest(name=name), patch.dict(os.environ, {name: "enabled"}, clear=True):
                with self.assertRaisesRegex(fusion.FusionError, name):
                    fusion.instrument(request())
        allowed = {name: "/tmp/ask-test" for name in fusion.ALLOWED_TRITON_ENVIRONMENT}
        allowed.update({name: "" for name in rejected})
        allowed["PATH"] = "/usr/bin"
        with patch.dict(os.environ, allowed, clear=True):
            source, contract = fusion.instrument(request())
            self.assertTrue(source)
            self.assertEqual(contract["shape"], [10])
        # The restriction is opt-in for fusion; ordinary export validation does
        # not start rejecting compiler environments that it previously accepted.
        normal = request()
        del normal["fusion"]
        with patch.dict(os.environ, {"TRITON_KERNEL_OVERRIDE": "1"}, clear=True):
            self.assertEqual(worker.validate_request(normal), normal)

    def test_instrumentation_is_identity_and_keeps_source_and_configuration(self):
        value = request()
        before = copy.deepcopy(value)
        source, contract = fusion.instrument(worker.validate_request(value))
        self.assertEqual(value, before)
        self.assertEqual(contract["original_source_sha256"],
                         hashlib.sha256(value["source"].encode()).hexdigest())
        self.assertEqual(contract["instrumented_source_sha256"],
                         hashlib.sha256(source.encode()).hexdigest())
        self.assertEqual(contract["grid"], value["grid"])
        self.assertEqual(contract["block"], value["constants"]["BLOCK"])
        self.assertEqual(contract["num_warps"], value["fixed_options"]["num_warps"])
        tree = ast.parse(source)
        traces = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "inline_asm_elementwise"]
        self.assertEqual(len(traces), 2)
        for node in traces:
            attributes = {item.arg: item.value for item in node.keywords}
            self.assertFalse(attributes["is_pure"].value)
            self.assertEqual(attributes["pack"].value, 1)
            self.assertEqual(attributes["constraints"].value, "=r,r,l,r,r")
            self.assertEqual(len(attributes["args"].elts), 4)
        # Instrumentation does not replace F's arithmetic with an ASK result.
        self.assertIn("value * 2", source)
        self.assertIn("tl.load(t + i", source)
        self.assertIn("tl.store(u + i", source)
        self.assertTrue(contract["proof"]["same_index"])
        self.assertTrue(contract["proof"]["same_mask"])

    def test_named_mask_and_pointwise_assignments_are_supported(self):
        value = request()
        value["source"] = value["source"].replace(
            "    value = tl.load", "    active = i < N\n    value = tl.load").replace(
            "mask=i < N", "mask=active").replace(
            "    tl.store(u + i, value * 2", "    doubled = value * 2\n    tl.store(u + i, doubled + 7")
        source, contract = fusion.instrument(worker.validate_request(value))
        self.assertIn("doubled = value * 2", source)
        self.assertIn("doubled + 7", source)
        self.assertEqual(contract["shape"], [10])

    def test_rejects_unproven_memory_effects_indices_and_control_flow(self):
        original = request()["source"]
        rejected = {
            "second_load": original.replace("value * 2", "value + tl.load(t + i, mask=i < N, other=0)"),
            "unmasked_load": original.replace(", mask=i < N, other=0", ""),
            "nonzero_other": original.replace("other=0", "other=1"),
            "volatile_load": original.replace("other=0)", "other=0, volatile=True)"),
            "wrong_program_axis": original.replace("tl.program_id(0)", "tl.program_id(1)"),
            "inclusive_mask": original.replace("i < N", "i <= N"),
            "different_store_index": original.replace("tl.store(u + i", "tl.store(u + i + 1"),
            "different_store_mask": original.replace("value * 2, mask=i < N", "value * 2, mask=i < N - 1"),
            "input_write": original.replace("tl.store(u + i", "tl.store(t + i"),
            "atomic": original.replace("tl.store(u + i, value * 2, mask=i < N)",
                                        "tl.atomic_add(u + i, value, mask=i < N)"),
            "pointer_escape": original.replace("value * 2", "t.to(tl.int64).to(tl.int32)"),
            "foreign_inline_asm": original.replace(
                "value * 2", 'tl.inline_asm_elementwise("trap;", "=r", [], tl.int32, False, 1)'),
            "reduction": original.replace("value * 2", "tl.sum(value, axis=0)"),
            "extra_effect": original.replace("    tl.store", "    tl.debug_barrier()\n    tl.store"),
            "shadowed_pointer": original.replace("    tl.store", "    t = u\n    tl.store"),
            "reassigned_index": original.replace("    tl.store", "    i = i + 1\n    tl.store"),
            "branch": original.replace("    tl.store", "    if N > 0:\n        value = value + 1\n    tl.store"),
            "loop": original.replace("    tl.store", "    for j in range(2):\n        value = value + 1\n    tl.store"),
            "float": original.replace("value * 2", "value * 2.0"),
            "division": original.replace("value * 2", "value // 2"),
        }
        for label, source in rejected.items():
            with self.subTest(label=label):
                value = request()
                value["source"] = source
                with self.assertRaises(fusion.FusionError):
                    fusion.instrument(worker.validate_request(value))

    def test_shape_and_launch_must_match_existing_strategy(self):
        for field, replacement in (("constants", {"N": 0, "BLOCK": 128}),
                                   ("constants", {"N": fusion.MAX_ELEMENTS + 1, "BLOCK": 128}),
                                   ("constants", {"N": 10, "BLOCK": 256}),
                                   ("grid", [2, 1, 1]),
                                   ("grid", [1, 2, 1])):
            with self.subTest(field=field, replacement=replacement):
                value = request()
                value[field] = replacement
                with self.assertRaises(fusion.FusionError):
                    fusion.instrument(worker.validate_request(value))

    def test_fusion_flag_is_strict_and_optional(self):
        value = request()
        del value["fusion"]
        self.assertEqual(worker.validate_request(value), value)
        value["fusion"] = {"enabled": False}
        self.assertEqual(worker.validate_request(value), value)
        for option in (True, {"enabled": 1}, {}, {"enabled": True, "ignore_errors": True}):
            with self.subTest(option=option):
                value["fusion"] = option
                with self.assertRaises(worker.WorkerError):
                    worker.validate_request(value)


@unittest.skipUnless(PYTHON, "set ASK_TEST_TRITON_PYTHON for real trace exports")
class RealBoundaryExportTests(unittest.TestCase):
    def test_real_trace_exports_bind_source_ir_layout_and_partial_final_block(self):
        with tempfile.TemporaryDirectory(prefix="ask-boundary-export-test-") as directory:
            directory = Path(directory)
            for count in (10, 129):
                with self.subTest(count=count):
                    value = request()
                    value["constants"]["N"] = count
                    value["grid"] = [(count + 127) // 128, 1, 1]
                    source = directory / f"request-{count}.json"
                    output = directory / f"artifact-{count}.json"
                    source.write_text(json.dumps(value))
                    result = subprocess.run([PYTHON, str(ROOT / "worker.py"), "--request", str(source),
                                             "--output", str(output)], text=True, capture_output=True, timeout=180)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    artifact = json.loads(output.read_text())
                    contract = artifact["fusion_contract"]
                    self.assertEqual(contract["protocol"], "ask.triton.boundary.v1")
                    self.assertEqual(contract["llir_sha256"],
                                     hashlib.sha256(artifact["payload"].encode()).hexdigest())
                    self.assertEqual(contract["original_source_sha256"],
                                     artifact["specializations"]["source_sha256"])
                    self.assertEqual(contract["instrumented_source_sha256"],
                                     hashlib.sha256(artifact["instrumented_source"].encode()).hexdigest())
                    self.assertTrue(contract["proof"]["layout"]["matched_uninstrumented_baseline"])
                    self.assertEqual(contract["shape"], [count])
                    self.assertEqual(artifact["invocation"]["grid"], value["grid"])
                    self.assertEqual(artifact["invocation"]["num_warps"], 4)
                    self.assertEqual(artifact["invocation"]["shared_bytes"], 0)
                    self.assertFalse(artifact["validation"]["device_execution_performed"])
                    for role in ("input", "output"):
                        boundary = contract[role]
                        self.assertEqual(artifact["payload"].count(fusion._llvm_string(boundary["marker_asm"])), 1)
                        self.assertEqual(boundary["arguments"], {"value": 0, "address": 1, "index": 2, "mask": 3})


if __name__ == "__main__":
    unittest.main()
