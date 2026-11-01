import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
spec = importlib.util.spec_from_file_location("ask_triton_worker_tests", ROOT / "worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def request():
    return json.loads((FIXTURES / "scale2.request.json").read_text())


def compiled(metadata_changes=None, llvm_ir=None):
    metadata = {"name": "scale2", "triton_version": "3.7.1",
                "target": {"backend": "cuda", "arch": 80, "warp_size": 32},
                "num_warps": 4, "warp_size": 32, "num_ctas": 1,
                "shared": 0, "global_scratch_size": 0, "global_scratch_align": 1,
                "profile_scratch_size": 0, "profile_scratch_align": 1,
                "tmem_size": 0, "tensordesc_meta": [], "instrumentation_mode": "",
                "launch_cooperative_grid": False, "launch_pdl": False, "extern_libs": []}
    metadata.update(metadata_changes or {})
    return SimpleNamespace(name="scale2", metadata=SimpleNamespace(_asdict=lambda: metadata),
                           asm={"llir": llvm_ir if llvm_ir is not None else
                                (FIXTURES / "scale2.ll").read_text()})


class RequestTests(unittest.TestCase):
    def test_valid_request(self):
        value = request()
        self.assertEqual(worker.validate_request(value), value)

    def test_rejects_boolean_protocol(self):
        value = request()
        value["protocol_version"] = True
        with self.assertRaisesRegex(worker.WorkerError, "protocol_version"):
            worker.validate_request(value)

    def test_rejects_unknown_fields(self):
        value = request()
        value["run"] = True
        with self.assertRaisesRegex(worker.WorkerError, "unknown"):
            worker.validate_request(value)

    def test_rejects_nonpointer_signature(self):
        value = request()
        value["signature"]["t"] = "i32"
        with self.assertRaisesRegex(worker.WorkerError, r"\*i32"):
            worker.validate_request(value)

    def test_rejects_missing_or_ambiguous_source_binding(self):
        for bad in ({"t": "input:0"}, {"t": "input:0", "u": "input:0"},
                    {"t": [], "u": "output:0"}):
            with self.subTest(binding=bad):
                value = request()
                value["source_bindings"] = bad
                with self.assertRaises(worker.WorkerError):
                    worker.validate_request(value)

    def test_constexpr_must_exist_in_source(self):
        value = request()
        value["constants"]["MISSING"] = 10
        with self.assertRaisesRegex(worker.WorkerError, "argument names"):
            worker.validate_request(value)

    def test_constants_cannot_be_runtime_parameters(self):
        value = request()
        value["source"] = value["source"].replace("N: tl.constexpr", "N")
        with self.assertRaisesRegex(worker.WorkerError, "tl.constexpr"):
            worker.validate_request(value)

    def test_source_must_be_decorated(self):
        value = request()
        value["source"] = value["source"].replace("@triton.jit\n", "")
        with self.assertRaisesRegex(worker.WorkerError, "@triton.jit"):
            worker.validate_request(value)

    def test_unknown_options_are_not_ignored(self):
        value = request()
        value["fixed_options"]["made_up"] = 1
        with self.assertRaisesRegex(worker.WorkerError, "fixed_options"):
            worker.validate_request(value)

    def test_grid_is_not_inferred(self):
        value = request()
        value["grid"] = [1, 1]
        with self.assertRaisesRegex(worker.WorkerError, "grid"):
            worker.validate_request(value)

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"protocol_version": 1, "protocol_version": 1}')
            with self.assertRaisesRegex(worker.WorkerError, "duplicate JSON"):
                worker.read_request(path)


class ArtifactTests(unittest.TestCase):
    def test_records_both_zero_resource_hidden_parameters(self):
        artifact = worker.build_artifact(request(), compiled(), "3.7.1", ["t", "u"])
        self.assertEqual(len(artifact["entry_signature"]["parameters"]), 4)
        self.assertEqual([arg["binding"] for arg in artifact["invocation"]["arguments"]],
                         ["input:0", "output:0", "null", "null"])
        self.assertEqual(artifact["port_map"][0]["source_parameter"], "t")
        self.assertFalse(artifact["validation"]["llvm_module_verified"])
        self.assertNotIn("N", [entry["source_parameter"] for entry in artifact["port_map"]])
        json.dumps(artifact, allow_nan=False)

    def test_mapping_uses_compiler_source_order_not_request_dict_order(self):
        value = request()
        value["signature"] = {"u": "*i32", "t": "*i32"}
        artifact = worker.build_artifact(value, compiled(), "3.7.1", ["t", "u"])
        self.assertEqual(artifact["port_map"][0]["logical_port"], "input:0")
        self.assertEqual(artifact["port_map"][0]["argument"], 0)

    def test_nonzero_scratch_does_not_silently_bind_null(self):
        with self.assertRaisesRegex(worker.WorkerError, "zero global_scratch_size"):
            worker.build_artifact(request(), compiled({"global_scratch_size": 16}),
                                  "3.7.1", ["t", "u"])

    def test_missing_resource_metadata_is_unknown_not_zero(self):
        with self.assertRaisesRegex(worker.WorkerError, "zero shared"):
            worker.build_artifact(request(), compiled({"shared": None}), "3.7.1", ["t", "u"])

    def test_changed_target_is_rejected(self):
        with self.assertRaisesRegex(worker.WorkerError, "target differs"):
            worker.build_artifact(request(), compiled({"target": {"backend": "cuda", "arch": 90,
                                                                   "warp_size": 32}}),
                                  "3.7.1", ["t", "u"])

    def test_hidden_argument_count_change_is_rejected(self):
        llvm_ir = (FIXTURES / "scale2.ll").read_text().replace(
            ", ptr addrspace(1) readnone captures(none) %3", "")
        with self.assertRaisesRegex(worker.WorkerError, "two scratch"):
            worker.build_artifact(request(), compiled(llvm_ir=llvm_ir), "3.7.1", ["t", "u"])

    def test_unknown_parameter_attributes_are_rejected(self):
        llvm_ir = (FIXTURES / "scale2.ll").read_text().replace(
            "ptr addrspace(1) %0", "ptr addrspace(1) byval(i32) %0")
        with self.assertRaisesRegex(worker.WorkerError, "unsupported attributes"):
            worker.parse_entry_signature(llvm_ir, "scale2")

    def test_unrecognized_address_space_is_rejected(self):
        llvm_ir = (FIXTURES / "scale2.ll").read_text().replace("addrspace(1) %0", "addrspace(3) %0")
        with self.assertRaisesRegex(worker.WorkerError, "address-space-1"):
            worker.parse_entry_signature(llvm_ir, "scale2")

    def test_missing_llir_is_not_a_successful_export(self):
        value = compiled()
        value.asm = {"cubin": b"binary is not LLVM IR"}
        with self.assertRaisesRegex(worker.WorkerError, "asm"):
            worker.build_artifact(request(), value, "3.7.1", ["t", "u"])


class CommandLineTests(unittest.TestCase):
    def test_invalid_request_needs_no_triton_and_publishes_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            output = Path(directory) / "response.json"
            path.write_text('{"protocol_version": 99}')
            result = subprocess.run([sys.executable, "-S", str(ROOT / "worker.py"),
                                     "--request", str(path), "--output", str(output)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())
            self.assertEqual(result.stdout, "")
            self.assertEqual(json.loads(result.stderr)["error"]["code"], "InvalidRequest")

    def test_loading_worker_does_not_import_triton(self):
        script = ("import importlib.util, sys; "
                  f"s=importlib.util.spec_from_file_location('worker', {str(ROOT / 'worker.py')!r}); "
                  "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                  "assert 'triton' not in sys.modules")
        result = subprocess.run([sys.executable, "-S", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
