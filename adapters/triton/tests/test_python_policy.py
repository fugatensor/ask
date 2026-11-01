"""Producer-specific Python policies remain inside the Triton adapter."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "python"))
from ask import CompilerError
from ask.adapters import CompilerEnvironment

spec = importlib.util.spec_from_file_location("triton_python_policy_tests", ROOT / "python/__init__.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)
_numerics, _validate_environment = policy._numerics, policy._validate_environment


class PolicyTests(unittest.TestCase):
    def test_dependency_contract_change_invalidates_cache_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("worker.py", "fusion.py", "dependencies.lock.json"):
                (root / name).write_bytes((ROOT / name).read_bytes())
            request = {"signature": {"a": "*i32", "b": "*i32"},
                       "source": "source", "target": {"backend": "cuda", "arch": 80}}
            environment = CompilerEnvironment("python", str(root / "worker.py"))
            with patch.object(policy.subprocess, "run", return_value=SimpleNamespace(stdout='{}')), \
                 patch.dict(os.environ, {}, clear=True):
                before = policy.create_adapter().cache_identity(request, environment, timeout=1)
                lock = json.loads((root / "dependencies.lock.json").read_text())
                lock["adapter_version"] = "0.7.1"
                (root / "dependencies.lock.json").write_text(json.dumps(lock))
                after = policy.create_adapter().cache_identity(request, environment, timeout=1)
            self.assertNotEqual(before["adapter"]["dependencies.lock.json"],
                                after["adapter"]["dependencies.lock.json"])
            self.assertEqual(before["adapter"]["worker.py"], after["adapter"]["worker.py"])

    def test_python_requirement_and_override_reject_before_probing_or_cache(self):
        adapter = policy.create_adapter()
        with patch.object(policy.subprocess, "run") as run:
            with self.assertRaisesRegex(CompilerError, "Python interpreter"):
                adapter.cache_identity({}, CompilerEnvironment(worker="exporter"), timeout=1)
            with patch.dict(os.environ, {"TRITON_LIBDEVICE_PATH": "/changed-in-place.bc"}, clear=True):
                with self.assertRaisesRegex(CompilerError, "TRITON_LIBDEVICE_PATH"):
                    adapter.cache_identity({}, CompilerEnvironment("python", "missing.py"), timeout=1)
            run.assert_not_called()

    def test_explicit_legacy_migration_preserves_policy_and_shape(self):
        import test_worker
        request = test_worker.request()
        original = test_worker.worker.build_artifact(request, test_worker.compiled(), "3.7.1", ["t", "u"])
        for enabled in (True, False):
            legacy = json.loads(json.dumps(original))
            legacy["invocation"].update(legacy.pop("producer_metadata"))
            legacy["invocation"]["version"] = 1
            legacy["invocation"]["compiler_metadata"].update(enable_fp_fusion=enabled, enable_reflect_ftz=True)
            del legacy["codegen"]
            for port in legacy["port_map"]:
                del port["shape"], port["layout"]
            updated = policy.normalize_legacy_artifact(legacy, shape=(2, 5))
            self.assertEqual(updated["invocation"]["version"], 2)
            self.assertEqual(updated["codegen"], {"fp_contract": "fast" if enabled else "off", "ptx_version": 80})
            self.assertEqual(updated["port_map"][0]["shape"], [2, 5])
            self.assertNotIn("compiler_metadata", updated["invocation"])
            self.assertEqual(legacy["invocation"]["version"], 1)
            with self.assertRaisesRegex(CompilerError, "element count"):
                policy.normalize_legacy_artifact(legacy, shape=(11,))
            legacy["invocation"]["grid"] = [2, 1, 1]
            with self.assertRaisesRegex(CompilerError, "grid"):
                policy.normalize_legacy_artifact(legacy)

    def test_missing_or_inconsistent_legacy_numerical_policy_is_rejected(self):
        import test_worker
        legacy = test_worker.worker.build_artifact(test_worker.request(), test_worker.compiled(), "3.7.1", ["t", "u"])
        legacy["invocation"].update(legacy.pop("producer_metadata"))
        legacy["invocation"]["version"] = 1
        legacy["invocation"]["compiler_metadata"]["enable_reflect_ftz"] = True
        legacy["fusion"] = {"ask_numerics": "i32-wrap", "producer_numerics": {
            "representation": "original-F-instruction-contract-flags", "producer_enable_fp_fusion": True,
            "producer_enable_reflect_ftz": True}}
        legacy["source_fusion_contract"] = {"dtype": "i32", "numerics": {
            "enable_fp_fusion": True, "enable_reflect_ftz": True}}
        legacy["validation"].update(stage="fused_llvm", llvm_module_verified=True)
        self.assertEqual(policy.normalize_legacy_artifact(legacy)["codegen"]["fp_contract"], "on")
        legacy["fusion"]["producer_numerics"]["producer_enable_fp_fusion"] = False
        with self.assertRaisesRegex(CompilerError, "normalization"):
            policy.normalize_legacy_artifact(legacy)
        del legacy["invocation"]["compiler_metadata"]["enable_fp_fusion"]
        with self.assertRaisesRegex(CompilerError, "explicit numerical"):
            policy.normalize_legacy_artifact(legacy)

    def test_cache_numerics_distinguish_dtype_and_float_rounding(self):
        integer = _numerics({"signature": {"a": "*i32", "b": "*i32"}})
        floating = _numerics({"signature": {"a": "*fp32", "b": "*fp32"}})
        self.assertNotEqual(integer, floating)
        self.assertEqual(integer["element_type"], "i32")
        self.assertEqual(floating["element_type"], "f32")
        self.assertEqual(floating["rounding"], "roundTiesToEven")
        self.assertFalse(floating["fast_math"])
        self.assertFalse(floating["contraction"])
        descriptors = [integer, floating]
        for signature, dtype in (("*fp16", "f16"), ("*bf16", "bf16")):
            descriptor = _numerics({"signature": {"a": signature, "b": signature}})
            self.assertEqual(descriptor["element_type"], dtype)
            self.assertEqual(descriptor["operation_result_type"], "explicit-per-SSA-value")
            self.assertEqual(descriptor["rounding"], "roundTiesToEven")
            self.assertNotIn(descriptor, descriptors)
            descriptors.append(descriptor)
        for signature, dtype in (("*i1", "bool"), ("*i64", "i64")):
            descriptor = _numerics({"signature": {"a": signature, "b": signature}})
            self.assertEqual(descriptor["element_type"], dtype)
            self.assertEqual(descriptor["cast_semantics"], "rne-saturating-fptosi-nan-zero-v1")
            self.assertNotIn(descriptor, descriptors)
            descriptors.append(descriptor)
        for signature in ({"a": "*i32", "b": "*fp32"}, {"a": "*fp64", "b": "*fp64"}):
            with self.assertRaises(CompilerError):
                _numerics({"signature": signature})

    def test_standalone_cache_identity_carries_each_pointer_type(self):
        request = {'signature': {'q': '*fp16', 'k': '*bf16', 'out': '*fp32'},
                   'port_shapes': {'input:0': [2, 32], 'input:1': [7, 32], 'output:0': [2, 7]}}
        numerics = _numerics(request)
        self.assertEqual(numerics['kind'], 'external-kernel')
        self.assertEqual(numerics['pointer_types'], request['signature'])
        self.assertEqual(numerics['arithmetic'], 'producer-source-and-fixed-options')
        request['signature']['out'] = '*fp64'
        with self.assertRaisesRegex(CompilerError, 'unsupported data-pointer'):
            _numerics(request)

    def test_allowed_cache_location_and_false_override(self):
        with patch.dict(os.environ, {"TRITON_CACHE_DIR": "/tmp/custom-cache",
                                     "TRITON_KERNEL_OVERRIDE": ""}, clear=True):
            _validate_environment()
        with patch.dict(os.environ, {"TRITON_KERNEL_OVERRIDE": "0"}, clear=True):
            with self.assertRaises(CompilerError):
                _validate_environment()
