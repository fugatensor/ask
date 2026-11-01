"""Compiler authorization and environment checks must happen before cache hits."""
import os
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ask import CompilerError
from ask.gpu import CUDAKernel, compile_graph, _numerics, _tool, _validate_environment


class GPUEntryTests(unittest.TestCase):
    def test_compile_graph_keeps_ir_and_executable_statistics_separate(self):
        compiler = Mock(executable="ask-opt", timeout=30)
        compiler.capabilities.return_value = {"third_party_compilers": ["triton"],
                                               "triton_llir_fusion": True, "llvm_revision": "test-llvm"}
        request = {"compiler": "triton", "source": "test DSL", "target": {"arch": 80},
                   "signature": {"t": "*i32", "u": "*i32"}}
        compiler.prepare.return_value = SimpleNamespace(report={"external_requests": [request],
                                                                 "shape_snapshot": [10]})
        ir_stats = {"status": "hit", "counters": {"hits": 1}}
        executable_stats = {"status": "miss", "counters": {"hits": 0, "builds": 1}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = root / "worker.py"
            worker.write_text("test worker")
            (root / "fusion.py").write_text("test fusion")
            with patch("ask.gpu._validate_environment"), \
                 patch("ask.gpu._tool", return_value={"path": "/test/tool", "sha256": "a" * 64}), \
                 patch("ask.gpu.subprocess.run", return_value=SimpleNamespace(stdout='{"name":"triton"}')), \
                 patch("ask.gpu.IRCache") as cache, \
                 patch("ask.gpu.compile_cuda", return_value={"executable_cache": executable_stats}) as codegen:
                cache.return_value.get_or_create.return_value = (
                    {"fused.ll": b"LLVM", "fused.artifact.json": b"{}", "report.json": b"{}"}, ir_stats)
                kernel = compile_graph(compiler, "graph", cache_dir=root / "cache", output_dir=root / "result",
                                       triton_python="python", triton_worker=worker, llc="llc", ptxas="ptxas")
            self.assertEqual(codegen.call_args.kwargs["cache_dir"], root / "cache/executables")
            self.assertEqual(kernel.cache, ir_stats)
            self.assertEqual(kernel.report["ir_cache"], ir_stats)
            self.assertEqual(kernel.report["executable_cache"], executable_stats)
            self.assertEqual(json.loads((root / "result/report.json").read_text()), kernel.report)
            compiler._run.assert_not_called()

    def test_runtime_uses_artifact_types_without_a_parallel_float_entry(self):
        kernel = CUDAKernel({"port_map": [{"element_type": "f32"}]}, {}, {}, {})
        with patch("ask.gpu.execute", return_value={"output": [1.5]}) as execute:
            self.assertEqual(kernel.run([1.5]), {"output": [1.5]})
        execute.assert_called_once_with(kernel.artifact, kernel.codegen, [1.5],
                                        device_ordinal=0, driver_library=None)

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

    def test_none_build_never_looks_up_cache_or_starts_worker(self):
        compiler = Mock()
        compiler.capabilities.return_value = {"third_party_compilers": []}
        with patch("ask.gpu.IRCache") as cache, patch("ask.gpu.subprocess.run") as run:
            with self.assertRaisesRegex(CompilerError, "does not enable"):
                compile_graph(compiler, "not parsed", cache_dir="unused", output_dir="unused",
                              triton_python="missing", triton_worker="missing", llc="missing", ptxas="missing")
            cache.assert_not_called()
            run.assert_not_called()
            compiler.prepare.assert_not_called()

    def test_override_cannot_bypass_gate_on_a_warm_cache(self):
        compiler = Mock()
        compiler.capabilities.return_value = {"third_party_compilers": ["triton"],
                                               "triton_llir_fusion": True}
        with patch.dict(os.environ, {"TRITON_LIBDEVICE_PATH": "/changed-in-place.bc"}, clear=True), \
             patch("ask.gpu.IRCache") as cache, patch("ask.gpu.subprocess.run") as run:
            with self.assertRaisesRegex(CompilerError, "TRITON_LIBDEVICE_PATH"):
                compile_graph(compiler, "not parsed", cache_dir="existing", output_dir="unused",
                              triton_python="missing", triton_worker="missing", llc="missing", ptxas="missing")
            cache.assert_not_called()
            run.assert_not_called()
            compiler.prepare.assert_not_called()

    def test_allowed_cache_location_and_false_override(self):
        with patch.dict(os.environ, {"TRITON_CACHE_DIR": "/tmp/custom-cache",
                                     "TRITON_KERNEL_OVERRIDE": ""}, clear=True):
            _validate_environment()
        with patch.dict(os.environ, {"TRITON_KERNEL_OVERRIDE": "0"}, clear=True):
            with self.assertRaises(CompilerError):
                _validate_environment()

    @unittest.skipUnless(os.name == "posix", "executable symlinks need POSIX")
    def test_tool_fingerprint_preserves_virtualenv_executable_path(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "python-real"
            link = Path(directory) / "venv-python"
            base.write_bytes(b"test executable")
            base.chmod(0o700)
            link.symlink_to(base)
            self.assertEqual(_tool(link)["path"], str(link))
            before = _tool(link)["sha256"]
            base.write_bytes(b"updated executable")
            self.assertNotEqual(_tool(link)["sha256"], before)


if __name__ == "__main__":
    unittest.main()
