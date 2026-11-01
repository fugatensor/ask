"""Generic orchestration and build authorization without importing a producer."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ask import CompilerError, CompilerEnvironment
from ask.adapters import AdapterRegistry, CompilerAdapter
from ask.gpu import CUDAKernel, compile_graph, _tool


def capabilities():
    return {"third_party_compilers": ["toy"], "external_llir_fusion": True,
            "compiler_adapters": [{"id": "toy", "protocol_version": 1, "supports_fusion": True}],
            "llvm_revision": "test-llvm"}


REGISTRATION = {"toy": {"id": "toy", "protocol_version": 1, "module": "test.toy_adapter", "factory": "create_adapter",
                         "aliases": {"toy_python": "python", "toy_worker": "worker"}}}


class ToyAdapter(CompilerAdapter):
    compiler_id = "toy"
    def cache_identity(self, request, environment, *, timeout):
        return {"producer": {"name": "toy"}, "adapter": {"revision": "test"}, "dsl": "opaque",
                "config": request, "target": {"backend": "cuda", "arch": 80},
                "numerics": {"element_type": "f32"}}


class GPUEntryTests(unittest.TestCase):
    def test_registered_policy_and_separate_cache_statistics(self):
        compiler = Mock(executable="ask-opt", timeout=30)
        compiler.capabilities.return_value = capabilities()
        # No signature, N/BLOCK or scheduling details: the policy owns its DSL.
        request = {"compiler": "toy", "opaque_payload": "third-party program"}
        compiler.prepare.return_value = SimpleNamespace(report={"external_requests": [request],
                                                                 "shape_snapshot": [10]})
        ir_stats = {"status": "hit", "counters": {"hits": 1}}
        executable_stats = {"status": "miss", "counters": {"hits": 0, "builds": 1}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("ask.adapters._build_registrations", return_value=REGISTRATION), \
                 patch("ask.adapters._import_module", return_value=SimpleNamespace(create_adapter=ToyAdapter)), \
                 patch("ask.gpu._tool", return_value={"path": "/test/tool", "sha256": "a" * 64}), \
                 patch("ask.gpu.IRCache") as cache, \
                 patch("ask.gpu.compile_cuda", return_value={"executable_cache": executable_stats}) as codegen:
                cache.return_value.get_or_create.return_value = (
                    {"fused.ll": b"LLVM", "fused.artifact.json": b"{}", "report.json": b"{}"}, ir_stats)
                kernel = compile_graph(compiler, "graph", cache_dir=root / "cache", output_dir=root / "result",
                                       compilers={"toy": CompilerEnvironment(worker=root / "exporter")},
                                       llc="llc", ptxas="ptxas")
            self.assertEqual(codegen.call_args.kwargs["cache_dir"], root / "cache/executables")
            self.assertEqual(kernel.cache, ir_stats)
            self.assertEqual(kernel.report["ir_cache"], ir_stats)
            self.assertEqual(kernel.report["executable_cache"], executable_stats)
            self.assertEqual(json.loads((root / "result/report.json").read_text()), kernel.report)
            self.assertEqual(kernel.report["ir_cache_key_inputs"]["config"]["compiler_id"], "toy")
            self.assertNotIn("python", kernel.report["ir_cache_key_inputs"]["tools"])
            compiler._run.assert_not_called()

    def test_registry_aliases_are_metadata_and_do_not_import_the_adapter(self):
        registry = AdapterRegistry(capabilities(), registrations=REGISTRATION)
        with patch("ask.adapters._import_module") as imported:
            result = registry.environments(None, {"toy_python": "python", "toy_worker": "export.py"})
            self.assertEqual(result, {"toy": CompilerEnvironment("python", "export.py")})
            with self.assertRaisesRegex(TypeError, "provided twice"):
                registry.environments(result, {"toy_python": "another", "toy_worker": "export.py"})
            with self.assertRaisesRegex(TypeError, "worker path"):
                registry.environments(None, {"toy_python": "python"})
            imported.assert_not_called()

    def test_none_and_unknown_fail_before_import_worker_or_cache(self):
        for caps, compiler_id in (({"third_party_compilers": []}, "toy"), (capabilities(), "unknown")):
            compiler = Mock()
            compiler.capabilities.return_value = caps
            with self.subTest(compiler_id=compiler_id), patch("ask.adapters._import_module") as imported, \
                 patch("ask.adapters._build_registrations") as registrations, \
                 patch("ask.gpu.IRCache") as cache, patch("ask.gpu._tool") as tools:
                with self.assertRaisesRegex(CompilerError, "does not enable"):
                    compile_graph(compiler, "not parsed", cache_dir="unused", output_dir="unused",
                                  compilers={compiler_id: CompilerEnvironment(worker="missing")},
                                  llc="missing", ptxas="missing")
                imported.assert_not_called()
                registrations.assert_not_called()
                cache.assert_not_called()
                tools.assert_not_called()
                compiler.prepare.assert_not_called()

    def test_ir_compiler_must_be_enabled_before_loading_adapter(self):
        compiler = Mock()
        compiler.capabilities.return_value = capabilities()
        compiler.prepare.return_value = SimpleNamespace(report={"external_requests": [{"compiler": "unknown"}]})
        with patch("ask.adapters._import_module") as imported, patch("ask.gpu.IRCache") as cache:
            with self.assertRaisesRegex(CompilerError, "does not enable"):
                compile_graph(compiler, "graph", cache_dir="unused", output_dir="unused", llc="llc", ptxas="ptxas",
                              compilers={"toy": CompilerEnvironment(worker="exporter")})
            imported.assert_not_called()
            cache.assert_not_called()

    def test_export_only_registration_cannot_claim_fusion(self):
        caps = capabilities()
        caps["compiler_adapters"][0]["supports_fusion"] = False
        with self.assertRaisesRegex(CompilerError, "does not support"):
            AdapterRegistry(caps, registrations=REGISTRATION).require_enabled("toy", fusion=True)

    def test_registration_protocol_mismatch_fails_before_module_import(self):
        registration = {"toy": {**REGISTRATION["toy"], "protocol_version": 2}}
        with patch("ask.adapters._import_module") as imported:
            with self.assertRaisesRegex(CompilerError, "protocol differs"):
                AdapterRegistry(capabilities(), registrations=registration).load("toy")
            imported.assert_not_called()

    def test_runtime_keeps_typed_execution_entry(self):
        kernel = CUDAKernel({"port_map": [{"element_type": "f32"}]}, {}, {}, {})
        with patch("ask.gpu.execute", return_value={"output": [1.5]}) as execute:
            self.assertEqual(kernel.run([1.5]), {"output": [1.5]})
        execute.assert_called_once_with(kernel.artifact, kernel.codegen, [1.5],
                                        device_ordinal=0, driver_library=None)

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
