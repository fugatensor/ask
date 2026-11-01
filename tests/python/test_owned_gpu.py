"""Native graph routing and multi-port execution checks without a CUDA device."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ask import CompilerError
from ask.cuda import CUDAError, _artifact_digest, _kernel_contract
from ask.gpu import compile_graph
import test_executable as executable_fixture
from test_executable import tensor


class OwnedGraphRoutingTests(unittest.TestCase):
    def compiler(self):
        compiler = Mock(executable="ask-opt")
        compiler.capabilities.return_value = {
            "third_party_compilers": [], "compiler_adapters": [],
            "cuda_graph_codegen": True, "llvm_revision": "test-llvm",
        }
        compiler.prepare.return_value = SimpleNamespace(
            report={"external_requests": [], "shape_snapshot": {}})
        return compiler

    def test_none_uses_owned_pipeline_and_never_resolves_an_adapter(self):
        compiler = self.compiler()
        stats = {"status": "hit", "counters": {"hits": 1}}
        with tempfile.TemporaryDirectory() as directory, \
             patch("ask.gpu.AdapterRegistry.load", side_effect=AssertionError("adapter loaded")), \
             patch("ask.adapters._build_registrations", side_effect=AssertionError("adapter discovery")), \
             patch("ask.gpu._tool", return_value={"path": "/test/tool", "sha256": "a" * 64}), \
             patch("ask.gpu.IRCache") as cache, \
             patch("ask.gpu.compile_cuda", return_value={"executable_cache": stats}):
            cache.return_value.get_or_create.return_value = (
                {"kernel.ll": b"LLVM", "kernel.artifact.json": b"{}", "report.json": b"{}"}, stats)
            result = compile_graph(compiler, "graph", target={"backend": "cuda", "arch": 86},
                                   cache_dir=Path(directory) / "cache", output_dir=Path(directory) / "result",
                                   llc="llc", ptxas="ptxas")
            key = cache.return_value.get_or_create.call_args.args[0]
            self.assertEqual(key["producer"], {"name": "ask"})
            self.assertIsNone(key["dsl"])
            self.assertIsNone(key["adapter"])
            self.assertEqual(set(key["tools"]), {"ask-opt", "llc", "ptxas"})
            self.assertEqual(result.cache["status"], "hit")
            compiler._run.assert_not_called()

    def test_invalid_native_policy_fails_before_cache_or_tools(self):
        cases = [{"target": None}, {"target": {"backend": "cuda", "arch": True}},
                 {"block_size": 0}, {"block_size": 1025}, {"block_size": True},
                 {"ptx_version": True}]
        for change in cases:
            with self.subTest(change=change), patch("ask.gpu.IRCache") as cache, \
                 patch("ask.gpu._tool") as tools:
                options = dict(target={"backend": "cuda", "arch": 86}, block_size=128, ptx_version=80)
                options.update(change)
                with self.assertRaises(CompilerError):
                    compile_graph(self.compiler(), "graph", cache_dir="unused", output_dir="unused",
                                  llc="missing", ptxas="missing", **options)
                cache.assert_not_called()
                tools.assert_not_called()


class MultiPortExecutableTests(unittest.TestCase):
    def setUp(self):
        fixture = executable_fixture.ResidentExecutableTests(methodName="test_load_once_and_submit_without_file_io")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.kernel, self.session = fixture.kernel, fixture.session
        artifact = self.kernel.artifact
        # Deliberately scramble physical pointer order; logical order is stable.
        specs = [("output:1", [], "i64"), ("input:1", [], "i64"),
                 ("output:0", [2, 5], "f32"), ("input:0", [2, 5], "i32")]
        artifact["port_map"] = [
            {"argument": index, "logical_port": binding, "shape": shape,
             "element_type": dtype, "layout": "contiguous_row_major"}
            for index, (binding, shape, dtype) in enumerate(specs)]
        artifact["invocation"]["arguments"] = [
            {"argument": index, "binding": binding, "llvm_type": "ptr addrspace(1)"}
            for index, (binding, _, _) in enumerate(specs)]
        self.kernel.codegen.update(element_type=None,
                                   invocation=copy.deepcopy(artifact["invocation"]),
                                   artifact_sha256=_artifact_digest(artifact))
        self.inputs = [tensor(), tensor(shape=(), dtype="i64", pointer=8192)]
        self.outputs = [tensor(dtype="f32", pointer=12288),
                        tensor(shape=(), dtype="i64", pointer=16384)]
        self.inputs[1].nbytes = self.outputs[1].nbytes = 8

    def test_mixed_ports_use_logical_order_and_metadata_only_hot_path(self):
        executable = self.kernel.prepare(self.session)
        self.assertEqual([p["element_type"] for p in executable.inputs], ["i32", "i64"])
        self.assertEqual([p["shape"] for p in executable.outputs], [[2, 5], []])
        executable.outputs[0]["shape"][0] = 999
        with patch.object(Path, "read_bytes", side_effect=AssertionError("hot file I/O")):
            executable.submit_many(self.inputs, self.outputs)
        self.assertEqual(self.session.submissions[0][0][1],
                         [self.outputs[1], self.inputs[1], self.outputs[0], self.inputs[0]])
        self.assertEqual(self.session.loads[0][1]["required_types"], ["f32", "i32", "i64"])

    def test_all_writes_must_be_disjoint_and_all_ports_present(self):
        executable = self.kernel.prepare(self.session)
        self.outputs[1].data_ptr = self.outputs[0].data_ptr + 4
        with self.assertRaisesRegex(CUDAError, "non-overlapping"):
            executable.submit_many(self.inputs, self.outputs)
        self.outputs[1].data_ptr = self.inputs[0].data_ptr + 4
        with self.assertRaisesRegex(CUDAError, "non-overlapping"):
            executable.submit_many(self.inputs, self.outputs)
        with self.assertRaisesRegex(CUDAError, "2 input"):
            executable.submit_many(self.inputs[:1], self.outputs)
        self.assertFalse(self.session.submissions)

    def test_sparse_or_duplicate_port_indices_are_rejected(self):
        for changed in ("input:3", "output:1"):
            artifact = copy.deepcopy(self.kernel.artifact)
            artifact["port_map"][1]["logical_port"] = changed
            artifact["invocation"]["arguments"][1]["binding"] = changed
            with self.subTest(changed=changed), self.assertRaises(CUDAError):
                _kernel_contract(artifact)


if __name__ == "__main__":
    unittest.main()
