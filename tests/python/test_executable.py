"""Resident package validation, without requiring CUDA for contract tests."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from ask.cuda import CUDAError, _artifact_digest, _digest
from ask.gpu import CUDAKernel
from test_cuda import artifact


class SessionFixture:
    def __init__(self):
        self.loads = []
        self.submissions = []

    def load(self, *args, **kwargs):
        self.loads.append((args, kwargs))
        return object()

    def submit(self, *args, **kwargs):
        self.submissions.append((args, kwargs))
        return object()


def tensor(shape=(2, 5), dtype="i32", pointer=4096, readonly=False):
    return SimpleNamespace(shape=shape, dtype=dtype, data_ptr=pointer,
                           nbytes=40, readonly=readonly)


class ResidentExecutableTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        directory = Path(self.temporary.name)
        value = artifact()
        for port in value["port_map"]:
            port["shape"] = [2, 5]
        source, binary = directory / "input.ll", directory / "kernel.cubin"
        source.write_text(value["payload"])
        binary.write_bytes(b"synthetic cubin for metadata-only test")
        receipt = {"entry": value["entry"], "arch": 80,
                   "invocation": copy.deepcopy(value["invocation"]),
                   "element_type": "i32", "fp_contract": "fast",
                   "artifact_sha256": _artifact_digest(value),
                   "llvm_sha256": _digest(source.read_bytes()), "llvm_ir": str(source),
                   "cubin_sha256": _digest(binary.read_bytes()), "cubin": str(binary)}
        report = {"shape_snapshot": {"shape_bindings": [
            {"type": "tensor<2x5xi32>", "shape": [2, 5]}]}}
        self.kernel = CUDAKernel(value, receipt, report, {})
        self.session = SessionFixture()

    def test_load_once_and_submit_without_file_io(self):
        executable = self.kernel.prepare(self.session)
        self.assertEqual(executable.shape, (2, 5))
        self.assertEqual(executable.dtype, "i32")
        source, destination = tensor(), tensor(pointer=8192)
        with patch.object(Path, "read_bytes", side_effect=AssertionError("hot path file I/O")):
            for _ in range(3):
                executable.submit(source, destination)
        self.assertEqual(len(self.session.loads), 1)
        self.assertEqual(self.session.loads[0][1]["expected_parameters"], 2)
        self.assertEqual(len(self.session.submissions), 3)
        args, kwargs = self.session.submissions[0]
        self.assertEqual(args[1], [source, destination])
        self.assertEqual(kwargs, {"grid": [1, 1, 1], "block": [128, 1, 1], "shared_bytes": 0})

    def test_arbitrary_null_arguments_and_launch_axes_are_preserved(self):
        value = self.kernel.artifact
        bindings = ["null", "output:0", "null", "input:0", "null"]
        value["invocation"].update(arguments=[{"argument": index, "binding": binding,
                                              "llvm_type": "ptr addrspace(1)"}
                                             for index, binding in enumerate(bindings)],
                                   grid=[2, 3, 4], block=[3, 5, 7], shared_bytes=256)
        for port in value["port_map"]:
            port["argument"] = bindings.index(port["logical_port"])
        self.kernel.codegen["invocation"] = copy.deepcopy(value["invocation"])
        self.kernel.codegen["artifact_sha256"] = _artifact_digest(value)
        executable = self.kernel.prepare(self.session)
        source, destination = tensor(), tensor(pointer=8192)
        executable.submit(source, destination)
        self.assertEqual(self.session.loads[0][1]["expected_parameters"], 5)
        self.assertEqual(self.session.submissions[0][0][1], [None, destination, None, source, None])
        self.assertEqual(self.session.submissions[0][1],
                         {"grid": [2, 3, 4], "block": [3, 5, 7], "shared_bytes": 256})

    def test_rejects_equal_size_but_wrong_shape_and_dtype(self):
        executable = self.kernel.prepare(self.session)
        for invalid, message in ((tensor(shape=(10,)), "shape"),
                                 (tensor(dtype="f32"), "dtype")):
            with self.subTest(message=message), self.assertRaisesRegex(CUDAError, message):
                executable.submit(invalid, tensor(pointer=8192))
        self.assertFalse(self.session.submissions)

    def test_rejects_readonly_and_overlapping_outputs(self):
        executable = self.kernel.prepare(self.session)
        for invalid, message in ((tensor(pointer=8192, readonly=True), "read-only"),
                                 (tensor(pointer=4100), "non-overlapping")):
            with self.subTest(message=message), self.assertRaisesRegex(CUDAError, message):
                executable.submit(tensor(), invalid)
        self.assertFalse(self.session.submissions)

    def test_rejects_descriptor_larger_than_backing_buffer(self):
        executable = self.kernel.prepare(self.session)
        short = tensor()
        short.nbytes = 4
        with self.assertRaisesRegex(CUDAError, "buffer size"):
            executable.submit(short, tensor(pointer=8192))
        self.assertFalse(self.session.submissions)

    def test_checks_binary_and_contract_before_loading(self):
        original = copy.deepcopy(self.kernel.codegen)
        self.kernel.codegen["artifact_sha256"] = "0" * 64
        with self.assertRaisesRegex(CUDAError, "artifact digest"):
            self.kernel.prepare(self.session)
        self.kernel.codegen.update(original)
        Path(self.kernel.codegen["cubin"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(CUDAError, "cubin digest"):
            self.kernel.prepare(self.session)
        self.assertFalse(self.session.loads)

    def test_prepared_contract_does_not_alias_mutable_package(self):
        executable = self.kernel.prepare(self.session)
        self.kernel.artifact["invocation"]["grid"][0] = 999
        self.kernel.artifact["port_map"][0]["shape"][0] = 99
        executable.submit(tensor(), tensor(pointer=8192))
        self.assertEqual(executable.shape, (2, 5))
        self.assertEqual(self.session.submissions[0][1]["grid"], [1, 1, 1])

    def test_reversed_pointer_mapping_and_intermediate_bf16_are_preserved(self):
        value = self.kernel.artifact
        value["required_element_types"] = ["bf16", "f32"]
        for item in value["invocation"]["arguments"][:2]:
            item["binding"] = "output:0" if item["argument"] == 0 else "input:0"
        for item in value["port_map"]:
            item["logical_port"] = "output:0" if item["argument"] == 0 else "input:0"
        self.kernel.codegen["invocation"] = copy.deepcopy(value["invocation"])
        self.kernel.codegen["artifact_sha256"] = _artifact_digest(value)
        executable = self.kernel.prepare(self.session)
        self.assertEqual(self.session.loads[0][1]["required_types"], ["bf16", "f32", "i32"])
        source, destination = tensor(), tensor(pointer=8192)
        executable.submit(source, destination)
        self.assertEqual(self.session.submissions[0][0][1], [destination, source])


if __name__ == "__main__":
    unittest.main()
