"""CUDA contract and artifact checks; these tests never require a GPU or Triton."""
import copy
import ctypes
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from ask.cuda import (CUDAError, compile_cuda, execute, execute_i32, _host_buffers,
                      _decode_output, _check_device_dtype)


def artifact(size=10, *, block=128, dtype="i32"):
    # This fixture deliberately has no producer name, DSL constants, scheduling
    # metadata or scratch arguments. The execution contract stands on its own.
    return {
        "format": "llvm-ir", "entry": "scale2", "payload": "define void @fixture() { ret void }\n",
        "target": {"backend": "cuda", "arch": 80},
        "port_map": [{"argument": index, "logical_port": binding, "element_type": dtype,
                      "shape": [size], "layout": "contiguous_row_major"}
                     for index, binding in enumerate(("input:0", "output:0"))],
        "invocation": {"kind": "cuda-kernel", "version": 2,
                       "grid": [max(1, (size + block - 1) // block), 1, 1], "block": [block, 1, 1],
                       "shared_bytes": 0,
                       "arguments": [{"argument": index, "binding": binding, "llvm_type": "ptr addrspace(1)"}
                                     for index, binding in enumerate(("input:0", "output:0"))]},
        "codegen": {"fp_contract": "fast", "ptx_version": 80},
        "required_element_types": [dtype],
    }


class CUDAContractTests(unittest.TestCase):
    def compile_fixture(self, directory, value):
        source = directory / "source.ll"
        source.write_text(value["payload"])

        def fake_compiler(command, **kwargs):
            # This only tests routing/hashes. No successful LLVM compilation or
            # GPU execution is inferred from these intentionally synthetic bytes.
            Path(command[-1]).write_bytes(b"fixture binary\n")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("ask.cuda.subprocess.run", side_effect=fake_compiler) as compiler:
            result = compile_cuda(source, value, directory / "codegen", llc="/test/llc", ptxas="/test/ptxas")
        self.assertEqual(compiler.call_count, 2)
        return result

    def test_codegen_binds_the_supplied_ir_and_records_tools(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            value = artifact()
            result = self.compile_fixture(directory, value)
            self.assertEqual(result["commands"][0][1:4],
                             ["-march=nvptx64", "-mcpu=sm_80", "-mattr=+ptx80"])
            self.assertEqual(result["commands"][0][4], "--fp-contract=fast")
            self.assertEqual(result["commands"][0][5], str(directory / "source.ll"))
            self.assertEqual(result["commands"][1][1], "--gpu-name=sm_80")
            self.assertFalse(result["original_producer_binary_used"])
            self.assertEqual(json.loads((directory / "codegen/codegen.json").read_text()), result)
            with self.assertRaisesRegex(CUDAError, "already contains"):
                compile_cuda(directory / "source.ll", value, directory / "codegen",
                             llc="/test/llc", ptxas="/test/ptxas")

    def test_zero_tail_and_multiple_blocks_preserve_explicit_grid(self):
        for size in (0, 1, 127, 128, 129, 257):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as temporary:
                value = artifact(size)
                result = self.compile_fixture(Path(temporary), value)
                self.assertEqual(result["invocation"]["grid"], value["invocation"]["grid"])

    def test_launch_dimensions_and_storage_shape_are_independent(self):
        for grid, block in (([2, 3, 4], [3, 5, 7]), ([1, 2, 1], [1, 16, 1]), ([8, 1, 1], [256, 1, 1])):
            with self.subTest(grid=grid, block=block), tempfile.TemporaryDirectory() as temporary:
                value = artifact(513)
                value["invocation"].update(grid=grid, block=block, shared_bytes=512)
                result = self.compile_fixture(Path(temporary), value)
                self.assertEqual(result["invocation"]["block"], block)
                self.assertEqual(result["invocation"]["grid"], grid)
                self.assertEqual(result["invocation"]["shared_bytes"], 512)

    def test_invalid_launch_and_abi_rejected_before_tools_or_driver(self):
        changes = [
            lambda a: a["invocation"].update(grid=[0, 1, 1]),
            lambda a: a["invocation"].update(grid=[True, 1, 1]),
            lambda a: a["invocation"].update(grid=[1, 1]),
            lambda a: a["invocation"].update(block=[1, -2, 1]),
            lambda a: a["invocation"].update(shared_bytes=-1),
            lambda a: a["invocation"].update(shared_bytes=True),
            lambda a: a["invocation"].update(version=1),
            lambda a: a["target"].update(arch="90a"),
            lambda a: a["invocation"]["arguments"][0].update(binding="output:0"),
            lambda a: a["invocation"]["arguments"][1].update(llvm_type="i64"),
            lambda a: a["invocation"]["arguments"].pop(),
            lambda a: a.update(target=None),
            lambda a: a.update(invocation=None),
            lambda a: a["invocation"]["arguments"][0].update(binding={}),
            lambda a: a.pop("payload"),
            lambda a: a.pop("port_map"),
            lambda a: a["port_map"][0].update(element_type="f64"),
            lambda a: a["port_map"][0].update(element_type=[]),
            lambda a: a["port_map"][0].update(element_type="f32"),
            lambda a: a["port_map"][0].update(argument=True),
            lambda a: a["port_map"][0].update(logical_port="output:0"),
            lambda a: a["port_map"][0].update(shape=[-1]),
            lambda a: a["port_map"][0].update(shape=[True]),
            lambda a: a["port_map"][0].update(shape=[2**63]),
            lambda a: a["port_map"][0].update(layout="strided"),
            lambda a: a["port_map"][0].update(storage_bytes=1),
            lambda a: a.pop("codegen"),
            lambda a: a["codegen"].update(fp_contract="unknown"),
            lambda a: a["codegen"].update(ptx_version=True),
            lambda a: a.update(required_element_types=["f64"]),
            lambda a: a.pop("required_element_types"),
        ]
        for index, change in enumerate(changes):
            with self.subTest(case=index), patch("ask.cuda._Driver") as driver:
                value = artifact()
                change(value)
                with self.assertRaises(CUDAError):
                    execute_i32(value, {}, [0] * 10)
                driver.assert_not_called()

    def test_different_ir_cannot_borrow_an_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "source.ll").write_text("different LLVM IR")
            with patch("ask.cuda.subprocess.run") as compiler:
                with self.assertRaisesRegex(CUDAError, "differs from artifact payload"):
                    compile_cuda(path / "source.ll", artifact(), path / "output",
                                 llc="/test/llc", ptxas="/test/ptxas")
                compiler.assert_not_called()

    def test_tampered_ir_binary_manifest_or_launch_never_reaches_driver(self):
        for kind in ("llvm", "cubin", "artifact", "codegen_digest", "launch"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                value = artifact()
                result = self.compile_fixture(Path(temporary), value)
                if kind == "llvm":
                    Path(result["llvm_ir"]).write_text("changed")
                elif kind == "cubin":
                    Path(result["cubin"]).write_bytes(b"changed")
                elif kind == "artifact":
                    value["payload"] += "; changed\n"
                elif kind == "codegen_digest":
                    result["llvm_sha256"] = "0" * 64
                else:
                    result["invocation"] = copy.deepcopy(result["invocation"])
                    result["invocation"]["grid"] = [2, 1, 1]
                with patch("ask.cuda._Driver") as driver:
                    with self.assertRaises(CUDAError):
                        execute_i32(value, result, [0] * 10)
                    driver.assert_not_called()

    def test_invalid_input_does_not_load_driver(self):
        for values in ([0] * 9, [0] * 9 + [True], [0] * 9 + [2**31]):
            with self.subTest(values=values), patch("ask.cuda._Driver") as driver:
                with self.assertRaises(CUDAError):
                    execute_i32(artifact(), {}, values)
                driver.assert_not_called()

    def test_float_buffers_round_once_and_preserve_special_values(self):
        values = [1.0 + 2**-24, 1.0 + 3 * 2**-24, -0.0, 2**-149,
                  float("inf"), float("-inf"), float("nan")]
        source, destination = _host_buffers(values, len(values), "f32")
        self.assertEqual(ctypes.sizeof(source), len(values) * 4)
        self.assertEqual(list(source)[:2], [1.0, 1.0 + 2**-22])
        self.assertEqual(math.copysign(1.0, source[2]), -1.0)
        self.assertEqual(source[3], 2**-149)
        self.assertEqual(list(source)[4:6], [float("inf"), float("-inf")])
        self.assertTrue(math.isnan(source[6]))
        self.assertEqual(list(destination), [0.0] * len(values))

    def test_large_integer_input_does_not_round_through_binary64(self):
        # This is just above an f32 midpoint. Converting it to Python float
        # first would erase +1 and incorrectly round the midpoint down.
        value = 2**100 + 2**76 + 1
        source, _ = _host_buffers([value], 1, "f32")
        self.assertEqual(source[0], float(2**100 + 2**77))

    def test_bool_and_i64_buffers_keep_their_storage_width_and_values(self):
        for dtype, values, width in (("bool", [False, True], 1),
                                     ("i64", [-(2**63), 2**63 - 1, 2**53 + 1, -1], 8)):
            source, _ = _host_buffers(values, len(values), dtype)
            self.assertEqual(ctypes.sizeof(source), len(values) * width)
            self.assertEqual(_decode_output(source, len(values), dtype), values)
        for dtype, value in (("bool", 1), ("bool", "false"), ("i64", True),
                             ("i64", 2**63), ("i64", -(2**63) - 1), ("i64", 1.0)):
            with self.subTest(dtype=dtype, value=value), self.assertRaises(CUDAError):
                _host_buffers([value], 1, dtype)

    def test_declared_target_drives_codegen_without_changing_producer_strategy(self):
        for arch in (75, 86, 90):
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as temporary:
                value = artifact(1)
                value["target"]["arch"] = arch
                codegen = self.compile_fixture(Path(temporary), value)
                self.assertEqual(codegen["arch"], arch)
                self.assertIn(f"-mcpu=sm_{arch}", codegen["commands"][0])
                self.assertIn(f"--gpu-name=sm_{arch}", codegen["commands"][1])

    def test_internal_bf16_requires_a_capable_compilation_target(self):
        value = artifact(1, dtype="i32")
        value["target"]["arch"] = 75
        value["required_element_types"] = ["i32", "bf16"]
        with tempfile.TemporaryDirectory() as temporary, patch("ask.cuda.subprocess.run") as compiler:
            with self.assertRaisesRegex(CUDAError, "bf16 compilation requires"):
                compile_cuda(Path(temporary) / "missing.ll", value, temporary,
                             llc="/test/llc", ptxas="/test/ptxas")
            compiler.assert_not_called()

    def test_low_precision_buffers_preserve_bits_and_decode_values(self):
        for dtype, fraction_bits, minimum in (("f16", 10, 2**-24), ("bf16", 7, 2**-133)):
            with self.subTest(dtype=dtype):
                half_ulp = 2**(-fraction_bits - 1)
                values = [1.0 + half_ulp, 1.0 + 3 * half_ulp, -0.0, minimum,
                          float("inf"), float("-inf"), float("nan")]
                source, destination = _host_buffers(values, len(values), dtype)
                self.assertEqual(ctypes.sizeof(source), len(values) * 2)
                actual = _decode_output(source, len(values), dtype)
                self.assertEqual(actual[:2], [1.0, 1.0 + 2**(1 - fraction_bits)])
                self.assertEqual(math.copysign(1.0, actual[2]), -1.0)
                self.assertEqual(actual[3], minimum)
                self.assertEqual(actual[4:6], [float("inf"), float("-inf")])
                self.assertTrue(math.isnan(actual[6]))
                self.assertEqual(list(destination), [0] * len(values))
                self.assertEqual(_decode_output(source, 0, dtype), [])

    def test_bf16_device_support_is_checked_before_context_or_gpu_memory(self):
        class UnsupportedDriver:
            def __init__(self):
                self.calls = []

            def call(self, name, *args):
                self.calls.append(name)
                if name == "cuDeviceGetName":
                    args[0].value = b"test pre-Ampere GPU"
                self.assert_metadata_only(name)

            @staticmethod
            def assert_metadata_only(name):
                if name not in {"cuInit", "cuDeviceGet", "cuDriverGetVersion", "cuDeviceGetName"}:
                    raise AssertionError("unsupported bf16 reached " + name)

            def attribute(self, device, attribute):
                return {75: 7, 76: 5}[attribute]

        for dtype, location in (("bf16", None), ("i32", "artifact"), ("f32", "artifact")):
            with self.subTest(dtype=dtype, location=location), tempfile.TemporaryDirectory() as temporary:
                value = artifact(1, dtype=dtype)
                if location == "artifact":
                    value["required_element_types"] = [dtype, "bf16"]
                codegen = self.compile_fixture(Path(temporary), value)
                driver = UnsupportedDriver()
                with patch("ask.cuda._Driver", return_value=driver):
                    with self.assertRaisesRegex(CUDAError, r"bf16 requires hardware support.*7\.5"):
                        execute(value, codegen, [1])
                self.assertNotIn("cuCtxCreate_v2", driver.calls)
                self.assertNotIn("cuModuleLoadDataEx", driver.calls)
                self.assertNotIn("cuMemAlloc_v2", driver.calls)
                self.assertNotIn("cuLaunchKernel", driver.calls)
        for capability in ((8, 0), (8, 6), (9, 0)):
            _check_device_dtype("bf16", *capability, "supported")
        _check_device_dtype("f16", 7, 5, "no bf16, still has fp16")

    def test_invalid_float_input_and_integer_wrapper_reject_before_driver(self):
        for values in ([True], ["1.0"], [1e100], [10**1000]):
            with self.subTest(values=values), patch("ask.cuda._Driver") as driver:
                with self.assertRaises(CUDAError):
                    execute(artifact(1, dtype="f32"), {}, values)
                driver.assert_not_called()
        with patch("ask.cuda._Driver") as driver:
            with self.assertRaisesRegex(CUDAError, "execute_i32 requires"):
                execute_i32(artifact(1, dtype="f32"), {}, [1])
            driver.assert_not_called()

    def test_float_codegen_and_dtype_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            value = artifact(1, dtype="f32")
            result = self.compile_fixture(Path(temporary), value)
            self.assertEqual(result["element_type"], "f32")
            result["element_type"] = "i32"
            with patch("ask.cuda._Driver") as driver:
                with self.assertRaisesRegex(CUDAError, "contracts disagree"):
                    execute(value, result, [1.0])
                driver.assert_not_called()

    def test_normalized_codegen_policy_is_independent_of_storage_dtype(self):
        for dtype in ("bool", "i32", "i64", "f32", "f16", "bf16"):
            for policy in ("off", "on", "fast"):
                with self.subTest(dtype=dtype, policy=policy), tempfile.TemporaryDirectory() as temporary:
                    value = artifact(1, dtype=dtype)
                    value["codegen"]["fp_contract"] = policy
                    result = self.compile_fixture(Path(temporary), value)
                    self.assertEqual(result["fp_contract"], policy)
                    self.assertIn("--fp-contract=" + policy, result["commands"][0])

    def test_legacy_contract_requires_explicit_adapter_migration(self):
        value = artifact()
        value["invocation"]["version"] = 1
        with patch("ask.cuda._Driver") as driver, self.assertRaisesRegex(CUDAError, "re-export legacy"):
            execute(value, {}, [0] * 10)
        driver.assert_not_called()

    def test_compiler_failure_has_no_success_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            value = artifact()
            (path / "source.ll").write_text(value["payload"])
            with patch("ask.cuda.subprocess.run", return_value=SimpleNamespace(
                    returncode=1, stdout="", stderr="invalid IR")):
                with self.assertRaisesRegex(CUDAError, "invalid IR"):
                    compile_cuda(path / "source.ll", value, path / "output", llc="/test/llc", ptxas="/test/ptxas")
            self.assertFalse((path / "output/codegen.json").exists())

    def test_failed_launch_is_not_replayed_and_resources_are_released(self):
        class FakeDriver:
            def __init__(self, major=8, minor=6):
                self.major, self.minor = major, minor
                self.calls, self.freed = [], []
                self.library = SimpleNamespace(
                    cuFuncGetParamInfo=lambda *a: 1,
                    cuMemFree_v2=lambda pointer: self.freed.append(pointer.value),
                    cuModuleUnload=lambda module: self.calls.append("unload"),
                    cuCtxDestroy_v2=lambda context: self.calls.append("destroy"))
                self.next_pointer = 1000

            def attribute(self, device, attribute):
                return {75: self.major, 76: self.minor, 10: 32}.get(attribute, 2**20)

            def function_attribute(self, function, attribute):
                return {0: 128, 1: 1024, 4: 14, 8: 48000}[attribute]

            def call(self, name, *args):
                self.calls.append(name)
                if name in ("cuCtxCreate_v2", "cuModuleLoadDataEx", "cuModuleGetFunction"):
                    args[0]._obj.value = 1
                elif name == "cuMemAlloc_v2":
                    args[0]._obj.value = self.next_pointer
                    self.next_pointer += 1000
                elif name == "cuFuncGetParamInfo":
                    args[2]._obj.value, args[3]._obj.value = args[1] * 8, 8
                elif name == "cuDeviceGetName":
                    args[0].value = b"fake"
                elif name == "cuLaunchKernel":
                    raise CUDAError("injected launch failure")

        for dtype, arch, capability in (("i32", 80, (8, 6)), ("bf16", 80, (8, 6)),
                                         ("bf16", 90, (9, 0))):
            with self.subTest(dtype=dtype, arch=arch), tempfile.TemporaryDirectory() as temporary:
                value = artifact(dtype=dtype)
                value["target"]["arch"] = arch
                result = self.compile_fixture(Path(temporary), value)
                driver = FakeDriver(*capability)
                with patch("ask.cuda._Driver", return_value=driver):
                    with self.assertRaisesRegex(CUDAError, "injected launch failure"):
                        execute(value, result, [0] * 10)
                self.assertEqual(driver.calls.count("cuLaunchKernel"), 1)
                self.assertEqual(driver.freed, [2000, 1000])
                self.assertEqual(driver.calls[-2:], ["unload", "destroy"])
                self.assertNotIn("cuMemcpyDtoH_v2", driver.calls)


if __name__ == "__main__":
    unittest.main()
