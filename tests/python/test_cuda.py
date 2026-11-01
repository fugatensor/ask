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
from ask.cuda import CUDAError, compile_cuda, execute, execute_i32, _host_buffers


def artifact(size=10, *, block=128, num_warps=4, dtype="i32"):
    return {
        "format": "llvm-ir", "entry": "scale2", "payload": "define void @fixture() { ret void }\n",
        "target": {"backend": "cuda", "arch": 80, "warp_size": 32},
        "specializations": {"constants": {"N": size, "BLOCK": block}},
        "port_map": [{"argument": 0, "logical_port": "input:0", "element_type": dtype},
                     {"argument": 1, "logical_port": "output:0", "element_type": dtype}],
        "invocation": {"kind": "cuda-kernel", "version": 1,
                       "grid": [max(1, (size + block - 1) // block), 1, 1], "block": [num_warps * 32, 1, 1],
                       "num_warps": num_warps, "num_ctas": 1, "shared_bytes": 0,
                       "compiler_metadata": {"enable_fp_fusion": True, "enable_reflect_ftz": True},
                       "arguments": [
                           {"argument": 0, "binding": "input:0", "llvm_type": "ptr addrspace(1)"},
                           {"argument": 1, "binding": "output:0", "llvm_type": "ptr addrspace(1)"},
                           {"argument": 2, "binding": "null", "resource": "global_scratch", "llvm_type": "ptr addrspace(1)"},
                           {"argument": 3, "binding": "null", "resource": "profile_scratch", "llvm_type": "ptr addrspace(1)"}],
                       "resources": {"global_scratch": {"size_bytes": 0},
                                     "profile_scratch": {"size_bytes": 0}}},
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
            self.assertEqual(result["commands"][0][4], "--fp-contract=on")
            self.assertEqual(result["commands"][0][5], str(directory / "source.ll"))
            self.assertEqual(result["commands"][1][1], "--gpu-name=sm_80")
            self.assertFalse(result["original_triton_binary_used"])
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

    def test_element_partition_and_thread_block_are_independent(self):
        # BLOCK counts logical elements. num_warps determines physical threads;
        # a thread may process several elements, or not own an active element.
        for block, warps in ((64, 4), (128, 8), (256, 4), (512, 8), (1024, 4)):
            with self.subTest(block=block, warps=warps), tempfile.TemporaryDirectory() as temporary:
                value = artifact(513, block=block, num_warps=warps)
                result = self.compile_fixture(Path(temporary), value)
                self.assertEqual(result["invocation"]["block"], [warps * 32, 1, 1])
                self.assertEqual(result["invocation"]["grid"], [(513 + block - 1) // block, 1, 1])

    def test_invalid_launch_and_abi_rejected_before_tools_or_driver(self):
        changes = [
            lambda a: a["invocation"].update(grid=[0, 1, 1]),
            lambda a: a["invocation"].update(grid=[1, 2, 1]),
            lambda a: a["invocation"].update(grid=[True, 1, 1]),
            lambda a: a["invocation"].update(grid=[1, 1]),
            lambda a: a["invocation"].update(grid=[2, 1, 1]),
            lambda a: a["invocation"].update(block=[64, 1, 1]),
            lambda a: a["invocation"].update(shared_bytes=4),
            lambda a: a["target"].update(arch=90),
            lambda a: a["specializations"]["constants"].update(N=-1),
            lambda a: a["specializations"]["constants"].update(N=True),
            lambda a: a["specializations"]["constants"].update(N=129),
            lambda a: a["specializations"]["constants"].update(BLOCK=0),
            lambda a: a["specializations"]["constants"].update(BLOCK=96),
            lambda a: a["specializations"]["constants"].update(BLOCK=True),
            lambda a: a["invocation"].update(num_warps=0),
            lambda a: a["invocation"].update(num_warps=3),
            lambda a: a["invocation"].update(num_warps=True),
            lambda a: a["invocation"].update(num_warps=8),
            lambda a: a["invocation"]["arguments"][0].update(binding="output:0"),
            lambda a: a["invocation"]["arguments"][1].update(llvm_type="i64"),
            lambda a: a["invocation"]["arguments"].pop(),
            lambda a: a["invocation"]["resources"]["global_scratch"].update(size_bytes=16),
            lambda a: a["invocation"].update(resources=None),
            lambda a: a.update(target=None),
            lambda a: a.update(invocation=None),
            lambda a: a.update(specializations=None),
            lambda a: a["invocation"]["arguments"][0].update(binding={}),
            lambda a: a.pop("payload"),
            lambda a: a.pop("port_map"),
            lambda a: a["port_map"][0].update(element_type="f64"),
            lambda a: a["port_map"][0].update(element_type=[]),
            lambda a: a["port_map"][0].update(element_type="f32"),
            lambda a: a["port_map"][0].update(argument=True),
            lambda a: a["port_map"][0].update(logical_port="output:0"),
            lambda a: a.update(source_fusion_contract={"dtype": "f32"}),
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

    def test_float_codegen_preserves_raw_and_fused_contraction_policies(self):
        for fused in (False, True):
            for enabled in (False, True):
                with self.subTest(fused=fused, enabled=enabled), tempfile.TemporaryDirectory() as temporary:
                    value = artifact(1, dtype="f32")
                    metadata = value["invocation"]["compiler_metadata"]
                    metadata["enable_fp_fusion"] = enabled
                    if fused:
                        value["validation"] = {"stage": "fused_llvm", "llvm_module_verified": True}
                        value["source_fusion_contract"] = {"dtype": "f32", "numerics": dict(metadata)}
                        value["fusion"] = {
                            "ask_numerics": "f32-rne-no-fast-math-no-contract",
                            "producer_numerics": {
                                "producer_enable_fp_fusion": enabled, "producer_enable_reflect_ftz": True,
                                "representation": "original-F-instruction-contract-flags"}}
                    result = self.compile_fixture(Path(temporary), value)
                    expected = "on" if fused else "fast" if enabled else "off"
                    self.assertEqual(result["fp_contract"], expected)
                    self.assertIn("--fp-contract=" + expected, result["commands"][0])
                    if fused:
                        value["fusion"]["producer_numerics"]["producer_enable_fp_fusion"] = not enabled
                        with patch("ask.cuda.subprocess.run") as compiler:
                            with self.assertRaisesRegex(CUDAError, "normalization"):
                                compile_cuda(Path(temporary) / "source.ll", value, Path(temporary) / "bad",
                                             llc="/test/llc", ptxas="/test/ptxas")
                            compiler.assert_not_called()

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
            def __init__(self):
                self.calls, self.freed = [], []
                self.library = SimpleNamespace(
                    cuFuncGetParamInfo=lambda *a: 1,
                    cuMemFree_v2=lambda pointer: self.freed.append(pointer.value),
                    cuModuleUnload=lambda module: self.calls.append("unload"),
                    cuCtxDestroy_v2=lambda context: self.calls.append("destroy"))
                self.next_pointer = 1000

            def attribute(self, device, attribute):
                return {75: 8, 76: 6, 10: 32}.get(attribute, 2**20)

            def function_attribute(self, function, attribute):
                return {0: 128, 1: 0, 4: 14}[attribute]

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

        with tempfile.TemporaryDirectory() as temporary:
            value = artifact()
            result = self.compile_fixture(Path(temporary), value)
            driver = FakeDriver()
            with patch("ask.cuda._Driver", return_value=driver):
                with self.assertRaisesRegex(CUDAError, "injected launch failure"):
                    execute_i32(value, result, [0] * 10)
            self.assertEqual(driver.calls.count("cuLaunchKernel"), 1)
            self.assertEqual(driver.freed, [2000, 1000])
            self.assertEqual(driver.calls[-2:], ["unload", "destroy"])
            self.assertNotIn("cuMemcpyDtoH_v2", driver.calls)


if __name__ == "__main__":
    unittest.main()
