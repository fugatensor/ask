"""Native CUDA graph lowering, CUDA compilation contract, adapter routing, and prepared executable validation."""
from __future__ import annotations

import base64
import copy
import ctypes
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT / "python"), str(ROOT / "tests")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from ask import CompilerEnvironment, CompilerError, Module, ScalarType, TensorType
from ask.cuda import (
    CUDAError,
    _artifact_digest,
    _check_device_dtype,
    _decode_output,
    _digest,
    _host_buffers,
    _kernel_contract,
    compile_cuda,
    install_external,
    execute,
    execute_i32,
)
from ask.gpu import CUDAKernel, compile_graph
from support.host_runner import run_ask_opt
from support.pointwise import DTYPES, build_module


def make_artifact(size: int = 10, *, block: int = 128, dtype: str = "i32") -> dict:
    return {
        "format": "llvm-ir",
        "entry": "scale2",
        "payload": "define void @fixture() { ret void }\n",
        "target": {"backend": "cuda", "arch": 80},
        "port_map": [
            {"argument": idx, "logical_port": binding, "element_type": dtype,
             "shape": [size], "layout": "contiguous_row_major"}
            for idx, binding in enumerate(("input:0", "output:0"))
        ],
        "invocation": {
            "kind": "cuda-kernel",
            "version": 2,
            "grid": [max(1, (size + block - 1) // block), 1, 1],
            "block": [block, 1, 1],
            "shared_bytes": 0,
            "arguments": [
                {"argument": idx, "binding": binding, "llvm_type": "ptr addrspace(1)"}
                for idx, binding in enumerate(("input:0", "output:0"))
            ],
        },
        "codegen": {"fp_contract": "fast", "ptx_version": 80},
        "required_element_types": [dtype],
    }


def make_external_artifact(format_="ptx", *, shape=(2, 5), payload=None, arch=80):
    if payload is None:
        payload = (".version 8.0\n.target sm_80\n.address_size 64\n" if format_ == "ptx"
                   else base64.b64encode(b"\x7fELF" + b"\0" * 60).decode())
    artifact = make_artifact()
    artifact.pop("codegen")
    artifact.update(protocol_version=2, format=format_, payload=payload,
                    producer={"name": "fixture", "version": "1"}, target={"backend": "cuda", "arch": arch})
    artifact["invocation"]["completion"] = "device-event"
    artifact["invocation"]["arguments"] = [
        {"argument": i, "binding": name, "kind": "pointer"}
        for i, name in enumerate(("input:0", "output:0"))]
    for port in artifact["port_map"]:
        port["shape"] = list(shape)
    return artifact


def mock_tensor(shape=(2, 5), dtype="i32", pointer=4096, nbytes=40, readonly=False):
    return SimpleNamespace(shape=shape, dtype=dtype, data_ptr=pointer, nbytes=nbytes, readonly=readonly)


class SessionRecorder:
    def __init__(self):
        self.loads = []
        self.submissions = []

    def load(self, *args, **kwargs):
        self.loads.append((args, kwargs))
        return object()

    def submit(self, *args, **kwargs):
        self.submissions.append((args, kwargs))
        return object()


class CUDAGraphLoweringTests(unittest.TestCase):
    def lower(self, module: Module, *, arch: int = 86, optimize: bool = True):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            flags = ["-", "--emit-cuda-llvm", f"--cuda-arch={arch}", "--report-json", report]
            if optimize:
                flags.append("--optimize")
            result = run_ask_opt(*flags, source=module.to_mlir())
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads(report.read_text(encoding="utf-8"))
        artifact = metadata["cuda_artifact"]
        self.assertEqual(artifact["invocation"]["version"], 2)
        self.assertEqual(run_ask_opt("-", "--check-llir", source=result.stdout).returncode, 0)
        return artifact, metadata

    def test_all_precisions_ports_empty_and_64bit_indexing(self):
        for dtype in DTYPES:
            artifact, metadata = self.lower(build_module(129, dtype=dtype, external=False))
            self.assertIn("define ptx_kernel void @pipeline", artifact["payload"])
            self.assertEqual(artifact["invocation"]["grid"], [2, 1, 1])
            self.assertIn(dtype, artifact["required_element_types"])
            self.assertEqual(metadata["cuda_lowering"]["temporary_buffer_count"], 0)

        multi = Module()
        vec, other = TensorType((129,), "f32"), TensorType((3,), "i64")
        g = multi.graph("calc", inputs={"x": vec, "s": ScalarType("f32"), "y": other},
                        results=[vec, TensorType((129,), "i64"), other, ScalarType("f32")])
        val = g.add(g.inputs["x"], g.splat(g.inputs["s"], (129,)))
        g.return_(val, g.cast(val, dtype="i64"), g.inputs["y"], g.inputs["s"])
        art, rep = self.lower(multi)
        self.assertEqual(rep["cuda_lowering"]["output_loop_count"], 3)
        self.assertEqual(rep["cuda_lowering"]["groups"][0]["outputs"], [0, 1])

        empty_art, empty_rep = self.lower(build_module(shape=(0, 3), dtype="f16", external=False))
        self.assertNotIn(" load ", empty_art["payload"])
        self.assertEqual(empty_rep["cuda_lowering"]["zero_element_outputs"], 1)

        large = Module()
        lt = TensorType((2**31 + 17,), "f32")
        lg = large.graph("large", inputs={"x": lt}, results=[lt])
        lg.return_(lg.inputs["x"])
        large_art, _ = self.lower(large)
        self.assertEqual(large_art["invocation"]["grid"], [65535, 1, 1])
        self.assertIn("icmp ult i64", large_art["payload"])

    def test_bf16_arch_and_flag_rejections(self):
        bf = Module()
        t = TensorType((3,), "f32")
        g = bf.graph("conv", inputs={"x": t}, results=[t])
        g.return_(g.cast(g.cast(g.inputs["x"], dtype="bf16"), dtype="f32"))
        rejected = run_ask_opt("-", "--emit-cuda-llvm", "--cuda-arch=75", source=bf.to_mlir())
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("bf16 compilation requires target sm80", rejected.stderr)

        src = build_module(3, external=False).to_mlir()
        for flags, needle in (
            (["--emit-cuda-llvm"], "explicit --cuda-arch"),
            (["--emit-cuda-llvm", "--cuda-arch=86", "--cuda-block-size=0"], "block size"),
            (["--emit-llvm", "--emit-cuda-llvm"], "select only one"),
        ):
            res = run_ask_opt("-", *flags, source=src)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn(needle, res.stderr)
        ext = run_ask_opt("-", "--emit-cuda-llvm", "--cuda-arch=86", source=build_module(3).to_mlir())
        self.assertNotEqual(ext.returncode, 0)
        self.assertIn("cannot compile ask.external", ext.stderr)


class CUDAContractAndRoutingTests(unittest.TestCase):
    def compile_fixture(self, directory: Path, value: dict):
        source = directory / "source.ll"
        source.write_text(value["payload"], encoding="utf-8")

        def fake_tool(command, **kwargs):
            Path(command[-1]).write_bytes(b"fixture binary\n")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("ask.cuda.subprocess.run", side_effect=fake_tool) as run_mock:
            result = compile_cuda(source, value, directory / "codegen", llc="/test/llc", ptxas="/test/ptxas")
        self.assertEqual(run_mock.call_count, 2)
        return result

    def test_codegen_commands_tamper_checks_and_host_buffer_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            val = make_artifact()
            res = self.compile_fixture(work, val)
            self.assertEqual(res["commands"][0][1:5],
                             ["-march=nvptx64", "-mcpu=sm_80", "-mattr=+ptx80", "--fp-contract=fast"])
            self.assertEqual(res["commands"][1][1], "--gpu-name=sm_80")
            Path(res["cubin"]).write_bytes(b"tampered")
            with patch("ask.cuda._Driver") as drv:
                with self.assertRaises(CUDAError):
                    execute_i32(val, res, [0] * 10)
                drv.assert_not_called()

        for mutate in (
            lambda a: a["invocation"].update(grid=[0, 1, 1]),
            lambda a: a["invocation"].update(version=1),
            lambda a: a["target"].update(arch="90a"),
            lambda a: a["port_map"][0].update(layout="strided"),
            lambda a: a["codegen"].update(fp_contract="unknown"),
        ):
            bad = make_artifact()
            mutate(bad)
            with patch("ask.cuda._Driver") as drv:
                with self.assertRaises(CUDAError):
                    execute(bad, {}, [0] * 10)
                drv.assert_not_called()

        # Large integer rounds directly to f32 without double intermediate loss.
        buf, _ = _host_buffers([2**100 + 2**76 + 1], 1, "f32")
        self.assertEqual(buf[0], float(2**100 + 2**77))
        for dtype, bits in (("f16", 10), ("bf16", 7)):
            half_ulp = 2**(-bits - 1)
            src, _ = _host_buffers([1.0 + half_ulp, 1.0 + 3 * half_ulp, -0.0], 3, dtype)
            self.assertEqual(ctypes.sizeof(src), 6)
            dec = _decode_output(src, 3, dtype)
            self.assertEqual(dec[:2], [1.0, 1.0 + 2**(1 - bits)])
            self.assertEqual(math.copysign(1.0, dec[2]), -1.0)
        _check_device_dtype("bf16", 8, 6, "Ampere")

    def test_owned_graph_routing(self):
        owned_compiler = Mock(executable="ask-opt")
        owned_compiler.capabilities.return_value = {
            "external_execution": True, "external_formats": ["cubin", "ptx"],
            "cuda_graph_codegen": True, "llvm_revision": "rev",
        }
        owned_compiler.prepare.return_value = SimpleNamespace(report={"external_requests": [], "shape_snapshot": {}})
        stats = {"status": "hit", "counters": {"hits": 1}}
        with tempfile.TemporaryDirectory() as directory, \
             patch("ask.gpu._tool", return_value={"path": "/t", "sha256": "a" * 64}), \
             patch("ask.gpu.IRCache") as cache_cls, \
             patch("ask.gpu.compile_cuda", return_value={"executable_cache": stats}):
            cache_cls.return_value.get_or_create.return_value = (
                {"kernel.ll": b"LL", "kernel.artifact.json": b"{}", "report.json": b"{}"}, stats
            )
            kernel = compile_graph(
                owned_compiler, "graph", target={"backend": "cuda", "arch": 86},
                cache_dir=Path(directory) / "c", output_dir=Path(directory) / "o", llc="llc", ptxas="ptxas",
            )
            key = cache_cls.return_value.get_or_create.call_args.args[0]
            self.assertEqual(key["producer"], {"name": "ask"})
            self.assertEqual(kernel.cache["status"], "hit")


class PreparedExecutableContractTests(unittest.TestCase):
    def make_kernel_package(self, directory: Path):
        val = make_artifact()
        for port in val["port_map"]:
            port["shape"] = [2, 5]
        src, cubin = directory / "in.ll", directory / "k.cubin"
        src.write_text(val["payload"], encoding="utf-8")
        cubin.write_bytes(b"synthetic cubin")
        receipt = {
            "schema_version": 1, "entry": val["entry"], "arch": 80, "invocation": copy.deepcopy(val["invocation"]),
            "element_type": "i32", "fp_contract": "fast", "artifact_sha256": _artifact_digest(val),
            "llvm_sha256": _digest(src.read_bytes()), "llvm_ir": str(src),
            "cubin_sha256": _digest(cubin.read_bytes()), "cubin": str(cubin),
        }
        return CUDAKernel(val, receipt, {"shape_snapshot": {"shape_bindings": []}}, {}), SessionRecorder()

    def test_single_and_multi_port_hot_path_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            kernel, session = self.make_kernel_package(Path(directory))
            exe = kernel.prepare(session)
            src, dst = mock_tensor(), mock_tensor(pointer=8192)
            with patch.object(Path, "read_bytes", side_effect=AssertionError("hot path I/O")):
                exe.submit(src, dst)
                exe.submit(src, dst)
            self.assertEqual((len(session.loads), len(session.submissions)), (1, 2))
            for bad, msg in (
                (mock_tensor(shape=(10,)), "shape"),
                (mock_tensor(dtype="f32"), "dtype"),
            ):
                with self.assertRaisesRegex(CUDAError, msg):
                    exe.submit(bad, dst)
            with self.assertRaisesRegex(CUDAError, "non-overlapping"):
                exe.submit(src, mock_tensor(pointer=4100))

            # Multi-port scrambled physical order vs stable logical order.
            m_kernel, m_session = self.make_kernel_package(Path(directory))
            specs = [("output:1", [], "i64"), ("input:1", [], "i64"),
                     ("output:0", [2, 5], "f32"), ("input:0", [2, 5], "i32")]
            m_art = m_kernel.artifact
            m_art["port_map"] = [
                {"argument": idx, "logical_port": bind, "shape": shp,
                 "element_type": dt, "layout": "contiguous_row_major"}
                for idx, (bind, shp, dt) in enumerate(specs)
            ]
            m_art["invocation"]["arguments"] = [
                {"argument": idx, "binding": bind, "llvm_type": "ptr addrspace(1)"}
                for idx, (bind, _, _) in enumerate(specs)
            ]
            m_kernel.codegen.update(element_type=None, invocation=copy.deepcopy(m_art["invocation"]),
                                    artifact_sha256=_artifact_digest(m_art))
            m_exe = m_kernel.prepare(m_session)
            ins = [mock_tensor(), mock_tensor(shape=(), dtype="i64", pointer=8192, nbytes=8)]
            outs = [mock_tensor(dtype="f32", pointer=12288),
                    mock_tensor(shape=(), dtype="i64", pointer=16384, nbytes=8)]
            m_exe.submit_many(ins, outs)
            self.assertEqual(m_session.submissions[0][0][1], [outs[1], ins[1], outs[0], ins[0]])

            bad_sparse = copy.deepcopy(m_art)
            bad_sparse["port_map"][1]["logical_port"] = "input:3"
            bad_sparse["invocation"]["arguments"][1]["binding"] = "input:3"
            with self.assertRaises(CUDAError):
                _kernel_contract(bad_sparse)


class ExternalExecutableTests(unittest.TestCase):
    def test_direct_formats_prepare_and_verify_without_compiler_tools(self):
        for format_ in ("cubin", "ptx"):
            with self.subTest(format=format_), tempfile.TemporaryDirectory() as directory:
                artifact = make_external_artifact(format_)
                with patch("ask.cuda.subprocess.run", side_effect=AssertionError("unexpected tool")):
                    receipt = install_external(artifact, directory)
                    session = SessionRecorder()
                    executable = CUDAKernel(artifact, receipt, {}, {}).prepare(session)
                    executable.submit(mock_tensor(), mock_tensor(pointer=8192))
                self.assertEqual(receipt["compiler_invocations"], 0)
                self.assertNotIn("llvm_ir", receipt)
                expected = (base64.b64decode(artifact["payload"]) if format_ == "cubin"
                            else artifact["payload"].encode())
                self.assertEqual(session.loads[0][0][0], expected)
                self.assertEqual(Path(receipt[format_]).read_bytes(), expected)
                with self.assertRaisesRegex(CUDAError, "already contains"):
                    install_external(artifact, directory)
                Path(receipt[format_]).write_bytes(b"tampered")
                with self.assertRaisesRegex(CUDAError, "digest"), patch("ask.cuda._Driver") as driver:
                    execute(artifact, receipt, [0] * 10)
                driver.assert_not_called()

    def test_external_contract_payload_and_receipt_rejections(self):
        mutations = [
            lambda a: a.update(protocol_version=True),
            lambda a: a.update(format="llvm-bitcode"),
            lambda a: a.update(payload=".version 8.0\0"),
            lambda a: a.update(format="cubin", payload="invalid base64"),
            lambda a: a.update(format="cubin", payload=base64.b64encode(b"not ELF").decode()),
            lambda a: a["invocation"]["arguments"][0].update(kind="scalar"),
            lambda a: a["invocation"]["arguments"][0].update(binding="scratch:0"),
            lambda a: a["invocation"].update(completion="host"),
            lambda a: a["port_map"][0].update(argument=1),
        ]
        for mutate in mutations:
            artifact = make_external_artifact()
            mutate(artifact)
            with self.subTest(artifact=artifact), self.assertRaises(CUDAError):
                _kernel_contract(artifact)
        with tempfile.TemporaryDirectory() as directory:
            artifact = make_external_artifact()
            receipt = install_external(artifact, directory)
            for field, value in (("format", "cubin"), ("schema_version", 1),
                                 ("payload_sha256", "0" * 64), ("llvm_ir", "/etc/passwd")):
                bad = dict(receipt, **{field: value})
                with self.subTest(field=field), self.assertRaises(CUDAError):
                    CUDAKernel(artifact, bad, {}, {}).prepare(SessionRecorder())


if __name__ == "__main__":
    unittest.main()
