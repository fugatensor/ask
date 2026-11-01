"""Hardware CUDA tests for native ASK CUDA kernels, raw ABI v2 kernels, and resident ExecutionSession."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import math
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import weakref

ROOT = Path(__file__).resolve().parents[2]
for entry in (str(ROOT / "python"), str(ROOT / "tests")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from ask import Compiler, ExecutionSession, Module, ScalarType, TensorType
from ask.cuda import CUDAError, compile_cuda, execute
from ask.gpu import CUDAKernel
from support.host_runner import ask_opt_path, llc_path, open_cuda_session, ptxas_path
from support.pointwise import (
    DTYPES,
    arithmetic_reference,
    build_module,
    cast_reference,
    expected_values,
    input_values,
    numerically_equal,
    round_reference,
)


def _generic_kernel_ir(arguments: str, incoming: str, outgoing: str, pointer_type: str) -> str:
    registers = ("tid.x", "tid.y", "tid.z", "ctaid.x", "ctaid.y", "ctaid.z")
    decls = "\n".join(f"declare i32 @llvm.nvvm.read.ptx.sreg.{r}()" for r in registers)
    reads = "\n".join(f"  %{r.replace('.', '_')} = call i32 @llvm.nvvm.read.ptx.sreg.{r}()" for r in registers)
    return f"""target triple = "nvptx64-nvidia-cuda"
@static = internal addrspace(3) global [12 x i32] undef, align 4
@dynamic = external addrspace(3) global [0 x i32], align 4
{decls}
define ptx_kernel void @generic_kernel({arguments}) {{
{reads}
  %by = mul i32 %ctaid_y, 2
  %bz = mul i32 %ctaid_z, 4
  %bxy = add i32 %ctaid_x, %by
  %b = add i32 %bxy, %bz
  %base = mul i32 %b, 12
  %ty = mul i32 %tid_y, 3
  %tz = mul i32 %tid_z, 6
  %txy = add i32 %tid_x, %ty
  %t = add i32 %txy, %tz
  %index = add i32 %base, %t
  %in = getelementptr i32, {pointer_type} %{incoming}, i32 %index
  %out = getelementptr i32, {pointer_type} %{outgoing}, i32 %index
  %sp = getelementptr [12 x i32], ptr addrspace(3) @static, i32 0, i32 %t
  %dp = getelementptr [0 x i32], ptr addrspace(3) @dynamic, i32 0, i32 %t
  %value = load i32, {pointer_type} %in, align 4
  store volatile i32 %value, ptr addrspace(3) %sp, align 4
  %v1 = load volatile i32, ptr addrspace(3) %sp, align 4
  %plus1 = add i32 %v1, 1
  store volatile i32 %plus1, ptr addrspace(3) %dp, align 4
  %v2 = load volatile i32, ptr addrspace(3) %dp, align 4
  %plus2 = add i32 %v2, 1
  store i32 %plus2, {pointer_type} %out, align 4
  ret void
}}
"""


class NativeCUDAHardwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llc = llc_path()
        cls.ptxas = ptxas_path()
        try:
            cls.compiler = Compiler(timeout=120)
        except Exception:
            cls.compiler = Compiler(ask_opt_path(), timeout=120)
        probe = open_cuda_session()
        cls.major, cls.minor = probe.capability
        cls.arch = cls.major * 10 + cls.minor
        cls.supports_bf16 = cls.major >= 8
        probe.close()
        cls._tmp = tempfile.TemporaryDirectory(prefix="ask-gpu-native-")
        cls.work = Path(cls._tmp.name)
        cls.cache_dir = cls.work / "cache"

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "_tmp"):
            cls._tmp.cleanup()

    def compile_owned(self, label: str, module: Module, *, block_size: int = 128, expected_hit: bool = False):
        kernel = self.compiler.compile_cuda(
            module,
            target={"backend": "cuda", "arch": self.arch},
            block_size=block_size,
            ptx_version=80,
            cache_dir=self.cache_dir,
            output_dir=self.work / label,
            llc=self.llc,
            ptxas=self.ptxas,
        )
        wanted = "hit" if expected_hit else "miss"
        self.assertEqual(kernel.cache["status"], wanted, (label, kernel.cache))
        self.assertEqual(kernel.codegen["executable_cache"]["status"], wanted, (label, kernel.codegen))
        return kernel

    def test_owned_precisions_shapes_casts_and_cache_hits(self):
        dtypes = [d for d in DTYPES if self.supports_bf16 or d != "bf16"]
        shape_cases = [
            *((dtype, "tail", {"size": 129}) for dtype in dtypes),
            *(
                (dtype, suffix, kwargs)
                for dtype in ("f32", "i64", "bool")
                for suffix, kwargs in (
                    ("shape3x5", {"shape": (3, 5)}),
                    ("rank0", {"shape": ()}),
                    ("empty", {"shape": (2, 0, 3)}),
                )
            ),
        ]
        with ExecutionSession() as session:
            for dtype, label_suffix, kwargs in shape_cases:
                with self.subTest(dtype=dtype, case=label_suffix):
                    mod = build_module(dtype=dtype, external=False, **kwargs)
                    shape = kwargs.get("shape", (kwargs.get("size"),))
                    vals = input_values(math.prod(shape), dtype)
                    exp = expected_values(vals, 1, 2, dtype=dtype, external=False)
                    cold = self.compile_owned(f"{dtype}-{label_suffix}-cold", mod)
                    kernel = cold
                    if label_suffix == "tail" and dtype in ("f32", "i32"):
                        warm = self.compile_owned(f"{dtype}-{label_suffix}-warm", mod, expected_hit=True)
                        self.assertEqual(cold.codegen["cubin_sha256"], warm.codegen["cubin_sha256"])
                        kernel = warm
                    exe = kernel.prepare(session)
                    x = session.from_host(vals, shape=shape, dtype=dtype)
                    y = session.tensor(shape, dtype)
                    exe.submit(x, y).wait()
                    self.assertTrue(numerically_equal(y.to_host(), exp, dtype), (dtype, label_suffix))

            # Direct multi-output 6x6 casts.
            for src in dtypes:
                mod = Module()
                g = mod.graph("direct_casts", inputs={"x": TensorType((65,), src)},
                              results=[TensorType((65,), dst) for dst in dtypes])
                g.return_(*(g.cast(g.inputs["x"], dtype=dst) for dst in dtypes))
                vals = input_values(65, src)
                expected = [[cast_reference(round_reference(v, src), src, dst) for v in vals] for dst in dtypes]
                exe = self.compile_owned(f"casts-{src}", mod).prepare(session)
                in_t = [session.from_host(vals, shape=(65,), dtype=src)]
                out_t = [session.tensor((65,), dst) for dst in dtypes]
                exe.submit_many(in_t, out_t).wait()
                for dst, out_buf, ref in zip(dtypes, out_t, expected):
                    self.assertTrue(numerically_equal(out_buf.to_host(), ref, dst), (src, dst))

            # Multi-port heterogeneous shapes, scalar argument, and aliasing rejection.
            multi = Module()
            g = multi.graph(
                "multi_ports",
                inputs={"x": TensorType((33,), "i64"), "s": ScalarType("f32")},
                results=[TensorType((33,), "i64"), TensorType((33,), "f32"), ScalarType("i64")],
            )
            doubled = g.add(g.inputs["x"], g.inputs["x"])
            shifted = g.add(g.cast(g.inputs["x"], dtype="f32"), g.splat(g.inputs["s"], (33,)))
            g.return_(doubled, shifted, g.cast(g.inputs["s"], dtype="i64"))
            exe = self.compile_owned("multi-ports", multi).prepare(session)
            x_vals = [2**50 + i for i in range(33)]
            ins = [session.from_host(x_vals, shape=(33,), dtype="i64"),
                   session.from_host([2.5], shape=(), dtype="f32")]
            outs = [session.tensor((33,), "i64"), session.tensor((33,), "f32"), session.tensor((), "i64")]
            exe.submit_many(ins, outs).wait()
            self.assertEqual(outs[0].to_host(), [arithmetic_reference("add", v, v, "i64") for v in x_vals])
            self.assertEqual(outs[2].to_host(), [2])
            with self.assertRaises((CUDAError, ValueError)):
                exe.submit_many(ins, [ins[0], outs[1], outs[2]])

    def test_raw_abi_v2_shared_memory_and_resident_session_lifecycle(self):
        with ExecutionSession() as session:
            bindings = ["null", "output:0", "null", "input:0", "null"]
            payload = _generic_kernel_ir(
                ", ".join(f"ptr addrspace(1) %arg{i}" for i in range(5)), "arg3", "arg1", "ptr addrspace(1)"
            )
            artifact = {
                "format": "llvm-ir", "entry": "generic_kernel", "payload": payload,
                "target": {"backend": "cuda", "arch": min(self.arch, 80)},
                "required_element_types": ["i32"], "codegen": {"fp_contract": "off", "ptx_version": 80},
                "port_map": [
                    {"argument": bindings.index(port), "logical_port": port, "shape": [2, 2, 2, 12],
                     "layout": "contiguous_row_major", "element_type": "i32"}
                    for port in ("input:0", "output:0")
                ],
                "invocation": {
                    "version": 2, "kind": "cuda-kernel", "grid": [2, 2, 2],
                    "block": [3, 2, 2], "shared_bytes": 48,
                    "arguments": [{"argument": i, "binding": b, "llvm_type": "ptr addrspace(1)"}
                                  for i, b in enumerate(bindings)],
                },
            }
            raw_dir = self.work / "raw-abi"
            raw_dir.mkdir()
            ll_path = raw_dir / "input.ll"
            ll_path.write_text(payload, encoding="utf-8")
            codegen = compile_cuda(ll_path, artifact, raw_dir / "codegen", llc=self.llc, ptxas=self.ptxas)
            vals = list(range(-48, 48))
            exp = [v + 2 for v in vals]
            self.assertEqual(execute(artifact, codegen, vals)["output"], exp)

            package = CUDAKernel(artifact, codegen, {}, {})
            prepared = package.prepare(session)
            src = session.from_host(vals, shape=(2, 2, 2, 12), dtype="i32")
            dst = session.tensor((2, 2, 2, 12), "i32")

            # 20 hot-path launches without extra allocations, copies, or module reloads.
            before = session.stats
            for _ in range(20):
                ev = prepared.submit(src, dst)
            ev.wait()
            after = session.stats
            self.assertEqual(after["launches"] - before["launches"], 20)
            for key in ("allocations", "module_loads", "host_to_device_copies", "device_to_host_copies"):
                self.assertEqual(after[key], before[key])
            self.assertEqual(dst.to_host(), exp)

            # External CAI owner retention until GPU completion.
            class ExternalOwner:
                def __init__(self, t):
                    self.t = t
                @property
                def __cuda_array_interface__(self):
                    return self.t.__cuda_array_interface__

            owner = ExternalOwner(session.from_host(vals, shape=(2, 2, 2, 12), dtype="i32"))
            owner_ref = weakref.ref(owner)
            borrowed = session.from_cuda_array(owner)
            pending = prepared.submit(borrowed, dst)
            del owner, borrowed
            gc.collect()
            self.assertIsNotNone(owner_ref())
            pending.wait()
            gc.collect()
            self.assertIsNone(owner_ref())

            # Multi-threaded concurrent submit + property reads.
            barrier = threading.Barrier(3)
            outputs = [session.tensor((2, 2, 2, 12), "i32") for _ in range(3)]

            def worker(idx: int):
                barrier.wait(timeout=5)
                for _ in range(10):
                    done = prepared.submit(src, outputs[idx])
                done.wait()
                return outputs[idx].to_host()

            with ThreadPoolExecutor(max_workers=3) as pool:
                for res in pool.map(worker, range(3)):
                    self.assertEqual(res, exp)

            # Oversized dynamic shared memory rejected cleanly without incrementing launch count.
            native_mod = session.load(Path(codegen["cubin"]).read_bytes(), "generic_kernel",
                                      required_types=["i32"], arch=min(self.arch, 80), expected_parameters=5)
            prev_launches = session.stats["launches"]
            with self.assertRaisesRegex(CUDAError, "dynamic shared memory"):
                session.submit(native_mod, [None, dst, None, src, None],
                               grid=[2, 2, 2], block=[3, 2, 2], shared_bytes=2**31)
            self.assertEqual(session.stats["launches"], prev_launches)


if __name__ == "__main__":
    unittest.main()
