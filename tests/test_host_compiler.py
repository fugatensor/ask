"""Host CPU backend lowering, multi-precision C ABI execution, and multi-output loop fusion."""
from __future__ import annotations

import ctypes
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT / "python"), str(ROOT / "tests")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from ask import Module, ScalarType, TensorType
from support.host_runner import (
    STORAGE,
    link_object_to_cdll,
    make_buffer,
    read_buffer,
    require_cc,
    run_ask_opt,
)
from support.pointwise import (
    DTYPES,
    build_module,
    cast_reference,
    expected_values,
    input_values,
    numerically_equal,
    round_reference,
)


def i32(value: int) -> int:
    return ctypes.c_int32(value).value


def shared_loop_graph(shape: tuple[int, ...], *, scalar_output: bool = False,
                      single_output: bool = False, repeat_output: bool = False) -> str:
    module = Module()
    tensor = TensorType(shape)
    results = [tensor] if single_output else [tensor, tensor]
    if scalar_output:
        results.insert(1, ScalarType())
    if repeat_output:
        results.append(tensor)
    graph = module.graph("calculate", inputs={"x": tensor, "s": ScalarType()}, results=results)
    scalar = graph.inputs["s"]
    biased = graph.add(graph.inputs["x"], graph.splat(scalar, shape))
    scalar_res = graph.add(scalar, graph.constant(1)) if scalar_output else None
    doubled = graph.mul(biased, graph.splat(graph.constant(2), shape))
    final = graph.add(doubled, graph.splat(graph.constant(3), shape))
    outputs = [final] if single_output else [biased, final]
    if scalar_output:
        outputs.insert(1, scalar_res)
    if repeat_output:
        outputs.append(biased)
    graph.return_(*outputs)
    return module.to_mlir()


class HostCodegenAndFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        caps_proc = run_ask_opt("--capabilities")
        if caps_proc.returncode != 0:
            raise RuntimeError(caps_proc.stderr)
        cls.caps = json.loads(caps_proc.stdout)
        if not cls.caps["host_codegen"]:
            raise unittest.SkipTest("host backend disabled in this ASK build")
        require_cc()

    def compile_host(self, source: str, work: Path, stem: str, *, optimize: bool = False,
                     object_file: bool = True, no_fusion: bool = False):
        out = work / f"{stem}{'.o' if object_file else '.ll'}"
        rep = work / f"{stem}.json"
        flags = ["-", "--emit-object" if object_file else "--emit-llvm", "-o", out, "--report-json", rep]
        if optimize:
            flags.append("--optimize")
        if no_fusion:
            flags.append("--no-fusion")
        res = run_ask_opt(*flags, source=source)
        self.assertEqual(res.returncode, 0, res.stderr)
        metadata = json.loads(rep.read_text(encoding="utf-8"))
        if not object_file:
            return metadata, out.read_text(encoding="utf-8")
        return metadata, link_object_to_cdll(out, work / f"{stem}.so")

    def test_shapes_scalars_empty_tensors_and_f32_non_contraction(self):
        module = Module()
        shapes = [(10,), (2, 3), (), (0,)]
        for idx, shape in enumerate(shapes):
            t = TensorType(shape)
            g = module.graph(f"calc{idx}", inputs={"x": t, "s": ScalarType()}, results=[t, ScalarType(), t])
            x, s = g.inputs["x"], g.inputs["s"]
            doubled = g.mul(g.add(x, g.splat(s, shape)), g.splat(g.constant(2), shape))
            g.return_(doubled, g.add(s, g.constant(1)), x)
        with tempfile.TemporaryDirectory() as directory:
            _, lib = self.compile_host(module.to_mlir(), Path(directory), "shapes")
            ptr = ctypes.POINTER(ctypes.c_int32)
            for idx, shape in enumerate(shapes):
                count = math.prod(shape)
                data = [2147483647, -2147483648, -7, 0, 1, 7, 42, 1024, -1024, 9][:count]
                x = (ctypes.c_int32 * max(count, 1))(*data)
                out = (ctypes.c_int32 * max(count, 1))(*([123] * max(count, 1)))
                copy_out = (ctypes.c_int32 * max(count, 1))(*([456] * max(count, 1)))
                scalar_out = ctypes.c_int32()
                fn = getattr(lib, f"calc{idx}")
                fn.argtypes = [ptr, ctypes.c_int32, ptr, ptr, ptr]
                fn(x, 2147483647, out, ctypes.byref(scalar_out), copy_out)
                self.assertEqual(list(out)[:count], [i32((v + 2147483647) * 2) for v in data])
                self.assertEqual(scalar_out.value, -2147483648)
                if count == 0:
                    self.assertEqual(out[0], 123)

        fmod = Module()
        ft, fs = TensorType((8,), "f32"), ScalarType("f32")
        fg = fmod.graph("fcalc", inputs={"x": ft, "a": fs}, results=[ft, ft])
        prod = fg.mul(fg.inputs["x"], fg.splat(fg.inputs["a"], (8,)))
        adj = fg.add(prod, fg.splat(fg.constant(-1.0, dtype="f32"), (8,)))
        fg.return_(prod, adj)
        with tempfile.TemporaryDirectory() as directory:
            _, llvm_text = self.compile_host(fmod.to_mlir(), Path(directory), "fll", object_file=False)
            self.assertNotIn("fmul contract", llvm_text)
            self.assertNotIn("fadd fast", llvm_text)
            meta, flib = self.compile_host(fmod.to_mlir(), Path(directory), "fobj", optimize=True)
            self.assertEqual(meta["abi"]["functions"][0]["lowering"]["loop_count"], 1)
            data = [1.0 + 2**-23, -0.0, float("inf"), float("-inf"), float("nan"), 2**-149, -2**-149, 0.0]
            coeff = 1.0 - 2**-23
            fptr = ctypes.POINTER(ctypes.c_float)
            flib.fcalc.argtypes = [fptr, ctypes.c_float, fptr, fptr]
            x, p_out, r_out = (ctypes.c_float * 8)(*data), (ctypes.c_float * 8)(), (ctypes.c_float * 8)()
            flib.fcalc(x, coeff, p_out, r_out)
            self.assertEqual(r_out[0], 0.0)
            self.assertEqual(struct.pack("f", p_out[1]), struct.pack("f", -0.0))

    def test_shared_pointwise_graph_and_full_cast_matrix(self):
        supports_bf16 = self.caps["host_bf16_hardware_supported"]
        shape, count = (2, 8), 16
        combos = [(d, None) for d in DTYPES] + [(d, "f32") for d in ("f16", "bf16")]
        for dtype, compute in combos:
            mod = build_module(shape=shape, dtype=dtype, external=False, compute_dtype=compute)
            with self.subTest(dtype=dtype, compute=compute), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                if dtype == "bf16" and not supports_bf16:
                    rejected = run_ask_opt("-", "--emit-object", "-o", work / "bad.o", source=mod.to_mlir())
                    self.assertNotEqual(rejected.returncode, 0)
                    self.assertIn("bf16 requires hardware support", rejected.stderr)
                    continue
                values = input_values(count, dtype)
                expected = expected_values(values, 1, 2, dtype=dtype, external=False, compute_dtype=compute)
                _, lib = self.compile_host(mod.to_mlir(), work, "pt", optimize=True)
                ctype = STORAGE[dtype]
                lib.pipeline.argtypes = [ctypes.POINTER(ctype), ctypes.POINTER(ctype)]
                in_buf = make_buffer(values, dtype)
                out_buf = (ctype * count)()
                lib.pipeline(in_buf, out_buf)
                actual = read_buffer(out_buf, count, dtype)
                self.assertTrue(numerically_equal(actual, expected, dtype), (dtype, actual, expected))

        dtypes = [d for d in DTYPES if d != "bf16" or supports_bf16]
        c_count = 24
        cast_mod = Module()
        data = {}
        for src in dtypes:
            vals = input_values(c_count, src)
            if src == "i64":
                vals[-6:] = [(2**40 + 2**32 + d) * s for d in (-1, 0, 1) for s in (-1, 1)]
            data[src] = [round_reference(v, src) for v in vals]
            for dst in dtypes:
                g = cast_mod.graph(f"cast_{src}_{dst}", inputs={"x": TensorType((c_count,), src)},
                                   results=[TensorType((c_count,), dst)])
                g.return_(g.cast(g.inputs["x"], dtype=dst))
        bool_g = cast_mod.graph("bool_abi", inputs={"x": TensorType((4,), "bool"), "s": ScalarType("bool")},
                                results=[TensorType((4,), "bool"), ScalarType("bool")])
        bool_g.return_(bool_g.inputs["x"], bool_g.inputs["s"])
        with tempfile.TemporaryDirectory() as directory:
            _, lib = self.compile_host(cast_mod.to_mlir(), Path(directory), "casts", optimize=True)
            for src in dtypes:
                in_buf = make_buffer(data[src], src)
                for dst in dtypes:
                    out_buf = (STORAGE[dst] * c_count)()
                    fn = getattr(lib, f"cast_{src}_{dst}")
                    fn.argtypes = [ctypes.POINTER(STORAGE[src]), ctypes.POINTER(STORAGE[dst])]
                    fn(in_buf, out_buf)
                    actual = read_buffer(out_buf, c_count, dst)
                    expected = [cast_reference(v, src, dst) for v in data[src]]
                    self.assertTrue(numerically_equal(actual, expected, dst), (src, dst))
            b_in, b_out, b_sc = (ctypes.c_uint8 * 4)(0, 1, 2, 255), (ctypes.c_uint8 * 4)(), ctypes.c_uint8()
            lib.bool_abi.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint8,
                                     ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8)]
            lib.bool_abi(b_in, 255, b_out, ctypes.byref(b_sc))
            self.assertEqual(list(b_out), [0, 1, 1, 1])
            self.assertEqual(b_sc.value, 1)

    def test_multi_output_loop_fusion_and_group_validation(self):
        source = shared_loop_graph((10,))
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            base_meta, base_ir = self.compile_host(source, work, "base", optimize=False, object_file=False)
            fused_meta, fused_ir = self.compile_host(source, work, "fused", optimize=True, object_file=False)
            base_low = base_meta["abi"]["functions"][0]["lowering"]
            fused_low = fused_meta["abi"]["functions"][0]["lowering"]
            self.assertEqual((base_low["loop_count"], fused_low["loop_count"]), (2, 1))
            self.assertEqual((base_ir.count("load i32"), fused_ir.count("load i32")), (2, 1))
            self.assertNotIn("add nsw", fused_ir)

            dis_meta, _ = self.compile_host(source, work, "dis", optimize=True, object_file=False, no_fusion=True)
            self.assertEqual(dis_meta["abi"]["functions"][0]["lowering"]["loop_count"], 2)

            r0_meta, _ = self.compile_host(
                shared_loop_graph((), scalar_output=True, repeat_output=True),
                work, "rank0", optimize=True, object_file=False,
            )
            self.assertEqual(r0_meta["abi"]["functions"][0]["lowering"]["groups"][0]["outputs"], [0, 2, 3])

            empty_meta, empty_lib = self.compile_host(shared_loop_graph((0,)), work, "empty", optimize=True)
            self.assertEqual(empty_meta["abi"]["functions"][0]["lowering"]["loop_count"], 0)
            ptr = ctypes.POINTER(ctypes.c_int32)
            empty_lib.calculate.argtypes = [ptr, ctypes.c_int32, ptr, ptr]
            empty_lib.calculate(None, 1, None, None)

        bad_group_base = """module {
  func.func @calculate(%x: tensor<10xi32>) -> tensor<10xi32> {
    %one = "ask.constant"() {value = 1 : i32} : () -> i32
    %ones = "ask.splat"(%one) {shape = array<i64: 10>, ask.fusion_group = "g"} : (i32) -> tensor<10xi32>
    %result = "ask.add"(%x, %ones) {overflow = "wrap", ask.fusion_group = "g"} : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
    return %result : tensor<10xi32>
  }
}"""
        self.assertEqual(run_ask_opt("-", "--emit-llvm", source=bad_group_base).returncode, 0)
        for bad in (
            bad_group_base.replace('ask.fusion_group = "g"', 'ask.fusion_group = ""'),
            bad_group_base.replace(', ask.fusion_group = "g"', "", 1),
        ):
            res = run_ask_opt("-", "--emit-llvm", source=bad)
            self.assertGreater(res.returncode, 0)
            self.assertIn("fusion_group", res.stderr)


if __name__ == "__main__":
    unittest.main()
