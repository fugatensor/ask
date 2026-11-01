"""ASK Python SDK builder, numerical codecs, client protocol, and native bindings."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT / "python"), str(ROOT / "tests")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from ask import Compiler, CompilerError, Module, ScalarType, TensorType, __version__
from ask.ir import mlir_string
from ask.numerics import decode_float, encode_f32, encode_float, round_f32, round_float
from support.host_runner import ask_opt_path

NATIVE_AVAILABLE = importlib.util.find_spec("ask._native") is not None
if os.environ.get("ASK_NATIVE_REQUIRED") and not NATIVE_AVAILABLE:
    raise RuntimeError("ask._native extension is required when ASK_NATIVE_REQUIRED=1")


def example_module() -> Module:
    spec = importlib.util.spec_from_file_location("ask_framework_example", ROOT / "examples/framework_frontend.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    return example.build_module()


def self_owned_module() -> Module:
    module = Module()
    tensor = TensorType((10,))
    graph = module.graph("main", inputs={"x": tensor}, results=[tensor])
    t = graph.add(graph.inputs["x"], graph.splat(graph.constant(1), (10,)), id="P")
    t = graph.mark(t, mark_id="after_P")
    y = graph.mul(t, graph.splat(graph.constant(2), (10,)), id="Q")
    graph.return_(y)
    return module


def mixed_f16_module() -> Module:
    module = Module()
    tensor = TensorType((2, 5), "f16")
    graph = module.graph("mixed", inputs={"x": tensor}, results=[tensor])
    wide = graph.cast(graph.inputs["x"], dtype="f32")
    bias = graph.splat(graph.constant(0.1, dtype="f32"), (2, 5))
    result = graph.cast(graph.add(wide, bias), dtype="f16")
    graph.return_(graph.mark(result, mark_id="output"))
    return module


class BuilderAndCodecTests(unittest.TestCase):
    def test_types_constants_and_explicit_casts(self):
        self.assertEqual(str(TensorType((None, 4))), "tensor<?x4xi32>")
        self.assertEqual(str(TensorType(())), "tensor<i32>")
        self.assertEqual(str(ScalarType("fp16")), "f16")
        self.assertEqual(str(ScalarType("bool")), "i1")
        for bad_shape in ((-1,), (True,), (2**63,), ("10",)):
            with self.assertRaises(ValueError):
                TensorType(bad_shape)
        with self.assertRaises(ValueError):
            ScalarType("f64")

        module = Module()
        graph = module.graph("mixed", inputs={"x": TensorType((2, 3), "bool")},
                             results=[TensorType((2, 3), "f32")])
        widened = graph.cast(graph.inputs["x"], dtype="i64", id="widen", value_id="int_value")
        out = graph.cast(widened, dtype="f32")
        graph.return_(out)
        mlir = module.to_mlir()
        self.assertEqual(mlir.count('"ask.cast"'), 2)
        self.assertIn('ask.value_id = "int_value"', mlir)

        consts = Module().graph("consts", inputs={}, results=[])
        true_val = consts.constant(True, dtype="bool")
        for op in (consts.add, consts.mul):
            with self.assertRaisesRegex(TypeError, "bool add/mul"):
                op(true_val, true_val)
        for bad_bool in (0, 1, "true"):
            with self.assertRaises(ValueError):
                consts.constant(bad_bool, dtype="bool")
        for bad_i64 in (True, 2**63, -(2**63) - 1):
            with self.assertRaises(ValueError):
                consts.constant(bad_i64, dtype="i64")
        consts.constant(-(2**63), dtype="i64")
        consts.constant(2**63 - 1, dtype="i64")
        consts.return_()

    def test_float_constants_and_direct_rne_codecs(self):
        for dtype, precision, zero, one in (
            ("f32", 23, "80000000", "3F800000"),
            ("f16", 10, "8000", "3C00"),
            ("bf16", 7, "8000", "3F80"),
        ):
            module = Module()
            graph = module.graph("main", inputs={}, results=[ScalarType(dtype)])
            graph.return_(graph.add(graph.constant(-0.0, dtype=dtype),
                                    graph.constant(1.0 + 2.0**-(precision + 1), dtype=dtype)))
            source = module.to_mlir()
            self.assertIn(f"0x{zero} : {dtype}", source)
            self.assertIn(f"0x{one} : {dtype}", source)
            self.assertNotIn("overflow", source)

        for dtype, fraction_bits, minimum, maximum in (
            ("f16", 10, -24, 65504.0),
            ("bf16", 7, -133, float.fromhex("0x1.fep127")),
        ):
            self.assertEqual(encode_float(-0.0, dtype), b"\x00\x80")
            self.assertEqual(encode_float(2.0**minimum, dtype), b"\x01\x00")
            self.assertEqual(encode_float(2.0**(minimum - 1), dtype), b"\x00\x00")
            self.assertEqual(round_float(maximum, dtype), maximum)
            with self.assertRaises(ValueError):
                round_float(maximum * 2, dtype)
            midpoint = 1.0 + 2**-(fraction_bits + 1)
            self.assertEqual(round_float(midpoint, dtype), 1.0)
            self.assertEqual(round_float(math.nextafter(midpoint, math.inf), dtype), 1.0 + 2**-fraction_bits)
            for bits in (0x0000, 0x0001, 0x3C00, 0x3F80, 0x7BFF, 0x7C00, 0x7E00, 0x8000, 0xFC00, 0xFFFF):
                encoded = bits.to_bytes(2, "little")
                reference = (struct.unpack("<e", encoded)[0] if dtype == "f16"
                             else struct.unpack("<f", bytes(2) + encoded)[0])
                actual = decode_float(encoded, dtype)
                if math.isnan(reference):
                    self.assertTrue(math.isnan(actual))
                else:
                    self.assertEqual(actual, reference)
                    self.assertEqual(encode_float(actual, dtype), encoded)

        bf16_mid = 2**80 + 2**72
        self.assertEqual(round_float(bf16_mid, "bf16"), float(2**80))
        self.assertEqual(round_float(bf16_mid + 1, "bf16"), float(2**80 + 2**73))

        f32_mid = 2**80 + 2**56
        self.assertEqual(round_f32(f32_mid), float(2**80))
        self.assertEqual(round_f32(f32_mid + 1), float(2**80 + 2**57))
        self.assertEqual(encode_f32(-0.0), bytes.fromhex("00000080"))

    def test_graph_scoping_escaping_and_markers(self):
        self.assertEqual(mlir_string('a"\\\n\x00中'), '"a\\22\\5C\\0A\\00\\E4\\B8\\AD"')
        for bad_sym in ("main) {", "main\n", "主函数", "main.with.dot"):
            with self.assertRaises(ValueError):
                Module().graph(bad_sym, inputs={}, results=[])

        module = Module()
        graph = module.graph("main", inputs={"a": TensorType((None, 4)), "b": TensorType((10, None)),
                                             "ask_v0": ScalarType()}, results=[TensorType((10, 4))])
        merged = graph.add(graph.inputs["a"], graph.inputs["b"], id="sum")
        self.assertEqual(merged.type, TensorType((10, 4)))
        self.assertNotEqual(graph.constant(1).name, graph.inputs["ask_v0"].name)
        marked = graph.mark(merged, mark_id="trace_sum")
        self.assertEqual(marked.value_id, merged.value_id)
        with self.assertRaises(ValueError):
            graph.constant(2, id="sum")
        graph.return_(marked)
        with self.assertRaises(ValueError):
            graph.constant(3)

        was_imported = "triton" in sys.modules
        external_mlir = example_module().to_mlir()
        self.assertIn('compiler = "triton"', external_mlir)
        self.assertEqual("triton" in sys.modules, was_imported)


class ClientProtocolTests(unittest.TestCase):
    def test_cli_driver_protocol_and_error_handling(self):
        def fake_run(command, **kwargs):
            self.assertNotIn("shell", kwargs)
            Path(command[command.index("-o") + 1]).write_text("module {}\n", encoding="utf-8")
            Path(command[command.index("--report-json") + 1]).write_text('{"phase": "prepared"}', encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch("ask.compiler.subprocess.run", side_effect=fake_run):
            result = Compiler("/path with spaces/ask-opt").prepare("module {}")
        self.assertEqual(result.report, {"phase": "prepared"})

        with patch("ask.compiler.subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "bad shape")):
            with self.assertRaises(CompilerError) as caught:
                Compiler("ask-opt").prepare("module {}")
        self.assertEqual(caught.exception.diagnostics, "bad shape")

        with patch("ask.compiler.subprocess.run", return_value=subprocess.CompletedProcess([], -11, "", "crash")):
            with self.assertRaisesRegex(CompilerError, "signal 11"):
                Compiler("ask-opt").prepare("module {}")

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(CompilerError, "not found"):
                Compiler(Path(directory) / "missing-ask-opt").prepare(self_owned_module())


class NativeCompilerAndBindingTests(unittest.TestCase):
    def test_cli_prepare_shape_refinement_and_casts(self):
        compiler = Compiler(ask_opt_path())
        owned = compiler.prepare(self_owned_module())
        self.assertNotIn('"ask.mark"', owned.ir)
        self.assertIn("after_P", json.dumps(owned.report))

        ext = compiler.prepare(example_module())
        self.assertIn('"ask.external"', ext.ir)
        self.assertEqual(len(ext.report["external_requests"]), 1)

        marker_id = '引号"\\\n\x00'
        escaped = self_owned_module().to_mlir().replace('mark_id = "after_P"', 'mark_id = ' + mlir_string(marker_id))
        self.assertEqual(compiler.prepare(escaped).report["markers"][0]["mark_id"], marker_id)

        dtypes = ("bool", "i32", "i64", "f32", "f16", "bf16")
        cast_mod = Module()
        for src in dtypes:
            g = cast_mod.graph("from_" + src, inputs={"x": TensorType((2, 5), src)},
                               results=[TensorType((2, 5), dst) for dst in dtypes])
            g.return_(*(g.cast(g.inputs["x"], dtype=dst) for dst in dtypes))
        prepared = compiler.prepare(cast_mod)
        self.assertEqual(prepared.ir.count('"ask.cast"'), 36)

    @unittest.skipUnless(NATIVE_AVAILABLE, "build ask._native for in-process binding checks")
    def test_in_process_bindings_and_cli_parity(self):
        with patch("ask.compiler.subprocess.run", side_effect=AssertionError("unexpected subprocess")):
            native = Compiler()
            caps = native.capabilities()
            prepared = native.prepare(mixed_f16_module())
            optimized = native.optimize(mixed_f16_module())
            with self.assertRaises(CompilerError) as caught:
                native.prepare("module { unknown }")
        self.assertEqual(caps["python_bindings"], "nanobind")
        self.assertEqual(caps["ask_version"], __version__)
        self.assertEqual(prepared.report["phase"], "prepared")
        self.assertEqual(optimized.report["phase"], "optimized")
        self.assertIn("<ask-python>", caught.exception.diagnostics)

        source = mixed_f16_module().to_mlir()
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(native.prepare, [source] * 8))
        self.assertTrue(all(r.ir == outputs[0].ir and r.report == outputs[0].report for r in outputs))

        if os.environ.get("ASK_OPT"):
            cli = Compiler(os.environ["ASK_OPT"])
            self.assertEqual(native.capabilities(), cli.capabilities())
            self.assertEqual(native.prepare(mixed_f16_module()), cli.prepare(mixed_f16_module()))
            native._check_companion()

        mismatched = dict(caps, build_id="another-build")
        with patch("ask.compiler.subprocess.run",
                   return_value=subprocess.CompletedProcess([], 0, json.dumps(mismatched), "")):
            with self.assertRaisesRegex(CompilerError, "different builds"):
                Compiler()._check_companion()

        from ask import _native
        with patch.object(_native, "capabilities", return_value=json.dumps(dict(caps, ask_version="0.0.0"))):
            with self.assertRaisesRegex(CompilerError, "different versions"):
                Compiler()


if __name__ == "__main__":
    unittest.main()
