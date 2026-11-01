from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from ask import Compiler, CompilerError, Module, ScalarType, TensorType
from ask.ir import mlir_string
from ask.numerics import encode_f32, round_f32, encode_float, decode_float, round_float


def example_module():
    spec = importlib.util.spec_from_file_location("ask_framework_example", ROOT / "examples/framework_frontend.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    return example.build_module()


def self_owned_module():
    module = Module()
    tensor = TensorType((10,))
    graph = module.graph("main", inputs={"x": tensor}, results=[tensor])
    one = graph.splat(graph.constant(1), (10,))
    t = graph.add(graph.inputs["x"], one, id="P")
    t = graph.mark(t, mark_id="after_P")
    two = graph.splat(graph.constant(2), (10,))
    y = graph.mul(t, two, id="Q")
    graph.return_(y)
    return module


class BuilderTests(unittest.TestCase):
    def test_tensor_dimensions_and_dtype(self):
        self.assertEqual(str(TensorType((None, 4))), "tensor<?x4xi32>")
        self.assertEqual(str(TensorType(())), "tensor<i32>")
        for shape in [(-1,), (True,), (2**63,), ("10",)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                TensorType(shape)
        for dtype in ("f32", "f16", "bf16"):
            self.assertEqual(str(ScalarType(dtype)), dtype)
            self.assertEqual(str(TensorType((2, 3), dtype)), f"tensor<2x3x{dtype}>")
        self.assertEqual(str(ScalarType("fp16")), "f16")
        self.assertEqual(TensorType((2,), "fp16"), TensorType((2,), "f16"))
        with self.assertRaises(ValueError):
            ScalarType("f64")

    def test_float_constants_preserve_rounded_bits_and_types(self):
        for dtype, precision, zero, one in [("f32", 23, "80000000", "3F800000"),
                                            ("f16", 10, "8000", "3C00"),
                                            ("bf16", 7, "8000", "3F80")]:
            with self.subTest(dtype=dtype):
                module = Module()
                graph = module.graph("main", inputs={}, results=[ScalarType(dtype)])
                negative_zero = graph.constant(-0.0, dtype=dtype)
                rounded = graph.constant(1.0 + 2.0 ** -(precision + 1), dtype=dtype)
                graph.return_(graph.add(negative_zero, rounded))
                source = module.to_mlir()
                self.assertIn(f"0x{zero} : {dtype}", source)
                self.assertIn(f"0x{one} : {dtype}", source)
                self.assertNotIn("overflow", source)
                other = Module().graph("other", inputs={}, results=[])
                for value in (True, "1", 2 ** 128):
                    with self.subTest(value=value), self.assertRaises(ValueError):
                        other.constant(value, dtype=dtype)
                with self.assertRaises(TypeError):
                    other.add(other.constant(1), other.constant(1.0, dtype=dtype))

    def test_low_precision_codecs_match_ieee_encodings(self):
        for dtype, fraction_bits, minimum, maximum in [("f16", 10, -24, 65504.0),
                                                       ("bf16", 7, -133, float.fromhex("0x1.fep127"))]:
            with self.subTest(dtype=dtype):
                self.assertEqual(encode_float(-0.0, dtype), b"\x00\x80")
                self.assertEqual(encode_float(2.0 ** minimum, dtype), b"\x01\x00")
                self.assertEqual(encode_float(2.0 ** (minimum - 1), dtype), b"\x00\x00")
                self.assertEqual(round_float(maximum, dtype), maximum)
                self.assertTrue(math.isnan(round_float(float("nan"), dtype)))
                self.assertEqual(round_float(float("-inf"), dtype), float("-inf"))
                with self.assertRaises(ValueError):
                    round_float(maximum * 2, dtype)
                midpoint = 1.0 + 2 ** -(fraction_bits + 1)
                self.assertEqual(round_float(midpoint, dtype), 1.0)
                # A conversion through f32 erases this difference. The direct
                # codec must select the upper representable lowp value instead.
                above = math.nextafter(midpoint, math.inf)
                self.assertEqual(round_float(above, dtype), 1.0 + 2 ** -fraction_bits)
                self.assertEqual(round_float(math.nextafter(midpoint, -math.inf), dtype), 1.0)
                for bits in range(65536):
                    encoded = bits.to_bytes(2, "little")
                    reference = (struct.unpack("<e", encoded)[0] if dtype == "f16" else
                                 struct.unpack("<f", bytes(2) + encoded)[0])
                    actual = decode_float(encoded, dtype)
                    if math.isnan(reference):
                        self.assertTrue(math.isnan(actual))
                    else:
                        self.assertEqual(actual, reference)
                        self.assertEqual(encode_float(actual, dtype), encoded)
        midpoint = 2 ** 80 + 2 ** 72
        self.assertEqual(float(midpoint - 1), float(midpoint + 1))
        self.assertEqual(round_float(midpoint, "bf16"), float(2 ** 80))
        self.assertEqual(round_float(midpoint + 1, "bf16"), float(2 ** 80 + 2 ** 73))

    def test_f32_encoding_rounds_large_integers_directly(self):
        midpoint = 2 ** 80 + 2 ** 56
        # All three integers become the same binary64 value, but they round
        # differently to binary32. The even lower significand wins the tie.
        self.assertEqual(float(midpoint - 1), float(midpoint + 1))
        self.assertEqual(round_f32(midpoint - 1), float(2 ** 80))
        self.assertEqual(round_f32(midpoint), float(2 ** 80))
        self.assertEqual(round_f32(midpoint + 1), float(2 ** 80 + 2 ** 57))
        self.assertEqual(round_f32(-midpoint - 1), -float(2 ** 80 + 2 ** 57))
        odd_midpoint = 2 ** 80 + 3 * 2 ** 56
        self.assertEqual(round_f32(odd_midpoint), float(2 ** 80 + 2 ** 58))
        self.assertEqual(encode_f32(-0.0), bytes.fromhex("00000080"))
        self.assertTrue(math.isnan(round_f32(float("nan"))))
        self.assertEqual(round_f32(float("inf")), float("inf"))
        for value in (1e40, 2 ** 128, True, "1.0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                encode_f32(value)
        module = Module()
        graph = module.graph("large", inputs={}, results=[ScalarType("f32")])
        graph.return_(graph.constant(midpoint + 1, dtype="f32"))
        self.assertIn("0x67800001 : f32", module.to_mlir())

    def test_utf8_string_escaping(self):
        self.assertEqual(mlir_string('a"\\\n\x00中'), '"a\\22\\5C\\0A\\00\\E4\\B8\\AD"')
        with self.assertRaises(ValueError):
            mlir_string("\ud800")

    def test_symbols_cannot_inject_ir(self):
        for name in ['main) {', 'main\n', '主函数', 'main.with.dot']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                Module().graph(name, inputs={}, results=[])

    def test_explicit_splat_and_wrapping_integer_ops(self):
        module = self_owned_module()
        source = module.to_mlir()
        self.assertIn('shape = array<i64: 10>', source)
        self.assertIn('overflow = "wrap"', source)
        self.assertIn('ask.schema_version = 1 : i64', source)
        graph = Module().graph("main", inputs={"x": TensorType((10,))}, results=[])
        with self.assertRaises(TypeError):
            graph.add(graph.inputs["x"], graph.constant(1))
        with self.assertRaises(TypeError):
            graph.add(graph.inputs["x"], graph.splat(graph.constant(1), (9,)))

    def test_marker_preserves_logical_identity_and_has_own_ssa(self):
        graph = Module().graph("main", inputs={"x": TensorType((10,))}, results=[])
        value = graph.inputs["x"]
        marked = graph.mark(value, mark_id="trace_x")
        self.assertEqual(marked.value_id, value.value_id)
        self.assertEqual(marked.type, value.type)
        self.assertNotEqual(marked.name, value.name)
        with self.assertRaises(ValueError):
            graph.mark(value, mark_id="input_x", role="input")

    def test_graph_scope_unique_ids_and_terminator(self):
        module = Module()
        graph = module.graph("main", inputs={}, results=[ScalarType()])
        other = module.graph("other", inputs={}, results=[])
        value = graph.constant(1, id="one")
        with self.assertRaises(ValueError):
            graph.constant(2, id="one")
        with self.assertRaises(ValueError):
            other.add(value, value)
        graph.return_(value)
        with self.assertRaises(ValueError):
            graph.constant(2)
        with self.assertRaises(ValueError):
            module.to_mlir()
        other.return_()
        self.assertIn("func.func @other", module.to_mlir())

    def test_dsl_is_data_and_external_shapes_follow_semantics(self):
        was_imported = "triton" in sys.modules
        source = example_module().to_mlir()
        self.assertIn('compiler = "triton"', source)
        self.assertIn('semantics = @scale2_i32_v1', source)
        self.assertIn('source_bindings = {t = "input:0", u = "output:0"}', source)
        self.assertIn('constants = {BLOCK = 128 : i64, N = 10 : i64}', source)
        self.assertEqual("triton" in sys.modules, was_imported)

    def test_dynamic_static_inputs_merge_known_dimensions(self):
        graph = Module().graph("main", inputs={"a": TensorType((None, 4)),
                                               "b": TensorType((10, None))}, results=[])
        value = graph.add(graph.inputs["a"], graph.inputs["b"])
        self.assertEqual(value.type, TensorType((10, 4)))

    def test_no_silent_return_type_cast(self):
        graph = Module().graph("main", inputs={"x": TensorType((10,))},
                               results=[TensorType((None,))])
        with self.assertRaises(TypeError):
            graph.return_(graph.inputs["x"])

    def test_ssa_names_do_not_collide_with_arguments(self):
        graph = Module().graph("main", inputs={"ask_v0": ScalarType()}, results=[])
        self.assertNotEqual(graph.constant(1).name, graph.inputs["ask_v0"].name)


class ClientProtocolTests(unittest.TestCase):
    def test_prepare_uses_native_driver_and_independent_files(self):
        calls = []
        def run(command, **kwargs):
            calls.append((command, kwargs))
            self.assertNotIn("shell", kwargs)
            self.assertEqual(Path(command[1]).read_text(), "module {}")
            Path(command[command.index("-o") + 1]).write_text("module {}\n")
            Path(command[command.index("--report-json") + 1]).write_text('{"phase": "prepared"}')
            return subprocess.CompletedProcess(command, 0, "", "")
        with patch("ask.compiler.subprocess.run", side_effect=run):
            result = Compiler("/path with spaces/ask-opt").prepare("module {}")
        self.assertEqual(result.ir, "module {}\n")
        self.assertEqual(result.report, {"phase": "prepared"})
        self.assertEqual(calls[0][0][0], "/path with spaces/ask-opt")
        self.assertFalse(Path(calls[0][0][1]).exists())

    def test_native_diagnostics_are_retained(self):
        with patch("ask.compiler.subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "bad shape")):
            with self.assertRaises(CompilerError) as caught:
                Compiler().prepare("module {}")
        self.assertEqual(caught.exception.returncode, 1)
        self.assertEqual(caught.exception.diagnostics, "bad shape")

    def test_native_crash_is_reported_as_a_signal(self):
        with patch("ask.compiler.subprocess.run", return_value=subprocess.CompletedProcess([], -11, "", "stack trace")):
            with self.assertRaisesRegex(CompilerError, "signal 11") as caught:
                Compiler().prepare("module {}")
        self.assertEqual(caught.exception.returncode, -11)

    def test_missing_compiler_is_not_emulated(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(CompilerError, "not found"):
                Compiler(Path(directory) / "missing-ask-opt").prepare(self_owned_module())

    def test_capabilities_requires_a_json_object(self):
        with patch("ask.compiler.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "[]", "")):
            with self.assertRaises(CompilerError):
                Compiler().capabilities()

    def test_success_exit_without_artifacts_is_an_error(self):
        with patch("ask.compiler.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "", "")):
            with self.assertRaises(CompilerError):
                Compiler().prepare("module {}")


@unittest.skipUnless(os.environ.get("ASK_OPT"), "set ASK_OPT to run real native compiler tests")
class NativeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.compiler = Compiler(os.environ["ASK_OPT"])

    def test_self_owned_graph_and_marker_extraction(self):
        result = self.compiler.prepare(self_owned_module())
        self.assertIn("func.func @main", result.ir)
        self.assertNotIn('"ask.mark"', result.ir)
        self.assertIn("shape_snapshot", result.report)
        self.assertIn("after_P", json.dumps(result.report))
        self.assertIn("P.result0", json.dumps(result.report))

    def test_external_example_prepares_without_importing_triton(self):
        was_imported = "triton" in sys.modules
        result = self.compiler.prepare(example_module())
        self.assertIn('"ask.external"', result.ir)
        self.assertIn("before_F", json.dumps(result.report))
        self.assertIn("external_requests", result.report)
        self.assertEqual("triton" in sys.modules, was_imported)

    def test_utf8_and_control_characters_survive_native_parser(self):
        module = self_owned_module()
        marker_id = '引号"\\\n\x00'
        source = module.to_mlir().replace('mark_id = "after_P"',
                                         'mark_id = ' + mlir_string(marker_id))
        result = self.compiler.prepare(source)
        self.assertEqual(result.report["markers"][0]["mark_id"], marker_id)

    def test_external_dynamic_result_is_refined_by_native_prepare(self):
        module = Module()
        tensor = TensorType((10,))
        dynamic = TensorType((None,))
        semantics = module.semantics("scale", inputs=[dynamic], results=[dynamic])
        implementation = module.implementation(
            "scale_impl", compiler="triton", source="# not executed during prepare\n",
            entry="scale", signature={"x": "*i32", "y": "*i32"},
            source_bindings={"x": "input:0", "y": "output:0"}, target={"backend": "cuda"},
        )
        graph = module.graph("main", inputs={"x": tensor}, results=[dynamic])
        result = graph.external(graph.inputs["x"], semantics=semantics,
                                implementation=implementation, result_type=dynamic, id="F")
        graph.return_(result)
        prepared = self.compiler.prepare(module)
        external_line = next(line for line in prepared.ir.splitlines() if '"ask.external"' in line)
        self.assertIn("-> tensor<10xi32>", external_line)
        self.assertNotIn("-> tensor<?xi32>", external_line)

    def test_float_external_shape_refinement_and_signature_validation(self):
        for dtype, signature in [("f32", "*fp32"), ("f16", "*fp16"), ("bf16", "*bf16")]:
            with self.subTest(dtype=dtype):
                module = Module()
                tensor, dynamic = TensorType((2, 5), dtype), TensorType((None, None), dtype)
                semantics = module.semantics("copy", inputs=[dynamic], results=[dynamic])
                implementation = module.implementation(
                    "copy_impl", compiler="triton", source="# DSL is not executed during prepare\n",
                    entry="copy", signature={"x": signature, "y": signature},
                    source_bindings={"x": "input:0", "y": "output:0"}, target={"backend": "cuda"})
                graph = module.graph("main", inputs={"x": tensor}, results=[dynamic])
                result = graph.external(graph.inputs["x"], semantics=semantics,
                                        implementation=implementation, result_type=dynamic, id="F")
                graph.return_(result)
                prepared = self.compiler.prepare(module)
                shapes = prepared.report["shape_snapshot"]["shape_bindings"]
                self.assertTrue(shapes)
                self.assertTrue(all(item["element_type"] == dtype and item["shape"] == [2, 5]
                                    for item in shapes))
                self.assertIn(f"-> tensor<2x5x{dtype}>", prepared.ir)
                with self.assertRaises(CompilerError):
                    self.compiler.prepare(module.to_mlir().replace(f'"{signature}"', '"*i32"'))


    def test_real_diagnostics_for_invalid_ir(self):
        with self.assertRaises(CompilerError) as caught:
            self.compiler.prepare("module { unknown }")
        self.assertGreater(caught.exception.returncode, 0)
        self.assertIn("error:", caught.exception.diagnostics)

    def test_capabilities_are_read_from_native_binary(self):
        self.assertIsInstance(self.compiler.capabilities(), dict)


if __name__ == "__main__":
    unittest.main()
