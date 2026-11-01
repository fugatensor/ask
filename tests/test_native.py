"""Exercise the native verifier, lowering and resulting C ABI as a client."""
import ctypes
import json
import math
from pathlib import Path
import shutil
import subprocess
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = str(Path(sys.argv.pop(1)).resolve())
sys.path.insert(0, str(ROOT / "python"))
from ask import Module, ScalarType, TensorType
from ask.numerics import encode_float, decode_float
sys.path.insert(0, str(ROOT / "tests"))
from support.pointwise import (build_module, input_values, expected_values, numerically_equal, DTYPES,
                               cast_reference, round_reference, FLOAT_FORMATS)


def run(*args, source=None):
    return subprocess.run([DRIVER, *map(str, args)], input=source, text=True,
                          capture_output=True, timeout=60)


class NativeTests(unittest.TestCase):
    def test_positive_fixtures(self):
        for name in ("pipeline", "refine-and-erase"):
            with self.subTest(name=name):
                result = run(ROOT / "tests/ir" / (name + ".mlir"), "--prepare")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn('"ask.mark"', result.stdout)
                repeated = run("-", "--prepare", source=result.stdout)
                self.assertEqual(repeated.returncode, 0, repeated.stderr)
                self.assertEqual(repeated.stdout, result.stdout)

    def test_negative_fixtures_have_diagnostics(self):
        for path in sorted((ROOT / "tests/ir").glob("*.mlir")):
            if path.stem in ("pipeline", "refine-and-erase"):
                continue
            with self.subTest(name=path.name):
                result = run(path, "--prepare")
                self.assertGreater(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("error", result.stderr.lower())

    def test_native_llvm_import_preserves_inline_ptx(self):
        source = '''target triple = "nvptx64-nvidia-cuda"
define void @f() {
  call void asm sideeffect "bar.sync 0;", ""()
  ret void
}
'''
        result = run("-", "--check-llir", source=source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('asm sideeffect "bar.sync 0;"', result.stdout)
        invalid = run("-", "--check-llir", source="define void @f() { ret i32 1 }")
        self.assertGreater(invalid.returncode, 0)
        self.assertIn("reader", invalid.stderr)

    def test_external_request_preserves_boolean_options(self):
        source = (ROOT / "tests/ir/pipeline.mlir").read_text().replace(
            'options = {num_warps = 4 : i64}',
            'options = {num_warps = 4 : i64, enable_fp_fusion = true}')
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            result = run("-", "--prepare", "--report-json", report, source=source)
            self.assertEqual(result.returncode, 0, result.stderr)
            options = json.loads(report.read_text())["external_requests"][0]["fixed_options"]
            self.assertIs(options["enable_fp_fusion"], True)

    def test_external_is_not_silently_lowered_as_semantic_identity(self):
        result = run(ROOT / "tests/ir/pipeline.mlir", "--emit-llvm")
        self.assertGreater(result.returncode, 0)
        self.assertIn("external", result.stderr)

    def test_disabled_exporter_rejected_before_artifacts(self):
        caps = json.loads(run("--capabilities").stdout)
        if "triton" in caps["third_party_compilers"]:
            self.skipTest("this build explicitly enables Triton")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "exports"
            result = run(ROOT / "tests/ir/pipeline.mlir", "--export-external",
                         "--export-dir", destination)
            self.assertGreater(result.returncode, 0)
            self.assertIn("not enabled", result.stderr)
            self.assertFalse(destination.exists())

    def test_host_objects_execute_correctly(self):
        if not json.loads(run("--capabilities").stdout)["host_codegen"]:
            self.skipTest("host backend disabled")
        cc = shutil.which("cc")
        self.assertIsNotNone(cc, "C compiler is required to link the emitted object")
        module = Module()
        shapes = [(10,), (2, 3), (), (0,)]
        for index, shape in enumerate(shapes):
            tensor = TensorType(shape)
            graph = module.graph(f"calculate{index}", inputs={"x": tensor, "s": ScalarType()},
                                 results=[tensor, ScalarType(), tensor])
            x, scalar = graph.inputs["x"], graph.inputs["s"]
            biased = graph.add(x, graph.splat(scalar, shape))
            marked = graph.mark(biased, mark_id=f"trace{index}")
            doubled = graph.mul(marked, graph.splat(graph.constant(2), shape))
            scalar_result = graph.add(scalar, graph.constant(1))
            graph.return_(doubled, scalar_result, x)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            obj, shared, report = path / "graph.o", path / "graph.so", path / "abi.json"
            result = run("-", "--emit-object", "-o", obj, "--report-json", report,
                         source=module.to_mlir())
            self.assertEqual(result.returncode, 0, result.stderr)
            linked = subprocess.run([cc, "-shared", str(obj), "-o", str(shared)],
                                    text=True, capture_output=True)
            self.assertEqual(linked.returncode, 0, linked.stderr)
            metadata = json.loads(report.read_text())
            self.assertEqual(metadata["phase"], "host_object")
            self.assertEqual(len(metadata["abi"]["functions"]), len(shapes))
            library = ctypes.CDLL(str(shared))
            pointer = ctypes.POINTER(ctypes.c_int32)
            for index, shape in enumerate(shapes):
                count = math.prod(shape)
                data = [2147483647, -2147483648, -7, 0, 1, 7, 42, 1024, -1024, 9][:count]
                x = (ctypes.c_int32 * max(count, 1))(*data)
                out = (ctypes.c_int32 * max(count, 1))(*([123] * max(count, 1)))
                copy = (ctypes.c_int32 * max(count, 1))(*([456] * max(count, 1)))
                scalar = ctypes.c_int32()
                function = getattr(library, f"calculate{index}")
                function.argtypes = [pointer, ctypes.c_int32, pointer, pointer, pointer]
                function.restype = None
                function(x, 2147483647, out, ctypes.byref(scalar), copy)
                self.assertEqual(list(out)[:count], [ctypes.c_int32((v + 2147483647) * 2).value for v in data])
                self.assertEqual(list(copy)[:count], data)
                self.assertEqual(scalar.value, -2147483648)
                if count == 0:
                    self.assertEqual(out[0], 123)
                    self.assertEqual(copy[0], 456)

    def test_float_host_abi_preserves_rounding_and_special_values(self):
        if not json.loads(run("--capabilities").stdout)["host_codegen"]:
            self.skipTest("host backend disabled")
        cc = shutil.which("cc")
        self.assertIsNotNone(cc)
        module = Module()
        tensor, scalar_type = TensorType((2, 4), "f32"), ScalarType("f32")
        graph = module.graph("calculate", inputs={"x": tensor, "a": scalar_type},
                             results=[tensor, tensor, scalar_type])
        product = graph.mul(graph.inputs["x"], graph.splat(graph.inputs["a"], (2, 4)))
        adjusted = graph.add(product, graph.splat(graph.constant(-1.0, dtype="f32"), (2, 4)))
        scalar = graph.add(graph.inputs["a"], graph.constant(-1.0, dtype="f32"))
        graph.return_(product, adjusted, scalar)
        source = module.to_mlir()
        llvm_result = run("-", "--emit-llvm", source=source)
        self.assertEqual(llvm_result.returncode, 0, llvm_result.stderr)
        self.assertIn("fadd float", llvm_result.stdout)
        self.assertIn("fmul float", llvm_result.stdout)
        self.assertNotIn("fadd fast", llvm_result.stdout)
        self.assertNotIn("fmul contract", llvm_result.stdout)
        data = [1.0 + 2 ** -23, -0.0, float("inf"), float("-inf"),
                float("nan"), 2 ** -149, -2 ** -149, 0.0]
        coefficient = 1.0 - 2 ** -23
        expected_products = [ctypes.c_float(value * coefficient).value for value in data]
        expected_outputs = [ctypes.c_float(value - 1.0).value for value in expected_products]
        self.assertEqual(expected_outputs[0], 0.0)  # Contracted FMA would differ.
        with tempfile.TemporaryDirectory() as directory:
            for optimized in (False, True):
                path = Path(directory)
                obj = path / f"float{optimized}.o"
                shared = path / f"float{optimized}.so"
                report = path / f"float{optimized}.json"
                args = ["-", "--emit-object", "-o", obj, "--report-json", report]
                if optimized:
                    args.append("--optimize")
                result = run(*args, source=source)
                self.assertEqual(result.returncode, 0, result.stderr)
                metadata = json.loads(report.read_text())
                abi = metadata["abi"]["functions"][0]
                self.assertTrue(all(item["dtype"] == "f32" for item in abi["inputs"] + abi["outputs"]))
                self.assertEqual(abi["lowering"]["loop_count"], 2 if optimized else 3)
                linked = subprocess.run([cc, "-shared", str(obj), "-o", str(shared)],
                                        text=True, capture_output=True)
                self.assertEqual(linked.returncode, 0, linked.stderr)
                function = ctypes.CDLL(str(shared)).calculate
                pointer = ctypes.POINTER(ctypes.c_float)
                function.argtypes = [pointer, ctypes.c_float, pointer, pointer, pointer]
                function.restype = None
                storage = ctypes.c_float * 8
                x, product_out, result_out, scalar_out = storage(*data), storage(), storage(), ctypes.c_float()
                function(x, coefficient, product_out, result_out, ctypes.byref(scalar_out))
                for actual, expected in zip(list(product_out) + list(result_out),
                                            expected_products + expected_outputs):
                    if math.isnan(expected):
                        self.assertTrue(math.isnan(actual))
                    else:
                        self.assertEqual(struct.pack("f", actual), struct.pack("f", expected))
                self.assertEqual(scalar_out.value, -2 ** -23)

    def test_shared_pointwise_graph_across_host_precisions(self):
        if not json.loads(run("--capabilities").stdout)["host_codegen"]:
            self.skipTest("host backend disabled")
        cc = shutil.which("cc")
        self.assertIsNotNone(cc)
        shape, count = (2, 8), 16
        # This builder is also used by the GPU precision matrix. The operation
        # graph is identical; dtype controls only types and numerical encoding.
        supports_bf16 = json.loads(run("--capabilities").stdout)["host_bf16_hardware_supported"]
        for dtype, compute in [(dtype, None) for dtype in DTYPES] + [(dtype, "f32") for dtype in ("f16", "bf16")]:
            module = build_module(shape=shape, dtype=dtype, external=False, compute_dtype=compute)
            prepared = run("-", "--prepare", source=module.to_mlir())
            self.assertEqual(prepared.returncode, 0, prepared.stderr)
            with self.subTest(dtype=dtype, compute=compute), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                if dtype == "bf16" and not supports_bf16:
                    for mode in ("--emit-llvm", "--emit-object"):
                        output = path / "unsupported"
                        rejected = run("-", mode, "-o", output, source=module.to_mlir())
                        self.assertNotEqual(rejected.returncode, 0)
                        self.assertIn("bf16 requires hardware support", rejected.stderr)
                        self.assertFalse(output.exists())
                    continue
                values = input_values(count, dtype)
                expected = expected_values(values, 1, 2, dtype=dtype, external=False, compute_dtype=compute)
                storage_type = {"bool": ctypes.c_uint8, "i32": ctypes.c_int32, "i64": ctypes.c_int64,
                                "f32": ctypes.c_float, "f16": ctypes.c_uint16, "bf16": ctypes.c_uint16}[dtype]
                storage = storage_type * count
                encoded = ([int.from_bytes(encode_float(value, dtype), "little") for value in values]
                           if dtype in ("f16", "bf16") else values)
                for optimize in (False, True):
                    obj, library_path, report = path / f"{optimize}.o", path / f"{optimize}.so", path / f"{optimize}.json"
                    arguments = ["-", "--emit-object", "-o", obj, "--report-json", report]
                    if optimize:
                        arguments.append("--optimize")
                    emitted = run(*arguments, source=module.to_mlir())
                    self.assertEqual(emitted.returncode, 0, emitted.stderr)
                    linked = subprocess.run([cc, "-shared", str(obj), "-o", str(library_path)],
                                            text=True, capture_output=True)
                    self.assertEqual(linked.returncode, 0, linked.stderr)
                    library = ctypes.CDLL(str(library_path))
                    function = library.pipeline
                    function.argtypes = [ctypes.POINTER(storage_type)] * 2
                    function.restype = None
                    inputs, outputs = storage(*encoded), storage()
                    function(inputs, outputs)
                    actual = ([decode_float(value.to_bytes(2, "little"), dtype) for value in outputs]
                              if dtype in ("f16", "bf16") else list(outputs))
                    self.assertTrue(numerically_equal(actual, expected, dtype), (dtype, actual, expected))
                    abi = json.loads(report.read_text())["abi"]["functions"][0]
                    width = ctypes.sizeof(storage_type)
                    self.assertEqual(abi["minimum_alignment"], width)
                    self.assertTrue(all(port["dtype"] == dtype and port["element_bytes"] == width
                                        and port["minimum_alignment"] == width
                                        for port in abi["inputs"] + abi["outputs"]))

    def test_cast_matrix_executes_and_mixed_outputs_share_a_loop(self):
        caps = json.loads(run("--capabilities").stdout)
        if not caps["host_codegen"]:
            self.skipTest("host backend disabled")
        dtypes = [dtype for dtype in DTYPES if dtype != "bf16" or caps["host_bf16_hardware_supported"]]
        storage_types = {"bool": ctypes.c_uint8, "i32": ctypes.c_int32, "i64": ctypes.c_int64,
                         "f32": ctypes.c_float, "f16": ctypes.c_uint16, "bf16": ctypes.c_uint16}
        count = 32
        module = Module()
        data = {}
        for source in dtypes:
            values = input_values(count, source)
            if source == "i64":
                values[-6:] = [(2**40 + 2**32 + delta) * sign for delta in (-1, 0, 1) for sign in (-1, 1)]
            if source == "f32":
                values[-10:] = [2**-134, 3 * 2**-134, 2**-133, 2**-24, 2**-25,
                                65520.0, 2**31, -(2**31), 2**63, -(2**63)]
            data[source] = [round_reference(value, source) for value in values]
            for target in dtypes:
                graph = module.graph(f"cast_{source}_{target}", inputs={"x": TensorType((count,), source)},
                                     results=[TensorType((count,), target)])
                graph.return_(graph.cast(graph.inputs["x"], dtype=target))
        # Return three dtypes from one same-shape chain. The f32 root is used
        # by both narrower outputs, so one loop must share its computation.
        tensor = TensorType((count,), "i64")
        mixed = module.graph("mixed", inputs={"x": tensor},
                             results=[TensorType((count,), dtype) for dtype in ("f32", "f16", "bool")])
        widened = mixed.cast(mixed.inputs["x"], dtype="f32")
        narrowed = mixed.cast(widened, dtype="f16")
        truth = mixed.cast(narrowed, dtype="bool")
        mixed.return_(widened, narrowed, truth)
        # Bool scalar input and tensor input both use one byte at the C ABI.
        boolean = module.graph("bool_abi", inputs={"x": TensorType((4,), "bool"), "s": ScalarType("bool")},
                               results=[TensorType((4,), "bool"), ScalarType("bool")])
        boolean.return_(boolean.inputs["x"], boolean.inputs["s"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for optimize in (False, True):
                obj, shared, report = (path / f"cast-{optimize}.{extension}" for extension in ("o", "so", "json"))
                args = ["-", "--emit-object", "-o", obj, "--report-json", report]
                if optimize:
                    args.append("--optimize")
                compiled = run(*args, source=module.to_mlir())
                self.assertEqual(compiled.returncode, 0, compiled.stderr)
                linked = subprocess.run(["cc", "-shared", str(obj), "-o", str(shared)], capture_output=True, text=True)
                self.assertEqual(linked.returncode, 0, linked.stderr)
                library = ctypes.CDLL(str(shared))
                for source in dtypes:
                    encoded = [int.from_bytes(encode_float(value, source), "little") for value in data[source]] if source in ("f16", "bf16") else data[source]
                    inputs = (storage_types[source] * count)(*encoded)
                    for target in dtypes:
                        with self.subTest(source=source, target=target, optimized=optimize):
                            outputs = (storage_types[target] * count)()
                            function = getattr(library, f"cast_{source}_{target}")
                            function.argtypes = [ctypes.POINTER(storage_types[source]), ctypes.POINTER(storage_types[target])]
                            function.restype = None
                            function(inputs, outputs)
                            actual = [decode_float(value.to_bytes(2, "little"), target) for value in outputs] if target in ("f16", "bf16") else list(outputs)
                            expected = [cast_reference(value, source, target) for value in data[source]]
                            self.assertTrue(numerically_equal(actual, expected, target), (source, target, actual, expected))
                inputs = (ctypes.c_int64 * count)(*data["i64"])
                floats, halves, bools = (ctypes.c_float * count)(), (ctypes.c_uint16 * count)(), (ctypes.c_uint8 * count)()
                library.mixed(inputs, floats, halves, bools)
                expected_floats = [cast_reference(value, "i64", "f32") for value in data["i64"]]
                expected_halves = [cast_reference(value, "f32", "f16") for value in expected_floats]
                self.assertTrue(numerically_equal(list(floats), expected_floats, "f32"))
                self.assertTrue(numerically_equal([decode_float(value.to_bytes(2, "little"), "f16") for value in halves], expected_halves, "f16"))
                self.assertEqual(list(bools), [bool(value) for value in expected_halves])
                byte_input, byte_output, scalar = (ctypes.c_uint8 * 4)(0, 1, 2, 255), (ctypes.c_uint8 * 4)(), ctypes.c_uint8()
                library.bool_abi.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint8, ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8)]
                library.bool_abi(byte_input, 255, byte_output, ctypes.byref(scalar))
                self.assertEqual(list(byte_output), [0, 1, 1, 1])
                self.assertEqual(scalar.value, 1)
                abi = json.loads(report.read_text())["abi"]
                self.assertNotEqual(abi["target_cpu"], "generic")
                metadata = next(item for item in abi["functions"] if item["name"] == "mixed")
                self.assertEqual(metadata["lowering"]["loop_count"], 1 if optimize else 3)
                metadata = next(item for item in abi["functions"] if item["name"] == "bool_abi")
                self.assertTrue(all(item["dtype"] == "bool" and item["element_bytes"] == 1
                                    for item in metadata["inputs"] + metadata["outputs"]))

    def test_float_type_mismatch_and_integer_only_options_are_rejected(self):
        source = '''module {
  func.func @f(%x: f32, %y: f32) -> f32 {
    %z = "ask.add"(%x, %y) : (f32, f32) -> f32
    return %z : f32
  }
}'''
        for attribute in ('overflow = "wrap"', 'fastmath = "fast"', 'fastmathFlags = "contract"'):
            invalid = source.replace('"ask.add"(%x, %y)', f'"ask.add"(%x, %y) {{{attribute}}}')
            with self.subTest(attribute=attribute):
                result = run("-", "--prepare", source=invalid)
                self.assertGreater(result.returncode, 0)
        splat = '''module {
  func.func @f(%x: f32) -> tensor<2xi32> {
    %z = "ask.splat"(%x) {shape = array<i64: 2>} : (f32) -> tensor<2xi32>
    return %z : tensor<2xi32>
  }
}'''
        result = run("-", "--prepare", source=splat)
        self.assertGreater(result.returncode, 0)
        self.assertIn("element types must match", result.stderr)


if __name__ == "__main__":
    unittest.main()
