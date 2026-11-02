"""Physical CUDA tests for direct external PTX/cubin and mixed execution plans."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "python", ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ask import Compiler, CompilerEnvironment, CUDAProgram, ExecutionSession, Module, TensorType
from ask.cuda import CUDAError, install_external
from ask.gpu import CUDAKernel
from ask.plan import validate_plan
from support.host_runner import ask_opt_path, llc_path, open_cuda_session, ptxas_path
from test_cuda_compilation import make_external_artifact

PTX = '''.version 8.0
.target sm_80
.address_size 64
.visible .entry scale2(.param .u64 input_ptr, .param .u64 output_ptr) {
    .reg .pred %p;
    .reg .b32 %i, %v, %r;
    .reg .b64 %x, %y, %offset, %xp, %yp;
    ld.param.u64 %x, [input_ptr];
    ld.param.u64 %y, [output_ptr];
    mov.u32 %i, %tid.x;
    setp.ge.u32 %p, %i, 10;
    @%p bra done;
    mul.wide.u32 %offset, %i, 4;
    add.u64 %xp, %x, %offset;
    add.u64 %yp, %y, %offset;
    ld.global.u32 %v, [%xp];
    add.u32 %r, %v, %v;
    st.global.u32 [%yp], %r;
done:
    ret;
}
'''

# This fixture models a compiler that already emits an executable. It is not
# registered in ASK and never imports a producer package or any LLVM machinery.
WORKER = '''import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--probe', action='store_true')
p.add_argument('--request')
p.add_argument('--output')
a = p.parse_args()
producer = {'name': 'fixture', 'version': '1'}
if a.probe:
    print(json.dumps({'producer': producer, 'adapter_version': '1'}))
else:
    request = json.loads(Path(a.request).read_text())
    code = json.loads(request['source'])
    artifact = {'protocol_version': 2, **code, 'entry': 'scale2',
        'target': request['target'], 'producer': producer,
        'launch': {'grid': request['grid'], 'block': [128, 1, 1], 'shared_bytes': 0},
        'arguments': ['input:0', 'output:0'], 'required_element_types': ['i32']}
    Path(a.output).write_text(json.dumps(artifact))
'''


def external_graph(arch, source, *, branches=False):
    module = Module()
    tensor = TensorType((10,), "i32")
    semantic = module.semantics("operator", inputs=[tensor], results=[tensor])
    implementation = module.implementation(
        "implementation", compiler="fixture", source=source, entry="scale2",
        signature={"x": "*i32", "out": "*i32"}, source_bindings={"x": "input:0", "out": "output:0"},
        constants={}, options={}, target={"backend": "cuda", "arch": arch, "warp_size": 32}, grid=(1, 1, 1))
    graph = module.graph("main", inputs={"x": tensor}, results=[tensor] * (4 if branches else 1))
    x = graph.inputs["x"]
    one = graph.splat(graph.constant(1), (10,))
    first = graph.external(graph.add(x, one), semantics=semantic, implementation=implementation, id="first")
    if branches:
        second = graph.external(x, semantics=semantic, implementation=implementation, id="second")
        final = graph.add(first, second)
        graph.return_(first, final, final, x)
    else:
        three = graph.splat(graph.constant(3), (10,))
        graph.return_(graph.add(first, three))
    return module


class CUDAExecutionPlanHardwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open_cuda_session() as session:
            major, minor = session.capability
            cls.arch = 10 * major + minor
        if cls.arch < 80:
            raise unittest.SkipTest("external fixture PTX requires sm80 or newer")
        cls.compiler = Compiler(ask_opt_path(), timeout=180)

    def payload(self, format_, root):
        if format_ == "ptx":
            return PTX
        source, binary = root / "producer.ptx", root / "producer.cubin"
        source.write_text(PTX)
        result = subprocess.run([ptxas_path(), f"--gpu-name=sm_{self.arch}", str(source), "-o", str(binary)],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        return base64.b64encode(binary.read_bytes()).decode()

    def test_direct_ptx_and_cubin_without_ask_codegen(self):
        for format_ in ("ptx", "cubin"):
            with self.subTest(format=format_), tempfile.TemporaryDirectory() as directory, ExecutionSession() as session:
                root = Path(directory)
                payload = self.payload(format_, root)
                artifact = make_external_artifact(format_, shape=(10,), payload=payload, arch=self.arch)
                with patch("ask.cuda.subprocess.run", side_effect=AssertionError("external execution invoked a compiler")):
                    codegen = install_external(artifact, root / "installed")
                    kernel = CUDAKernel(artifact, codegen, {}, {})
                    x, y = session.from_host(list(range(10)), dtype="i32"), session.tensor((10,), "i32")
                    before = session.stats["launches"]
                    executable = kernel.prepare(session)
                    executable.submit(x, y).wait()
                    self.assertEqual(y.to_host(), [2 * n for n in range(10)])
                    self.assertEqual(session.stats["launches"] - before, 1)
                    hits = session.stats["module_hits"]
                    kernel.prepare(session)
                    self.assertEqual(session.stats["module_hits"], hits + 1)
                    self.assertEqual(kernel.run(list(range(10)))["output"], [2 * n for n in range(10)])

    def test_external_only_worker_compile_needs_no_ask_codegen(self):
        with tempfile.TemporaryDirectory() as directory, ExecutionSession() as session:
            root = Path(directory)
            worker = root / "worker.py"
            worker.write_text(WORKER)
            module = Module()
            tensor = TensorType((10,), "i32")
            semantic = module.semantics("operator", inputs=[tensor], results=[tensor])
            implementation = module.implementation(
                "implementation", compiler="fixture", source=json.dumps({"format": "ptx", "payload": PTX}),
                entry="scale2", signature={"x": "*i32", "out": "*i32"},
                source_bindings={"x": "input:0", "out": "output:0"}, constants={}, options={},
                target={"backend": "cuda", "arch": self.arch, "warp_size": 32}, grid=(1, 1, 1))
            graph = module.graph("main", inputs={"x": tensor}, results=[tensor])
            graph.return_(graph.external(graph.inputs["x"], semantics=semantic, implementation=implementation))
            # This public compilation path requires neither local compiler tool:
            # only the worker and the already-compiled PTX are supplied.
            with patch("ask.gpu._tool", side_effect=AssertionError("unexpected ASK compiler tool")):
                kernel = self.compiler.compile_cuda(module, cache_dir=root / "cache", output_dir=root / "output",
                    compilers={"fixture": CompilerEnvironment(python=sys.executable, worker=worker)})
            self.assertIsInstance(kernel, CUDAKernel)
            x, y = session.from_host(list(range(10)), dtype="i32"), session.tensor((10,), "i32")
            kernel.prepare(session).submit(x, y).wait()
            self.assertEqual(y.to_host(), [2 * value for value in range(10)])

    def test_mixed_scalar_abi_preserves_bits_and_checks_layout(self):
        # Independent PTX observes the exact scalar encodings. Parameter widths
        # deliberately force padding before both scalar and pointer arguments.
        samples = [("i1", True, 1), ("i8", -5, 251), ("i16", -1234, 64302),
                   ("i32", -7, 4294967289), ("i64", -9, -9), ("u8", 255, 255),
                   ("u16", 65535, 65535), ("u32", 2**32 - 1, 2**32 - 1),
                   ("u64", 2**64 - 1, -1), ("fp16", 1.5, 0x3e00),
                   ("bf16", -2.5, 0xc020), ("fp32", 1.25, 0x3fa00000),
                   ("fp64", -0.0, -(2**63))]
        widths = [1, 1, 2, 4, 8, 1, 2, 4, 8, 2, 2, 4, 8]
        parameters = [".param .u64 input_ptr"]
        arguments = [{"argument": 0, "kind": "pointer", "binding": "input:0"}]
        output_index = 0
        body = ["ld.param.u64 %out, [output_ptr];"]
        for index, ((dtype, value, _), width) in enumerate(zip(samples, widths)):
            if index == 2:
                output_index = len(arguments)
                parameters.append(".param .u64 output_ptr")
                arguments.append({"argument": output_index, "kind": "pointer", "binding": "output:0"})
            parameters.append(f".param .u{width * 8} scalar{index}")
            arguments.append({"argument": len(arguments), "kind": "scalar", "type": dtype, "value": value})
            if width == 8:
                body.append(f"ld.param.u64 %wide, [scalar{index}];")
            else:
                body.extend([f"ld.param.u{width * 8} %small, [scalar{index}];", "cvt.u64.u32 %wide, %small;"])
            body.append(f"st.global.u64 [%out+{index * 8}], %wide;")
        payload = (".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry scale2(" +
                   ", ".join(parameters) + ") {\n.reg .b32 %small;\n.reg .b64 %wide, %out;\n" +
                   "\n".join(body) + "\nret;\n}\n")
        with tempfile.TemporaryDirectory() as directory, ExecutionSession() as session:
            artifact = make_external_artifact(payload=payload, shape=(len(samples),), arch=self.arch)
            artifact["invocation"].update(arguments=arguments, block=[1, 1, 1])
            artifact["port_map"][1]["argument"] = output_index
            for port in artifact["port_map"]:
                port["element_type"] = "i64"
            artifact["required_element_types"] = ["i64"]
            kernel = CUDAKernel(artifact, install_external(artifact, directory), {}, {})
            x = session.from_host([0] * len(samples), dtype="i64")
            y = session.tensor((len(samples),), "i64")
            prepared = kernel.prepare(session)
            prepared.submit(x, y).wait()
            expected = [value for _, _, value in samples]
            self.assertEqual(y.to_host(), expected)
            # The synchronous host runner uses the same encoding/layout helpers.
            self.assertEqual(kernel.run([0] * len(samples))["output"], expected)
            launches = session.stats["launches"]
            raw = prepared._parameters([x], [y])
            raw[1] = b"\x01\x00"
            with self.assertRaisesRegex(CUDAError, "byte width"):
                session.submit(prepared._kernel, raw, grid=[1, 1, 1], block=[1, 1, 1])
            raw = prepared._parameters([x], [y])
            raw[0] = b"\x00" * 8
            with self.assertRaisesRegex(CUDAError, "pointer argument"):
                session.submit(prepared._kernel, raw, grid=[1, 1, 1], block=[1, 1, 1])
            self.assertEqual(session.stats["launches"], launches)
            with ExecutionSession() as other:
                wrong_types = [arg["type"] if arg["kind"] == "scalar" else "pointer" for arg in arguments]
                wrong_types[1] = "i16"
                with self.assertRaisesRegex(CUDAError, "layout"):
                    other.load(payload.encode(), "scale2", required_types=["i64"], arch=self.arch,
                               parameter_types=wrong_types)

    def test_external_module_parameter_abi_is_checked_before_launch(self):
        with tempfile.TemporaryDirectory() as directory, ExecutionSession() as session:
            # Remove one declared pointer but keep the normalized artifact's two
            # ports: the driver-reported ABI must reject the false description.
            payload = PTX.replace(", .param .u64 output_ptr", "").replace("ld.param.u64 %y, [output_ptr];", "mov.u64 %y, 0;")
            artifact = make_external_artifact(payload=payload, shape=(10,), arch=self.arch)
            kernel = CUDAKernel(artifact, install_external(artifact, directory), {}, {})
            launches = session.stats["launches"]
            with self.assertRaisesRegex(CUDAError, "parameter|ABI|ParamInfo"):
                kernel.prepare(session)
            self.assertEqual(session.stats["launches"], launches)

    def test_mixed_plans_keep_owned_regions_and_relocate_both_formats(self):
        for format_ in ("ptx", "cubin"):
            for branches in (False, True):
                with self.subTest(format=format_, branches=branches), tempfile.TemporaryDirectory() as directory, ExecutionSession() as session:
                    root = Path(directory)
                    worker = root / "worker.py"
                    worker.write_text(WORKER)
                    source = json.dumps({"format": format_, "payload": self.payload(format_, root)})
                    module = external_graph(self.arch, source, branches=branches)
                    plan = validate_plan(self.compiler.partition(module).report["execution_plan"])
                    external_steps = [step for step in plan["steps"] if step["kind"] == "external"]
                    if branches:
                        self.assertEqual(len(external_steps), 2)
                        self.assertNotIn(external_steps[0]["id"], external_steps[1]["depends_on"])
                    else:
                        self.assertEqual([step["kind"] for step in plan["steps"]], ["owned", "external", "owned"])
                    def compile(name):
                        return self.compiler.compile_cuda(module, target={"backend": "cuda", "arch": self.arch},
                            compilers={"fixture": CompilerEnvironment(python=sys.executable, worker=worker)},
                            llc=llc_path(), ptxas=ptxas_path(), cache_dir=root / "cache", output_dir=root / name)
                    cold, warm = compile("cold"), compile("warm")
                    self.assertIsInstance(cold, CUDAProgram)
                    self.assertTrue(all(kernel.cache["status"] == "hit" for kernel in warm.kernels.values()))
                    shutil.move(root / "warm", root / "relocated")
                    shutil.rmtree(root / "cold")
                    shutil.rmtree(root / "cache")
                    worker.unlink()
                    executable = CUDAProgram.load(root / "relocated").prepare(session)
                    x = session.from_host(list(range(10)), dtype="i32")
                    y = session.tensor((10,), "i32")
                    before = session.stats["launches"]
                    if branches:
                        first = session.tensor((10,), "i32")
                        executable.submit_many([x], [first, y, y, x]).wait()
                        self.assertEqual(first.to_host(), [2 * (n + 1) for n in range(10)])
                        self.assertEqual(y.to_host(), [4 * n + 2 for n in range(10)])
                    else:
                        executable.submit(x, y).wait()
                        self.assertEqual(y.to_host(), [2 * n + 5 for n in range(10)])
                    self.assertEqual(session.stats["launches"] - before, len(plan["steps"]))


if __name__ == "__main__":
    unittest.main()
