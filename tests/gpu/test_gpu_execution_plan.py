"""Real CUDA plan scheduling, fusion fallback, package relocation and reuse."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "python", ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ask import Compiler, CompilerEnvironment, CUDAProgram, ExecutionSession, Module, TensorType
from ask.plan import validate_plan
from support.host_runner import ask_opt_path, llc_path, open_cuda_session, ptxas_path


def external_graph(arch, *, reduction=False, branches=False):
    module = Module()
    tensor = TensorType((10,), "i32")
    semantic = module.semantics("operator", inputs=[tensor], results=[tensor])
    expression = "tl.sum(value, 0)" if reduction else "value * 2"
    source = f'''import triton
import triton.language as tl
@triton.jit
def operation(x, out, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x + i, i < N, other=0)
    tl.store(out + i, {expression}, i < N)
'''
    implementation = module.implementation(
        "implementation", compiler="triton", source=source, entry="operation",
        signature={"x": "*i32", "out": "*i32"}, source_bindings={"x": "input:0", "out": "output:0"},
        constants={"N": 10, "BLOCK": 128}, options={"num_warps": 4},
        target={"backend": "cuda", "arch": arch, "warp_size": 32}, grid=(1, 1, 1))
    graph = module.graph("main", inputs={"x": tensor}, results=[tensor] * (4 if branches else 1))
    x = graph.inputs["x"]
    one = graph.splat(graph.constant(1), (10,))
    first = graph.external(graph.add(x, one), semantics=semantic, implementation=implementation, id="first")
    if branches:
        # Independent external branch: sequential ordering is in steps, not a
        # spurious data edge from the first branch's output.
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
        cls.compiler = Compiler(ask_opt_path(), timeout=180)
        if "triton" not in cls.compiler.capabilities().get("third_party_compilers", []):
            raise unittest.SkipTest("this build does not enable Triton")
        python = os.environ.get("ASK_TEST_TRITON_PYTHON")
        if not python:
            raise unittest.SkipTest("set ASK_TEST_TRITON_PYTHON for external CUDA plan tests")
        cls.environment = {"triton": CompilerEnvironment(python=python, worker=ROOT / "adapters/triton/worker.py")}
        cls.llc, cls.ptxas = llc_path(), ptxas_path()
        with open_cuda_session() as session:
            major, minor = session.capability
            cls.arch = 10 * major + minor

    def compile(self, module, root, name):
        return self.compiler.compile_cuda(module, target={"backend": "cuda", "arch": self.arch},
                                          compilers=self.environment, llc=self.llc, ptxas=self.ptxas,
                                          cache_dir=root / "cache", output_dir=root / name)

    def test_independent_branches_aliases_cache_and_relocated_package(self):
        with tempfile.TemporaryDirectory() as directory, ExecutionSession() as session:
            root = Path(directory)
            module = external_graph(self.arch, branches=True)
            plan = validate_plan(self.compiler.partition(module).report["execution_plan"])
            externals = [step for step in plan["steps"] if step["kind"] != "owned"]
            self.assertEqual(len(externals), 2)
            self.assertNotIn(externals[0]["id"], externals[1]["depends_on"])
            cold = self.compile(module, root, "cold")
            self.assertIsInstance(cold, CUDAProgram)
            warm = self.compile(module, root, "warm")
            self.assertTrue(all(kernel.cache["status"] == "hit" for kernel in warm.kernels.values()))
            self.assertTrue(all(kernel.codegen["executable_cache"]["status"] == "hit" for kernel in warm.kernels.values()))
            shutil.move(root / "warm", root / "relocated")
            executable = CUDAProgram.load(root / "relocated").prepare(session)
            x = session.from_host(list(range(10)), dtype="i32")
            first, final = session.tensor((10,), "i32"), session.tensor((10,), "i32")
            before = session.stats["launches"]
            executable.submit_many([x], [first, final, final, x]).wait()
            self.assertEqual(session.stats["launches"] - before, len(plan["steps"]))
            self.assertEqual(first.to_host(), [2 * (n + 1) for n in range(10)])
            self.assertEqual(final.to_host(), [4 * n + 2 for n in range(10)])

    def test_unsupported_pointwise_proof_falls_back_to_three_kernels(self):
        with tempfile.TemporaryDirectory() as directory, ExecutionSession() as session:
            program = self.compile(external_graph(self.arch, reduction=True), Path(directory), "fallback")
            self.assertIsInstance(program, CUDAProgram)
            self.assertEqual([step["kind"] for step in program.plan["steps"]], ["owned", "external", "owned"])
            self.assertEqual(program.report["fusion_fallbacks"][0]["external_id"], "first")
            self.assertIn("UnsupportedFusion", program.report["fusion_fallbacks"][0]["reason"])
            x, y = session.from_host(list(range(10)), dtype="i32"), session.tensor((10,), "i32")
            program.prepare(session).submit(x, y).wait()
            self.assertEqual(y.to_host(), [58] * 10)


if __name__ == "__main__":
    unittest.main()
