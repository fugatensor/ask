"""Actual tiled attention kernels: mixed port shapes, tail masks and online softmax."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
for entry in (str(ROOT / 'python'), str(ROOT / 'tests')):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from ask import Compiler, CompilerEnvironment, ExecutionSession
from support.host_runner import ask_opt_path, llc_path, open_cuda_session, ptxas_path

spec = importlib.util.spec_from_file_location('ask_attention_demo', ROOT / 'examples/flash_attention2.py')
attention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(attention)


class FlashAttentionCUDAHardwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.compiler = Compiler(timeout=180)
        except Exception:
            cls.compiler = Compiler(ask_opt_path(), timeout=180)
        if 'triton' not in cls.compiler.capabilities().get('third_party_compilers', []):
            raise unittest.SkipTest('ASK build does not enable the Triton adapter')
        cls.python = os.environ.get('ASK_TEST_TRITON_PYTHON') or os.environ.get('ASK_TRITON_PYTHON') or sys.executable
        cls.environment = CompilerEnvironment(cls.python, str(ROOT / 'adapters/triton/worker.py'))
        cls.llc, cls.ptxas = llc_path(), ptxas_path()
        with open_cuda_session() as session:
            major, minor = session.capability
            cls.arch = major * 10 + minor
        if cls.arch < 80:
            raise unittest.SkipTest('attention example requires Ampere or newer')

    def test_forward_precisions_causal_rectangles_tail_tiles_and_cache(self):
        # Every precision/mask combination crosses at least two KV tiles;
        # unequal D/DV demonstrates this is not an elementwise boundary kernel.
        cases = [(dtype, causal, dict(batch=1, heads=2, m=37, n=55, d=32, dv=64), 1.0)
                 for dtype in ('f16', 'bf16') for causal in (False, True)]
        cases += [('f16', False, dict(batch=2, heads=2, m=65, n=73, d=64, dv=32), 1.0),
                  ('bf16', True, dict(batch=1, heads=1, m=35, n=19, d=128, dv=128), 5.0)]
        with tempfile.TemporaryDirectory(prefix='ask-attention-test-') as work, ExecutionSession() as session:
            directory = Path(work)
            for index, (dtype, causal, shape, magnitude) in enumerate(cases):
                with self.subTest(dtype=dtype, causal=causal, shape=shape):
                    module = attention.build_module(**shape, dtype=dtype, causal=causal, arch=self.arch)
                    kernel = self.compiler.compile_cuda(module, llc=self.llc, ptxas=self.ptxas,
                        compilers={'triton': self.environment}, cache_dir=directory / 'cache',
                        output_dir=directory / f'case-{index}')
                    self.assertEqual([step['kind'] for step in kernel.report['execution_plan']['steps']], ['external'])
                    self.assertGreater(kernel.artifact['invocation']['shared_bytes'], 0)
                    self.assertEqual([port['logical_port'] for port in kernel.artifact['port_map']],
                                     ['input:0', 'input:1', 'input:2', 'output:0'])
                    b, h, m, n, d, dv = (shape[name] for name in ('batch', 'heads', 'm', 'n', 'd', 'dv'))
                    arrays = attention.sample_inputs(**shape, scale=magnitude)
                    inputs = [session.from_host(values, shape=dims, dtype=dtype)
                              for values, dims in zip(arrays, ((b, h, m, d), (b, h, n, d), (b, h, n, dv)))]
                    output = session.tensor((b, h, m, dv), dtype)
                    kernel.prepare(session).submit_many(inputs, [output]).wait()
                    expected = attention.reference(*(value.to_host() for value in inputs), **shape, causal=causal)
                    attention.check_result(output.to_host(), expected, dtype)
                    # The native plan must retain the opaque kernel's own ABI,
                    # not feed it to the pointwise source instrumentation pass.
                    if index == 0:
                        warm = self.compiler.compile_cuda(module, llc=self.llc, ptxas=self.ptxas,
                            compilers={'triton': self.environment}, cache_dir=directory / 'cache',
                            output_dir=directory / 'warm')
                        self.assertEqual(warm.report['ir_cache']['status'], 'hit')
                        self.assertEqual(warm.report['executable_cache']['status'], 'hit')
                        another = session.tensor((b, h, m, dv), dtype)
                        warm.prepare(session).submit_many(inputs, [another]).wait()
                        self.assertEqual(another.to_host(), output.to_host())


if __name__ == '__main__':
    unittest.main()
