"""Hardware CUDA tests for ASK + Triton value-boundary fusion, multi-tier caching, and cross-stream CAI."""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
for entry in (str(ROOT / "python"), str(ROOT / "tests")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from ask import Compiler, CompilerEnvironment, ExecutionSession
from support.host_runner import ask_opt_path, llc_path, open_cuda_session, ptxas_path
from support.pointwise import DTYPES, build_module, expected_values, input_values, numerically_equal, plan_matrix


class TritonFusedCUDAHardwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.compiler = Compiler(timeout=120)
        except Exception:
            cls.compiler = Compiler(ask_opt_path(), timeout=120)
        caps = cls.compiler.capabilities()
        if "triton" not in caps.get("third_party_compilers", []):
            raise unittest.SkipTest("this ASK build does not enable the Triton adapter")
        cls.triton_python = (
            os.environ.get("ASK_TEST_TRITON_PYTHON")
            or os.environ.get("ASK_TRITON_PYTHON")
            or sys.executable
        )
        cls.triton_worker = str(ROOT / "adapters/triton/worker.py")
        cls.llc = llc_path()
        cls.ptxas = ptxas_path()
        probe = open_cuda_session()
        cls.major, cls.minor = probe.capability
        cls.arch = cls.major * 10 + cls.minor
        cls.supports_bf16 = cls.major >= 8
        probe.close()
        cls._tmp = tempfile.TemporaryDirectory(prefix="ask-gpu-triton-")
        cls.work = Path(cls._tmp.name)
        cls.cache_dir = cls.work / "cache"

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "_tmp"):
            cls._tmp.cleanup()

    def compile_fused(self, label: str, module):
        return self.compiler.compile_cuda(
            module,
            cache_dir=self.cache_dir,
            output_dir=self.work / label,
            compilers={"triton": CompilerEnvironment(python=self.triton_python, worker=self.triton_worker)},
            llc=self.llc,
            ptxas=self.ptxas,
        )

    def test_fused_precisions_fma_policy_cache_and_cross_stream_cai(self):
        dtypes = [d for d in DTYPES if self.supports_bf16 or d != "bf16"]
        selected_cases = [
            "same-shape",
            "shape-3x5",
            "multiple-sites",
            "F-contract-enabled",
            "F-contract-disabled",
            "mixed-accumulate",
        ]
        plan = plan_matrix(dtypes, selected_cases)
        cases = [
            item for item in plan.eligible
            if item.graph.name == "same-shape"
            or (item.graph.name == "mixed-accumulate" and item.dtype == "f16")
            or item.dtype == "f32"
        ]
        with ExecutionSession(device=0) as session:
            for item in cases:
                dtype, options = item.dtype, item.options
                label = f"{dtype}-{item.graph.name}"
                with self.subTest(case=label):
                    module = build_module(dtype=dtype, arch=self.arch, **options)
                    shape = tuple(options.get("shape", (options.get("size"),)))
                    size = math.prod(shape)
                    values = input_values(size, dtype)
                    ref_opts = {
                        k: options[k]
                        for k in ("bias", "factor", "pre_factor", "post_factor", "offset",
                                  "tail_offset", "external_offset", "enable_fp_fusion",
                                  "compute_dtype", "roundtrip_dtype")
                        if k in options
                    }
                    ref_opts.setdefault("bias", 1)
                    ref_opts.setdefault("factor", 2)
                    expected = expected_values(values, dtype=dtype, **ref_opts)

                    cold = self.compile_fused(f"{label}-cold", module)
                    warm = self.compile_fused(f"{label}-warm", module)
                    self.assertEqual((cold.cache["status"], warm.cache["status"]), ("miss", "hit"))
                    self.assertEqual(cold.codegen["cubin_sha256"], warm.codegen["cubin_sha256"])

                    exe = cold.prepare(session)
                    before_hits = session.stats["module_hits"]
                    warm.prepare(session)
                    self.assertEqual(session.stats["module_hits"], before_hits + 1)

                    x = session.from_host(values, shape=shape, dtype=dtype)
                    y = session.tensor(shape, dtype)
                    before_launches = session.stats["launches"]
                    for _ in range(4):
                        ev = exe.submit(x, y)
                    ev.wait()
                    self.assertEqual(session.stats["launches"] - before_launches, 4)
                    self.assertTrue(numerically_equal(y.to_host(), expected, dtype), label)

                    if item.graph.name == "same-shape" and dtype == "f32":
                        with ExecutionSession(device=0) as consumer:
                            next_exe = cold.prepare(consumer)
                            exe.submit(x, y)
                            imported = consumer.from_cuda_array(y)
                            self.assertEqual(imported.data_ptr, y.data_ptr)
                            out2 = consumer.tensor(shape, dtype)
                            next_exe.submit(imported, out2).wait()
                            expected_twice = expected_values(expected, dtype=dtype, **ref_opts)
                            self.assertTrue(numerically_equal(out2.to_host(), expected_twice, dtype))


if __name__ == "__main__":
    unittest.main()
