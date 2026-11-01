"""Independent rational IEEE-754 RNE numerical oracle and test matrix planner checks."""
from __future__ import annotations

from fractions import Fraction
import math
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT / "python"), str(ROOT / "tests")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from support.pointwise import (
    DTYPES,
    arithmetic_reference,
    build_module,
    cast_reference,
    expected_values,
    fma_reference,
    numerically_equal,
    plan_matrix,
    round_reference,
)


class PrecisionReferenceTests(unittest.TestCase):
    def test_rne_subnormals_sticky_bits_and_overflow(self):
        for dtype, step, smallest in (
            ("f16", 2.0**-10, 2.0**-24),
            ("bf16", 2.0**-7, 2.0**-133),
            ("f32", 2.0**-23, 2.0**-149),
        ):
            self.assertEqual(round_reference(1 + step / 2, dtype), 1)
            self.assertEqual(round_reference(1 + 3 * step / 2, dtype), 1 + 2 * step)
            self.assertEqual(round_reference(smallest, dtype), smallest)
            self.assertEqual(round_reference(smallest / 2, dtype), 0)
            self.assertEqual(round_reference(3 * smallest / 2, dtype), 2 * smallest)
            neg_zero = round_reference(-smallest / 2, dtype)
            self.assertEqual(math.copysign(1, neg_zero), -1)

        self.assertEqual(round_reference(2**100 + 2**76 + 1, "f32"), float(2**100 + 2**77))
        self.assertEqual(round_reference(Fraction(1) + Fraction(1, 2**24) + Fraction(1, 2**80), "f32"), 1 + 2**-23)
        self.assertEqual(round_reference(65512, "f16"), 65504)
        self.assertEqual(round_reference(65520, "f16"), math.inf)

    def test_special_floats_fma_boundary_and_saturating_casts(self):
        for dtype, step, residual in (
            ("f16", 2.0**-10, -2.0**-20),
            ("bf16", 2.0**-7, -2.0**-14),
            ("f32", 2.0**-23, -2.0**-46),
        ):
            self.assertFalse(numerically_equal([0.0], [-0.0], dtype))
            self.assertTrue(numerically_equal([math.nan], [math.nan], dtype))
            self.assertEqual(fma_reference(1 + step, 1 - step, -1.0, dtype), residual)
            enabled = expected_values([1 + step], -2 * step, 1 + step, dtype=dtype,
                                      external_offset=-1, offset=-0.0, enable_fp_fusion=True)
            disabled = expected_values([1 + step], -2 * step, 1 + step, dtype=dtype,
                                       external_offset=-1, offset=-0.0, enable_fp_fusion=False)
            self.assertEqual((enabled, disabled), ([residual], [0.0]))

        for target, width in (("i32", 32), ("i64", 64)):
            lower, upper = -(1 << (width - 1)), (1 << (width - 1)) - 1
            self.assertEqual(cast_reference(math.nan, "f32", target), 0)
            self.assertEqual(cast_reference(math.inf, "f32", target), upper)
            self.assertEqual(cast_reference(-math.inf, "f32", target), lower)
            self.assertEqual(arithmetic_reference("add", upper, 1, target), lower)
        self.assertTrue(cast_reference(math.nan, "f32", "bool"))
        self.assertFalse(cast_reference(-0.0, "f32", "bool"))


class MatrixPlannerTests(unittest.TestCase):
    def test_matrix_deduplication_aliases_and_topology_sharing(self):
        plan = plan_matrix(DTYPES)
        self.assertEqual((len(plan.eligible), len(plan.skipped), plan.gpu_executions), (104, 28, 208))

        dedup = plan_matrix(["f32", "f16", "bf16"], ["special", "mixed-special"])
        self.assertEqual(dedup.gpu_executions, 10)
        self.assertEqual(dedup.skipped[0]["case"], "mixed-special")

        alias = plan_matrix(["fp16", "f16"], ["same-shape"])
        self.assertEqual(alias.dtypes, ("f16",))
        for bad_dtypes, bad_cases in ((["i32"], ["fractions"]), ([], ["same-shape"]), (["f64"], ["same-shape"])):
            with self.assertRaises(ValueError):
                plan_matrix(bad_dtypes, bad_cases)

        chain = plan_matrix(DTYPES, ["expression-chain"])
        topologies = [re.findall(r'"(ask\.[a-z_]+)"\(', build_module(dtype=c.dtype, **c.options).to_mlir())
                      for c in chain.eligible]
        self.assertTrue(all(t == topologies[0] for t in topologies))


if __name__ == "__main__":
    unittest.main()
