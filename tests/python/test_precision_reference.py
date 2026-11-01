"""Landmark checks for the independent oracle used by every precision matrix."""
from fractions import Fraction
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "python")]
from support.pointwise import (arithmetic_reference, expected_values, fma_reference,
                               numerically_equal, round_reference)


class PrecisionReferenceTests(unittest.TestCase):
    def test_rounding_halfway_to_even_and_minimum_subnormals(self):
        # The spacings are known IEEE format properties, not values obtained
        # from the implementation's encoder or its format descriptor table.
        for dtype, step, smallest in (("f16", 2.0**-10, 2.0**-24),
                                      ("bf16", 2.0**-7, 2.0**-133),
                                      ("f32", 2.0**-23, 2.0**-149)):
            with self.subTest(dtype=dtype):
                self.assertEqual(round_reference(1 + step / 2, dtype), 1)
                self.assertEqual(round_reference(1 + 3 * step / 2, dtype), 1 + 2 * step)
                self.assertEqual(round_reference(smallest, dtype), smallest)
                self.assertEqual(round_reference(smallest / 2, dtype), 0)
                self.assertEqual(round_reference(3 * smallest / 2, dtype), 2 * smallest)
                rounded = round_reference(-smallest / 2, dtype)
                self.assertEqual(rounded, 0)
                self.assertEqual(math.copysign(1, rounded), -1)

    def test_direct_rounding_keeps_sticky_bits_below_binary32_and_binary64(self):
        self.assertEqual(round_reference(1 + 2**-11 + 2**-40, "f16"), 1 + 2**-10)
        self.assertEqual(round_reference(1 + 2**-8 + 2**-30, "bf16"), 1 + 2**-7)
        integer = 2**100 + 2**76 + 1
        self.assertEqual(round_reference(integer, "f32"), float(2**100 + 2**77))
        # An exact rational is also allowed for intermediate arithmetic. Its
        # sticky bit would disappear if the oracle first converted to double.
        exact = Fraction(1) + Fraction(1, 2**24) + Fraction(1, 2**80)
        self.assertEqual(round_reference(exact, "f32"), 1 + 2**-23)

    def test_low_precision_overflow_is_arithmetic_infinity(self):
        self.assertEqual(round_reference(65504, "f16"), 65504)
        self.assertEqual(round_reference(65512, "f16"), 65504)
        self.assertEqual(round_reference(65520, "f16"), math.inf)
        self.assertEqual(round_reference(-65520, "f16"), -math.inf)
        bf16_largest = 255 * 2**120
        self.assertEqual(round_reference(bf16_largest, "bf16"), float(bf16_largest))
        self.assertEqual(round_reference(bf16_largest + 2**118, "bf16"), float(bf16_largest))
        self.assertEqual(round_reference(bf16_largest + 2**119, "bf16"), math.inf)

    def test_special_values_and_signed_zero_are_observable(self):
        for dtype in ("f16", "bf16", "f32"):
            with self.subTest(dtype=dtype):
                negative = arithmetic_reference("add", -0.0, -0.0, dtype)
                positive = arithmetic_reference("add", -0.0, 0.0, dtype)
                self.assertEqual(math.copysign(1, negative), -1)
                self.assertEqual(math.copysign(1, positive), 1)
                self.assertEqual(math.copysign(1, arithmetic_reference("mul", -0.0, 2.0, dtype)), -1)
                self.assertTrue(math.isnan(arithmetic_reference("mul", 0.0, math.inf, dtype)))
                self.assertTrue(math.isnan(arithmetic_reference("add", math.inf, -math.inf, dtype)))
                self.assertEqual(arithmetic_reference("mul", math.inf, -2.0, dtype), -math.inf)
                self.assertTrue(math.isnan(fma_reference(0.0, math.inf, 1.0, dtype)))
                self.assertEqual(math.copysign(1, fma_reference(-0.0, 2.0, -0.0, dtype)), -1)
                self.assertFalse(numerically_equal([0.0], [-0.0], dtype))
                self.assertTrue(numerically_equal([math.nan], [math.nan], dtype))

    def test_fma_retains_residual_that_separate_low_precision_operations_lose(self):
        for dtype, step, residual in (("f16", 2.0**-10, -2.0**-20),
                                      ("bf16", 2.0**-7, -2.0**-14),
                                      ("f32", 2.0**-23, -2.0**-46)):
            with self.subTest(dtype=dtype):
                product = arithmetic_reference("mul", 1 + step, 1 - step, dtype)
                separate = arithmetic_reference("add", product, -1.0, dtype)
                self.assertEqual(separate, 0.0)
                self.assertEqual(fma_reference(1 + step, 1 - step, -1.0, dtype), residual)
                # The shared graph reference must preserve the ASK/F boundary:
                # only F's multiply-add is allowed to use the fused oracle.
                enabled = expected_values([1 + step], -2 * step, 1 + step, dtype=dtype,
                                          external_offset=-1, offset=-0.0, enable_fp_fusion=True)
                disabled = expected_values([1 + step], -2 * step, 1 + step, dtype=dtype,
                                           external_offset=-1, offset=-0.0, enable_fp_fusion=False)
                host = expected_values([1 + step], -2 * step, 1 + step, dtype=dtype,
                                       external_offset=-1, offset=-0.0, external=False)
                self.assertEqual(enabled, [residual])
                self.assertEqual(disabled, [0.0])
                self.assertEqual(host, [0.0])

    def test_integer_graph_reference_wraps_after_every_operation(self):
        self.assertEqual(arithmetic_reference("add", 2147483647, 1, "i32"), -2147483648)
        self.assertEqual(arithmetic_reference("mul", 1073741824, 4, "i32"), 0)
        self.assertEqual(expected_values([2147483647, -2147483648], 1, 2), [3, 5])


if __name__ == "__main__":
    unittest.main()
