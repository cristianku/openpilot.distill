"""CPU-only checks of the avg miss metric of evaluation/metrics.py (numpy only, no data)."""

from __future__ import annotations

import importlib.util
import unittest

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_CV2 = importlib.util.find_spec("cv2") is not None

if HAS_NUMPY and HAS_CV2:
    import numpy as np

    from evaluation import metrics
    from evaluation.segment import T_IDXS


@unittest.skipUnless(HAS_NUMPY and HAS_CV2, "numpy/cv2 not installed")
class MetricTest(unittest.TestCase):
    def test_mask_skips_point_zero_and_respects_horizon(self):
        mask = metrics.plan_point_mask(np.array([0.0]), last_time=100.0, horizon_s=10.0)
        self.assertFalse(mask[0, 0])
        self.assertTrue(mask[0, 1:].all())

        mask = metrics.plan_point_mask(np.array([0.0]), last_time=100.0, horizon_s=2.5)
        expected = (np.arange(33) >= 1) & (T_IDXS <= 2.5)
        np.testing.assert_array_equal(mask[0], expected)

    def test_mask_stops_at_the_end_of_the_segment(self):
        # Frame at 55 s of a segment ending at 60 s: only the points up to +5 s are measured.
        mask = metrics.plan_point_mask(np.array([55.0]), last_time=60.0, horizon_s=10.0)
        expected = (np.arange(33) >= 1) & (T_IDXS <= 5.0)
        np.testing.assert_array_equal(mask[0], expected)
        self.assertFalse(metrics.full_horizon_frames(np.array([55.0]), 60.0, 10.0)[0])
        self.assertTrue(metrics.full_horizon_frames(np.array([50.0]), 60.0, 10.0)[0])

    def test_constant_lateral_offset(self):
        truth = np.zeros((2, 33, 3))
        truth[:, :, 0] = 15.0 * T_IDXS
        predicted = truth.copy()
        predicted[0, :, 1] += 0.5
        predicted[1, :, 1] -= 2.0
        mask = metrics.plan_point_mask(np.array([0.0, 0.05]), last_time=100.0)
        miss, lateral = metrics.frame_misses(predicted, truth, mask)
        np.testing.assert_allclose(miss, [0.5, 2.0])
        np.testing.assert_allclose(lateral, [0.5, 2.0])

    def test_miss_is_the_mean_of_the_point_distances(self):
        truth = np.zeros((1, 33, 3))
        predicted = np.zeros((1, 33, 3))
        # 3-4-5 triangle on the last point only, z ignored.
        predicted[0, 32, 0:3] = [3.0, 4.0, 100.0]
        mask = metrics.plan_point_mask(np.array([0.0]), last_time=100.0)
        miss, lateral = metrics.frame_misses(predicted, truth, mask)
        self.assertAlmostEqual(miss[0], 5.0 / 32)
        self.assertAlmostEqual(lateral[0], 4.0 / 32)

    def test_frames_without_points_or_plan_are_nan(self):
        truth = np.zeros((2, 33, 3))
        predicted = np.zeros((2, 33, 3))
        predicted[1] = np.nan
        mask = np.zeros((2, 33), dtype=bool)
        mask[1, 1:] = True
        miss, lateral = metrics.frame_misses(predicted, truth, mask)
        self.assertTrue(np.isnan(miss[0]))
        self.assertTrue(np.isnan(miss[1]))
        self.assertTrue(np.isnan(lateral[1]))

    def test_average_ignores_nan(self):
        self.assertAlmostEqual(metrics.average([1.0, np.nan, 3.0]), 2.0)
        self.assertIsNone(metrics.average([np.nan]))
        self.assertIsNone(metrics.average([]))
        self.assertIsNone(metrics.rounded(None))
        self.assertIsNone(metrics.rounded(float("nan")))
        self.assertEqual(metrics.rounded(1.234567, 2), 1.23)


if __name__ == "__main__":
    unittest.main()
