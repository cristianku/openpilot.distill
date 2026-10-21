"""CPU-only checks of the drawing of the comparison video (numpy + cv2, synthetic frames)."""

from __future__ import annotations

import importlib.util
import unittest

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_CV2 = importlib.util.find_spec("cv2") is not None

if HAS_NUMPY and HAS_CV2:
    import numpy as np

    from evaluation import render
    from evaluation.segment import T_IDXS

MICI_NARROW_K = [[1141.5, 0.0, 672.0], [0.0, 1141.5, 380.0], [0.0, 0.0, 1.0]]


@unittest.skipUnless(HAS_NUMPY and HAS_CV2, "numpy/cv2 not installed")
class RenderTest(unittest.TestCase):
    def straight_plan(self) -> np.ndarray:
        plan = np.zeros((33, 3))
        plan[:, 0] = 15.0 * T_IDXS
        return plan

    def test_panel_draws_a_copy(self):
        frame = np.full((760, 1344, 3), 80, dtype=np.uint8)
        original = frame.copy()
        plan = self.straight_plan()
        scale = render.inset_scale(plan[None], np.ones((1, 33), dtype=bool))
        panel = render.draw_panel(
            frame, np.zeros(3), np.array(MICI_NARROW_K), "studente", render.GREEN, plan, plan, True, 1.234, 12.0, 3.5,
            scale,
        )
        np.testing.assert_array_equal(frame, original)
        self.assertEqual(panel.shape, frame.shape)
        # The line starts at the bottom centre of the image.
        bottom_centre = panel[740:760, 662:682].reshape(-1, 3)
        self.assertTrue(np.any(np.all(bottom_centre == render.GREEN, axis=1)))
        # The inset (bottom right) was drawn into the panel, not into a copy of the region.
        inset = panel[760 - render.INSET_HEIGHT - render.INSET_MARGIN : 760 - render.INSET_MARGIN, -300:]
        self.assertTrue(np.any(np.all(inset == render.GREEN, axis=-1)))

    def test_path_far_outside_the_image_does_not_fail(self):
        frame = np.zeros((760, 1344, 3), dtype=np.uint8)
        pixels = np.array([[1e9, -1e9], [-5e6, 3e7], [672.0, 380.0]])
        render.draw_path(frame, pixels, render.WHITE, 5)
        render.draw_path(frame, np.zeros((1, 2)), render.WHITE, 5)

    def test_inset_scale_from_the_truth(self):
        plan = self.straight_plan()
        plan[:, 1] = np.linspace(0.0, 7.0, 33)
        scale = render.inset_scale(plan[None], np.ones((1, 33), dtype=bool))
        self.assertEqual(scale.forward_m, 150.0)
        self.assertEqual(scale.lateral_m, 10.0)

    def test_footer_has_even_height(self):
        for show_truth in (False, True):
            footer = render.footer_image(1344, show_truth)
            self.assertEqual(footer.shape[1], 1344)
            self.assertEqual(footer.shape[0] % 2, 0)

    def test_miss_text(self):
        self.assertEqual(render.miss_text(None), "avg miss su questo tratto: --")
        self.assertEqual(render.miss_text(1.234), "avg miss su questo tratto: 1.23 m")


if __name__ == "__main__":
    unittest.main()
