"""CPU-only checks of the ground truth, packing and projection of evaluation/ on synthetic data.

The copies of openpilot / supervised math in evaluation/segment.py are also compared with the
originals when those import (on the server venv); elsewhere those tests are skipped.
"""

from __future__ import annotations

import importlib.util
import unittest

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_CV2 = importlib.util.find_spec("cv2") is not None
HAS_OPENPILOT = importlib.util.find_spec("openpilot") is not None

if HAS_NUMPY and HAS_CV2:
    import numpy as np

    from evaluation import render, segment

MICI_NARROW_K = [[1141.5, 0.0, 672.0], [0.0, 1141.5, 380.0], [0.0, 0.0, 1.0]]


def _module_imports(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


def circle_drive(speed: float, yaw_rate: float, frames: int, fps: float = 20.0):
    """frame_t, frame_states (N, 43) of a car on a flat circle, device z = world z (down)."""
    frame_t = 100.0 + np.arange(frames) / fps
    elapsed = frame_t - frame_t[0]
    heading = yaw_rate * elapsed
    radius = speed / yaw_rate
    states = np.zeros((frames, 43))
    states[:, 0] = radius * np.sin(heading)
    states[:, 1] = radius * (1.0 - np.cos(heading))
    # ecef_from_device = rotation about z by the heading: quaternion (w, x, y, z).
    states[:, 3] = np.cos(heading / 2)
    states[:, 6] = np.sin(heading / 2)
    states[:, 7] = speed * np.cos(heading)
    states[:, 8] = speed * np.sin(heading)
    states[:, 12] = yaw_rate
    states[:, 20] = speed * yaw_rate
    return frame_t, states


@unittest.skipUnless(HAS_NUMPY and HAS_CV2, "numpy/cv2 not installed")
class GroundTruthTest(unittest.TestCase):
    def test_straight_drive(self):
        frame_t = np.arange(400) / 20.0
        states = np.zeros((400, 43))
        states[:, 0] = 12.0 * frame_t
        states[:, 3] = 1.0
        paths = segment.ground_truth_paths(frame_t, states, np.zeros(3))
        np.testing.assert_allclose(paths[10, :, 0], 12.0 * segment.T_IDXS, atol=1e-4)
        np.testing.assert_allclose(paths[10, :, 1:], 0.0, atol=1e-6)

    def test_right_turn_is_positive_y(self):
        speed = 10.0
        yaw_rate = 0.2
        frame_t, states = circle_drive(speed, yaw_rate, 600)
        paths = segment.ground_truth_paths(frame_t, states, np.zeros(3))
        radius = speed / yaw_rate
        turned = yaw_rate * segment.T_IDXS
        expected_x = radius * np.sin(turned)
        expected_y = radius * (1.0 - np.cos(turned))
        # Any frame with the whole 10 s inside the drive: the path seen from the car is the same.
        for frame_index in (0, 57, 300):
            np.testing.assert_allclose(paths[frame_index, :, 0], expected_x, atol=2e-3)
            np.testing.assert_allclose(paths[frame_index, :, 1], expected_y, atol=2e-3)
        self.assertGreater(paths[57, 32, 1], 5.0)

    def test_calibration_rotates_the_path(self):
        frame_t, states = circle_drive(10.0, 0.1, 400)
        rpy = np.array([0.0, 0.03, -0.02])
        plain = segment.ground_truth_paths(frame_t, states, np.zeros(3))
        calibrated = segment.ground_truth_paths(frame_t, states, rpy)
        calib_from_device = segment.rot_from_euler(rpy).T
        np.testing.assert_allclose(calibrated[5], plain[5] @ calib_from_device.T, atol=1e-4)

    def test_rotations(self):
        np.testing.assert_allclose(segment.rot_from_euler(np.zeros(3)), np.eye(3))
        yaw = segment.rot_from_euler(np.array([0.0, 0.0, np.pi / 2]))
        # Yaw +90 degrees turns forward (x) into right (y).
        np.testing.assert_allclose(yaw @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)
        quaternion = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        np.testing.assert_allclose(segment.rot_from_quat(quaternion), yaw, atol=1e-12)
        eulers = np.random.default_rng(0).uniform(-0.5, 0.5, size=(5, 3))
        matrices = segment.rot_from_euler(eulers)
        for matrix in matrices:
            np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)


@unittest.skipUnless(HAS_NUMPY and HAS_CV2, "numpy/cv2 not installed")
class PackAndProjectionTest(unittest.TestCase):
    def test_pack_image_layout(self):
        rgb = np.random.default_rng(1).integers(0, 256, size=(256, 512, 3), dtype=np.uint8)
        pack = segment.pack_image(rgb)
        self.assertEqual(pack.shape, (6, 128, 256))
        y = segment.cv2.cvtColor(rgb, segment.cv2.COLOR_RGB2YUV_I420)[:256]
        # Same channel order as rl/actor.py and modeld.
        np.testing.assert_array_equal(pack[0], y[0::2, 0::2])
        np.testing.assert_array_equal(pack[1], y[1::2, 0::2])
        np.testing.assert_array_equal(pack[2], y[0::2, 1::2])
        np.testing.assert_array_equal(pack[3], y[1::2, 1::2])

    def test_straight_path_projects_on_the_centre_line(self):
        distances = np.array([0.5, 2.0, 10.0, 50.0])
        points = np.zeros((4, 3))
        points[:, 0] = distances
        pixels = render.project_to_image(points, np.zeros(3), np.array(MICI_NARROW_K))
        # The point at 0.5 m is behind the cut (MIN_FORWARD_M) and dropped.
        self.assertEqual(len(pixels), 3)
        np.testing.assert_allclose(pixels[:, 0], 672.0)
        np.testing.assert_allclose(pixels[:, 1], 380.0 + 1141.5 * 1.22 / distances[1:])
        # Starts below the bottom of the 760 px frame, then climbs towards the horizon.
        self.assertGreater(pixels[0, 1], 760.0)
        self.assertLess(pixels[2, 1], 420.0)

    def test_point_to_the_right_is_right_of_centre(self):
        pixels = render.project_to_image(np.array([[20.0, 2.0, 0.0]]), np.zeros(3), np.array(MICI_NARROW_K))
        self.assertAlmostEqual(pixels[0, 0], 672.0 + 1141.5 * 2.0 / 20.0)


@unittest.skipUnless(HAS_NUMPY and HAS_CV2 and HAS_OPENPILOT, "openpilot not installed")
class OpenpilotCopiesTest(unittest.TestCase):
    def test_constants_match_openpilot(self):
        from openpilot.common.transformations.camera import view_frame_from_device_frame
        from openpilot.selfdrive.modeld.constants import ModelConstants

        np.testing.assert_allclose(segment.T_IDXS, ModelConstants.T_IDXS)
        np.testing.assert_array_equal(segment.VIEW_FRAME_FROM_DEVICE_FRAME, view_frame_from_device_frame)
        np.testing.assert_allclose(segment.narrow_camera_intrinsics(), MICI_NARROW_K)

    def test_rotations_match_openpilot(self):
        from openpilot.common.transformations.orientation import rot_from_euler, rot_from_quat

        generator = np.random.default_rng(2)
        eulers = generator.uniform(-1.0, 1.0, size=(6, 3))
        np.testing.assert_allclose(segment.rot_from_euler(eulers), rot_from_euler(eulers), atol=1e-12)
        quaternions = generator.normal(size=(6, 4))
        np.testing.assert_allclose(segment.rot_from_quat(quaternions), rot_from_quat(quaternions), atol=1e-12)

    @unittest.skipUnless(HAS_OPENPILOT and _module_imports("helpers.video_helpers"), "PyNvVideoCodec not importable")
    def test_warps_match_video_helpers(self):
        from openpilot.common.transformations.camera import DEVICE_CAMERAS
        from openpilot.common.transformations.model import medmodel_intrinsics, sbigmodel_intrinsics

        from helpers.video_helpers import calibration_view_eulers, calibration_warp_matrix

        rpy = np.array([0.0, -0.0192, 0.0137])
        cameras = DEVICE_CAMERAS[("mici", "os04c10")]
        view_euler = calibration_view_eulers(rpy)
        narrow, wide = segment.model_warp_matrices(rpy)
        expected_narrow = calibration_warp_matrix(cameras.narrow_road.intrinsics, medmodel_intrinsics, view_euler)
        expected_wide = calibration_warp_matrix(cameras.wide_road.intrinsics, sbigmodel_intrinsics, view_euler)
        np.testing.assert_allclose(narrow, expected_narrow, atol=1e-9)
        np.testing.assert_allclose(wide, expected_wide, atol=1e-9)

    @unittest.skipUnless(HAS_OPENPILOT and _module_imports("supervised.dataset"), "supervised.dataset not importable")
    def test_ground_truth_and_pack_match_the_training_data(self):
        from supervised.dataset import _load_targets, _pack_image

        frame_t, states = circle_drive(14.0, 0.08, 500)
        localizer = {
            "frame_t": frame_t,
            "frame_states": states,
            "rpy": np.array([0.0, -0.02, 0.015]),
            "wide_from_device_euler": np.zeros(3),
        }
        frames = np.arange(0, 500, 37)
        count = len(frames)
        targets = _load_targets(
            localizer,
            frames[:, None],
            np.broadcast_to(np.eye(3), (count, 3, 3)),
            np.full((count, 2), 0.3, dtype=np.float32),
        )
        mine = segment.ground_truth_paths(frame_t, states, localizer["rpy"])[frames]
        np.testing.assert_allclose(mine, targets["plan"][:, 0, :, 0:3], atol=1e-4)

        rgb = np.random.default_rng(3).integers(0, 256, size=(256, 512, 3), dtype=np.uint8)
        np.testing.assert_array_equal(segment.pack_image(rgb), _pack_image(rgb))


if __name__ == "__main__":
    unittest.main()
