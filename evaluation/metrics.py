"""avg miss: how far the predicted path is from where the car really went.

miss of one frame = mean over the plan points i >= 1 with T_IDXS[i] <= horizon and
t + T_IDXS[i] <= the last localizer time, of the distance on the road plane (x, y) between the
predicted point i and the real position at the same time. lateral miss = same mean of |y_pred - y_real|.
avg miss = mean of the frame misses over the evaluated frames.
"""

from __future__ import annotations

import numpy as np

from .segment import T_IDXS

DEFAULT_HORIZON_S = 10.0
# Bump when a change in the evaluation code changes the numbers: holdout results with another
# version are recomputed instead of read from outputs/eval/.
EVAL_CODE_VERSION = 1
# Float slack for the time comparisons (T_IDXS[32] is exactly 10.0, frame times are float64).
_TIME_EPSILON = 1e-6


def plan_point_mask(frame_times: np.ndarray, last_time: float, horizon_s: float = DEFAULT_HORIZON_S) -> np.ndarray:
    """(N, 33) bool: the plan points that count in the miss of each frame."""
    frame_times = np.asarray(frame_times, dtype=np.float64).reshape(-1)
    # Point 0 is "now": always at the origin for both, it would only dilute the mean.
    index_ok = np.arange(len(T_IDXS)) >= 1
    horizon_ok = T_IDXS <= horizon_s + _TIME_EPSILON
    point_ok = index_ok & horizon_ok
    # Past the end of the segment the ground truth is extrapolated, not measured.
    time_ok = frame_times[:, None] + T_IDXS[None, :] <= last_time + _TIME_EPSILON
    return point_ok[None, :] & time_ok


def full_horizon_frames(frame_times: np.ndarray, last_time: float, horizon_s: float = DEFAULT_HORIZON_S) -> np.ndarray:
    """(N,) bool: frames whose whole horizon is inside the segment (same points for every frame)."""
    frame_times = np.asarray(frame_times, dtype=np.float64).reshape(-1)
    return frame_times + horizon_s <= last_time + _TIME_EPSILON


def frame_misses(predicted: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """miss and lateral miss in metres for N frames.

    predicted, truth: (N, 33, >=2) positions in the calibrated frame (x forward, y right).
    mask: (N, 33) from plan_point_mask. A frame without points (or with a NaN plan) gives NaN.
    """
    predicted = np.asarray(predicted, dtype=np.float64)[..., 0:2]
    truth = np.asarray(truth, dtype=np.float64)[..., 0:2]
    mask = np.asarray(mask, dtype=bool)

    distance = np.linalg.norm(predicted - truth, axis=-1)
    lateral = np.abs(predicted[..., 1] - truth[..., 1])
    counts = mask.sum(axis=-1)

    miss = np.full(len(counts), np.nan)
    lateral_miss = np.full(len(counts), np.nan)
    has_points = counts > 0
    miss[has_points] = np.where(mask, distance, 0.0).sum(axis=-1)[has_points] / counts[has_points]
    lateral_miss[has_points] = np.where(mask, lateral, 0.0).sum(axis=-1)[has_points] / counts[has_points]
    return miss, lateral_miss


def average(values) -> float | None:
    """Mean ignoring NaN; None when there is nothing to average (JSON null)."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    return float(values.mean())


def rounded(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return round(float(value), digits)
