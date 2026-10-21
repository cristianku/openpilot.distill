"""Drawing of the comparison video: plan on the camera frame, header, footer, top-down inset.

Images are BGR (cv2). Positions are in the calibrated frame of the model: x forward, y right,
z down, origin at the device; the road is ROAD_HEIGHT_M below it.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .segment import ROAD_HEIGHT_M, VIEW_FRAME_FROM_DEVICE_FRAME, rot_from_euler

WHITE = (255, 255, 255)
GREEN = (60, 230, 60)
YELLOW = (0, 215, 255)
GREY = (170, 170, 170)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

PATH_THICKNESS = 7
TRUTH_THICKNESS = 3
# Points closer than this (in front of the camera) are dropped: they project far below the frame.
MIN_FORWARD_M = 1.0
HEADER_HEIGHT = 86
FOOTER_LINE_HEIGHT = 34

FOOTER_TEXT = "avg miss = quanto il percorso previsto dista da dove ho guidato davvero, piu' basso e' meglio"
TRUTH_LEGEND = "linea gialla = dove ho guidato davvero"

INSET_WIDTH = 270
INSET_HEIGHT = 330
INSET_MARGIN = 16
_FORWARD_RANGES_M = (25.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0)
_LATERAL_RANGES_M = (5.0, 10.0, 20.0, 50.0, 100.0, 200.0)


def project_to_image(points: np.ndarray, rpy: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """(K, 2) pixels of the road points in front of the camera, in path order.

    Same as the openpilot UI: K @ view_frame_from_device_frame @ device_from_calib @ (p + [0, 0, 1.22]).
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    road_points = points + np.array([0.0, 0.0, ROAD_HEIGHT_M])
    view_from_calib = VIEW_FRAME_FROM_DEVICE_FRAME @ rot_from_euler(rpy)
    view = road_points @ view_from_calib.T
    in_front = np.isfinite(view).all(axis=1) & (view[:, 2] > MIN_FORWARD_M)
    view = view[in_front]
    pixels = view @ np.asarray(intrinsics, dtype=np.float64).T
    return pixels[:, 0:2] / pixels[:, 2:3]


def draw_path(image: np.ndarray, pixels: np.ndarray, colour: tuple[int, int, int], thickness: int) -> None:
    """Anti-aliased polyline with a dark outline, sub-pixel precise (cv2 shift = 4 bits)."""
    if len(pixels) < 2:
        return
    # Far outside the frame cv2 still clips, but keep the fixed-point values well inside int32.
    pixels = np.clip(pixels, -20000.0, 20000.0)
    fixed_point = np.round(pixels * 16.0).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [fixed_point], False, BLACK, thickness + 4, cv2.LINE_AA, shift=4)
    cv2.polylines(image, [fixed_point], False, colour, thickness, cv2.LINE_AA, shift=4)


def _darken(image: np.ndarray, top: int, bottom: int, left: int, right: int, factor: float) -> None:
    region = image[top:bottom, left:right]
    image[top:bottom, left:right] = (region.astype(np.float32) * factor).astype(np.uint8)


def _put_text(image, text, origin, scale, colour, thickness) -> None:
    cv2.putText(image, text, origin, FONT, scale, BLACK, thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, FONT, scale, colour, thickness, cv2.LINE_AA)


def _text_width(text: str, scale: float, thickness: int) -> int:
    (width, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    return width


def miss_text(avg_miss_m: float | None) -> str:
    if avg_miss_m is None:
        return "avg miss su questo tratto: --"
    return f"avg miss su questo tratto: {avg_miss_m:.2f} m"


def draw_header(
    image: np.ndarray,
    label: str,
    colour: tuple[int, int, int],
    avg_miss_m: float | None,
    speed_mps: float,
    time_s: float,
) -> None:
    """Model name (in the colour of its line), running avg miss, speed and time on a dark band."""
    width = image.shape[1]
    _darken(image, 0, HEADER_HEIGHT, 0, width, 0.4)
    _put_text(image, label, (18, 36), 1.0, colour, 2)
    _put_text(image, miss_text(avg_miss_m), (18, 74), 1.0, WHITE, 2)

    speed_text = f"{speed_mps * 3.6:.0f} km/h"
    time_text = f"t {time_s:.1f} s"
    _put_text(image, speed_text, (width - 18 - _text_width(speed_text, 0.8, 2), 36), 0.8, WHITE, 2)
    _put_text(image, time_text, (width - 18 - _text_width(time_text, 0.8, 2), 74), 0.8, GREY, 2)


def footer_image(width: int, show_truth: bool) -> np.ndarray:
    """Black band under the panels with the explanation of avg miss (and of the yellow line)."""
    lines = [FOOTER_TEXT]
    if show_truth:
        lines.append(TRUTH_LEGEND)
    height = 14 + FOOTER_LINE_HEIGHT * len(lines)
    height += height % 2  # yuv420p wants an even frame height
    footer = np.zeros((height, width, 3), dtype=np.uint8)
    for line_number, text in enumerate(lines):
        scale = 0.75
        while scale > 0.3 and _text_width(text, scale, 2) > width - 36:
            scale -= 0.05
        if line_number == 0:
            colour = WHITE
        else:
            colour = YELLOW
        cv2.putText(footer, text, (18, 34 + FOOTER_LINE_HEIGHT * line_number), FONT, scale, colour, 2, cv2.LINE_AA)
    return footer


@dataclass(frozen=True, slots=True)
class InsetScale:
    forward_m: float
    lateral_m: float


def inset_scale(paths: np.ndarray, masks: np.ndarray) -> InsetScale:
    """Fixed ranges for a whole clip, from the real paths, so the inset does not jump frame to frame."""
    paths = np.asarray(paths, dtype=np.float64)
    masks = np.asarray(masks, dtype=bool)
    forward = 0.0
    lateral = 0.0
    if masks.any():
        forward = float(np.max(paths[..., 0][masks]))
        lateral = float(np.max(np.abs(paths[..., 1][masks])))
    forward_range = _FORWARD_RANGES_M[-1]
    for candidate in _FORWARD_RANGES_M:
        if forward <= candidate:
            forward_range = candidate
            break
    # Lateral axis exaggerated (labelled): the misses are lateral, metres matter there.
    lateral_range = _LATERAL_RANGES_M[-1]
    for candidate in _LATERAL_RANGES_M:
        if lateral * 1.2 <= candidate:
            lateral_range = candidate
            break
    return InsetScale(forward_m=forward_range, lateral_m=lateral_range)


def draw_inset(
    image: np.ndarray,
    scale: InsetScale,
    lines: list[tuple[np.ndarray, tuple[int, int, int], int]],
) -> None:
    """Top-down view at the bottom right: x forward = up, y right = right. lines: (points (M, 3), colour, px)."""
    height, width = image.shape[:2]
    left = width - INSET_WIDTH - INSET_MARGIN
    top = height - INSET_HEIGHT - INSET_MARGIN
    _darken(image, top, top + INSET_HEIGHT, left, left + INSET_WIDTH, 0.3)
    cv2.rectangle(image, (left, top), (left + INSET_WIDTH - 1, top + INSET_HEIGHT - 1), GREY, 1, cv2.LINE_AA)

    plot_left = left + 14
    plot_right = left + INSET_WIDTH - 14
    plot_top = top + 34
    plot_bottom = top + INSET_HEIGHT - 34
    centre_x = (plot_left + plot_right) / 2.0
    half_width = (plot_right - plot_left) / 2.0
    plot_height = plot_bottom - plot_top

    cv2.putText(image, "vista dall'alto", (left + 10, top + 22), FONT, 0.55, WHITE, 1, cv2.LINE_AA)
    # Straight-ahead axis and the car at the bottom.
    cv2.line(image, (int(centre_x), plot_top), (int(centre_x), plot_bottom), (90, 90, 90), 1, cv2.LINE_AA)
    car = np.array([[centre_x, plot_bottom - 8], [centre_x - 6, plot_bottom + 4], [centre_x + 6, plot_bottom + 4]])
    cv2.fillPoly(image, [np.round(car).astype(np.int32)], WHITE, cv2.LINE_AA)

    forward_label = f"{scale.forward_m:.0f} m"
    cv2.putText(image, forward_label, (plot_left, plot_top + 4), FONT, 0.45, GREY, 1, cv2.LINE_AA)
    lateral_label = f"laterale +-{scale.lateral_m:.0f} m"
    cv2.putText(image, lateral_label, (left + 10, top + INSET_HEIGHT - 10), FONT, 0.45, GREY, 1, cv2.LINE_AA)

    for points, colour, thickness in lines:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 2:
            continue
        pixel_x = centre_x + points[:, 1] / scale.lateral_m * half_width
        pixel_y = plot_bottom - points[:, 0] / scale.forward_m * plot_height
        pixels = np.column_stack((pixel_x, pixel_y))
        pixels = np.clip(pixels, -20000.0, 20000.0)
        fixed_point = np.round(pixels * 16.0).astype(np.int32).reshape(-1, 1, 2)
        # Clip to the plot area: draw on a view of it, shifted.
        plot = image[plot_top - 6 : plot_bottom + 6, left + 1 : left + INSET_WIDTH - 1]
        offset = np.array([left + 1, plot_top - 6], dtype=np.int32) * 16
        cv2.polylines(plot, [fixed_point - offset], False, colour, thickness, cv2.LINE_AA, shift=4)


def draw_panel(
    frame_bgr: np.ndarray,
    rpy: np.ndarray,
    intrinsics: np.ndarray,
    label: str,
    colour: tuple[int, int, int],
    plan_points: np.ndarray,
    truth_points: np.ndarray,
    show_truth: bool,
    avg_miss_m: float | None,
    speed_mps: float,
    time_s: float,
    inset: InsetScale | None,
) -> np.ndarray:
    """One panel of the video: a copy of the camera frame with the plan of one model on it."""
    panel = frame_bgr.copy()
    if show_truth:
        draw_path(panel, project_to_image(truth_points, rpy, intrinsics), YELLOW, TRUTH_THICKNESS)
    draw_path(panel, project_to_image(plan_points, rpy, intrinsics), colour, PATH_THICKNESS)
    draw_header(panel, label, colour, avg_miss_m, speed_mps, time_s)
    if inset is not None:
        draw_inset(panel, inset, [(truth_points, YELLOW, 2), (plan_points, colour, 3)])
    return panel
