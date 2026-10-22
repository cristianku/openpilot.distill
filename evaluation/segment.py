"""One recorded segment for the evaluation: calibration, ground-truth path and streamed model inputs.

The math mirrors supervised/dataset.py (warps, YUV packing, _load_targets) so the models see exactly
what they saw in training. Only numpy and cv2 are imported here; safetensors, openpilot and
PyNvVideoCodec are imported inside the functions that need them, so the module (and the tests of
the geometry) also load on a Mac.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

CAMERA_FPS = 20
# 20 Hz camera, 5 Hz model steps: CAMERA_FPS // DatasetConfig.fps in supervised/dataset.py.
FRAME_SKIP = 4
# Height of the camera above the road, the value the openpilot UI uses to draw the path
# (selfdrive/locationd/calibrationd.py HEIGHT_INIT).
ROAD_HEIGHT_M = 1.22
# Model input size of both cameras (openpilot MEDMODEL_INPUT_SIZE == SBIGMODEL_INPUT_SIZE).
MODEL_INPUT_SIZE = (512, 256)
# The comma four (mici) cameras: rl/env.py and the training data only use this device.
MICI_CAMERAS = ("mici", "os04c10")
MICI_DEVICE_TYPE = 7  # openpilot cereal log.InitData.DeviceType.mici

# Copy of openpilot ModelConstants.T_IDXS: the 33 plan times, 10 * (i / 32) ** 2 seconds.
T_IDXS = 10.0 * (np.arange(33, dtype=np.float64) / 32.0) ** 2

# Copy of openpilot common/transformations/camera.py view_frame_from_device_frame.
# device: x forward, y right, z down. view: x right, y down, z forward.
VIEW_FRAME_FROM_DEVICE_FRAME = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ]
)

# Same 4 road points as helpers/video_helpers.py _WARP_POINTS.
_WARP_POINTS = np.asarray(
    [[-1.0, 1.22, 100.0], [1.0, 1.22, 100.0], [-1.0, 1.22, 200.0], [1.0, 1.22, 200.0]],
    dtype=np.float32,
)


def rot_from_euler(euler: np.ndarray) -> np.ndarray:
    """Copy of openpilot orientation.rot_from_euler (roll, pitch, yaw -> Rz @ Ry @ Rx), any leading shape."""
    euler = np.asarray(euler, dtype=np.float64)
    cos_roll = np.cos(euler[..., 0])
    sin_roll = np.sin(euler[..., 0])
    cos_pitch = np.cos(euler[..., 1])
    sin_pitch = np.sin(euler[..., 1])
    cos_yaw = np.cos(euler[..., 2])
    sin_yaw = np.sin(euler[..., 2])

    matrix = np.empty((*euler.shape[:-1], 3, 3), dtype=np.float64)
    matrix[..., 0, 0] = cos_yaw * cos_pitch
    matrix[..., 0, 1] = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll
    matrix[..., 0, 2] = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll
    matrix[..., 1, 0] = sin_yaw * cos_pitch
    matrix[..., 1, 1] = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
    matrix[..., 1, 2] = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll
    matrix[..., 2, 0] = -sin_pitch
    matrix[..., 2, 1] = cos_pitch * sin_roll
    matrix[..., 2, 2] = cos_pitch * cos_roll
    return matrix


def rot_from_quat(quaternion: np.ndarray) -> np.ndarray:
    """Copy of openpilot orientation.rot_from_quat (w, x, y, z), any leading shape, not normalized."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    w = quaternion[..., 0]
    x = quaternion[..., 1]
    y = quaternion[..., 2]
    z = quaternion[..., 3]

    matrix = np.empty((*quaternion.shape[:-1], 3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - w * z)
    matrix[..., 0, 2] = 2 * (x * z + w * y)
    matrix[..., 1, 0] = 2 * (x * y + w * z)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - w * x)
    matrix[..., 2, 0] = 2 * (x * z - w * y)
    matrix[..., 2, 1] = 2 * (y * z + w * x)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def ground_truth_paths(frame_t: np.ndarray, frame_states: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    """(N, 33, 3) path the car really drove after each camera frame, at the plan times T_IDXS.

    Calibrated frame of that camera frame (x forward, y right, z down, origin at the device): the
    Plan.POSITION target of supervised/dataset.py::_load_targets without augmentation, for all
    frames at once. Points past the end of the segment are extrapolated like there: mask them
    with evaluation.metrics.plan_point_mask.
    """
    frame_t = np.asarray(frame_t, dtype=np.float64)
    query_t = frame_t[:, None] + T_IDXS[None, :]
    indexes = np.clip(np.searchsorted(frame_t, query_t) - 1, 0, len(frame_t) - 2)
    distance = (query_t - frame_t[indexes]) / (frame_t[indexes + 1] - frame_t[indexes])

    # Columns 0:3 ECEF position, 3:7 quaternion ecef_from_device (localizer/localizer.py).
    pose = np.asarray(frame_states[:, 0:7], dtype=np.float64)
    future = pose[indexes] * (1.0 - distance)[..., None] + pose[indexes + 1] * distance[..., None]

    calib_from_device = rot_from_euler(rpy).T
    device_from_ecef = rot_from_quat(future[:, 0, 3:7]).swapaxes(-1, -2)
    calib_from_ecef = np.einsum("ij,njk->nik", calib_from_device, device_from_ecef)
    offsets = future[:, :, 0:3] - future[:, 0:1, 0:3]
    paths = np.einsum("nij,ntj->nti", calib_from_ecef, offsets)
    return paths.astype(np.float32)


def pack_image(rgb: np.ndarray) -> np.ndarray:
    """(6, H/2, W/2) YUV420 planes of one RGB image: copy of supervised/dataset.py::_pack_image."""
    height, width = rgb.shape[:2]
    yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420)
    y = yuv[:height].reshape(height // 2, 2, width // 2, 2).transpose(3, 1, 0, 2).reshape(4, height // 2, width // 2)
    uv = yuv[height:].reshape(2, height // 2, width // 2)
    return np.concatenate((y, uv))


def model_warp_matrices(rpy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera -> model input warps (narrow, wide), as supervised/dataset.py::_load_images without augmentation.

    Copy of calibration_view_eulers + calibration_warp_matrix of helpers/video_helpers.py: that module
    imports PyNvVideoCodec when it loads, which a CPU-only host may not have.
    """
    from openpilot.common.transformations.camera import DEVICE_CAMERAS, denormalize
    from openpilot.common.transformations.model import medmodel_intrinsics, sbigmodel_intrinsics
    from openpilot.common.transformations.orientation import euler_from_rot

    # calibration_view_eulers(rpy) with augment_from_calib = identity.
    augment_from_device = rot_from_euler(rpy).T
    view_euler = VIEW_FRAME_FROM_DEVICE_FRAME @ euler_from_rot(augment_from_device.T)

    cameras = DEVICE_CAMERAS[MICI_CAMERAS]
    # The wide camera ignores wide_from_device_euler, like the training data and modeld.
    narrow = _warp_matrix(cameras.narrow_road.intrinsics, medmodel_intrinsics, view_euler, denormalize)
    wide = _warp_matrix(cameras.wide_road.intrinsics, sbigmodel_intrinsics, view_euler, denormalize)
    return narrow, wide


def _warp_matrix(source_intrinsics, target_intrinsics, view_euler, denormalize) -> np.ndarray:
    after = _WARP_POINTS @ rot_from_euler(view_euler)
    before_pixels = denormalize(_WARP_POINTS[:, :2] / _WARP_POINTS[:, 2, None], intrinsics=source_intrinsics)
    after_pixels = denormalize(after[:, :2] / after[:, 2, None], intrinsics=target_intrinsics)
    return cv2.getPerspectiveTransform(before_pixels.astype(np.float32), after_pixels.astype(np.float32))


def narrow_camera_intrinsics() -> np.ndarray:
    """K of the raw comma four road camera (1344x760, focal 1141.5)."""
    from openpilot.common.transformations.camera import DEVICE_CAMERAS

    return np.asarray(DEVICE_CAMERAS[MICI_CAMERAS].narrow_road.intrinsics, dtype=np.float64)


class ModelInputPacker:
    """Raw RGB frames -> the (6, 128, 256) YUV packs the models take, one frame per camera."""

    def __init__(self, rpy: np.ndarray) -> None:
        self.narrow_matrix, self.wide_matrix = model_warp_matrices(rpy)

    def pack(self, narrow_rgb: np.ndarray, wide_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        narrow = cv2.warpPerspective(narrow_rgb, self.narrow_matrix, MODEL_INPUT_SIZE, borderMode=cv2.BORDER_REPLICATE)
        wide = cv2.warpPerspective(wide_rgb, self.wide_matrix, MODEL_INPUT_SIZE, borderMode=cv2.BORDER_REPLICATE)
        return pack_image(narrow), pack_image(wide)


@dataclass(slots=True)
class Segment:
    name: str
    directory: Path
    frame_info: dict[str, np.ndarray]
    frame_t: np.ndarray
    frame_states: np.ndarray
    rpy: np.ndarray
    frame_count: int
    speeds: np.ndarray
    ground_truth: np.ndarray

    @property
    def duration_s(self) -> float:
        return self.frame_count / CAMERA_FPS


def load_segment(root: Path, name: str) -> Segment:
    """data/<name>/{frame_info,localizer}.safetensors of a comma four segment."""
    from safetensors.numpy import load_file

    directory = root / "data" / name
    for file_name in ("frame_info.safetensors", "localizer.safetensors", "fcamera.hevc", "ecamera.hevc"):
        if not (directory / file_name).exists():
            raise ValueError(f"segment {name}: {file_name} missing in {directory}")
    frame_info = load_file(directory / "frame_info.safetensors")
    if int(frame_info["device_type"].item()) != MICI_DEVICE_TYPE:
        raise ValueError(f"segment {name}: not a comma four (mici)")
    localizer = load_file(directory / "localizer.safetensors")

    frame_t = np.asarray(localizer["frame_t"], dtype=np.float64)
    frame_states = np.asarray(localizer["frame_states"], dtype=np.float64)
    # Roll is already 0 in localizer.safetensors (localizer/localizer.py): the same calibration
    # is used for the warps, the ground truth and the drawing, so all three agree.
    rpy = np.asarray(localizer["rpy"], dtype=np.float64)
    # Road and wide camera can differ by one frame: use the frames both have.
    frame_count = min(len(frame_info["fcamera/t"]), len(frame_info["ecamera/t"]), len(frame_t))
    frame_t = frame_t[:frame_count]
    frame_states = frame_states[:frame_count]
    return Segment(
        name=name,
        directory=directory,
        frame_info=frame_info,
        frame_t=frame_t,
        frame_states=frame_states,
        rpy=rpy,
        frame_count=frame_count,
        speeds=np.linalg.norm(frame_states[:, 7:10], axis=1),
        ground_truth=ground_truth_paths(frame_t, frame_states, rpy),
    )


def nvdec_available() -> bool:
    try:
        import PyNvVideoCodec  # noqa: F401
    except Exception:
        return False
    return True


def stream_frames(
    segment: Segment, first_frame: int, end_frame: int, use_nvdec: bool
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """(frame index, narrow RGB, wide RGB) for first_frame <= index < end_frame, one frame at a time.

    The decoders always start at the beginning of the file (frames before first_frame are decoded
    and dropped), so the frame index is exact. Never more than a few frames in memory.
    """
    if use_nvdec:
        narrow = _nvdec_frames(segment.directory / "fcamera.hevc", segment.frame_info["fcamera/index"], end_frame)
        wide = _nvdec_frames(segment.directory / "ecamera.hevc", segment.frame_info["ecamera/index"], end_frame)
    else:
        narrow = _ffmpeg_frames(segment.directory / "fcamera.hevc", _frame_size(segment, "fcamera"), end_frame)
        wide = _ffmpeg_frames(segment.directory / "ecamera.hevc", _frame_size(segment, "ecamera"), end_frame)
    try:
        for frame_index in range(end_frame):
            narrow_rgb = next(narrow)
            wide_rgb = next(wide)
            if frame_index >= first_frame:
                yield frame_index, narrow_rgb, wide_rgb
    finally:
        narrow.close()
        wide.close()


def _frame_size(segment: Segment, camera: str) -> tuple[int, int]:
    width = int(segment.frame_info[f"{camera}/width"].item())
    height = int(segment.frame_info[f"{camera}/height"].item())
    return width, height


def _nvdec_frames(path: Path, index: np.ndarray, end_frame: int, gpu_id: int = 0) -> Iterator[np.ndarray]:
    """RGB frames from NVDEC, feeding the HEVC file one frame at a time (index = frame_info byte offsets)."""
    import PyNvVideoCodec

    decoder = PyNvVideoCodec.CreateDecoder(
        gpuid=gpu_id,
        codec=PyNvVideoCodec.cudaVideoCodec.HEVC,
        usedevicememory=False,
        outputColorType=PyNvVideoCodec.OutputColorType.RGB,
    )
    produced = 0
    with open(path, "rb") as video:
        for frame_number in range(len(index) - 1):
            # The first chunk also carries the global prefix (VPS/SPS/PPS) before frame 0.
            if frame_number == 0:
                start = 0
            else:
                start = int(index[frame_number, 1])
            end = int(index[frame_number + 1, 1])
            video.seek(start)
            chunk = video.read(end - start)
            for frame in _nvdec_decode(PyNvVideoCodec, decoder, chunk):
                yield frame
                produced += 1
                if produced >= end_frame:
                    return
    # An empty packet flushes the frames still inside the decoder.
    for frame in _nvdec_decode(PyNvVideoCodec, decoder, b""):
        yield frame
        produced += 1
        if produced >= end_frame:
            return
    raise RuntimeError(f"{path}: the decoder gave {produced} frames, {end_frame} were needed")


def _nvdec_decode(module, decoder, chunk: bytes) -> list[np.ndarray]:
    # Same packet setup as helpers/video_helpers.py::decode_frames.
    packet_array = np.frombuffer(chunk, dtype=np.uint8)
    packet = module.PacketData()
    packet.bsl = len(chunk)
    packet.bsl_data = packet_array.ctypes.data
    frames = []
    for frame in decoder.Decode(packet):
        frames.append(np.from_dlpack(frame).copy())
    return frames


def _ffmpeg_frames(path: Path, size: tuple[int, int], end_frame: int) -> Iterator[np.ndarray]:
    """RGB frames from an ffmpeg pipe (CPU). Colours can differ slightly from NVDEC."""
    width, height = size
    command = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-f", "hevc", "-i", str(path),
        "-frames:v", str(end_frame), "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    frame_bytes = width * height * 3
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    try:
        for _ in range(end_frame):
            data = process.stdout.read(frame_bytes)
            if len(data) != frame_bytes:
                raise RuntimeError(f"{path}: ffmpeg ended before the {end_frame} frames asked for")
            yield np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.kill()
        process.wait()
