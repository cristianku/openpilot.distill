# Video and calibration helpers

This directory contains **Python functions, not standalone shell scripts**. They are imported by [`supervised/dataset.py`](../supervised/dataset.py). Run the examples from the repository root, not from `helpers/`.

## Prerequisites

1. Install [`uv`](https://docs.astral.sh/uv/) and use **Python >=3.12.3,<3.13** (the range in `pyproject.toml`). From the repository root run `uv sync`.
2. Video decoding uses `PyNvVideoCodec` / NVIDIA **NVDEC** with an available supported NVIDIA GPU and working driver. Installing Python dependencies alone does not provide NVDEC hardware or a GPU driver.
3. For local comma-four recordings, place a complete segment under `data/<segment-id>/` with `fcamera.hevc`, `ecamera.hevc` and `rlog.zst`. The localizer currently requires comma four (`mici`), full-HEVC camera streams, complete frame indices and `gpsLocationExternal` in the log. Run `uv run python -m localizer.run --segment <segment-id>` first.
4. The localizer writes `data/<segment-id>/frame_info.safetensors` (camera metadata and the video indices) and `localizer.safetensors` (localization). Do **not** invent an index from frame numbers: the decoder needs actual HEVC byte offsets and I-frame markers.

See [localizer instructions](../localizer/README.md) and [supervised training](../supervised/README.md) for the full pipeline.

## Functions

| Function | Input | Output / role |
| --- | --- | --- |
| `decode_frames(path, index, wanted, gpu_id=0, fs=None)` | HEVC filename, corresponding `frame_info["fcamera/index"]` or `frame_info["ecamera/index"]`, set of zero-based frame numbers | `{frame_number: RGB numpy array}`; reads only the required video prefix using NVDEC |
| `calibration_view_eulers(device_from_calib_euler, augment_from_calib=None)` | Camera calibration Euler angles and optional rotation/augmentation matrix | View-space Euler angles usable for camera warps |
| `calibration_warp_matrix(source_intrinsics, target_intrinsics, view_euler)` | Source and destination 3x3 camera intrinsics plus view Euler angles | OpenCV 3x3 perspective warp matrix |

### Decode two frames from a prepared local segment

Replace the segment ID with a segment for which the localizer has already produced `frame_info.safetensors`. The example requests frames 0 and 5, so the video must contain at least six frames.

```bash
uv run python - <<'PY'
from pathlib import Path
from safetensors.numpy import load_file
from helpers.video_helpers import decode_frames

segment = Path('data/<segment-id>')
frame_info = load_file(str(segment / 'frame_info.safetensors'))
frames = decode_frames(
    segment / 'fcamera.hevc',
    frame_info['fcamera/index'],
    wanted={0, 5},
    gpu_id=0,
)
for number, rgb in sorted(frames.items()):
    print(number, rgb.shape, rgb.dtype)  # H x W x 3, uint8
PY
```

`wanted` is a set of **zero-based decoded-frame positions** and must be in `0..len(index)-2`; an empty set returns `{}`. `fs` is optional: without it `path` is opened locally; with it the object must provide `cat_file(path, end=byte_count)` (for example an `fsspec` filesystem). An invalid range raises `IndexError`; missing decoded frames raise `RuntimeError`.

### Build a camera perspective transform

```bash
uv run python - <<'PY'
import numpy as np
from openpilot.common.transformations.model import medmodel_intrinsics
from helpers.video_helpers import calibration_view_eulers, calibration_warp_matrix

view_euler = calibration_view_eulers(np.zeros(3, dtype=np.float64))
warp = calibration_warp_matrix(medmodel_intrinsics, medmodel_intrinsics, view_euler)
print(warp.shape)  # (3, 3); pass to cv2.warpPerspective
PY
```

For real driving data, **use the calibration and intrinsics for that camera**, as shown in [`supervised/dataset.py`](../supervised/dataset.py) (`_load_images` / `_calibration_augmentations`); zeros above are a shape/API demonstration, not a valid measured calibration.

## Troubleshooting and agent guidance

- `ModuleNotFoundError`: run `uv sync`, and execute `uv run python ...` **from the repository root**. Read `pyproject.toml` before changing dependency versions.
- CUDA/NVDEC or codec errors: check the NVIDIA driver, GPU decode capability and the source HEVC stream. A CPU-only Python installation does not make `decode_frames` work.
- Wrong camera or video/index pair: use the index belonging to the **same** `.hevc` file and segment; never silently substitute `qcamera` or inferred timestamps.
- Missing `frame_info.safetensors`: run the localizer on a **supported, complete** local segment; consult its errors rather than fabricating metadata.
- For agents (Codex/Claude): inspect the current source and these docs before calling helpers. Do not download private recordings, start costly GPU jobs, publish models, modify devices or commit credentials unless explicitly authorized.

## Relationship to deployment scripts

The shared [`openpilot_scripts`](https://github.com/cristianku/openpilot_scripts) repository contains the [`OP-update-distilled`](https://github.com/cristianku/openpilot_scripts/blob/main/OP-update-distilled/SKILL.md) skill. It **merges upstream Sunnypilot into `cristianku/sunnypilot:distilled` and mirrors the identical branch to `cristianku/openpilot:distilled`**. It does **not** run these helpers, train a model or publish model weights. This repository is the training/experimentation source, not the comma installation branch.
