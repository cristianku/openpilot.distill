# Agent guide — openpilot.distill

This repository is **training/experimentation code for driving-model distillation**, not the installable Sunnypilot branch or the speed-limit-sign detector. Read the relevant source file and guide before suggesting or running commands.

## Where to start

- [`README.md`](README.md): environment setup, supported local segments and workflow overview.
- [`localizer/README.md`](localizer/README.md): localizer prerequisites and generated `frame_info.safetensors` / `localizer.safetensors`.
- [`helpers/README.md`](helpers/README.md): how to **import and call** `helpers/video_helpers.py`, input shapes, video indices, NVDEC requirements and examples.
- [`supervised/README.md`](supervised/README.md): download the pinned teacher and run supervised training.
- [`rl/README.md`](rl/README.md): optional FastAPI world-model server and RL simulator.
- [`perception_repo/`](https://github.com/cristianku/openpilot.perception) (submodule, imported through the `perception` link): independent 3D perception and real-sensor labels. Its PSA radar reader keeps only frames received on bus 1 (`src 1`); frames openpilot sent (`src 129/193`) come from the vision model and are never labels. See its `docs/psa-radar.md`.

Use `uv sync` with Python `>=3.12.3,<3.13` (see `pyproject.toml`) from the repository root. The helper module is **not** a CLI; don't invent `helpers --help` or a helper executable. `decode_frames` requires a matching full-HEVC video/index pair and an NVIDIA NVDEC-capable GPU. Never fabricate missing timestamps/calibration or assume every comma recording is supported.

## Keep concerns separate

- `cristianku/openpilot.distill` provides training and data tooling; it does not itself publish an installable comma branch.
- [`cristianku/openpilot_scripts`](https://github.com/cristianku/openpilot_scripts) contains [`OP-update-distilled`](https://github.com/cristianku/openpilot_scripts/blob/main/OP-update-distilled/SKILL.md), which merges official Sunnypilot into `cristianku/sunnypilot:distilled` and mirrors that commit into `cristianku/openpilot:distilled`. **It does not train, distill or upload model weights.**
- [`cristianku/sunnypilot.speedlimits`](https://github.com/cristianku/sunnypilot.speedlimits) is a separate speed-sign/speed-limit dataset and model project; its `nnslr` CLI and GPU scripts are not Distill video helpers.

## Safety / evidence

- Inspect current code and documentation before making claims about commands, model formats or supported devices.
- Never start GPU-heavy jobs, download private logs, publish weights, update a comma device, change unrelated repositories or push release branches without the relevant explicit authorization.
- Never commit W&B/Hugging Face credentials or private vehicle recordings; keep secret material outside tracked files.
