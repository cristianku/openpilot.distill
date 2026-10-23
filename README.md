# 🌰 openpilot.distill

You too, can train state-of-the-art autonomous driving models!

With a comma four, and a few logged segments, distill the openpilot driving model into a new model and make it drive your own driving style.

Drop your segments video files and rlogs into the `data/` folder.

From your own comma, `helpers/fetch_comma_segments.sh` downloads them (`--help` for the options). In **automatic** mode it only takes dashcam routes of at least **300 seconds** measured on the complete route (including the segments that will be discarded), then excludes the first two and last two *available* segments and **every remaining segment containing reverse gear** in its full `rlog.zst`. Missing/unreadable logs are excluded, rather than silently accepted. Manual selection by route/`--segments` is unchanged. This fetcher and its log inspector are kept identical to `sunnypilot.speedlimits/main`. Tests: `uv run python -m unittest helpers/test_comma_fetch_log.py`. The raw files (`rlog.zst`, `fcamera.hevc`, `ecamera.hevc`) are stored once in a shared folder, `COMMA_VIDEOS_DIR` (default `comma-videos` next to this repo), one `<route>--<segment>/` folder per segment as on the comma, so other projects can use the same files. `data/<segment>/` only holds absolute links to them, plus the localizer outputs.

You don't have a comma four, no problem! You can still download videos and labels from https://huggingface.co/datasets/commaai/comma1M and train your own driving models.

## Install everything (Linux server with an NVIDIA GPU)

One command, from a fresh clone to the control site running:

```bash
git clone --recurse-submodules https://github.com/cristianku/openpilot.distill.git
cd openpilot.distill
sudo ./install.sh            # --dry-run to only see what it would do
```

It installs the missing system packages and [`uv`](https://docs.astral.sh/uv/), picks the right PyTorch build for the GPU (a Tesla V100 gets the CUDA 12 `v100` extra), creates the Python environment and starts the control site as the `distill-web` systemd service (port 8766, started at every boot). It is safe to run again. At the end it prints the address to open and, until the administrator exists, the setup code: the first visit creates the administrator (name and password); Hugging Face, Weights & Biases and the comma address are then set from the Settings page.

## Get started and find the helpers

Use [`uv`](https://docs.astral.sh/uv/) and Python **3.12.3 or newer, but below 3.13**, as specified in [`pyproject.toml`](pyproject.toml). From a clean checkout:

```bash
git clone --recurse-submodules https://github.com/cristianku/openpilot.distill.git
cd openpilot.distill
uv sync
```

For a **supported comma four** segment, prepare `data/<segment-id>/fcamera.hevc`, `data/<segment-id>/ecamera.hevc` and `data/<segment-id>/rlog.zst`, then:

```bash
uv run python -m localizer.run --segment <segment-id>
```

This produces `frame_info.safetensors` (including the video byte-offset indices) and `localizer.safetensors` in the segment folder. Local localization currently requires a complete comma-four (`mici`) full-HEVC log with external GNSS; other device/log formats are not silently supported.

| Where | What to run or read |
| --- | --- |
| [`localizer/README.md`](localizer/README.md) | Prepare localization and frame metadata for a recorded segment. |
| [`helpers/README.md`](helpers/README.md) | **Python video/calibration helpers**, function signatures, NVDEC requirements, runnable examples and troubleshooting. These are imports, not executable scripts. |
| [`supervised/README.md`](supervised/README.md) | Download the ONNX teacher model and start supervised training with `uv run torchrun ...`. |
| [`rl/README.md`](rl/README.md) | Start the world-model server and optional simulator/post-training workflows. |
| [`app/android/`](app/android/README.md) | **Distill for Android**: the companion app of the control site: native screens to follow the server and start trainings, evaluations and installations on the comma, plus the whole site. Built on the server (`helpers/build_android.sh`, then Settings › Distill for Android on the phone) or by GitHub Actions, in the [releases](https://github.com/cristianku/openpilot.distill/releases). |
| [`perception_repo/`](https://github.com/cristianku/openpilot.perception) | Submodule [openpilot.perception](https://github.com/cristianku/openpilot.perception), imported as `perception`: independent 3D object perception, with the car's radar (logged in `carState`) as a depth label when the car port provides it, vision-only otherwise. `uv run python -m perception.dataset data/<segment>/rlog.zst` shows what a segment offers. |

A clone made without `--recurse-submodules` gets `perception_repo/` from `sudo ./install.sh`, or by hand with `git submodule update --init`; the code update from the site keeps it in step.

The `supervised/dataset.py` loader already imports the functions in `helpers/video_helpers.py`; you do **not** run the helper module as a separate training stage. Training/video decoding requires a compatible NVIDIA GPU/driver (NVDEC), not just a Python virtual environment. For external data such as comma1M, follow the relevant dataset preparation steps; raw video alone is insufficient for the supervised loader.

### Training repo vs. installable Sunnypilot branch

This repository contains Distill training and experiment code. The separate [`openpilot_scripts`](https://github.com/cristianku/openpilot_scripts) automation repository maintains the [`OP-update-distilled`](https://github.com/cristianku/openpilot_scripts/blob/main/OP-update-distilled/SKILL.md) skill: it syncs `cristianku/sunnypilot:distilled` from the official Sunnypilot master, then mirrors the same commit to `cristianku/openpilot:distilled` for installation. **That skill does not train or export this repository's weights.** Verify the source branch, installable mirror and model publication separately before claiming a model is available on-device.

## Step 1: Run a simple localization pipeline

Run the localizer.

<img width="512" height="256" alt="2333e255f1de1fe83ea5975b24a3cc67_frame_700" src="https://github.com/user-attachments/assets/395683ff-7d96-4e07-bf6e-12a3969b487f" />

## Step 2: Distill the model

Train a smaller driving model using the localized targets, and distillation targets from the openpilot model.

<img width="1366" height="777" alt="Screenshot from 2026-08-31 17-28-31" src="https://github.com/user-attachments/assets/d53a1e0e-3b79-4923-a777-9368c34234ed" />

https://api.wandb.ai/links/yassiney/qme6ln4z

## Step 3: Fine tune the driving model in the simulator

DAgger style post-training in a learned world model as a simulator.

https://github.com/user-attachments/assets/45ba3950-9a90-45c2-af15-f7e55f87785e

Have fun!
