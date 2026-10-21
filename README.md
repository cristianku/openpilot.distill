# 🌰 openpilot.distill

You too, can train state-of-the-art autonomous driving models!

With a comma four, and a few logged segments, distill the openpilot driving model into a new model and make it drive your own driving style.

Drop your segments video files and rlogs into the `data/` folder.

From your own comma, `helpers/fetch_comma_segments.sh` downloads them (`--help` for the options). In **automatic** mode it only takes dashcam routes of at least **300 seconds** measured on the complete route (including the segments that will be discarded), then excludes the first two and last two *available* segments and **every remaining segment containing reverse gear** in its full `rlog.zst`. Missing/unreadable logs are excluded, rather than silently accepted. Manual selection by route/`--segments` is unchanged. This fetcher and its log inspector are kept identical to `sunnypilot.speedlimits/main`. Tests: `uv run python -m unittest helpers/test_comma_fetch_log.py`. The raw files (`rlog.zst`, `fcamera.hevc`, `ecamera.hevc`) are stored once in a shared folder, `COMMA_VIDEOS_DIR` (default `comma-videos` next to this repo), one `<route>--<segment>/` folder per segment as on the comma, so other projects can use the same files. `data/<segment>/` only holds absolute links to them, plus the localizer outputs.

You don't have a comma four, no problem! You can still download videos and labels from https://huggingface.co/datasets/commaai/comma1M and train your own driving models.

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
