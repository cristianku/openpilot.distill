# Supervised learning

`wandb login` or set `WANDB_MODE=offline`.

## Download the teacher model

The default teacher is **MD v3**, the big driving model that openpilot master runs since commit `1e219d778` ("MD v3"). It is published on Hugging Face in [`commaai/openpilot_driving_models`](https://huggingface.co/commaai/openpilot_driving_models) as `1365126a-e71e-4778-be00-8ab92dc82c09/12864`. Download it at the pinned revision, so the file is always the same:

```bash
HF_REVISION=8dbc57a6480111bd58487ecc248e30d50fb2f8dd
mkdir -p models
curl --fail --location --continue-at - \
  "https://huggingface.co/commaai/openpilot_driving_models/resolve/${HF_REVISION}/1365126a-e71e-4778-be00-8ab92dc82c09/12864/big_driving_supercombo.onnx" \
  --output models/big_driving_supercombo_1365126a_12864.onnx
sha256sum models/big_driving_supercombo_1365126a_12864.onnx
# 1cfd2c45353d876b5b257b24d77537de58ab90a44ae595f15b1f64b6ea3fd770
```

The old teacher of upstream openpilot.distill (openpilot `084747c75`) is still supported, to resume the runs started with it and for comparisons:

```bash
OPENPILOT_SHA=084747c75d2cbd23af65ab7a9e770bbd7b98bac9
mkdir -p models
curl --fail --location --continue-at - \
  "https://media.githubusercontent.com/media/commaai/openpilot/${OPENPILOT_SHA}/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx" \
  --output models/big_driving_supercombo.onnx
sha256sum models/big_driving_supercombo.onnx
# a501760a9d1d5fef0eab2b8c5d122d06124fc26dc8e0782e0aa94b82a208f0ff
```

`helpers/train_supervised.sh --teacher=md-v3|084747c7` (default `md-v3`) does the download and the sha256 check by itself.

### How the queue-format teacher is used

The Hugging Face models (openpilot PR #38916) keep their history inside the ONNX: modeld calls them at 20 Hz with one new frame per camera and feeds every `next_state_x` output back as `state_x`. `ORTSupercomboForInference.from_supercombo` recognises them by the `state_feat_q` input and `supervised/onnx_helpers.py` (`split_queue_supercombo` + `make_dense_queue`) cuts them, by the structure of the graph, into the same two graphs as the old teacher:

- vision: `img`, `big_img` `[batch, 12, 128, 256]` uint8 (frames t-4 and t) → `features [batch, 32, 512]`;
- policies: `features [batch, 9, 32, 512]`, `desire_pulse [batch, 33, 8]` (5 Hz), `traffic_convention`, `action_t` → `outputs [batch, 9, 18452]`, the outputs of all 9 steps at once.

The temporal transformers of these models are causal, so step k of the dense outputs sees steps 0..k only, like the student. The last step is the model as modeld runs it at that frame; the earlier steps see a shorter history than modeld would. Models with other inputs (the N16 nav model: `nav_map`, `navigation` and their queues) are refused, because the student never sees them.

Before training with a new model of that repository, check the cut on CPU (a few minutes, no GPU):

```bash
uv run python -m helpers.check_teacher models/big_driving_supercombo_1365126a_12864.onnx
```

It runs the full model at 20 Hz and compares it, output by output, with the split vision + dense policies at the last 5 Hz step; it also checks the vision features, causality and batching. Exit code 0 means every check passed.

The `distill/hidden_state` loss follows the teacher: with MD v3 it compares the 32 tokens x 512 of every step one by one (weight 1.0), with `084747c7` the average of the student's tokens with the teacher's pooled 512 vector (weight 10.0). The two values are not comparable between runs with different teachers.

## Training

```bash
uv run torchrun --standalone --nproc-per-node=1 -m supervised.train --steps 5000 --batch-size 8 --workers 1 \
  --teacher models/big_driving_supercombo_1365126a_12864.onnx
```


## Training modes and recoverable checkpoints

The supervised trainer supports the same three modes as `sunnypilot.speedlimits`:

- `fresh` (default) creates an independent model using the configured initialization. Select a **new** output path for each run.
- `continue` creates a **new run** initialized from the reviewed, SHA256-verified PyTorch weights referenced by `training/stable.json` in a Hugging Face model repository. It does not restore the previous optimizer.
- `resume` requires the **same output path**, total `--steps`, dataset and teacher as an interrupted run and restores the model and AdamW optimizer from its local `.training.pt` checkpoint. A completed checkpoint is refused; use `continue` to begin a new run. The error message shows the `--teacher` and `--dataset` the run was started with; `helpers/train_supervised.sh --resume` reads them from the checkpoint by itself.
- `continue` may use a different teacher than the run that produced the weights (e.g. MD v3 after `084747c7`): expect `distill/hidden_state` to start high.

```bash
# Completely new run:
uv run torchrun --standalone --nproc-per-node=1 -m supervised.train \
  --mode fresh --steps 5000 --output outputs/supervised/v1.pt

# Fine-tune a reviewed stable release (replace with the real HF model repo):
uv run torchrun --standalone --nproc-per-node=1 -m supervised.train \
  --mode continue --hf-repo OWNER/MODEL_REPO --steps 5000 \
  --output outputs/supervised/v2.pt

# Recover the last local in-progress checkpoint:
uv run torchrun --standalone --nproc-per-node=1 -m supervised.train \
  --mode resume --steps 5000 --output outputs/supervised/v2.pt
```

Every `--checkpoint-interval` steps (default 500) and on successful completion, the trainer atomically saves `<output>.training.pt` with optimizer state and a `status` field. The final `<output>.pt` remains the model-only state dict. Resume restarts from the last saved step; an iterable dataset's exact order/RNG and partial steps are **not** restored, so this is not bit-for-bit deterministic.

### Publishing community fine-tuning weights

`helpers/publish_training_weights.py` stages independently versioned, model-only `.pt` weights. **Dry run is the default**, and neither it nor training uploads private recordings or a full optimizer checkpoint:

```bash
uv run python -m helpers.publish_training_weights --repo OWNER/MODEL_REPO \
  --release v1 --model supercombo=outputs/supervised/v1.pt

# After validating the model and checking its redistribution rights:
uv run python -m helpers.publish_training_weights --repo OWNER/MODEL_REPO \
  --release v1 --model supercombo=outputs/supervised/v1.pt --upload --stable
```

The optional `--stable` updates `training/stable.json` in the same Hugging Face commit as the new immutable release, retaining other model entries. `continue` requires that manifest to exist; it never silently falls back to random weights. Uploading and advancing `stable` is a separate, explicit action following testing. Check weights, training data rights and provenance before publishing.

### From the helpers

`helpers/publish_model.sh` publishes the `supercombo.pt` as `training/releases/<name>/supercombo.pt` in the same commit as the ONNX/compiled model (`--stage-dir` of the script above); `--stable` also moves `training/stable.json` to it. `helpers/train_supervised.sh --continue` then starts a new run from those weights, and `helpers/train_supervised.sh --resume=outputs/supervised/<run>` restarts an interrupted run from its `supercombo.training.pt`.
