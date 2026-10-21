# Supervised learning

`wandb login` or set `WANDB_MODE=offline`.

## Download the teacher model

```bash
OPENPILOT_SHA=084747c75d2cbd23af65ab7a9e770bbd7b98bac9
mkdir -p models
curl --fail --location --continue-at - \
  "https://media.githubusercontent.com/media/commaai/openpilot/${OPENPILOT_SHA}/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx" \
  --output models/big_driving_supercombo.onnx
```

## Training

```bash
uv run torchrun --standalone --nproc-per-node=1 -m supervised.train --steps 5000 --batch-size 8 --workers 1
```


## Training modes and recoverable checkpoints

The supervised trainer supports the same three modes as `sunnypilot.speedlimits`:

- `fresh` (default) creates an independent model using the configured initialization. Select a **new** output path for each run.
- `continue` creates a **new run** initialized from the reviewed, SHA256-verified PyTorch weights referenced by `training/stable.json` in a Hugging Face model repository. It does not restore the previous optimizer.
- `resume` requires the **same output path**, total `--steps`, dataset and teacher as an interrupted run and restores the model and AdamW optimizer from its local `.training.pt` checkpoint. A completed checkpoint is refused; use `continue` to begin a new run.

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
