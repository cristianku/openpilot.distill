"""Train Supercombo on logged comma1M segments."""

import argparse
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import wandb

from .checkpoints import atomic_torch_save, model_weights, stable_weights, torch_load
from .dataset import Comma1MDataset, DatasetConfig, prefill_supercombo_batch
from .model import Supercombo
from .model_for_inference import InputQueues, ORTSupercomboForInference


ROOT = Path(__file__).resolve().parents[1]
TARGET_OUTPUTS = ("wide_from_device_euler", "pose", "plan", "action")
LOSS_WEIGHTS = {"gt/imgs": 100.0, "distill/lead": 0.05, "distill/hidden_state": 10.0}


def loss_fn(predictions, targets, teacher_targets):
    predicted_images = predictions["imgs"][:, -1].float() / 127.5 - 1
    target_images = targets["imgs"].permute(0, 3, 1, 2).float() / 127.5 - 1
    target_losses = {"imgs": F.mse_loss(predicted_images, target_images)}
    target_losses |= {
        name: F.mse_loss(
            predictions[name][..., : targets[name].shape[-1]].float(),
            targets[name].float(),
        )
        for name in TARGET_OUTPUTS
    }
    distill_losses = {
        name: F.mse_loss(
            (predictions["hidden_state"].mean(-2) if name == "hidden_state" else predictions[name]).float(),
            target.float(),
        )
        for name, target in teacher_targets.items()
    }
    target_loss = sum(LOSS_WEIGHTS.get(f"gt/{name}", 1.0) * value for name, value in target_losses.items())
    distill_loss = sum(LOSS_WEIGHTS.get(f"distill/{name}", 1.0) * value for name, value in distill_losses.items())
    return target_loss + distill_loss, {
        "gt": target_loss,
        "distill": distill_loss,
        **{f"gt/{name}": value for name, value in target_losses.items()},
        **{f"distill/{name}": value for name, value in distill_losses.items()},
    }


def _identity(value):
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", default=str(ROOT), help="comma1M dataset root")
    parser.add_argument(
        "--teacher",
        default=str(ROOT / "models/big_driving_supercombo.onnx"),
        help="teacher Supercombo ONNX model",
    )
    parser.add_argument("--steps", type=int, default=1, help="total training steps for this run")
    parser.add_argument("--mode", choices=("fresh", "continue", "resume"), default="fresh",
                        help="fresh: new model; continue: stable HF weights; resume: local interrupted run")
    parser.add_argument("--hf-repo", default=os.environ.get("HF_REPO"),
                        help="Hugging Face model repository containing training/stable.json (continue only)")
    parser.add_argument("--hf-revision", default="main", help="Hugging Face revision for stable manifest")
    parser.add_argument("--checkpoint-interval", type=int, default=500,
                        help="write resumable state every N steps (in addition to final state)")
    parser.add_argument("--batch-size", type=int, default=8, help="samples per training batch")
    parser.add_argument("--workers", type=int, default=0, help="data-loader workers")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "outputs/supervised/supercombo.pt",
        help="checkpoint output path",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16"),
        default="auto",
        help="autocast dtype; auto picks fp16 on GPUs without native bf16 (e.g. Tesla V100)",
    )
    args = parser.parse_args()
    if args.steps < 1 or args.checkpoint_interval < 1:
        parser.error("--steps and --checkpoint-interval must be positive")
    if args.mode == "continue" and not args.hf_repo:
        parser.error("continue requires --hf-repo or HF_REPO")
    checkpoint_path = args.output.with_suffix(".training.pt")
    if args.mode == "resume":
        if not checkpoint_path.is_file():
            parser.error(f"no resumable training checkpoint: {checkpoint_path}")
    elif checkpoint_path.exists() or args.output.exists():
        parser.error(f"output already exists; choose a new --output (found {checkpoint_path} or {args.output})")

    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = rank == 0
    if distributed:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    random.seed(rank)
    np.random.seed(rank)
    torch.manual_seed(rank)
    torch.set_float32_matmul_precision("high")

    # V100 (sm_70) has no native bf16: bf16 autocast falls back to slow emulation.
    # fp16 needs a GradScaler to keep small gradients from underflowing.
    precision = args.precision
    if precision == "auto":
        if torch.cuda.is_bf16_supported(including_emulation=False):
            precision = "bf16"
        else:
            precision = "fp16"
    if precision == "bf16":
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))

    train_dataset = Comma1MDataset(
        DatasetConfig(args.dataset, batch_size=args.batch_size),
        local_rank=local_rank,
        global_rank=rank,
        global_world_size=world_size,
    )
    batch_iterator = iter(
        DataLoader(
            train_dataset,
            batch_size=None,
            num_workers=args.workers,
            collate_fn=_identity,
            multiprocessing_context="spawn" if args.workers else None,
            persistent_workers=bool(args.workers),
        )
    )

    raw_model = Supercombo().to(device)
    model = DDP(raw_model, device_ids=[local_rank]) if distributed else raw_model
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-3,
        fused=True,
    )

    starting_step = 0
    if args.mode == "continue":
        saved, origin = stable_weights(args.hf_repo, "supercombo", revision=args.hf_revision)
        raw_model.load_state_dict(model_weights(saved), strict=True)
        if is_main:
            print(f"continue from verified stable Hugging Face weights: {origin}")
    elif args.mode == "resume":
        saved = torch_load(checkpoint_path)
        if saved.get("task") != "supercombo" or saved.get("format_version") != 1:
            parser.error("invalid supercombo training checkpoint")
        if saved.get("status") == "completed":
            parser.error("training already completed; use --mode continue for a new run")
        if saved.get("status") != "in_progress" or saved.get("total_steps") != args.steps:
            parser.error("resume requires an interrupted run with the same --steps")
        if saved.get("teacher") != args.teacher or saved.get("dataset") != args.dataset:
            parser.error("resume requires the same --teacher and --dataset")
        starting_step = saved["step"]
        if not isinstance(starting_step, int) or not 0 <= starting_step < args.steps:
            parser.error("invalid resume step")
        raw_model.load_state_dict(model_weights(saved), strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        if is_main:
            print(f"resume local training from step {starting_step}/{args.steps}")

    if is_main:
        print(f"student: {sum(p.numel() for p in raw_model.parameters()) / 1e6:.1f}M parameters")
        print(f"precision: {precision}")
        args.output.parent.mkdir(parents=True, exist_ok=True)

    def save_training_state(step: int, status: str) -> None:
        if is_main:
            atomic_torch_save({
                "format_version": 1,
                "task": "supercombo",
                "status": status,
                "step": step,
                "total_steps": args.steps,
                "dataset": args.dataset,
                "teacher": args.teacher,
                "state_dict": {key: value.detach().cpu() for key, value in raw_model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
            }, checkpoint_path)

    if is_main and args.mode != "resume":
        save_training_state(0, "in_progress")

    teacher = ORTSupercomboForInference.from_supercombo(
        args.teacher,
        providers=[("CUDAExecutionProvider", {"device_id": local_rank}), "CPUExecutionProvider"],
    )
    if is_main:
        print(f"teacher: {teacher.vision_session.get_providers()[0]}")
    run = (
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "openpilot.distill.supervised"),
            dir=args.output.parent,
            config={**vars(args), "world_size": world_size, "loss_weights": LOSS_WEIGHTS},
        )
        if is_main
        else None
    )

    for step in range(starting_step + 1, args.steps + 1):
        step_start = time.perf_counter()
        inputs, targets = next(batch_iterator)
        while not len(inputs["img"]):
            inputs, targets = next(batch_iterator)

        queues = InputQueues.from_model(teacher, batch_size=len(inputs["img"]))
        teacher_predictions = prefill_supercombo_batch(teacher, queues, inputs)
        teacher_targets = {name: torch.from_numpy(value).to(device) for name, value in teacher_predictions.items()}
        inputs = {name: torch.from_numpy(value).to(device) for name, value in inputs.items()}
        targets = {name: torch.from_numpy(value).to(device) for name, value in targets.items()}

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=autocast_dtype):
            predictions = model(inputs)
            loss, loss_metrics = loss_fn(predictions, targets, teacher_targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if is_main:
            torch.cuda.synchronize()
            metrics = {
                "loss": loss.item(),
                **{name: value.item() for name, value in loss_metrics.items()},
                "grad_norm": grad_norm.item(),
                "step_time": time.perf_counter() - step_start,
            }
            if step == 1 or step % 100 == 0:
                target_images = targets["imgs"][0].cpu().numpy()
                predicted_images = (
                    predictions["imgs"][0, -1]
                    .detach()
                    .clamp(0, 255)
                    .to(torch.uint8)
                    .permute(1, 2, 0)
                    .contiguous()
                    .cpu()
                    .numpy()
                )
                metrics |= {
                    "imgs/gt_narrow": wandb.Image(np.ascontiguousarray(target_images[..., :3])),
                    "imgs/pred_narrow": wandb.Image(np.ascontiguousarray(predicted_images[..., :3])),
                    "imgs/gt_wide": wandb.Image(np.ascontiguousarray(target_images[..., 3:])),
                    "imgs/pred_wide": wandb.Image(np.ascontiguousarray(predicted_images[..., 3:])),
                }
            print(
                f"step {step}/{args.steps}: loss={metrics['loss']:.6g} "
                f"gt={metrics['gt']:.6g} distill={metrics['distill']:.6g} "
                f"time={metrics['step_time']:.2f}s"
            )
            run.log(metrics, step=step)
            if step % args.checkpoint_interval == 0:
                save_training_state(step, "in_progress")

    if is_main:
        atomic_torch_save(
            {name: value.detach().cpu() for name, value in raw_model.state_dict().items()},
            args.output,
        )
        save_training_state(args.steps, "completed")
        run.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
