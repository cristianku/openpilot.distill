"""avg miss of one model on the held-out segments (the ones training never sees).

Usage: python -m evaluation.holdout --model <spec> [--segments heldout|all|<seg>,<seg>,...]
         [--rate 5|20] [--horizon 10] [--device cuda|cpu] [--force] [--out file.json]

heldout = the segments supervised/dataset.py keeps out of training (Comma1MDataset(val=True), the
same code, not a copy of the rule). On each segment the model runs from the first frame; the frames
evaluated are those after the warm-up (EVAL_WARMUP_FRAMES) whose whole horizon is inside the
segment, at 5 Hz (frames 0, 4, 8, ...) or 20 Hz. Same frames for every model.

Output: outputs/eval/<model key>.json (heldout, --rate 5, --horizon 10), otherwise
outputs/eval/<model key>__<segments>_r<rate>_h<horizon>.json. When that file already holds the
result for the same model file (sha256), segments, settings and code version it is not recomputed
(--force recomputes). Lines for the web site on stdout:
  PROGRESS <done>/<total>      frames processed over all segments
  OUTPUT_JSON=<absolute path>  at the end
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .metrics import (
    DEFAULT_HORIZON_S,
    EVAL_CODE_VERSION,
    average,
    frame_misses,
    full_horizon_frames,
    plan_point_mask,
    rounded,
)
from .runners import ACTION_T, BLOCK_FRAMES, EVAL_WARMUP_FRAMES, TRAFFIC_RIGHT_HAND, build_runner
from .segment import CAMERA_FPS, ModelInputPacker, Segment, load_segment, nvdec_available, stream_frames
from .specs import parse_model_spec, resolve_model, validate_segment_name

ROOT = Path(__file__).resolve().parents[1]
PROGRESS_EVERY_FRAMES = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="stock, teacher, logged, run:<folder> or hf:<name>")
    parser.add_argument("--segments", default="heldout", help="heldout (default), all or a comma separated list")
    parser.add_argument("--rate", type=int, choices=(5, 20), default=5, help="frames evaluated per second (default 5)")
    parser.add_argument("--horizon", type=float, default=DEFAULT_HORIZON_S, help="seconds of plan in the miss")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--force", action="store_true", help="computes again even when the result already exists")
    parser.add_argument("--out", type=Path, help="json to write (default outputs/eval/<model>.json)")
    return parser.parse_args()


def _exit_on_sigterm(signal_number, frame) -> None:
    raise SystemExit(143)


def heldout_segments() -> list[str]:
    """The validation split of supervised/dataset.py: the same Comma1MDataset code decides."""
    from supervised.dataset import Comma1MDataset, DatasetConfig

    return list(Comma1MDataset(DatasetConfig(dataset_path=str(ROOT)), val=True).segments)


def all_segments() -> list[str]:
    # Same glob as Comma1MDataset: every data/<segment>/fcamera.hevc.
    names = []
    for video in sorted((ROOT / "data").glob("*/fcamera.hevc")):
        names.append(video.parent.name)
    return names


def select_segments(selection: str) -> tuple[list[str], str]:
    """(segment names, tag for the output file name)."""
    if selection == "heldout":
        return heldout_segments(), "heldout"
    if selection == "all":
        return all_segments(), "all"
    names = []
    for name in selection.split(","):
        name = name.strip()
        if not name:
            continue
        try:
            validate_segment_name(name)
        except ValueError as error:
            raise SystemExit(f"ERROR: {error}") from None
        if not (ROOT / "data" / name).is_dir():
            raise SystemExit(f"ERROR: segment {name} not found in data/")
        if name not in names:
            names.append(name)
    if not names:
        raise SystemExit("ERROR: --segments is empty")
    digest = hashlib.sha256(",".join(names).encode()).hexdigest()[:8]
    return names, f"lista-{digest}"


def evaluated_frames(segment: Segment, rate: int, horizon_s: float) -> np.ndarray:
    """Frame indexes that count: after the warm-up, whole horizon inside the segment, on the rate grid."""
    step = CAMERA_FPS // rate
    usable_end = (segment.frame_count // BLOCK_FRAMES) * BLOCK_FRAMES
    full = full_horizon_frames(segment.frame_t, segment.frame_t[-1], horizon_s)
    frames = []
    for frame_index in range(EVAL_WARMUP_FRAMES, usable_end, step):
        if full[frame_index]:
            frames.append(frame_index)
    return np.asarray(frames, dtype=np.int64)


def decode_end(frames: np.ndarray) -> int:
    """First frame after the last block that holds an evaluated frame."""
    if len(frames) == 0:
        return 0
    return ((int(frames[-1]) // BLOCK_FRAMES) + 1) * BLOCK_FRAMES


def cached_result(path: Path, sha256: str, segments: list[str], settings: dict) -> dict | None:
    if not path.exists():
        return None
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if saved.get("code_version") != EVAL_CODE_VERSION:
        return None
    if saved.get("model", {}).get("sha256") != sha256:
        return None
    if saved.get("segments") != segments:
        return None
    if saved.get("settings") != settings:
        return None
    return saved


def evaluate_segment(runner, segment: Segment, frames: np.ndarray, horizon_s: float, use_nvdec: bool, progress) -> dict:
    """Run the model over one segment; per-frame misses of the evaluated frames. progress(n): n frames done."""
    wanted = set()
    for frame_index in frames:
        wanted.add(int(frame_index))
    end_frame = decode_end(frames)
    predicted = {}
    packer = ModelInputPacker(segment.rpy)
    runner.reset(segment)
    block_first = 0
    narrow_packs = []
    wide_packs = []
    for frame_index, narrow_rgb, wide_rgb in stream_frames(segment, 0, end_frame, use_nvdec):
        if not narrow_packs:
            block_first = frame_index
        narrow_pack, wide_pack = packer.pack(narrow_rgb, wide_rgb)
        narrow_packs.append(narrow_pack)
        wide_packs.append(wide_pack)
        if len(narrow_packs) < BLOCK_FRAMES:
            continue
        plans = runner.step_block(block_first, np.stack(narrow_packs), np.stack(wide_packs))
        for lane in range(runner.lanes):
            if block_first + lane in wanted:
                predicted[block_first + lane] = plans[lane, :, 0:3]
        narrow_packs = []
        wide_packs = []
        progress(BLOCK_FRAMES)

    ordered_plans = []
    for frame_index in frames:
        ordered_plans.append(predicted[int(frame_index)])
    plans = np.stack(ordered_plans)
    masks = plan_point_mask(segment.frame_t[frames], segment.frame_t[-1], horizon_s)
    misses, laterals = frame_misses(plans, segment.ground_truth[frames], masks)
    return {"misses": misses, "laterals": laterals}


class Progress:
    """PROGRESS <done>/<total> lines, every PROGRESS_EVERY_FRAMES frames and at the end."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.done = 0
        self.printed = 0

    def __call__(self, frames: int) -> None:
        self.done += frames
        if self.done - self.printed >= PROGRESS_EVERY_FRAMES or self.done >= self.total:
            print(f"PROGRESS {self.done}/{self.total}", flush=True)
            self.printed = self.done


def main() -> None:
    args = parse_args()
    try:
        spec = parse_model_spec(args.model)
    except ValueError as error:
        raise SystemExit(f"ERROR: {error}") from None
    if args.horizon <= 0 or args.horizon > 10.0:
        raise SystemExit("ERROR: --horizon between 0 and 10 seconds (the plan reaches 10 s)")
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    started = time.time()

    segments, selection_tag = select_segments(args.segments)
    settings = {"rate_hz": args.rate, "horizon_s": args.horizon, "warmup_frames": EVAL_WARMUP_FRAMES}
    if args.out is not None:
        output_path = args.out.resolve()
    elif selection_tag == "heldout" and args.rate == 5 and args.horizon == DEFAULT_HORIZON_S:
        output_path = ROOT / "outputs" / "eval" / f"{spec.key}.json"
    else:
        output_path = ROOT / "outputs" / "eval" / f"{spec.key}__{selection_tag}_r{args.rate}_h{args.horizon:g}.json"

    model = resolve_model(spec, ROOT)
    print(f"model: {spec.label} = {model.path} (sha256 {model.sha256[:16]})", flush=True)
    print(f"segments ({selection_tag}): {len(segments)}", flush=True)
    if not args.force:
        saved = cached_result(output_path, model.sha256, segments, settings)
        if saved is not None:
            saved_miss = saved["overall"]["avg_miss_m"]
            print(f"result already computed: avg miss {saved_miss} m (--force to compute it again)", flush=True)
            print(f"OUTPUT_JSON={output_path}", flush=True)
            return

    # Frames to evaluate and to decode, before loading the model: PROGRESS needs the total.
    plan = []
    skipped = []
    total_frames = 0
    for name in segments:
        try:
            segment = load_segment(ROOT, name)
        except (ValueError, OSError) as error:
            print(f"segment {name} skipped: {error}", flush=True)
            skipped.append({"segment": name, "error": str(error)})
            continue
        frames = evaluated_frames(segment, args.rate, args.horizon)
        if len(frames) == 0:
            print(f"segment {name} skipped: too short for a {args.horizon:g} s horizon", flush=True)
            skipped.append({"segment": name, "error": "too short"})
            continue
        plan.append(name)
        total_frames += decode_end(frames)

    lanes = 1
    if args.rate == 20:
        lanes = BLOCK_FRAMES
    runner = build_runner(model, args.device, lanes)
    if runner.warmup_frames > EVAL_WARMUP_FRAMES:
        print(f"WARNING: {runner.name} needs {runner.warmup_frames} warm-up frames", flush=True)
    use_nvdec = args.device == "cuda" and nvdec_available()
    decoder_name = "ffmpeg"
    if use_nvdec:
        decoder_name = "nvdec"
    progress = Progress(total_frames)

    per_segment = []
    all_misses = []
    all_laterals = []
    for name in plan:
        segment_started = time.time()
        segment = load_segment(ROOT, name)
        frames = evaluated_frames(segment, args.rate, args.horizon)
        result = evaluate_segment(runner, segment, frames, args.horizon, use_nvdec, progress)
        segment_miss = average(result["misses"])
        segment_lateral = average(result["laterals"])
        all_misses.append(result["misses"])
        all_laterals.append(result["laterals"])
        elapsed = time.time() - segment_started
        per_segment.append({
            "segment": name,
            "frames": int(np.isfinite(result["misses"]).sum()),
            "avg_miss_m": rounded(segment_miss),
            "avg_lateral_miss_m": rounded(segment_lateral),
            "elapsed_s": round(elapsed, 1),
        })
        print(
            f"segment {name}: avg miss {_metres(segment_miss)}, lateral {_metres(segment_lateral)}, "
            f"{len(frames)} frames, {elapsed:.1f} s",
            flush=True,
        )

    overall_miss = None
    overall_lateral = None
    evaluated = 0
    if all_misses:
        overall_miss = average(np.concatenate(all_misses))
        overall_lateral = average(np.concatenate(all_laterals))
        evaluated = int(np.isfinite(np.concatenate(all_misses)).sum())
    segment_means = []
    for record in per_segment:
        if record["avg_miss_m"] is not None:
            segment_means.append(record["avg_miss_m"])
    elapsed = time.time() - started
    path = None
    if model.path is not None:
        path = str(model.path)
    result = {
        "schema": 1,
        "kind": "holdout",
        "code_version": EVAL_CODE_VERSION,
        "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": {
            "spec": spec.text,
            "key": spec.key,
            "label": spec.label,
            "format": model.format,
            "path": path,
            "sha256": model.sha256,
        },
        "selection": selection_tag,
        "segments": segments,
        "settings": settings,
        "inputs": {
            "action_t": list(ACTION_T),
            "traffic_convention": list(TRAFFIC_RIGHT_HAND),
            "desire": 0,
            "decoder": decoder_name,
            "device": args.device,
        },
        "per_segment": per_segment,
        "skipped": skipped,
        "overall": {
            "segments": len(per_segment),
            "frames": evaluated,
            "avg_miss_m": rounded(overall_miss),
            "avg_lateral_miss_m": rounded(overall_lateral),
            "mean_of_segment_avg_miss_m": rounded(average(segment_means)),
        },
        "elapsed_s": round(elapsed, 1),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.name}.tmp")
    temporary.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(
        f"total: avg miss {_metres(overall_miss)}, lateral {_metres(overall_lateral)} over {len(per_segment)} "
        f"segments ({evaluated} frames) in {elapsed:.0f} s",
        flush=True,
    )
    print(f"OUTPUT_JSON={output_path}", flush=True)


def _metres(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.2f} m"


if __name__ == "__main__":
    main()
