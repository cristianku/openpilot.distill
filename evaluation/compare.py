"""Comparison video of two models on one recorded segment, like the Route42 posts.

Usage: python -m evaluation.compare --segment <name> --reference <spec> --student <spec>
         [--start s] [--duration s] [--speed 0.5] [--inset|--no-inset] [--show-gt]
         [--out file.mp4] [--json file.json] [--device cuda|cpu]

Two panels stacked on the same frames: the reference on top (white line), the student below
(green line), each with its running "avg miss" on the stretch shown. Specs: evaluation/specs.py.
The models also run on the 5.2 s before --start (warm-up), so their history is full when the clip
begins; --start is moved to 5.2 s when it is earlier.

Output: outputs/compare/<id>.mp4 and <id>.json. Lines for the web site on stdout:
  PROGRESS <done>/<total>      frames processed (warm-up included)
  OUTPUT_MP4=<absolute path>   at the end
  OUTPUT_JSON=<absolute path>  at the end
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from .metrics import DEFAULT_HORIZON_S, EVAL_CODE_VERSION, frame_misses, plan_point_mask, rounded
from .render import GREEN, WHITE, draw_panel, footer_image, inset_scale
from .runners import ACTION_T, BLOCK_FRAMES, EVAL_WARMUP_FRAMES, TRAFFIC_RIGHT_HAND, build_runner
from .segment import CAMERA_FPS, T_IDXS, ModelInputPacker, load_segment, narrow_camera_intrinsics, nvdec_available
from .segment import stream_frames
from .specs import parse_model_spec, resolve_model, validate_segment_name

ROOT = Path(__file__).resolve().parents[1]
SPEEDS = (0.25, 0.5, 1.0)
MAX_DURATION_S = 60.0
PROGRESS_EVERY_FRAMES = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segment", required=True, help="cartella in data/, es. 00000126f84ad127f010")
    parser.add_argument("--reference", default="stock", help="modello in alto, linea bianca (default stock)")
    parser.add_argument("--student", required=True, help="modello in basso, linea verde, es. run:2026-10-21_1544")
    parser.add_argument("--start", type=float, default=10.0, help="inizio del tratto in secondi (default 10)")
    parser.add_argument("--duration", type=float, default=15.0, help="durata del tratto in secondi (default 15)")
    parser.add_argument("--speed", type=float, default=0.5, help="velocita' del video: 0.25, 0.5 o 1 (default 0.5)")
    parser.add_argument("--inset", action=argparse.BooleanOptionalAction, default=True, help="vista dall'alto")
    parser.add_argument("--show-gt", action="store_true", help="disegna anche il percorso reale (giallo)")
    parser.add_argument("--horizon", type=float, default=DEFAULT_HORIZON_S, help="secondi di piano nel miss")
    parser.add_argument("--out", type=Path, help="mp4 da scrivere (default outputs/compare/<id>.mp4)")
    parser.add_argument("--json", type=Path, help="json da scrivere (default accanto al mp4)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def _exit_on_sigterm(signal_number, frame) -> None:
    # The web site's Stop sends SIGTERM: leave through the finally blocks (partial video removed).
    raise SystemExit(143)


def clip_frames(frame_count: int, start_s: float, duration_s: float) -> tuple[int, int, int, int]:
    """(first decoded frame, first clip frame, clip end, last decoded frame + 1); blocks of 4 frames."""
    usable_end = (frame_count // BLOCK_FRAMES) * BLOCK_FRAMES
    start_frame = int(round(start_s * CAMERA_FPS))
    if start_frame < EVAL_WARMUP_FRAMES:
        print(
            f"inizio spostato a {EVAL_WARMUP_FRAMES / CAMERA_FPS:.1f} s: prima i modelli si riscaldano",
            flush=True,
        )
        start_frame = EVAL_WARMUP_FRAMES
    if start_frame >= usable_end:
        raise SystemExit(f"ERRORE: --start oltre la fine del segmento ({usable_end / CAMERA_FPS:.1f} s)")
    clip_end = min(start_frame + int(round(duration_s * CAMERA_FPS)), usable_end)
    decode_start = ((start_frame - EVAL_WARMUP_FRAMES) // BLOCK_FRAMES) * BLOCK_FRAMES
    end_frame = ((clip_end + BLOCK_FRAMES - 1) // BLOCK_FRAMES) * BLOCK_FRAMES
    return decode_start, start_frame, clip_end, end_frame


def start_encoder(path: Path, width: int, height: int, speed: float) -> subprocess.Popen:
    """x264 encoder reading raw BGR frames on stdin. Slow motion = lower input frame rate."""
    command = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-framerate", f"{CAMERA_FPS * speed:g}", "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-r", str(CAMERA_FPS),
        "-f", "mp4", str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


class RunningMiss:
    """Running avg miss of one model over the clip frames seen so far."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def add(self, value: float) -> None:
        if np.isfinite(value):
            self.total += float(value)
            self.count += 1

    @property
    def average(self) -> float | None:
        if self.count == 0:
            return None
        return self.total / self.count


def model_record(model, runner, running_miss: RunningMiss, lateral_values: list[float]) -> dict:
    lateral = np.asarray(lateral_values, dtype=np.float64)
    lateral = lateral[np.isfinite(lateral)]
    lateral_average = None
    if len(lateral):
        lateral_average = float(lateral.mean())
    path = None
    if model.path is not None:
        path = str(model.path)
    return {
        "spec": model.spec.text,
        "key": model.spec.key,
        "label": model.spec.label,
        "format": model.format,
        "path": path,
        "sha256": model.sha256,
        "warmup_frames": runner.warmup_frames,
        "avg_miss_m": rounded(running_miss.average),
        "avg_lateral_miss_m": rounded(lateral_average),
        "frames": running_miss.count,
    }


def main() -> None:
    args = parse_args()
    try:
        validate_segment_name(args.segment)
        reference_spec = parse_model_spec(args.reference)
        student_spec = parse_model_spec(args.student)
    except ValueError as error:
        raise SystemExit(f"ERRORE: {error}") from None
    if args.speed not in SPEEDS:
        raise SystemExit(f"ERRORE: --speed deve essere 0.25, 0.5 o 1, non {args.speed:g}")
    if args.start < 0:
        raise SystemExit("ERRORE: --start deve essere >= 0")
    if args.duration <= 0 or args.duration > MAX_DURATION_S:
        raise SystemExit(f"ERRORE: --duration tra 0 e {MAX_DURATION_S:.0f} secondi")
    if not (ROOT / "data" / args.segment).is_dir():
        raise SystemExit(f"ERRORE: segmento {args.segment} non trovato in data/")
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    started = time.time()

    segment = load_segment(ROOT, args.segment)
    decode_start, start_frame, clip_end, end_frame = clip_frames(segment.frame_count, args.start, args.duration)
    print(
        f"segmento {segment.name} ({segment.duration_s:.1f} s): tratto {start_frame / CAMERA_FPS:.2f}"
        f"-{clip_end / CAMERA_FPS:.2f} s, riscaldamento da {decode_start / CAMERA_FPS:.2f} s",
        flush=True,
    )

    reference = resolve_model(reference_spec, ROOT)
    student = resolve_model(student_spec, ROOT)
    for role, model in (("riferimento", reference), ("studente", student)):
        print(f"{role}: {model.spec.label} = {model.path} (sha256 {model.sha256[:16]})", flush=True)
    reference_runner = build_runner(reference, args.device, BLOCK_FRAMES)
    student_runner = build_runner(student, args.device, BLOCK_FRAMES)
    for runner in (reference_runner, student_runner):
        if runner.warmup_frames > EVAL_WARMUP_FRAMES:
            print(f"ATTENZIONE: {runner.name} vuole {runner.warmup_frames} frame di riscaldamento", flush=True)
        runner.reset(segment)

    if args.out is not None:
        video_path = args.out.resolve()
    else:
        clip_id = (
            f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}_{segment.name}"
            f"_{reference_spec.key}_vs_{student_spec.key}"
        )
        video_path = ROOT / "outputs" / "compare" / f"{clip_id}.mp4"
    if args.json is not None:
        json_path = args.json.resolve()
    else:
        json_path = video_path.with_suffix(".json")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = video_path.with_name(f"{video_path.stem}.partial.mp4")

    masks = plan_point_mask(segment.frame_t, segment.frame_t[-1], args.horizon)
    first_point = np.arange(len(T_IDXS)) == 0
    inset = None
    if args.inset:
        inset = inset_scale(segment.ground_truth[start_frame:clip_end], masks[start_frame:clip_end])
    intrinsics = narrow_camera_intrinsics()
    packer = ModelInputPacker(segment.rpy)
    use_nvdec = args.device == "cuda" and nvdec_available()
    width = int(segment.frame_info["fcamera/width"].item())
    height = int(segment.frame_info["fcamera/height"].item())
    footer = footer_image(width, args.show_gt)
    video_height = 2 * height + footer.shape[0]

    reference_miss = RunningMiss()
    student_miss = RunningMiss()
    reference_lateral = []
    student_lateral = []
    per_frame = []
    total_frames = end_frame - decode_start
    done_frames = 0
    encoder = start_encoder(partial_path, width, video_height, args.speed)
    finished = False
    try:
        block_first = decode_start
        narrow_rgbs = []
        narrow_packs = []
        wide_packs = []
        for frame_index, narrow_rgb, wide_rgb in stream_frames(segment, decode_start, end_frame, use_nvdec):
            if not narrow_packs:
                block_first = frame_index
            narrow_pack, wide_pack = packer.pack(narrow_rgb, wide_rgb)
            narrow_rgbs.append(narrow_rgb)
            narrow_packs.append(narrow_pack)
            wide_packs.append(wide_pack)
            if len(narrow_packs) < BLOCK_FRAMES:
                continue

            narrow_block = np.stack(narrow_packs)
            wide_block = np.stack(wide_packs)
            reference_plans = reference_runner.step_block(block_first, narrow_block, wide_block)
            student_plans = student_runner.step_block(block_first, narrow_block, wide_block)
            for lane in range(BLOCK_FRAMES):
                clip_index = block_first + lane
                if clip_index < start_frame or clip_index >= clip_end:
                    continue
                truth = segment.ground_truth[clip_index]
                mask = masks[clip_index]
                plans = np.stack((reference_plans[lane, :, 0:3], student_plans[lane, :, 0:3]))
                misses, laterals = frame_misses(plans, np.stack((truth, truth)), np.stack((mask, mask)))
                reference_miss.add(misses[0])
                student_miss.add(misses[1])
                reference_lateral.append(float(laterals[0]))
                student_lateral.append(float(laterals[1]))
                speed_mps = float(segment.speeds[clip_index])
                time_s = float(segment.frame_t[clip_index] - segment.frame_t[0])
                per_frame.append({
                    "frame": clip_index,
                    "t": round(time_s, 3),
                    "speed_mps": round(speed_mps, 3),
                    "points": int(mask.sum()),
                    "reference_miss_m": rounded(misses[0]),
                    "reference_lateral_miss_m": rounded(laterals[0]),
                    "student_miss_m": rounded(misses[1]),
                    "student_lateral_miss_m": rounded(laterals[1]),
                })

                frame_bgr = cv2.cvtColor(narrow_rgbs[lane], cv2.COLOR_RGB2BGR)
                truth_points = truth[mask | first_point]
                top = draw_panel(
                    frame_bgr, segment.rpy, intrinsics, reference.spec.label, WHITE, reference_plans[lane, :, 0:3],
                    truth_points, args.show_gt, reference_miss.average, speed_mps, time_s, inset,
                )
                bottom = draw_panel(
                    frame_bgr, segment.rpy, intrinsics, student.spec.label, GREEN, student_plans[lane, :, 0:3],
                    truth_points, args.show_gt, student_miss.average, speed_mps, time_s, inset,
                )
                encoder.stdin.write(np.vstack((top, bottom, footer)).tobytes())

            narrow_rgbs = []
            narrow_packs = []
            wide_packs = []
            done_frames += BLOCK_FRAMES
            if done_frames % PROGRESS_EVERY_FRAMES == 0 or done_frames >= total_frames:
                print(f"PROGRESS {done_frames}/{total_frames}", flush=True)

        encoder.stdin.close()
        if encoder.wait() != 0:
            raise SystemExit("ERRORE: ffmpeg non e' riuscito a scrivere il video")
        partial_path.replace(video_path)
        finished = True
    finally:
        if not finished:
            if encoder.poll() is None:
                encoder.kill()
            encoder.wait()
            partial_path.unlink(missing_ok=True)

    elapsed = time.time() - started
    result = {
        "schema": 1,
        "kind": "compare",
        "code_version": EVAL_CODE_VERSION,
        "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "segment": segment.name,
        "video": str(video_path),
        "args": {
            "reference": reference_spec.text,
            "student": student_spec.text,
            "start_s": args.start,
            "duration_s": args.duration,
            "speed": args.speed,
            "inset": args.inset,
            "show_gt": args.show_gt,
            "horizon_s": args.horizon,
            "device": args.device,
        },
        "clip": {
            "fps": CAMERA_FPS,
            "first_frame": start_frame,
            "end_frame": clip_end,
            "warmup_from_frame": decode_start,
            "start_s": round(start_frame / CAMERA_FPS, 3),
            "end_s": round(clip_end / CAMERA_FPS, 3),
        },
        "inputs": {
            "action_t": list(ACTION_T),
            "traffic_convention": list(TRAFFIC_RIGHT_HAND),
            "desire": 0,
            "calibration_rpy": [float(value) for value in segment.rpy],
            "decoder": "nvdec" if use_nvdec else "ffmpeg",
        },
        "models": {
            "reference": model_record(reference, reference_runner, reference_miss, reference_lateral),
            "student": model_record(student, student_runner, student_miss, student_lateral),
        },
        "per_frame": per_frame,
        "elapsed_s": round(elapsed, 1),
    }
    temporary_json = json_path.with_name(f"{json_path.name}.tmp")
    temporary_json.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    temporary_json.replace(json_path)

    print(
        f"avg miss: {reference.spec.label} {_metres(reference_miss.average)}, "
        f"{student.spec.label} {_metres(student_miss.average)} ({reference_miss.count} frame, {elapsed:.0f} s)",
        flush=True,
    )
    print(f"OUTPUT_MP4={video_path}", flush=True)
    print(f"OUTPUT_JSON={json_path}", flush=True)


def _metres(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.2f} m"


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        print("ERRORE: ffmpeg si e' fermato (vedi il messaggio sopra)", file=sys.stderr)
        raise SystemExit(1) from None
