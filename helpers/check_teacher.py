"""Check on CPU that a queue-format big model of Hugging Face can be the distill teacher.

Usage: uv run python -m helpers.check_teacher <big_driving_supercombo.onnx> [--ticks <n>] [--tolerance <x>] [--seed <n>]

The big models of commaai/openpilot_driving_models ("MD v3" and later, openpilot PR #38916) keep their
history inside the ONNX: modeld runs them at 20 Hz with ONE new frame per camera and gives every
"next_state_x" output back as the "state_x" input of the next frame. The training cannot use them
like that: ORTSupercomboForInference.from_supercombo (supervised/onnx_helpers.py, split_queue_supercombo
+ make_dense_queue) cuts them into a vision graph and a policies graph that take the 5 Hz sequence of
the dataset (9 steps, every image = frames t-4 and t) and give the outputs of all 9 steps at once.
This script checks, with random frames, that the cut model is still the same model:

  vision:       the features of each 5 Hz step == the hidden_state of the full model at that frame
  equivalence:  the full model runs at 20 Hz for --ticks frames (next_state_x given back, a few desire
                pulses); the 9-step sequence that ends at the last frame goes through the teacher as
                supervised/train.py does it (from_supercombo, InputQueues, prefill dense): its last step
                == the outputs of the full model at the last frame, output slice by output slice
  hidden_state: the hidden_state of every dense step == the vision features of that step
  causality:    step k of the dense outputs only sees the steps 0..k: changing the features of step 8 or
                of step 4, or the newest desire, leaves the earlier steps exactly as they were
  batch:        2 different sequences in one batch == each sequence alone

Run it on a new model before training with it (and before adding it to helpers/train_supervised.sh).
The differences of vision and equivalence are float16 rounding (~1e-3 of the largest value of each
output): onnxruntime fuses the full graph and the two split graphs differently. A wrong cut (wrong
frames, wrong desire window, wrong step) gives errors of several percent.

--ticks:     20 Hz frames of the full model (default 48; at least 37, so that the oldest step has its
             t-4 frame and a full image queue)
--tolerance: largest accepted max_abs / max|reference| of each output (default 1e-2)
--seed:      seed of the random frames (default 1234)

Exit code 0 when every check passes, 1 otherwise or when the model cannot be the teacher (e.g. the
N16 nav model, which also needs a map and navigation input that the dataset does not have).
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from supervised.model import ModelInputs
from supervised.model_for_inference import InputQueues, ORTSupercomboForInference
from supervised.onnx_helpers import is_queue_supercombo


# 20 Hz camera, 5 Hz dataset steps (supervised/dataset.py): every image is the frames t-4 and t and
# the history steps are 4 frames apart.
FRAME_SKIP = 4
DEFAULT_TICKS = 48
DEFAULT_TOLERANCE = 1e-2
# Steps that must not move at all (causality, hidden_state): same inputs in the same CPU session.
UNCHANGED_TOLERANCE = 1e-6


def cpu_session(path: Path) -> ort.InferenceSession:
    """onnxruntime on CPU, with the same thread settings as ORTSupercomboForInference."""

    ort.set_default_logger_severity(3)
    options = ort.SessionOptions()
    # An explicit thread count stops onnxruntime from pinning its threads to cores, which an LXC
    # container refuses (see supervised/model_for_inference.py).
    if hasattr(os, "sched_getaffinity"):
        options.intra_op_num_threads = len(os.sched_getaffinity(0))
    else:
        options.intra_op_num_threads = os.cpu_count() or 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def numpy_dtype(type_name: str):
    if type_name == "tensor(uint8)":
        return np.uint8
    if type_name == "tensor(float)":
        return np.float32
    raise SystemExit(f"input di tipo {type_name}: questo controllo non lo sa riempire")


def run_full_model(session: ort.InferenceSession, frames: np.ndarray, desire_ticks: np.ndarray,
                   traffic: np.ndarray, action_t: np.ndarray) -> np.ndarray:
    """The model as modeld runs it: one call per frame, every next_state_x given back as state_x.

    Returns the "outputs" vector of every frame, [ticks, size].
    """

    states = {}
    for value in session.get_inputs():
        if value.name.startswith("state_"):
            states[value.name] = np.zeros(value.shape, dtype=numpy_dtype(value.type))
    output_names = []
    for value in session.get_outputs():
        output_names.append(value.name)

    outputs = []
    for tick in range(len(frames)):
        feed = {
            "new_img": frames[tick],
            "desire": desire_ticks[tick],
            "traffic_convention": traffic,
            "action_t": action_t,
        }
        for name, value in states.items():
            feed[name] = value
        values = session.run(output_names, feed)
        for name, value in zip(output_names, values, strict=True):
            if name == "outputs":
                outputs.append(value[0])
            else:
                states[name.removeprefix("next_")] = value
    return np.stack(outputs)


def step_images(frames: np.ndarray, last_tick: int, camera: int, steps: int) -> np.ndarray:
    """The 5 Hz images of one camera, as supervised/dataset.py gives them: [steps, 12, H, W].

    Step k is the frame t = last_tick - (steps - 1 - k) * 4; its image is the frames t-4 and t.
    """

    images = []
    for step in range(steps):
        tick = last_tick - (steps - 1 - step) * FRAME_SKIP
        images.append(np.concatenate((frames[tick - FRAME_SKIP, camera], frames[tick, camera])))
    return np.stack(images)


def desire_5hz(desire_ticks: np.ndarray, last_tick: int, temporal_len: int) -> np.ndarray:
    """The 5 Hz desire the model sees at last_tick: the max of each group of 4 frames of its queue.

    The queue holds the last temporal_len * 4 frames; the newest group is the frames last_tick-3 .. last_tick.
    """

    desire = np.zeros((temporal_len, desire_ticks.shape[1]), dtype=np.float32)
    for group in range(temporal_len):
        for offset in range(FRAME_SKIP):
            tick = last_tick - FRAME_SKIP * temporal_len + 1 + FRAME_SKIP * group + offset
            if tick >= 0:
                desire[group] = np.maximum(desire[group], desire_ticks[tick])
    return desire


def prefill(teacher: ORTSupercomboForInference, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Same as supervised.dataset.prefill_supercombo_batch, the call of supervised/train.py.

    Not imported from there: supervised/dataset.py needs the video and openpilot packages.
    """

    queues = InputQueues.from_model(teacher, batch_size=len(inputs[ModelInputs.IMG]))
    queues.reset()
    return teacher.prefill(inputs, queues, dense=True)


def errors(value: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """max_abs and max_abs / max|reference| (relative to the largest value of the output)."""

    value = value.astype(np.float64)
    reference = reference.astype(np.float64)
    max_abs = float(np.max(np.abs(value - reference)))
    scale = float(np.max(np.abs(reference)))
    if scale == 0:
        scale = 1.0
    return max_abs, max_abs / scale


class Checks:
    """Prints every comparison and remembers the failed ones."""

    def __init__(self) -> None:
        self.failed = []

    def compare(self, label: str, value: np.ndarray, reference: np.ndarray, tolerance: float) -> None:
        max_abs, relative = errors(value, reference)
        if relative <= tolerance:
            verdict = "ok"
        else:
            verdict = "FALLITO"
            self.failed.append(label)
        print(f"   {label:46s} max_abs {max_abs:.2e}  relativo {relative:.2e}  {verdict}")

    def require(self, label: str, ok: bool, detail: str) -> None:
        if ok:
            verdict = "ok"
        else:
            verdict = "FALLITO"
            self.failed.append(label)
        print(f"   {label:46s} {detail}  {verdict}")


def check_causality(checks: Checks, teacher: ORTSupercomboForInference, feed: dict[str, np.ndarray],
                    base: np.ndarray, changed_name: str, changed_value: np.ndarray, first_changed: int,
                    label: str) -> None:
    """Change one input of the dense policies: the steps before first_changed must stay the same."""

    changed = dict(feed)
    changed[changed_name] = changed_value
    outputs = teacher.policies_session.run(None, changed)[0][0]
    # The largest change of the steps that must stay the same, the smallest of the ones that must move.
    worst_unchanged = 0.0
    least_changed = None
    for step in range(len(base)):
        max_abs, relative = errors(outputs[step], base[step])
        if step < first_changed:
            if relative > worst_unchanged:
                worst_unchanged = relative
        elif least_changed is None or max_abs < least_changed:
            least_changed = max_abs
    ok = worst_unchanged <= UNCHANGED_TOLERANCE and least_changed > 0
    detail = (f"step 0..{first_changed - 1} relativo {worst_unchanged:.2e} (uguali), "
              f"step {first_changed}..{len(base) - 1} max_abs >= {least_changed:.2e} (cambiati)")
    checks.require(label, ok, detail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="big_driving_supercombo.onnx of commaai/openpilot_driving_models")
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS, help="20 Hz frames of the full model")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help="largest accepted max_abs / max|reference| per output")
    parser.add_argument("--seed", type=int, default=1234, help="seed of the random frames")
    args = parser.parse_args()

    model = onnx.load(str(args.model))
    checkpoint = ""
    for entry in model.metadata_props:
        if entry.key == "model_checkpoint":
            checkpoint = entry.value
    print(f"Modello: {args.model} ({args.model.stat().st_size / 1e6:.1f} MB), checkpoint {checkpoint or '?'}")
    for value in model.graph.input:
        dims = []
        for dim in value.type.tensor_type.shape.dim:
            dims.append(dim.dim_param or dim.dim_value)
        print(f"   input {value.name} {dims}")
    if not is_queue_supercombo(model):
        raise SystemExit(
            f"{args.model}: nessun input state_feat_q, non e' un modello a code di Hugging Face "
            "(il teacher vecchio 084747c7 non ha bisogno di questo controllo)"
        )

    start = time.time()
    try:
        teacher = ORTSupercomboForInference.from_supercombo(model, providers=["CPUExecutionProvider"])
    except ValueError as error:
        raise SystemExit(f"NON si puo' usare come teacher: {error}") from error
    del model
    steps = teacher.history_len
    temporal_len = teacher.temporal_len
    print(f"Diviso in vision + policies dense in {time.time() - start:.1f} s: {steps} step, "
          f"desire di {temporal_len} step, feature {teacher.feature_shape}")

    minimum_ticks = steps * FRAME_SKIP + 1
    if args.ticks < minimum_ticks:
        raise SystemExit(f"--ticks deve essere almeno {minimum_ticks} (il frame t-4 dello step piu' vecchio)")

    full = cpu_session(args.model)
    new_img_shape = None
    desire_size = None
    for value in full.get_inputs():
        if value.name == "new_img":
            new_img_shape = value.shape
        elif value.name == "desire":
            desire_size = value.shape[-1]

    # Random frames (both cameras), a desire pulse that only the newest 5 Hz entry sees and one that
    # every step sees, fixed traffic convention and action_t (the dataset keeps them constant too).
    rng = np.random.default_rng(args.seed)
    frames = rng.integers(0, 256, size=(args.ticks, *new_img_shape), dtype=np.uint8)
    last_tick = args.ticks - 1
    desire_ticks = np.zeros((args.ticks, desire_size), dtype=np.float32)
    desire_ticks[last_tick - 2, 3] = 1.0
    desire_ticks[last_tick - 13, 1] = 1.0
    traffic = np.asarray([[0.0, 1.0]], dtype=np.float32)
    action_t = np.asarray([[0.35, 0.55]], dtype=np.float32)

    start = time.time()
    full_outputs = run_full_model(full, frames, desire_ticks, traffic, action_t)
    print(f"Modello completo a 20 Hz: {args.ticks} frame in {time.time() - start:.1f} s")
    del full

    checks = Checks()
    output_slices = teacher.output_slices
    hidden = output_slices["hidden_state"]

    # Camera 0 of new_img is img, camera 1 is big_img (vision_input_names in openpilot's modeld.py).
    images = step_images(frames, last_tick, 0, steps)
    big_images = step_images(frames, last_tick, 1, steps)
    features = teacher.encode({ModelInputs.IMG: images[None], ModelInputs.BIG_IMG: big_images[None]})[0]
    print("[vision] feature di ogni step 5 Hz = hidden_state del modello completo a quel frame")
    for step in range(steps):
        tick = last_tick - (steps - 1 - step) * FRAME_SKIP
        reference = full_outputs[tick, hidden].reshape(features.shape[1:])
        checks.compare(f"step {step} (frame {tick})", features[step], reference, args.tolerance)

    # The sequence as the dataset gives it: 5 Hz images, desire, traffic_convention, action_t.
    desire = desire_5hz(desire_ticks, last_tick, temporal_len)
    inputs = {
        ModelInputs.IMG: images[None],
        ModelInputs.BIG_IMG: big_images[None],
        ModelInputs.DESIRE: desire[None],
        ModelInputs.TRAFFIC: np.tile(traffic.reshape(1, 1, -1), (1, temporal_len, 1)),
        ModelInputs.ACTION_T: np.tile(action_t.reshape(1, 1, -1), (1, temporal_len, 1)),
    }
    start = time.time()
    predictions = prefill(teacher, inputs)
    print(f"[equivalenza] ultimo step del teacher (prefill dense, {time.time() - start:.1f} s) = "
          f"modello completo al frame {last_tick}")
    for name, output_slice in output_slices.items():
        if name == "pad":
            continue
        value = predictions[name][0, -1].reshape(-1)
        checks.compare(name, value, full_outputs[last_tick, output_slice], args.tolerance)

    print("[hidden_state] hidden_state di ogni step denso = feature della vision di quello step")
    checks.compare(f"step 0..{steps - 1}", predictions["hidden_state"][0], features, UNCHANGED_TOLERANCE)

    # Causality on the dense policies graph itself: one run with the vision features of the
    # sequence, then one input changed at a time.
    feed = {
        "features": features[None],
        ModelInputs.DESIRE: desire[None],
        ModelInputs.TRAFFIC: traffic,
        ModelInputs.ACTION_T: action_t,
    }
    base = teacher.policies_session.run(None, feed)[0][0]
    print("[causalita'] lo step k vede solo gli step 0..k")
    scale = float(np.std(features.astype(np.float32)))
    for changed_step in (steps - 1, steps // 2):
        changed_features = features.copy()
        noise = rng.normal(0.0, scale, size=features.shape[1:])
        changed_features[changed_step] = noise.astype(features.dtype)
        check_causality(checks, teacher, feed, base, "features", changed_features[None], changed_step,
                        f"feature dello step {changed_step} cambiate")
    changed_desire = desire.copy()
    changed_desire[-1] = 0.0
    changed_desire[-1, 6] = 1.0
    check_causality(checks, teacher, feed, base, ModelInputs.DESIRE, changed_desire[None], steps - 1,
                    "desire piu' recente cambiato")

    # Batch: a second, different sequence; both in one batch == each one alone.
    other_desire = np.zeros_like(desire)
    other_desire[temporal_len // 2, 2] = 1.0
    other_inputs = {
        ModelInputs.IMG: rng.integers(0, 256, size=images.shape, dtype=np.uint8)[None],
        ModelInputs.BIG_IMG: rng.integers(0, 256, size=big_images.shape, dtype=np.uint8)[None],
        ModelInputs.DESIRE: other_desire[None],
        ModelInputs.TRAFFIC: np.tile(np.asarray([1.0, 0.0], dtype=np.float32).reshape(1, 1, -1), (1, temporal_len, 1)),
        ModelInputs.ACTION_T: np.tile(np.asarray([0.1, 0.9], dtype=np.float32).reshape(1, 1, -1), (1, temporal_len, 1)),
    }
    other_predictions = prefill(teacher, other_inputs)
    batch_inputs = {}
    for name in inputs:
        batch_inputs[name] = np.concatenate((inputs[name], other_inputs[name]))
    batch_predictions = prefill(teacher, batch_inputs)
    print("[batch] 2 sequenze diverse in un batch = ogni sequenza da sola")
    alone = (predictions, other_predictions)
    for sequence in range(2):
        worst = 0.0
        worst_name = ""
        for name in predictions:
            _, relative = errors(batch_predictions[name][sequence], alone[sequence][name][0])
            if relative >= worst:
                worst = relative
                worst_name = name
        checks.require(f"sequenza {sequence}", worst <= args.tolerance,
                       f"peggiore relativo {worst:.2e} ({worst_name})")
    sequence_difference, _ = errors(other_predictions["plan"], predictions["plan"])
    checks.require("le due sequenze danno plan diversi", sequence_difference > 0, f"max_abs {sequence_difference:.2e}")

    print()
    if checks.failed:
        print(f"RISULTATO: {len(checks.failed)} controlli FALLITI (tolleranza relativa {args.tolerance:.0e}):")
        for label in checks.failed:
            print(f"   {label}")
        raise SystemExit(1)
    print(f"RISULTATO: tutti i controlli ok (tolleranza relativa {args.tolerance:.0e}): "
          f"{args.model.name} si puo' usare come teacher")


if __name__ == "__main__":
    main()
