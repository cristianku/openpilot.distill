"""Check that supervised/big_model.py BigSupercombo gives the outputs of the teacher ONNX.

Usage: uv run --extra v100 python -m helpers.check_big_model [<big_driving_supercombo.onnx>] [--batch <n>] [--seed <n>]
       (default model: supervised/teachers.py DEFAULT_TEACHER)

The big model is cut by supervised/onnx_helpers.py into a vision graph and a dense 9-step policies
graph (helpers/check_teacher.py checks that the cut is the same model). This script runs the same
random 9-step sequences through onnxruntime on those two graphs and through BigSupercombo, and
compares:

  vision     the 32 x 512 features of every step (BigVision, the timm convnext_xlarge rebuilt)
  policies   every output slice of every step, the policies fed the onnxruntime features
  end to end every output slice, BigSupercombo alone from the frames

in float32 (the training) and float16 (the export). Differences are rounding: onnxruntime runs the
graphs in float16. --tolerance is the largest accepted max_abs / max|reference| of each output
(default 1e-2, as check_teacher.py).

Exit code 0 when every check passes, 1 otherwise.
"""

import argparse
import time

import numpy as np
import onnxruntime as ort
import torch

from helpers.check_teacher import Checks, cpu_session
from supervised.big_model import BigSupercombo
from supervised.model import ModelInputs
from supervised.model_for_inference import ORTSupercomboForInference
from supervised.onnx_helpers import make_dense_queue, split_queue_supercombo
from supervised.teachers import DEFAULT_TEACHER

DEFAULT_TOLERANCE = 1e-2
POLICY_INPUTS = (ModelInputs.DESIRE, ModelInputs.TRAFFIC, ModelInputs.ACTION_T)


def random_inputs(rng: np.random.Generator, batch: int, steps: int, temporal_len: int) -> dict[str, np.ndarray]:
    desire = np.zeros((batch, temporal_len, 8), dtype=np.float32)
    desire[:, rng.integers(0, temporal_len, size=3), rng.integers(1, 8, size=3)] = 1.0
    traffic = np.zeros((batch, 2), dtype=np.float32)
    traffic[:, 0] = 1.0
    return {
        ModelInputs.IMG: rng.integers(0, 256, size=(batch, steps, 12, 128, 256), dtype=np.uint8),
        ModelInputs.BIG_IMG: rng.integers(0, 256, size=(batch, steps, 12, 128, 256), dtype=np.uint8),
        ModelInputs.DESIRE: desire,
        ModelInputs.TRAFFIC: traffic,
        ModelInputs.ACTION_T: rng.normal(0.0, 0.5, size=(batch, 2)).astype(np.float32),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", nargs="?", default=DEFAULT_TEACHER)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()

    import onnx

    started = time.time()
    model = onnx.load(args.model)
    vision_graph, policies_graph = split_queue_supercombo(model)
    policies_graph = make_dense_queue(policies_graph)
    del model
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else None
    if providers:
        teacher = ORTSupercomboForInference(vision_graph, policies_graph, providers=providers)
    else:
        teacher = ORTSupercomboForInference(cpu_session_from(vision_graph), cpu_session_from(policies_graph))
    big = BigSupercombo.from_split(vision_graph, policies_graph)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    provider = teacher.vision_session.get_providers()[0]
    print(f"loaded in {time.time() - started:.0f} s, torch on {device}, onnxruntime {provider}")
    vision_params = sum(p.numel() for p in big.vision.parameters())
    policy_params = sum(p.numel() for p in big.policies.parameters())
    print(f"BigSupercombo: vision {vision_params / 1e6:.1f} M, policies {policy_params / 1e6:.1f} M parameters")

    rng = np.random.default_rng(args.seed)
    inputs = random_inputs(rng, args.batch, teacher.history_len, teacher.temporal_len)
    steps = teacher.history_len
    flat = {name: inputs[name].reshape(-1, 12, 128, 256) for name in teacher.image_names}
    reference_features = teacher.vision_session.run(None, flat)[0].reshape(args.batch, steps, 32, 512)
    feed = {
        teacher.policies_session.get_inputs()[0].name: reference_features,
        ModelInputs.DESIRE: inputs[ModelInputs.DESIRE],
        ModelInputs.TRAFFIC: inputs[ModelInputs.TRAFFIC],
        ModelInputs.ACTION_T: inputs[ModelInputs.ACTION_T],
    }
    reference_outputs = teacher.policies_session.run(None, feed)[0]

    checks = Checks()
    for dtype in (torch.float32, torch.float16):
        print(f"\n{str(dtype).removeprefix('torch.')}:")
        big.to(device=device, dtype=dtype)
        torch_inputs = {name: torch.from_numpy(value).to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            features = big.encode(torch_inputs[ModelInputs.IMG], torch_inputs[ModelInputs.BIG_IMG])
            checks.compare(f"vision {dtype}", features.float().cpu().numpy(), reference_features, args.tolerance)
            policy_inputs = {name: torch_inputs[name].to(dtype) for name in POLICY_INPUTS}
            policy_inputs[big.features_name] = torch.from_numpy(reference_features).to(device, dtype)
            policy_outputs = big.policies(**policy_inputs)["outputs"].float().cpu().numpy()
            end_to_end = big(torch_inputs)["outputs"].float().cpu().numpy()
        for label, value in (("policies", policy_outputs), ("end to end", end_to_end)):
            for name, part in big.output_slices.items():
                checks.compare(f"{label} {name}", value[..., part], reference_outputs[..., part], args.tolerance)

    if checks.failed:
        print(f"\nFAILED: {len(checks.failed)} checks: {', '.join(checks.failed)}")
        return 1
    print(f"\nok: BigSupercombo gives the outputs of the teacher ({time.time() - started:.0f} s)")
    return 0


def cpu_session_from(graph) -> ort.InferenceSession:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "graph.onnx"
        path.write_bytes(graph.SerializeToString())
        return cpu_session(path)


if __name__ == "__main__":
    raise SystemExit(main())
