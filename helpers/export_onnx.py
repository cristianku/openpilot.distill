"""Export a trained supercombo.pt as the driving_supercombo.onnx that openpilot runs on the comma.

Usage: uv run python -m helpers.export_onnx <supercombo.pt> --reference <original.onnx> --output <file.onnx>

openpilot (selfdrive/modeld/modeld.py) runs the model once per camera frame (20 Hz) with ONE new frame
per camera, and keeps the history through state inputs: every "state_x" input has a "next_state_x"
output that modeld gives back as input on the next frame. The student was trained on whole sequences
instead (supervised/dataset.py, 5 Hz): every image is 2 frames 4 apart (t-4, t) and the policies see
9 feature steps 4 frames apart. The wrapper rebuilds that from the 20 Hz stream:
  state_img_q:  the last 4 frames of each camera, the oldest one is the t-4 frame
  state_feat_q: the vision features of the last 32 frames, every 4th one is a history step
The outputs go in one "outputs" vector laid out like the original model of the comma: modeld reads
the layout from the "output_slices" metadata and the parser expects the same names and sizes.

--reference: the original driving_supercombo.onnx of the comma; only its inputs and output_slices are
             read, so the export follows the openpilot version that runs on the comma.
--no-check: skips the check (onnxruntime on random frames vs the trained model run as in training).

Desire (lane change) is always 0: training only saw desire 0 ("TODO: desire not supported" in
supervised/dataset.py), so the model gets 0 here too and does not do lane changes.
"""

import argparse
import base64
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import onnx
import torch
from openpilot.system.loggerd.config import CAMERA_FPS
from torch import nn

from supervised.dataset import DatasetConfig
from supervised.model import ModelInputs, Supercombo, SupercomboConfig, Unvision, VisionConfig


# Order of the cameras in new_img, as modeld fills it (vision_input_names in modeld.py).
CAMERA_INPUTS = (ModelInputs.IMG, ModelInputs.BIG_IMG)
# Inputs that modeld writes itself every frame; everything else must be a state.
DEVICE_INPUTS = ("new_img", "desire", "traffic_convention", "action_t")
CHECK_FRAMES = 40


def read_reference(path: Path) -> tuple[dict[str, list[int]], dict[str, str], int]:
    """Input shapes, metadata and size of the "outputs" vector of the comma's original model."""

    model = onnx.load(str(path), load_external_data=False)
    input_shapes = {}
    for value in model.graph.input:
        input_shapes[value.name] = [dim.dim_value for dim in value.type.tensor_type.shape.dim]

    metadata = {}
    for entry in model.metadata_props:
        metadata[entry.key] = entry.value

    output_size = None
    for value in model.graph.output:
        if value.name == "outputs":
            output_size = value.type.tensor_type.shape.dim[-1].dim_value
    if output_size is None:
        raise SystemExit(f"{path}: no 'outputs' output, it does not look like openpilot's driving_supercombo.onnx")
    if "output_slices" not in metadata:
        raise SystemExit(f"{path}: output_slices missing in the metadata")

    for name in input_shapes:
        if name not in DEVICE_INPUTS and not name.startswith("state_"):
            raise SystemExit(f"{path}: input '{name}' unknown to this export (openpilot changed?)")
    for name in DEVICE_INPUTS:
        if name not in input_shapes:
            raise SystemExit(f"{path}: input '{name}' missing (openpilot changed?)")
    return input_shapes, metadata, output_size


def output_layout(output_slices: dict[str, slice], output_size: int) -> list[tuple[int, int, str]]:
    """(start, stop, name) of every piece of "outputs", in order, checking that they cover it all."""

    layout = []
    for name, value in output_slices.items():
        start, stop, _ = value.indices(output_size)
        layout.append((start, stop, name))
    layout.sort()

    position = 0
    for start, stop, name in layout:
        if start != position:
            raise SystemExit(f"output_slices: '{name}' starts at {start}, expected {position}")
        position = stop
    if position != output_size:
        raise SystemExit(f"output_slices: they cover {position} values of {output_size}")
    return layout


class OnDeviceSupercombo(nn.Module):
    """The student with the one-frame-at-a-time interface of openpilot's modeld."""

    def __init__(self, model: Supercombo, frame_skip: int, layout: list[tuple[int, int, str]]) -> None:
        super().__init__()
        self.model = model
        self.frame_skip = frame_skip
        self.layout = layout
        config = model.config
        self.n_frames = config.vision.n_frames
        self.history_len = config.history_len
        self.temporal_len = config.temporal_len
        self.desire_dim = config.desire_dim
        self.img_queue_len = (self.n_frames - 1) * frame_skip
        self.feat_queue_len = (self.history_len - 1) * frame_skip

        # image_policy (Unvision) only rebuilds the camera images for the training loss.
        self.policy_names = []
        for name in model.policy_order:
            if not isinstance(getattr(model, name), Unvision):
                self.policy_names.append(name)

    def named_outputs(
        self,
        new_img: torch.Tensor,
        desire: torch.Tensor,
        traffic_convention: torch.Tensor,
        action_t: torch.Tensor,
        state_img_q: torch.Tensor,
        state_feat_q: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        # Image of each camera: the frames t-4 and t on the channels, as in dataset._load_images.
        vision_inputs = {}
        for camera, input_name in enumerate(CAMERA_INPUTS):
            frames = []
            for frame_number in range(self.n_frames):
                offset = (frame_number + 1 - self.n_frames) * self.frame_skip
                if offset == 0:
                    frames.append(new_img[camera])
                else:
                    frames.append(state_img_q[camera, self.img_queue_len + offset])
            vision_inputs[input_name] = torch.cat(frames, dim=0)[None]
        current = self.model.vision(vision_inputs)[0]

        # History: frames t-32, t-28, ..., t-4 from the queue, then the current one.
        steps = []
        for step in range(self.history_len - 1):
            steps.append(state_feat_q[step * self.frame_skip])
        steps.append(current)
        features = torch.stack(steps)[None]

        # Desire always 0 like in training; "+ desire.sum() * 0" only keeps the input in the graph,
        # modeld writes it every frame.
        desire_pulse = torch.zeros(1, self.temporal_len, self.desire_dim) + desire.sum() * 0
        traffic = traffic_convention.reshape(1, 1, -1)
        action = action_t.reshape(1, 1, -1)

        outputs = {}
        for name in self.policy_names:
            policy_outputs = getattr(self.model, name)(features, desire_pulse, traffic, action)
            for output_name, value in policy_outputs.items():
                outputs[output_name] = value[:, -1]
        # openpilot's driving_supercombo.onnx has room for 512 values of hidden_state: the 32 tokens
        # are averaged (the distill loss does the same with the old 084747c7 teacher; with the
        # queue-format teachers it compares the tokens one by one).
        outputs["hidden_state"] = features[:, -1].mean(dim=1)

        next_state_img_q = torch.cat((state_img_q[:, 1:], new_img[:, None]), dim=1)
        next_state_feat_q = torch.cat((state_feat_q[1:], current[None]), dim=0)
        return outputs, next_state_img_q, next_state_feat_q

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs, next_state_img_q, next_state_feat_q = self.named_outputs(*inputs)
        pieces = []
        for start, stop, name in self.layout:
            if name == "pad":
                pieces.append(torch.zeros(1, stop - start))
            else:
                pieces.append(outputs[name])
        return torch.cat(pieces, dim=1), next_state_img_q, next_state_feat_q


def check_outputs(wrapper: OnDeviceSupercombo, inputs: dict[str, torch.Tensor]) -> None:
    """Every piece of the comma's layout must exist in the student with the same size."""

    with torch.no_grad():
        outputs, _, _ = wrapper.named_outputs(*inputs.values())
    for start, stop, name in wrapper.layout:
        if name == "pad":
            continue
        if name not in outputs:
            raise SystemExit(f"the trained model does not have the output '{name}' that openpilot expects")
        if outputs[name].shape[-1] != stop - start:
            raise SystemExit(f"output '{name}': {outputs[name].shape[-1]} values, openpilot wants {stop - start}")


def example_inputs(wrapper: OnDeviceSupercombo, input_shapes: dict[str, list[int]]) -> dict[str, torch.Tensor]:
    cameras, channels, height, width = input_shapes["new_img"]
    vision = wrapper.model.vision
    return {
        "new_img": torch.zeros(cameras, channels, height, width, dtype=torch.uint8),
        "desire": torch.zeros(input_shapes["desire"]),
        "traffic_convention": torch.zeros(input_shapes["traffic_convention"]),
        "action_t": torch.zeros(input_shapes["action_t"]),
        "state_img_q": torch.zeros(cameras, wrapper.img_queue_len, channels, height, width, dtype=torch.uint8),
        "state_feat_q": torch.zeros(wrapper.feat_queue_len, wrapper.model.spatial_size, vision.output_dim),
    }


def check(wrapper: OnDeviceSupercombo, path: Path, inputs: dict[str, torch.Tensor]) -> None:
    """Runs the exported model frame by frame with onnxruntime and compares the last frame with the
    trained model run on the whole sequence, built as supervised/dataset.py does."""

    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    options = ort.SessionOptions()
    # An explicit thread count stops onnxruntime from pinning its threads to cores, which an
    # LXC container refuses ("pthread_setaffinity_np failed" once per thread).
    options.intra_op_num_threads = os.cpu_count() or 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(CHECK_FRAMES, *inputs["new_img"].shape), dtype=np.uint8)
    traffic = np.asarray([[1.0, 0.0]], dtype=np.float32).reshape(inputs["traffic_convention"].shape)
    action_t = np.asarray([[0.3, 0.5]], dtype=np.float32).reshape(inputs["action_t"].shape)

    state_img_q = inputs["state_img_q"].numpy()
    state_feat_q = inputs["state_feat_q"].numpy()
    for t in range(CHECK_FRAMES):
        feed = {
            "new_img": frames[t],
            "desire": inputs["desire"].numpy(),
            "traffic_convention": traffic,
            "action_t": action_t,
            "state_img_q": state_img_q,
            "state_feat_q": state_feat_q,
        }
        packed, state_img_q, state_feat_q = session.run(None, feed)

    last = CHECK_FRAMES - 1
    skip = wrapper.frame_skip
    dense = {}
    for camera, input_name in enumerate(CAMERA_INPUTS):
        images = []
        for step in range(wrapper.history_len):
            frame_t = last - (wrapper.history_len - 1 - step) * skip
            channels = []
            for frame_number in range(wrapper.n_frames):
                channels.append(frames[frame_t + (frame_number + 1 - wrapper.n_frames) * skip, camera])
            images.append(np.concatenate(channels))
        dense[input_name] = torch.from_numpy(np.stack(images)[None])
    temporal_len = wrapper.temporal_len
    dense[ModelInputs.DESIRE] = torch.zeros(1, temporal_len, wrapper.desire_dim)
    dense[ModelInputs.TRAFFIC] = torch.from_numpy(np.tile(traffic.reshape(1, 1, -1), (1, temporal_len, 1)))
    dense[ModelInputs.ACTION_T] = torch.from_numpy(np.tile(action_t.reshape(1, 1, -1), (1, temporal_len, 1)))
    with torch.no_grad():
        expected = wrapper.model(dense)

    worst = 0.0
    for start, stop, name in wrapper.layout:
        if name == "pad":
            continue
        if name == "hidden_state":
            reference = expected["hidden_state"][:, -1].mean(dim=1)
        else:
            reference = expected[name][:, -1]
        got = packed[0, start:stop]
        reference = reference[0].float().numpy()
        error = float(np.max(np.abs(got - reference)) / (np.max(np.abs(reference)) + 1e-6))
        worst = max(worst, error)
        if error > 1e-3:
            raise SystemExit(f"check failed on '{name}': relative error {error:.2e}")
    print(f"check ok: {CHECK_FRAMES} frames with onnxruntime = trained model (max error {worst:.1e})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="supercombo.pt from supervised/train.py")
    parser.add_argument("--reference", type=Path, required=True, help="original driving_supercombo.onnx of the comma")
    parser.add_argument("--output", type=Path, required=True, help="onnx to write")
    parser.add_argument("--no-check", action="store_true", help="skip the onnxruntime check")
    args = parser.parse_args()

    input_shapes, metadata, output_size = read_reference(args.reference)
    output_slices = pickle.loads(base64.b64decode(metadata["output_slices"]))
    layout = output_layout(output_slices, output_size)

    # Same model as supervised/train.py; the weights come from the checkpoint, no timm download.
    model = Supercombo(SupercomboConfig(vision=VisionConfig(pretrained=False)))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    frame_skip = CAMERA_FPS // DatasetConfig(dataset_path="").fps
    wrapper = OnDeviceSupercombo(model, frame_skip, layout).eval()
    inputs = example_inputs(wrapper, input_shapes)
    names = list(inputs)

    check_outputs(wrapper, inputs)

    # The new torch.export-based exporter needs onnxscript, not installed: the TorchScript one is used.
    warnings.filterwarnings("ignore", message="You are using the legacy TorchScript-based ONNX export")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            tuple(inputs.values()),
            str(args.output),
            input_names=names,
            output_names=["outputs", "next_state_img_q", "next_state_feat_q"],
            opset_version=17,
            dynamo=False,
        )

    exported = onnx.load(str(args.output))
    entry = exported.metadata_props.add()
    entry.key = "output_slices"
    entry.value = metadata["output_slices"]
    entry = exported.metadata_props.add()
    entry.key = "model_checkpoint"
    entry.value = str(args.checkpoint.resolve())
    onnx.checker.check_model(exported)
    onnx.save(exported, str(args.output))
    print(f"{args.output}: {args.output.stat().st_size / 1e6:.1f} MB, input {names}")

    if not args.no_check:
        check(wrapper, args.output, inputs)


if __name__ == "__main__":
    main()
