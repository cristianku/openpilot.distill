"""Model runners: camera frames in, plans out, for every kind of model spec.

Every runner gets the frames in blocks of 4 consecutive camera frames (one 5 Hz model step,
FRAME_SKIP = 4) starting at a frame index that is a multiple of 4, and returns the plan mean
(33, 15) of the first `lanes` frames of the block: lanes = 4 gives a plan at every 20 Hz frame,
lanes = 1 only at the 5 Hz frames 0, 4, 8, ...

Lanes (teacher and .pt students, which run like in training, one 5 Hz step per call): lane p sees
the frames 4k + p and keeps its own 5 Hz history (features, image t-4), so the 4 lanes are 4
independent copies of the model, run as one batch of 4. Same plans as running the model at 20 Hz,
batched. Streaming ONNX models (stock CD210, exported students) run at 20 Hz like modeld anyway.

Constant inputs for every model: desire 0 (training never saw anything else), traffic convention
right-hand [1, 0] (Italy), and ACTION_T below.
"""

from __future__ import annotations

import base64
import pickle
from pathlib import Path

import numpy as np

from .segment import FRAME_SKIP, Segment
from .specs import FORMAT_LOGGED, FORMAT_ONNX_STREAMING, FORMAT_ONNX_TEACHER, FORMAT_TORCH, ResolvedModel

BLOCK_FRAMES = FRAME_SKIP
PLAN_POINTS = 33
PLAN_WIDTH = 15
PLAN_MEAN_SIZE = PLAN_POINTS * PLAN_WIDTH  # the "plan" output is mean (495) then std (495)

# action_t (lateral, longitudinal) in seconds, the same for every model. modeld feeds
# lateral delay + 0.075 s and longitudinal delay + 0.3 s smoothing + 0.075 s (modeld.py), about
# 0.3 and 0.5 on a typical car; training draws both in [0, 1] (supervised/dataset.py). Same pair as
# the check in helpers/export_onnx.py.
ACTION_T = (0.3, 0.5)
TRAFFIC_RIGHT_HAND = (1.0, 0.0)
DESIRE_SIZE = 8

# Frames every model has warmed up after: CD210 keeps 96 frames of features plus 5 images,
# the lane models 9 steps of 4 frames plus the t-4 image. Multiple of BLOCK_FRAMES.
EVAL_WARMUP_FRAMES = 104


def ort_providers(device: str) -> list[object]:
    if device == "cuda":
        return [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _ort_session(path: Path, device: str):
    # Same session options as the training teacher (thread count for the LXC, CUDA check).
    from supervised.model_for_inference import ORTSupercomboForInference

    return ORTSupercomboForInference._session(path, ort_providers(device))


def _ort_dtype(type_name: str) -> np.dtype:
    name = type_name.removeprefix("tensor(").removesuffix(")")
    if name == "float":
        return np.dtype("float32")
    if name == "double":
        return np.dtype("float64")
    return np.dtype(name)


def _plan_mean(plan_output: np.ndarray) -> np.ndarray:
    """(B, 990) plan output -> (B, 33, 15) mean (first half, openpilot parse_mdn)."""
    plan_output = np.asarray(plan_output, dtype=np.float32)
    return plan_output[:, :PLAN_MEAN_SIZE].reshape(-1, PLAN_POINTS, PLAN_WIDTH)


class ModelRunner:
    """Common interface. reset() at the start of every segment, then step_block() per 4 frames."""

    name: str
    lanes: int
    warmup_frames: int

    def reset(self, segment: Segment) -> None:
        raise NotImplementedError

    def step_block(self, first_frame: int, narrow_packs: np.ndarray, wide_packs: np.ndarray) -> np.ndarray:
        """narrow_packs, wide_packs: (4, 6, 128, 256) uint8 of frames first_frame..first_frame+3.

        Returns (lanes, 33, 15): the plan of frames first_frame .. first_frame + lanes - 1.
        """
        raise NotImplementedError


class LaneRunner(ModelRunner):
    """Teacher ONNX or student .pt through supervised/model_for_inference.py, 5 Hz per lane."""

    def __init__(self, name: str, model, lanes: int, device: str) -> None:
        from supervised.model_for_inference import InputQueues

        self.name = name
        self.model = model
        self.lanes = lanes
        self.device = device
        self.is_torch = model.backend == "torch"
        image_names = set(model.image_names)
        if image_names != {"img", "big_img"}:
            raise ValueError(f"{name}: input immagine inattesi {sorted(image_names)}")
        # history_len steps plus the t-4 image of the first one.
        self.warmup_frames = BLOCK_FRAMES * (model.history_len + 1)
        if self.is_torch:
            self.queues = InputQueues.from_model(model, batch_size=lanes, device=device)
        else:
            self.queues = InputQueues.from_model(model, batch_size=lanes)
        self.previous_narrow = None
        self.previous_wide = None
        self.desire = np.zeros((lanes, DESIRE_SIZE), dtype=np.float32)
        self.traffic = np.tile(np.asarray(TRAFFIC_RIGHT_HAND, dtype=np.float32), (lanes, 1))
        self.action_t = np.tile(np.asarray(ACTION_T, dtype=np.float32), (lanes, 1))

    def reset(self, segment: Segment) -> None:
        self.queues.reset()
        self.previous_narrow = None
        self.previous_wide = None

    def step_block(self, first_frame: int, narrow_packs: np.ndarray, wide_packs: np.ndarray) -> np.ndarray:
        current_narrow = narrow_packs[: self.lanes]
        current_wide = wide_packs[: self.lanes]
        if self.previous_narrow is None:
            # Before the first block there is no t-4 image: zeros, like the empty queues. These
            # plans are inside the warm-up and never evaluated.
            self.previous_narrow = np.zeros_like(current_narrow)
            self.previous_wide = np.zeros_like(current_wide)
        # Channels 0-5 the frame 4 earlier (same lane, previous block), 6-11 the current frame,
        # as frame_number 0 / 1 in supervised/dataset.py::_load_images.
        narrow = np.ascontiguousarray(np.concatenate((self.previous_narrow, current_narrow), axis=1))
        wide = np.ascontiguousarray(np.concatenate((self.previous_wide, current_wide), axis=1))
        self.previous_narrow = current_narrow.copy()
        self.previous_wide = current_wide.copy()

        inputs = {
            "img": narrow,
            "big_img": wide,
            "desire_pulse": self.desire,
            "traffic_convention": self.traffic,
            "action_t": self.action_t,
        }
        if self.is_torch:
            import torch

            torch_inputs = {}
            for input_name, value in inputs.items():
                torch_inputs[input_name] = torch.from_numpy(value).to(self.device)
            with torch.inference_mode():
                outputs = self.model.decode(torch_inputs, self.queues)
            plan = outputs["plan"].float().cpu().numpy()
        else:
            outputs = self.model.decode(inputs, self.queues)
            plan = outputs["plan"]
        return _plan_mean(plan)


def load_teacher(path: Path, device: str, lanes: int, name: str) -> LaneRunner:
    from supervised.model_for_inference import ORTSupercomboForInference

    model = ORTSupercomboForInference.from_supercombo(path, providers=ort_providers(device))
    return LaneRunner(name, model, lanes, device)


def load_torch_student(path: Path, device: str, lanes: int, name: str) -> LaneRunner:
    import torch

    from supervised.checkpoints import model_weights, torch_load
    from supervised.model import Supercombo, SupercomboConfig, Unvision, VisionConfig
    from supervised.model_for_inference import TorchSupercomboForInference

    # Same construction as helpers/export_onnx.py: weights from the checkpoint, no timm download.
    model = Supercombo(SupercomboConfig(vision=VisionConfig(pretrained=False)))
    model.load_state_dict(model_weights(torch_load(path)))
    # image_policy (Unvision) only rebuilds the camera images for the training loss: skip it.
    kept_policies = []
    for policy_name in model.policy_order:
        if not isinstance(getattr(model, policy_name), Unvision):
            kept_policies.append(policy_name)
    model.policy_order = tuple(kept_policies)
    torch_device = torch.device("cuda" if device == "cuda" else "cpu")
    model = model.to(torch_device).eval()
    return LaneRunner(name, TorchSupercomboForInference(model), lanes, str(torch_device))


class OnnxStreamingRunner(ModelRunner):
    """driving_supercombo.onnx as modeld runs it: one frame per call, next_state_* fed back as state_*."""

    def __init__(self, name: str, path: Path, device: str, lanes: int) -> None:
        self.name = name
        self.lanes = lanes
        self.session = _ort_session(path, device)
        self.output_names = []
        for output in self.session.get_outputs():
            self.output_names.append(output.name)
        metadata = self.session.get_modelmeta().custom_metadata_map
        output_slices = pickle.loads(base64.b64decode(metadata["output_slices"]))
        self.plan_slice = output_slices["plan"]

        self.state_shapes = {}
        self.state_dtypes = {}
        self.constants = {}
        for model_input in self.session.get_inputs():
            shape = tuple(int(size) for size in model_input.shape)
            dtype = _ort_dtype(model_input.type)
            if model_input.name.startswith("state_"):
                if f"next_{model_input.name}" not in self.output_names:
                    raise ValueError(f"{name}: manca l'output next_{model_input.name}")
                self.state_shapes[model_input.name] = shape
                self.state_dtypes[model_input.name] = dtype
            elif model_input.name == "new_img":
                self.image_shape = shape
            elif model_input.name == "desire":
                self.constants["desire"] = np.zeros(shape, dtype=dtype)
            elif model_input.name == "traffic_convention":
                self.constants["traffic_convention"] = np.asarray(TRAFFIC_RIGHT_HAND, dtype=dtype).reshape(shape)
            elif model_input.name == "action_t":
                self.constants["action_t"] = np.asarray(ACTION_T, dtype=dtype).reshape(shape)
            else:
                raise ValueError(f"{name}: input '{model_input.name}' sconosciuto")

        # Frames until the feature queue holds only real frames (each feature needs its t-4 image).
        feature_frames = 0
        image_frames = 0
        if "state_feat_q" in self.state_shapes:
            feature_frames = self.state_shapes["state_feat_q"][0]
        if "state_img_q" in self.state_shapes:
            image_frames = self.state_shapes["state_img_q"][1]
        self.warmup_frames = feature_frames + image_frames
        self.states = {}

    def reset(self, segment: Segment) -> None:
        self.states = {}
        for state_name, shape in self.state_shapes.items():
            self.states[state_name] = np.zeros(shape, dtype=self.state_dtypes[state_name])

    def step_block(self, first_frame: int, narrow_packs: np.ndarray, wide_packs: np.ndarray) -> np.ndarray:
        plans = []
        for frame_offset in range(BLOCK_FRAMES):
            # Camera order of new_img: road (img), wide (big_img), like modeld and export_onnx.py.
            feed = {"new_img": np.stack((narrow_packs[frame_offset], wide_packs[frame_offset]))}
            for input_name, value in self.constants.items():
                feed[input_name] = value
            for state_name, value in self.states.items():
                feed[state_name] = value
            values = self.session.run(self.output_names, feed)
            outputs = dict(zip(self.output_names, values, strict=True))
            for state_name in self.states:
                self.states[state_name] = outputs[f"next_{state_name}"]
            if frame_offset < self.lanes:
                plans.append(outputs["outputs"][:, self.plan_slice])
        return _plan_mean(np.concatenate(plans, axis=0))


class LoggedModelV2Runner(ModelRunner):
    """The plan the car really ran: modelV2.position from the segment rlog, matched on timestampEof."""

    # modelV2.timestampEof is the road camera frame's; frame_info "fcamera/t" is that minus 8 ms.
    _EOF_OFFSET_S = 0.008
    _MATCH_TOLERANCE_S = 0.010

    def __init__(self, name: str, lanes: int) -> None:
        self.name = name
        self.lanes = lanes
        self.warmup_frames = 0
        self.plans = np.zeros((0, PLAN_POINTS, PLAN_WIDTH), dtype=np.float32)

    def reset(self, segment: Segment) -> None:
        from openpilot.tools.lib.logreader import LogReader

        rlog = segment.directory / "rlog.zst"
        if not rlog.exists():
            raise ValueError(f"segmento {segment.name}: manca rlog.zst per il modello 'logged'")
        frame_eof = segment.frame_t + self._EOF_OFFSET_S
        self.plans = np.full((segment.frame_count, PLAN_POINTS, PLAN_WIDTH), np.nan, dtype=np.float32)
        for event in LogReader(str(rlog), sort_by_time=True, only_union_types=True):
            if event.which() != "modelV2":
                continue
            model = event.modelV2
            if len(model.position.x) != PLAN_POINTS:
                continue
            eof = model.timestampEof * 1e-9
            frame_index = int(np.argmin(np.abs(frame_eof - eof)))
            if abs(frame_eof[frame_index] - eof) > self._MATCH_TOLERANCE_S:
                continue
            # Plan.POSITION columns 0:3; the rest stays NaN (only the position is evaluated).
            self.plans[frame_index, :, 0] = np.asarray(model.position.x, dtype=np.float32)
            self.plans[frame_index, :, 1] = np.asarray(model.position.y, dtype=np.float32)
            self.plans[frame_index, :, 2] = np.asarray(model.position.z, dtype=np.float32)

    def step_block(self, first_frame: int, narrow_packs: np.ndarray, wide_packs: np.ndarray) -> np.ndarray:
        block = np.full((self.lanes, PLAN_POINTS, PLAN_WIDTH), np.nan, dtype=np.float32)
        for lane in range(self.lanes):
            frame_index = first_frame + lane
            if frame_index < len(self.plans):
                block[lane] = self.plans[frame_index]
        return block


def build_runner(model: ResolvedModel, device: str, lanes: int) -> ModelRunner:
    """The runner of a resolved spec. lanes: 4 = plan at every 20 Hz frame, 1 = 5 Hz."""
    name = model.spec.label
    if model.format == FORMAT_ONNX_TEACHER:
        return load_teacher(model.path, device, lanes, name)
    if model.format == FORMAT_TORCH:
        return load_torch_student(model.path, device, lanes, name)
    if model.format == FORMAT_ONNX_STREAMING:
        return OnnxStreamingRunner(name, model.path, device, lanes)
    if model.format == FORMAT_LOGGED:
        return LoggedModelV2Runner(name, lanes)
    raise ValueError(f"formato di modello sconosciuto: {model.format}")
