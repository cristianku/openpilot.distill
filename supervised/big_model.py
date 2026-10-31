"""The teacher's big model (MD v3, supervised/teachers.py DEFAULT_TEACHER) as a trainable PyTorch model.

The starting point of our own big model: same weights as the ONNX, then new heads (speed limits,
perception) and fine-tuning on our drives. It runs where the big model runs in the car, on comma's
external GPU (Chestnut), so it keeps the big model's size.

BigVision     the vision part, rebuilt as the timm network it was exported from: convnext_xlarge
              (depths 3/3/27/3, dims 256..2048), 24 input channels (img + big_img, 2 frames of 6 YUV
              channels each), layer norm eps 1e-5, and a 1x1 conv head 2048 -> 512. Output: 4 x 8 = 32
              spatial tokens of 512 per step, the hidden_state of the teacher.
OnnxModule    the policies part (point/off/on policy transformers, 46 M parameters) run node by node
              from its ONNX graph, with every float initializer as an nn.Parameter: exact by
              construction, and trainable. The dense 9-step graph of supervised/onnx_helpers.py
              make_dense_queue is the one used, so it takes the 5 Hz sequence of the dataset.
BigSupercombo vision + policies: uint8 frames of 9 steps -> outputs of the 9 steps (+ hidden_state).

helpers/check_big_model.py checks that it gives the outputs of onnxruntime on the same graphs.
"""

from __future__ import annotations

import base64
import pickle
from pathlib import Path

import numpy as np
import onnx
import timm
import torch
import torch.nn.functional as F
from onnx import numpy_helper
from torch import nn

from .model import ModelInputs

VISION_MODEL_NAME = "convnext_xlarge"
VISION_PREFIX = "model.vision."
_INT64_MAX = np.iinfo(np.int64).max


class BigVision(nn.Module):
    def __init__(self, in_chans: int = 24, output_dim: int = 512, norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            VISION_MODEL_NAME, pretrained=False, in_chans=in_chans, num_classes=0, global_pool="", norm_eps=norm_eps
        )
        # timm's head with global_pool="" is norm + identity: the big model adds a 1x1 conv on the 4 x 8 grid.
        self.encoder.head.fc = nn.Conv2d(self.encoder.num_features, output_dim, kernel_size=1)
        self.output_dim = output_dim
        self.register_buffer("mean", torch.full((1, in_chans, 1, 1), 127.5))
        self.register_buffer("std", torch.full((1, in_chans, 1, 1), 63.75))

    def forward(self, img: torch.Tensor, big_img: torch.Tensor) -> torch.Tensor:
        dtype = self.encoder.stem[0].weight.dtype
        x = torch.cat((img, big_img), dim=1).to(dtype)
        x = self.encoder((x - self.mean.to(dtype)) / self.std.to(dtype))
        return x.permute(0, 2, 3, 1).flatten(1, 2)

    @torch.no_grad()
    def load_onnx(self, vision: onnx.ModelProto) -> None:
        """Copy the weights of the vision graph of split_queue_supercombo (names of the export)."""

        graph = vision.graph
        weights = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
        consumers: dict[str, list[onnx.NodeProto]] = {}
        for node in graph.node:
            for name in node.input:
                consumers.setdefault(name, []).append(node)

        state: dict[str, np.ndarray] = {}
        block = None
        for node in graph.node:
            named = [name for name in node.input if name.startswith(VISION_PREFIX)]
            if node.op_type == "Conv" and named and "conv_dw" in named[0]:
                block = named[0].removesuffix("conv_dw.weight")
            if node.op_type == "MatMul" and node.input[1] in weights:
                # Linear exported as MatMul [in, out] + Add bias: the name is on the bias.
                (add,) = consumers[node.output[0]]
                bias = add.input[1] if add.input[0] == node.output[0] else add.input[0]
                state[bias.removesuffix("bias") + "weight"] = weights[node.input[1]].T
            elif node.op_type == "Mul" and node.input[1] in weights and not named:
                # Layer scale of the current block, exported as a [1, C, 1, 1] constant.
                state[block + "gamma"] = weights[node.input[1]].reshape(-1)
        for name, value in weights.items():
            if name.startswith(VISION_PREFIX):
                state[name] = value

        own = self.state_dict()
        loaded = {}
        for name, value in state.items():
            key = name.removeprefix(VISION_PREFIX).removeprefix("_")
            if key not in own:
                raise ValueError(f"vision weight {name}: no {key} in BigVision")
            if tuple(own[key].shape) != value.shape:
                raise ValueError(f"vision weight {name}: shape {value.shape}, BigVision {tuple(own[key].shape)}")
            loaded[key] = torch.from_numpy(np.array(value))
        missing = sorted(set(own) - set(loaded))
        if missing:
            raise ValueError(f"vision weights missing from the ONNX: {missing[:5]} ({len(missing)})")
        self.load_state_dict({k: v.to(own[k].dtype) for k, v in loaded.items()})


def _tensor_list(value) -> list[int]:
    return [int(v) for v in value.reshape(-1).tolist()]


class OnnxModule(nn.Module):
    """Runs an ONNX graph with torch ops; float initializers become parameters, the rest buffers.

    Only the ops of the big model's policies graph are implemented (an unknown op raises). Float
    casts go to the dtype of the parameters, so the same graph runs in float32 for training and in
    float16 like the export.
    """

    def __init__(self, model: onnx.ModelProto) -> None:
        super().__init__()
        graph = model.graph
        self.nodes = list(graph.node)
        self.input_names = [value.name for value in graph.input]
        self.output_names = [value.name for value in graph.output]
        self.params = nn.ParameterDict()
        self.consts = nn.Module()
        self._keys: dict[str, tuple[str, str]] = {}
        for index, tensor in enumerate(graph.initializer):
            value = torch.from_numpy(np.array(numpy_helper.to_array(tensor)))
            key = f"w{index}"
            if value.is_floating_point():
                self.params[key] = nn.Parameter(value.float())
                self._keys[tensor.name] = ("params", key)
            else:
                self.consts.register_buffer(key, value)
                self._keys[tensor.name] = ("consts", key)
        self.initializer_names = {key: name for name, (_, key) in self._keys.items()}
        for node in self.nodes:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    raise NotImplementedError(f"tensor attribute in {node.op_type}")
        self._attributes = [{a.name: onnx.helper.get_attribute_value(a) for a in node.attribute} for node in self.nodes]

    def _value(self, name: str) -> torch.Tensor:
        group, key = self._keys[name]
        return self.params[key] if group == "params" else getattr(self.consts, key)

    def forward(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        dtype = next(iter(self.params.values())).dtype
        env: dict[str, torch.Tensor | None] = {"": None}
        env.update(inputs)
        for node, attrs in zip(self.nodes, self._attributes, strict=True):
            args = []
            for name in node.input:
                if name not in env:
                    env[name] = self._value(name)
                args.append(env[name])
            outputs = self._run(node.op_type, args, attrs, dtype)
            if not isinstance(outputs, (tuple, list)):
                outputs = (outputs,)
            for name, value in zip(node.output, outputs, strict=False):
                env[name] = value
        return {name: env[name] for name in self.output_names}

    @staticmethod
    def _run(op: str, x: list, a: dict, dtype: torch.dtype):
        if op == "Add":
            return x[0] + x[1]
        if op == "Sub":
            return x[0] - x[1]
        if op == "Mul":
            return x[0] * x[1]
        if op == "Div":
            return x[0] / x[1]
        if op == "MatMul":
            return x[0] @ x[1]
        if op == "Gemm":
            left = x[0].transpose(-1, -2) if a.get("transA", 0) else x[0]
            right = x[1].transpose(-1, -2) if a.get("transB", 0) else x[1]
            out = a.get("alpha", 1.0) * (left @ right)
            return out + a.get("beta", 1.0) * x[2] if len(x) > 2 and x[2] is not None else out
        if op == "Gelu":
            return F.gelu(x[0], approximate=a.get("approximate", b"none").decode())
        if op == "Sigmoid":
            return torch.sigmoid(x[0])
        if op == "Tanh":
            return torch.tanh(x[0])
        if op == "Softmax":
            return torch.softmax(x[0], dim=a.get("axis", -1))
        if op == "LayerNormalization":
            axis = a.get("axis", -1) % x[0].dim()
            shape = x[0].shape[axis:]
            bias = x[2] if len(x) > 2 else None
            return F.layer_norm(x[0], shape, x[1], bias, a.get("epsilon", 1e-5))
        if op == "ReduceMean":
            axes = _tensor_list(x[1]) if len(x) > 1 and x[1] is not None else list(range(x[0].dim()))
            return x[0].mean(dim=axes, keepdim=bool(a.get("keepdims", 1)))
        if op == "Cast":
            to = a["to"]
            if to in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16):
                return x[0].to(dtype)
            if to == onnx.TensorProto.INT64:
                return x[0].long()
            if to == onnx.TensorProto.BOOL:
                return x[0].bool()
            raise NotImplementedError(f"Cast to {to}")
        if op == "Not":
            return ~x[0]
        if op == "Where":
            return torch.where(x[0], x[1], x[2])
        if op == "Reshape":
            shape = _tensor_list(x[1])
            if not a.get("allowzero", 0):
                shape = [x[0].shape[i] if s == 0 else s for i, s in enumerate(shape)]
            return x[0].reshape(shape)
        if op == "Transpose":
            return x[0].permute(a["perm"])
        if op == "Squeeze":
            axes = _tensor_list(x[1]) if len(x) > 1 else [i for i, s in enumerate(x[0].shape) if s == 1]
            out = x[0]
            for axis in sorted((axis % out.dim() for axis in axes), reverse=True):
                out = out.squeeze(axis)
            return out
        if op == "Unsqueeze":
            out = x[0]
            rank = out.dim() + len(_tensor_list(x[1]))
            for axis in sorted(axis % rank for axis in _tensor_list(x[1])):
                out = out.unsqueeze(axis)
            return out
        if op == "Concat":
            return torch.cat(x, dim=a["axis"])
        if op == "Split":
            axis = a.get("axis", 0)
            if len(x) > 1 and x[1] is not None:
                return torch.split(x[0], _tensor_list(x[1]), dim=axis)
            parts = a["num_outputs"]
            return torch.split(x[0], -(-x[0].shape[axis] // parts), dim=axis)
        if op == "Gather":
            axis = a.get("axis", 0) % x[0].dim()
            indices = x[1].long() % x[0].shape[axis]
            out = torch.index_select(x[0], axis, indices.reshape(-1))
            return out.reshape(*x[0].shape[:axis], *indices.shape, *x[0].shape[axis + 1 :])
        if op == "Slice":
            starts, ends = _tensor_list(x[1]), _tensor_list(x[2])
            axes = _tensor_list(x[3]) if len(x) > 3 and x[3] is not None else list(range(len(starts)))
            steps = _tensor_list(x[4]) if len(x) > 4 and x[4] is not None else [1] * len(starts)
            index = [slice(None)] * x[0].dim()
            for start, end, axis, step in zip(starts, ends, axes, steps, strict=True):
                if step <= 0:
                    raise NotImplementedError("Slice with a negative step")
                index[axis % x[0].dim()] = slice(start, None if end >= _INT64_MAX // 2 else end, step)
            return x[0][tuple(index)]
        if op == "Expand":
            shape = torch.broadcast_shapes(x[0].shape, tuple(_tensor_list(x[1])))
            return x[0].expand(shape)
        if op == "Shape":
            shape = torch.tensor(x[0].shape, dtype=torch.long, device=x[0].device)
            return shape[a.get("start", 0) : a.get("end", None)]
        if op == "Contiguous":
            return x[0].contiguous()
        raise NotImplementedError(f"ONNX op {op}")


class BigSupercombo(nn.Module):
    """The teacher's big model in PyTorch: 5 Hz sequence of 9 steps -> outputs of each step."""

    def __init__(self, vision: BigVision, policies: OnnxModule, output_slices: dict) -> None:
        super().__init__()
        self.vision = vision
        self.policies = policies
        self.output_slices = output_slices
        self.features_name = policies.input_names[0]

    @classmethod
    def from_onnx(cls, model: str | Path | onnx.ModelProto) -> BigSupercombo:
        from .onnx_helpers import make_dense_queue, split_queue_supercombo

        model = model if hasattr(model, "graph") else onnx.load(str(model))
        vision_graph, policies_graph = split_queue_supercombo(model)
        policies_graph = make_dense_queue(policies_graph)
        return cls.from_split(vision_graph, policies_graph)

    @classmethod
    def from_split(cls, vision_graph: onnx.ModelProto, policies_graph: onnx.ModelProto) -> BigSupercombo:
        vision = BigVision()
        vision.load_onnx(vision_graph)
        metadata = {prop.key: prop.value for prop in policies_graph.metadata_props}
        output_slices = pickle.loads(base64.b64decode(metadata["output_slices"]))
        return cls(vision, OnnxModule(policies_graph), output_slices)

    def encode(self, img: torch.Tensor, big_img: torch.Tensor) -> torch.Tensor:
        """[B, T, 12, 128, 256] uint8 frames -> [B, T, 32, 512] features."""

        batch, steps = img.shape[:2]
        features = self.vision(img.flatten(0, 1), big_img.flatten(0, 1))
        return features.reshape(batch, steps, *features.shape[1:])

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = self.encode(inputs[ModelInputs.IMG], inputs[ModelInputs.BIG_IMG])
        dtype = features.dtype
        outputs = self.policies(
            **{
                self.features_name: features,
                ModelInputs.DESIRE: inputs[ModelInputs.DESIRE].to(dtype),
                ModelInputs.TRAFFIC: inputs[ModelInputs.TRAFFIC].to(dtype),
                ModelInputs.ACTION_T: inputs[ModelInputs.ACTION_T].to(dtype),
            }
        )
        outputs["hidden_state"] = features
        return outputs
