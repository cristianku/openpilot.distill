"""supervised/big_model.py OnnxModule on a small graph with the ops of the big model's policies.

The full check against the teacher is helpers/check_big_model.py (0.8 GB model, GPU).
"""

import unittest

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper

from supervised.big_model import OnnxModule


def _graph() -> onnx.ModelProto:
    rng = np.random.default_rng(0)

    def init(name, value):
        return numpy_helper.from_array(np.asarray(value), name)

    initializers = [
        init("w", rng.normal(size=(16, 8)).astype(np.float32)),
        init("b", rng.normal(size=(16,)).astype(np.float32)),
        init("ln_w", rng.normal(size=(16,)).astype(np.float32)),
        init("ln_b", rng.normal(size=(16,)).astype(np.float32)),
        init("mask", rng.random((3, 3)) > 0.3),
        init("neg", np.array(-1e4, dtype=np.float32)),
        init("idx", np.array([2, 0, -1], dtype=np.int64)),
        init("shape", np.array([0, 2, 8], dtype=np.int64)),
        init("axes1", np.array([1], dtype=np.int64)),
        init("starts", np.array([1], dtype=np.int64)),
        init("ends", np.array([np.iinfo(np.int64).max], dtype=np.int64)),
        init("axes_last", np.array([-1], dtype=np.int64)),
    ]
    nodes = [
        helper.make_node("Gemm", ["x", "w", "b"], ["g"], transB=1),
        helper.make_node("LayerNormalization", ["g", "ln_w", "ln_b"], ["ln"], axis=-1, epsilon=1e-5),
        helper.make_node("Gelu", ["ln"], ["ge"], approximate="tanh"),
        helper.make_node("Reshape", ["ge", "shape"], ["r"]),
        helper.make_node("Split", ["r"], ["s0", "s1"], axis=1, num_outputs=2),
        helper.make_node("Add", ["s0", "s1"], ["q"]),
        helper.make_node("Squeeze", ["q", "axes1"], ["sq"]),
        helper.make_node("Transpose", ["sq"], ["sq_t"], perm=[1, 0]),
        helper.make_node("MatMul", ["sq", "sq_t"], ["att"]),
        helper.make_node("Not", ["mask"], ["nmask"]),
        helper.make_node("Where", ["nmask", "neg", "att"], ["masked"]),
        helper.make_node("Softmax", ["masked"], ["sm"], axis=-1),
        helper.make_node("Gather", ["sm", "idx"], ["ga"], axis=1),
        helper.make_node("Slice", ["ga", "starts", "ends", "axes1"], ["sl"]),
        helper.make_node("ReduceMean", ["sl", "axes_last"], ["out"], keepdims=0),
        helper.make_node("Sigmoid", ["out"], ["sig"]),
    ]
    graph = helper.make_graph(
        nodes,
        "small",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 8])],
        [helper.make_tensor_value_info("sig", TensorProto.FLOAT, None)],
        initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])


class OnnxModuleTest(unittest.TestCase):
    def test_matches_onnxruntime(self):
        model = _graph()
        x = np.random.default_rng(1).normal(size=(3, 8)).astype(np.float32)
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        (reference,) = session.run(None, {"x": x})

        value = OnnxModule(model)(x=torch.from_numpy(x))["sig"].detach().numpy()
        np.testing.assert_allclose(value, reference, rtol=1e-5, atol=1e-5)

    def test_weights_are_trainable(self):
        module = OnnxModule(_graph())
        x = torch.from_numpy(np.random.default_rng(2).normal(size=(3, 8)).astype(np.float32))
        module(x=x)["sig"].sum().backward()
        weights = {module.initializer_names[key]: p for key, p in module.params.items()}
        self.assertGreater(float(weights["w"].grad.abs().sum()), 0.0)
        self.assertFalse(any(torch.is_floating_point(b) for b in module.consts.buffers()))


if __name__ == "__main__":
    unittest.main()
