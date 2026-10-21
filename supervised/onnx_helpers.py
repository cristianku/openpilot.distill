from __future__ import annotations

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.utils import Extractor


def _add_int64(graph: onnx.GraphProto, name: str, value) -> str:
    graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=np.int64), name))
    return name


def make_batch_dynamic(model: onnx.ModelProto, batch_dim: str = "batch") -> onnx.ModelProto:
    """Make an exported batch-1 graph accept a dynamic leading batch dimension.

    The rewrite handles constant ``Reshape`` targets and batch-1 constants
    concatenated along a non-batch axis. It mutates and returns ``model``.
    """

    graph = model.graph
    initializers = {value.name: value for value in graph.initializer}

    reshape_nodes: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        if node.op_type != "Reshape" or node.input[1] not in initializers:
            continue
        shape = numpy_helper.to_array(initializers[node.input[1]])
        if shape.ndim == 1 and len(shape) and shape[0] == 1:
            reshape_nodes.setdefault(node.input[1], []).append(node)

    for name, nodes in reshape_nodes.items():
        shape = numpy_helper.to_array(initializers[name]).copy()
        shape[0] = 0 if np.any(shape[1:] == -1) else -1
        initializers[name].CopyFrom(numpy_helper.from_array(shape, name))
        for node in nodes:
            for attribute in node.attribute:
                if attribute.name == "allowzero":
                    attribute.i = 0

    nodes = []
    for node_index, node in enumerate(graph.node):
        if node.op_type == "Concat":
            axis = next(attribute.i for attribute in node.attribute if attribute.name == "axis")
            source = next((name for name in node.input if name not in initializers), None)
            if axis != 0 and source is not None:
                for input_index, name in enumerate(node.input):
                    if name not in initializers:
                        continue
                    value = numpy_helper.to_array(initializers[name])
                    if value.ndim < 2 or value.shape[0] != 1:
                        continue

                    prefix = f"_dynamic_batch_{node_index}_{input_index}"
                    batch_shape = f"{prefix}_batch_shape"
                    target_shape = f"{prefix}_target_shape"
                    expanded = f"{prefix}_expanded"
                    nodes.extend(
                        (
                            helper.make_node("Shape", [source], [batch_shape], name=f"{prefix}_batch", start=0, end=1),
                            helper.make_node(
                                "Concat",
                                [batch_shape, _add_int64(graph, f"{prefix}_tail", value.shape[1:])],
                                [target_shape],
                                name=f"{prefix}_target",
                                axis=0,
                            ),
                            helper.make_node("Expand", [name, target_shape], [expanded], name=f"{prefix}_expand"),
                        )
                    )
                    node.input[input_index] = expanded
        nodes.append(node)

    del graph.node[:]
    graph.node.extend(nodes)

    initializer_names = set(initializers)
    for value in (*graph.input, *graph.output):
        if value.name in initializer_names:
            continue
        dimensions = value.type.tensor_type.shape.dim
        if dimensions and dimensions[0].dim_value == 1:
            dimensions[0].dim_param = batch_dim

    del graph.value_info[:]
    return model


def _rename_value(graph: onnx.GraphProto, old: str, new: str) -> None:
    for node in graph.node:
        for names in (node.input, node.output):
            for index, name in enumerate(names):
                if name == old:
                    names[index] = new
    for values in (graph.input, graph.output, graph.value_info, graph.initializer):
        for value in values:
            if value.name == old:
                value.name = new


def _prune_initializers(graph: onnx.GraphProto) -> None:
    used = {name for node in graph.node for name in node.input}
    initializers = [value for value in graph.initializer if value.name in used]
    del graph.initializer[:]
    graph.initializer.extend(initializers)


def split_supercombo(model: onnx.ModelProto) -> tuple[onnx.ModelProto, onnx.ModelProto]:
    """Split a streaming Supercombo into dynamic-batch vision and policy graphs."""

    graph = model.graph
    producers = {output: node for node in graph.node for output in node.output}
    feature_concat = next(node for node in graph.node if "features_buffer" in node.input)
    current = producers[next(name for name in feature_concat.input if name != "features_buffer")].input[0]
    images = [value.name for value in graph.input if "img" in value.name]
    excluded = {*images, "features_buffer"}
    policy_inputs = [
        feature_concat.output[0],
        current,
        *(value.name for value in graph.input if value.name not in excluded),
    ]
    outputs = [value.name for value in graph.output]

    extractor = Extractor(model)
    vision = extractor.extract_model(images, [current])
    policies = extractor.extract_model(policy_inputs, outputs)

    policies.graph.input.remove(next(value for value in policies.graph.input if value.name == current))
    _rename_value(vision.graph, current, "features")
    _rename_value(policies.graph, feature_concat.output[0], "features")
    _add_int64(policies.graph, "_last_feature", -1)
    policies.graph.node.insert(
        0,
        helper.make_node("Gather", ["features", "_last_feature"], [current], axis=1, name="last_feature"),
    )

    vision.graph.name = "vision"
    policies.graph.name = "policies"
    policies.metadata_props.extend(model.metadata_props)
    for part in (vision, policies):
        make_batch_dynamic(part)
    features = next(value for value in policies.graph.input if value.name == "features")
    features.type.tensor_type.shape.dim[1].dim_param = "time"
    return vision, policies


def make_dense(policies: onnx.ModelProto) -> onnx.ModelProto:
    """Make a split policy graph return its dense training outputs."""

    graph = policies.graph
    initializers = {value.name: value for value in graph.initializer}
    producers = {output: node for node in graph.node for output in node.output}
    features = next(value for value in graph.input if value.name == "features")
    desire = next(value for value in graph.input if value.name == "desire_pulse")
    feature_dim = features.type.tensor_type.shape.dim[-1].dim_value
    desire_dim = desire.type.tensor_type.shape.dim[-1].dim_value
    desire_window = desire.type.tensor_type.shape.dim[1].dim_value

    temporal_selects = []
    for node in graph.node:
        if node.op_type != "Gather" or node.input[1] not in initializers:
            continue
        axis = next(attribute.i for attribute in node.attribute if attribute.name == "axis")
        index = numpy_helper.to_array(initializers[node.input[1]])
        if axis == 1 and index.ndim == 0 and index >= 0:
            temporal_selects.append(node)
    timesteps = int(numpy_helper.to_array(initializers[temporal_selects[0].input[1]])) + 1

    flat_shape = _add_int64(graph, "_dense_flat_shape", [-1, feature_dim])
    point_select = next(node for node in graph.node if node.op_type == "Gather" and node.input[0] == "features")
    for node in (point_select, *temporal_selects):
        node.op_type = "Reshape"
        node.input[:] = [node.input[0], flat_shape]
        del node.attribute[:]

    desire_reshape = next(node for node in graph.node if "desire_pulse" in node.input)
    for node in graph.node:
        producer = producers.get(node.input[0]) if node.input else None
        if node.op_type == "Reshape" and producer and any("desire" in name for name in producer.input):
            node.input[1] = _add_int64(graph, f"_dense_{node.output[0]}_shape", [-1, timesteps, feature_dim])

    desire_indices = _add_int64(
        graph,
        "_dense_desire_indices",
        np.arange(timesteps)[:, None] + np.arange(desire_window)[None],
    )
    dense_desire = "_dense_desire"
    node_index = next(index for index, node in enumerate(graph.node) if node.name == desire_reshape.name)
    graph.node.insert(
        node_index,
        helper.make_node("Gather", ["desire_pulse", desire_indices], [dense_desire], axis=1),
    )
    desire_reshape.input[0] = dense_desire
    shape_name = desire_reshape.input[1]
    initializers[shape_name].CopyFrom(
        numpy_helper.from_array(np.asarray([-1, desire_window * desire_dim], dtype=np.int64), shape_name)
    )

    output = graph.output[0]
    output_size = output.type.tensor_type.shape.dim[-1].dim_value
    flat_output = f"_{output.name}_flat"
    output_node = next(node for node in graph.node if output.name in node.output)
    output_node.output[:] = [flat_output if name == output.name else name for name in output_node.output]
    output_shape = _add_int64(graph, "_dense_output_shape", [-1, timesteps, output_size])
    graph.node.append(helper.make_node("Reshape", [flat_output, output_shape], [output.name], name="dense_outputs"))

    features.type.tensor_type.shape.dim[1].dim_value = timesteps
    desire.type.tensor_type.shape.dim[1].dim_value = desire_window + timesteps - 1
    output.CopyFrom(
        helper.make_tensor_value_info(
            output.name,
            output.type.tensor_type.elem_type,
            ["batch", timesteps, output_size],
        )
    )
    _prune_initializers(graph)
    return policies


# Queue-format big models: commaai/openpilot_driving_models on Hugging Face ("MD v3" and later,
# openpilot PR #38916). modeld calls them at 20 Hz with ONE new frame per camera and keeps the
# history inside the ONNX: every "state_x" input has a "next_state_x" output that modeld gives back
# as input on the next frame. split_queue_supercombo + make_dense_queue turn them into the same two
# graphs that split_supercombo + make_dense give for the old interface:
#   vision:   img, big_img [batch, 12, 128, 256] uint8 (frames t-4 and t)  ->  features [batch, 32, 512]
#   policies: features [batch, 9, 32, 512], desire_pulse [batch, 33, 8] (5 Hz),
#             traffic_convention [batch, 2], action_t [batch, 2]  ->  outputs [batch, 9, 18452]
# The cut points are found from the structure of the graph, never from internal tensor names (they
# change with every torch export): only the names modeld itself relies on are used.

# Inputs that modeld writes itself every frame; every other input must be a state_x queue.
QUEUE_DEVICE_INPUTS = ("new_img", "desire", "traffic_convention", "action_t")
# Order of the cameras in new_img, as modeld fills it (vision_input_names in modeld.py).
QUEUE_CAMERA_INPUTS = ("img", "big_img")
# 20 Hz camera, 5 Hz dataset steps (supervised/dataset.py): every image of the dataset is the frames
# t-4 and t, and the history steps are 4 frames apart. The model must cut its queues the same way.
QUEUE_FRAME_SKIP = 4


def _value_dims(value: onnx.ValueInfoProto) -> list:
    """Dimensions of a graph input/output/value_info: an int, or the name of a symbolic dimension."""

    dims = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append(dim.dim_value)
    return dims


def _shape(model: onnx.ModelProto, name: str) -> list:
    """Shape of a tensor of the model, from its value_info (the torch export writes it for every tensor)."""

    for values in (model.graph.input, model.graph.output, model.graph.value_info):
        for value in values:
            if value.name == name:
                return _value_dims(value)
    raise ValueError(f"no shape for tensor {name} (value_info missing from the export?)")


def _constant(graph: onnx.GraphProto, name: str):
    """Value of an initializer or of a Constant node, None when the tensor is computed."""

    for value in graph.initializer:
        if value.name == name:
            return numpy_helper.to_array(value)
    for node in graph.node:
        if node.op_type == "Constant" and node.output[0] == name:
            for attribute in node.attribute:
                if attribute.name == "value":
                    return numpy_helper.to_array(attribute.t)
    return None


def _attribute(node: onnx.NodeProto, name: str, default=None):
    for attribute in node.attribute:
        if attribute.name == name:
            return helper.get_attribute_value(attribute)
    return default


def _consumers(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    """For every tensor, the nodes that read it."""

    consumers = {}
    for node in graph.node:
        for name in node.input:
            if name not in consumers:
                consumers[name] = []
            consumers[name].append(node)
    return consumers


def _producers(graph: onnx.GraphProto) -> dict[str, onnx.NodeProto]:
    """For every computed tensor, the node that writes it."""

    producers = {}
    for node in graph.node:
        for name in node.output:
            producers[name] = node
    return producers


def is_queue_supercombo(model: onnx.ModelProto) -> bool:
    """True for the queue-format interface of the Hugging Face big models (history kept in state_feat_q)."""

    for value in model.graph.input:
        if value.name == "state_feat_q":
            return True
    return False


def _check_frame_slice(model: onnx.ModelProto, node: onnx.NodeProto, what: str) -> None:
    """A Slice on axis 0 that keeps every QUEUE_FRAME_SKIP-th entry, from the first to the end of the queue."""

    graph = model.graph
    if len(node.input) < 5:
        raise ValueError(f"{what}: Slice without explicit axes/steps")
    starts = _constant(graph, node.input[1])
    ends = _constant(graph, node.input[2])
    axes = _constant(graph, node.input[3])
    steps = _constant(graph, node.input[4])
    if starts is None or ends is None or axes is None or steps is None:
        raise ValueError(f"{what}: Slice with computed starts/ends/axes/steps")
    if starts.tolist() != [0] or axes.tolist() != [0] or steps.tolist() != [QUEUE_FRAME_SKIP]:
        raise ValueError(
            f"{what}: Slice starts={starts.tolist()} axes={axes.tolist()} steps={steps.tolist()}, "
            f"expected [0] [0] [{QUEUE_FRAME_SKIP}] (5 Hz steps of a 20 Hz camera, like supervised/dataset.py)"
        )
    # The end must reach the end of the queue (INT64_MAX in the torch export): a shorter one would
    # drop the newest entries, and the kept ones would no longer end at the current frame.
    queue_length = _shape(model, node.input[0])[0]
    end_values = ends.tolist()
    if not isinstance(queue_length, int) or len(end_values) != 1 or end_values[0] < queue_length:
        raise ValueError(f"{what}: Slice ends={end_values} on a queue of {queue_length}, expected the whole queue")


def find_queue_boundaries(model: onnx.ModelProto) -> dict:
    """Tensors where a queue-format model is cut, found from the structure of the graph.

    Returned keys:
      cameras:  {"img": tensor, "big_img": tensor}, the [1, 12, H, W] uint8 image (frames t-4, t) of each camera
      current:  the [1, S, D] vision features of the current frame
      tokens:   the [1, T, S, D] features of the T history steps that enter the policies
      desire:   the [1, N, 8] desire at 5 Hz (after the max over 4 frames), before its cast to f16
      outputs:  the graph output with the driving outputs

    Raises ValueError, saying why, for a model the distill teacher cannot use.
    """

    graph = model.graph
    consumers = _consumers(graph)
    producers = _producers(graph)

    input_names = []
    for value in graph.input:
        input_names.append(value.name)
    output_names = []
    for value in graph.output:
        output_names.append(value.name)

    # Every input must be one that the dataset can give (cameras, desire, traffic_convention,
    # action_t) or a queue that the model fills itself.
    states = []
    unknown = []
    for name in input_names:
        if name in QUEUE_DEVICE_INPUTS:
            continue
        if "next_" + name in output_names:
            states.append(name)
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(
            f"inputs {', '.join(unknown)}: the distill teacher cannot feed them. The dataset and the student "
            "only have the two cameras, desire, traffic_convention and action_t; a model that also reads other "
            "inputs (e.g. the N16 nav model: nav_map, navigation and their queues) would give policy targets "
            "that depend on a context the student never sees"
        )
    for name in QUEUE_DEVICE_INPUTS:
        if name not in input_names:
            raise ValueError(f"no input '{name}': not the queue-format interface of openpilot's modeld")

    outputs = None
    for name in output_names:
        if not name.startswith("next_"):
            if outputs is not None:
                raise ValueError(f"more than one driving output: {outputs}, {name}")
            outputs = name
    if outputs is None:
        raise ValueError("no driving output (every output is a next_ state)")

    # Image queue: the uint8 state. Its next_ output (the old frames + the new one) is read once per
    # camera by a Gather on axis 0; the chain after it keeps the frames t-4 and t and reshapes them
    # to the 12-channel image of that camera.
    image_queue = None
    for name in states:
        for value in graph.input:
            if value.name == name and value.type.tensor_type.elem_type == onnx.TensorProto.UINT8:
                if image_queue is not None:
                    raise ValueError(f"two uint8 queues: {image_queue}, {name}")
                image_queue = name
    if image_queue is None:
        raise ValueError("no uint8 image queue")

    cameras = {}
    for node in consumers.get("next_" + image_queue, []):
        if node.op_type != "Gather" or _attribute(node, "axis", 0) != 0:
            continue
        index = _constant(graph, node.input[1])
        if index is None or index.ndim != 0:
            continue
        camera = int(index)
        if camera < 0 or camera >= len(QUEUE_CAMERA_INPUTS):
            raise ValueError(f"image queue: camera index {camera}")
        queue_frames = _shape(model, node.output[0])[0]
        if queue_frames != QUEUE_FRAME_SKIP + 1:
            raise ValueError(f"image queue of {queue_frames} frames, expected {QUEUE_FRAME_SKIP + 1} (t-4 .. t)")
        frame_slices = 0
        tensor = node.output[0]
        while True:
            next_nodes = consumers.get(tensor, [])
            if len(next_nodes) != 1:
                break
            next_node = next_nodes[0]
            if next_node.op_type == "Slice":
                _check_frame_slice(model, next_node, f"camera {camera}")
                frame_slices = frame_slices + 1
            elif next_node.op_type != "Reshape":
                break
            tensor = next_node.output[0]
        if frame_slices != 1:
            raise ValueError(f"camera {camera}: {frame_slices} frame Slices, expected 1 (frames t-4 and t)")
        shape = _shape(model, tensor)
        if len(shape) != 4 or shape[0] != 1:
            raise ValueError(f"camera {camera}: unexpected image shape {shape}")
        cameras[QUEUE_CAMERA_INPUTS[camera]] = tensor
    for name in QUEUE_CAMERA_INPUTS:
        if name not in cameras:
            raise ValueError(f"no image of camera '{name}' cut from the image queue")

    # Desire queue: the state whose next_ output is the Concat that appends the "desire" input.
    desire_queue = None
    for node in consumers.get("desire", []):
        for next_node in consumers.get(node.output[0], []):
            if next_node.op_type == "Concat" and next_node.output[0].startswith("next_"):
                desire_queue = next_node.output[0].removeprefix("next_")
    if desire_queue not in states:
        raise ValueError("no desire queue fed by the 'desire' input")

    # Feature queue: the only other state. More queues (map, navigation...) are not supported.
    others = []
    for name in states:
        if name != image_queue and name != desire_queue:
            others.append(name)
    if len(others) != 1:
        raise ValueError(f"expected one feature queue besides {image_queue} and {desire_queue}, found {others}")
    feature_queue = others[0]

    # The history keeps every 4th feature of the queue, ending 4 frames before the current one.
    queue_length = _shape(model, feature_queue)[0]
    history_slices = 0
    for node in consumers.get(feature_queue, []):
        if node.op_type != "Slice" or len(node.input) < 5:
            continue
        steps = _constant(graph, node.input[4])
        if steps is not None and steps.tolist() != [1]:
            _check_frame_slice(model, node, feature_queue)
            history_slices = history_slices + 1
    if history_slices != 1 or queue_length % QUEUE_FRAME_SKIP != 0:
        raise ValueError(
            f"{feature_queue} [{queue_length}, ...]: {history_slices} history Slices, expected one "
            f"[0::{QUEUE_FRAME_SKIP}] on a queue whose length is a multiple of {QUEUE_FRAME_SKIP}"
        )

    # History concat: walking forward from the feature queue, the first Concat that does not build a
    # next_ state. One input is the history, the other one the Unsqueeze of the current features.
    history_concat = None
    frontier = [feature_queue]
    visited = set()
    while frontier and history_concat is None:
        next_frontier = []
        for tensor in frontier:
            for node in consumers.get(tensor, []):
                if node.output[0] in visited:
                    continue
                visited.add(node.output[0])
                if node.op_type == "Concat" and node.output[0] not in output_names:
                    history_concat = node
                    break
                for output in node.output:
                    next_frontier.append(output)
            if history_concat is not None:
                break
        frontier = next_frontier
    if history_concat is None:
        raise ValueError(f"no history Concat after {feature_queue}")

    current = None
    for name in history_concat.input:
        producer = producers.get(name)
        if producer is not None and producer.op_type == "Unsqueeze":
            current = producer.input[0]
    if current is None:
        raise ValueError("history Concat without the Unsqueeze of the current features")
    current_shape = _shape(model, current)
    if len(current_shape) != 3 or current_shape[0] != 1:
        raise ValueError(f"current features {current} have shape {current_shape}, expected [1, S, D]")
    feature_size = current_shape[1] * current_shape[2]
    if _shape(model, feature_queue)[-1] != feature_size:
        raise ValueError(f"{feature_queue} does not hold the {current_shape[1:]} current features")

    # The current features are also what the outputs carry as hidden_state: a Reshape of them is one
    # input of the Concat that builds the outputs vector.
    output_node = producers[outputs]
    walked = 0
    while output_node.op_type != "Concat" and walked < 4:
        output_node = producers[output_node.input[0]]
        walked = walked + 1
    if output_node.op_type != "Concat":
        raise ValueError(f"no Concat builds {outputs}")
    hidden_found = False
    for name in output_node.input:
        producer = producers.get(name)
        if producer is not None and producer.op_type == "Reshape" and producer.input[0] == current:
            hidden_found = True
    if not hidden_found:
        raise ValueError("the hidden_state of the outputs is not the current features")

    # Tokens: from the history concat follow the chain of single-consumer nodes (the window of the
    # last T steps: Transpose, GatherND, Transpose in MD v3); the tensor read by several nodes is the
    # input of the policies.
    tokens = history_concat.output[0]
    while len(consumers.get(tokens, [])) == 1:
        tokens = consumers[tokens][0].output[0]
    token_shape = _shape(model, tokens)
    if len(token_shape) != 4 or token_shape[0] != 1 or token_shape[2:] != current_shape[1:]:
        raise ValueError(f"policy tokens {tokens} have shape {token_shape}, current features {current_shape}")
    timesteps = token_shape[1]

    # Desire at 5 Hz: the desire queue is grouped by 4 frames and reduced with a ReduceMax; the first
    # [1, N, 8] tensor after it is the 5 Hz sequence, taken before its cast to f16.
    desire = None
    seen_reduce = False
    tensor = "next_" + desire_queue
    for _ in range(10):
        next_nodes = consumers.get(tensor, [])
        if len(next_nodes) != 1:
            break
        if next_nodes[0].op_type == "ReduceMax":
            seen_reduce = True
        tensor = next_nodes[0].output[0]
        shape = _shape(model, tensor)
        if seen_reduce and len(shape) == 3 and shape[0] == 1:
            desire = tensor
            break
    if desire is None:
        raise ValueError("no 5 Hz desire after the ReduceMax of the desire queue")
    desire_steps = _shape(model, desire)[1]

    # The desire windows: a Gather on axis 1 with constant indices, a few nodes after the 5 Hz
    # desire. Step k must see the 5 Hz entries k .. k + W - 1, so that the policies of every step
    # are already there (make_dense_queue does not touch the desire).
    windows = None
    frontier = [desire]
    for _ in range(4):
        next_frontier = []
        for tensor in frontier:
            for node in consumers.get(tensor, []):
                if node.op_type == "Gather" and _attribute(node, "axis", 0) == 1:
                    index = _constant(graph, node.input[1])
                    if index is not None and index.ndim == 1:
                        windows = index
                for output in node.output:
                    next_frontier.append(output)
        if windows is not None:
            break
        frontier = next_frontier
    if windows is None:
        raise ValueError("no Gather of the desire windows after the 5 Hz desire")
    window = desire_steps - timesteps + 1
    expected = []
    for step in range(timesteps):
        for offset in range(window):
            expected.append(step + offset)
    if windows.tolist() != expected:
        raise ValueError(f"the desire windows are not {timesteps} windows of {window} steps, one per history step")

    return {
        "cameras": cameras,
        "current": current,
        "tokens": tokens,
        "desire": desire,
        "outputs": outputs,
    }


def _check_computed(part: onnx.ModelProto) -> None:
    """Every node input must come from a node, an initializer or a graph input of the part.

    Extractor keeps only the given inputs: a node that still needs a queue (or anything else that
    was not cut) would read a tensor that nothing computes.
    """

    available = set()
    for value in part.graph.input:
        available.add(value.name)
    for value in part.graph.initializer:
        available.add(value.name)
    for node in part.graph.node:
        for name in node.input:
            if name != "" and name not in available:
                raise ValueError(f"{part.graph.name}: {node.name} reads {name}, which the split graph does not compute")
        for name in node.output:
            available.add(name)


def split_queue_supercombo(model: onnx.ModelProto) -> tuple[onnx.ModelProto, onnx.ModelProto]:
    """Split a queue-format Supercombo into dynamic-batch vision and last-step policy graphs.

    The policies graph reads the current features with a Gather of the last step of "features", like
    split_supercombo does for the old interface: make_dense_queue turns it into every step.
    """

    found = find_queue_boundaries(model)
    images = [found["cameras"]["img"], found["cameras"]["big_img"]]
    current = found["current"]
    tokens = found["tokens"]
    desire = found["desire"]
    outputs = found["outputs"]

    extractor = Extractor(model)
    vision = extractor.extract_model(images, [current])
    policies = extractor.extract_model([tokens, current, desire, "traffic_convention", "action_t"], [outputs])
    vision.graph.name = "vision"
    policies.graph.name = "policies"
    _check_computed(vision)
    _check_computed(policies)

    for value in policies.graph.input:
        if value.name == current:
            policies.graph.input.remove(value)
            break
    _rename_value(vision.graph, images[0], QUEUE_CAMERA_INPUTS[0])
    _rename_value(vision.graph, images[1], QUEUE_CAMERA_INPUTS[1])
    _rename_value(vision.graph, current, "features")
    _rename_value(policies.graph, tokens, "features")
    _rename_value(policies.graph, desire, "desire_pulse")
    _add_int64(policies.graph, "_last_feature", -1)
    policies.graph.node.insert(
        0,
        helper.make_node("Gather", ["features", "_last_feature"], [current], axis=1, name="last_feature"),
    )

    # output_slices (and model_checkpoint) for ORTSupercomboForInference.
    policies.metadata_props.extend(model.metadata_props)
    make_batch_dynamic(vision)
    make_batch_dynamic(policies)

    # Vision contract: img and big_img with the same [batch, C, H, W] uint8, features [batch, S, D].
    vision_inputs = {}
    for value in vision.graph.input:
        vision_inputs[value.name] = value
    if sorted(vision_inputs) != sorted(QUEUE_CAMERA_INPUTS):
        raise ValueError(f"vision inputs {sorted(vision_inputs)}")
    image_dims = _value_dims(vision_inputs[QUEUE_CAMERA_INPUTS[0]])
    for name in QUEUE_CAMERA_INPUTS:
        if vision_inputs[name].type.tensor_type.elem_type != onnx.TensorProto.UINT8:
            raise ValueError(f"vision input {name} is not uint8")
        if _value_dims(vision_inputs[name]) != image_dims or len(image_dims) != 4:
            raise ValueError(f"vision input {name} has shape {_value_dims(vision_inputs[name])}")
    if len(vision.graph.output) != 1 or len(_value_dims(vision.graph.output[0])) != 3:
        raise ValueError("vision must have one [batch, S, D] output")
    return vision, policies


def make_dense_queue(policies: onnx.ModelProto) -> onnx.ModelProto:
    """Make the split policies of a queue-format model return the outputs of every history step.

    The temporal transformers use a causal mask over the time-major tokens (step * S + spatial) and
    their heads read the last token. The last token of step k has seen exactly the steps 0..k:
    reshaping the tokens to [batch * T, S, D] before those Gathers gives every step at once, and the
    heads (Gemm on [rows, D]) work on batch * T rows. The point policy and hidden_state read the
    current features, which become all the T steps the same way. The desire windows are already cut
    for every step. Same meaning as the old make_dense: step k sees the steps 0..k only.
    """

    graph = policies.graph
    features = None
    for value in graph.input:
        if value.name == "features":
            features = value
    if features is None:
        raise ValueError("no 'features' input: not a graph of split_queue_supercombo")
    dims = _value_dims(features)
    if len(dims) != 4:
        raise ValueError(f"features has shape {dims}, expected [batch, T, S, D]")
    for dim in dims[1:]:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"features has shape {dims}: T, S and D must be fixed")
    timesteps = dims[1]
    spatial = dims[2]
    feature_dim = dims[3]

    # Tensors computed from the features: the Gathers of traffic_convention and action_t have the
    # same form (axis 1, scalar index) but do not depend on the features.
    consumers = _consumers(graph)
    from_features = set()
    frontier = ["features"]
    while frontier:
        next_frontier = []
        for tensor in frontier:
            for node in consumers.get(tensor, []):
                for output in node.output:
                    if output not in from_features:
                        from_features.add(output)
                        next_frontier.append(output)
        frontier = next_frontier

    # Shapes of the token tensors, to check that a Gather really reads [batch, T * S, D] tokens.
    inferred = onnx.shape_inference.infer_shapes(policies)
    token_shapes = {}
    for value in inferred.graph.value_info:
        token_shapes[value.name] = _value_dims(value)

    tokens_shape = _add_int64(graph, "_dense_tokens_shape", [-1, spatial, feature_dim])
    nodes = []
    token_selects = 0
    for node in graph.node:
        index = None
        if node.op_type == "Gather" and node.input[0] in from_features and _attribute(node, "axis", 0) == 1:
            index = _constant(graph, node.input[1])
        if node.op_type == "Gather" and node.input[0] == "features":
            # Current features of the last step -> current features of every step, [batch * T, S, D].
            node.op_type = "Reshape"
            node.input[:] = ["features", tokens_shape]
            del node.attribute[:]
        elif index is not None and index.ndim == 0:
            shape = token_shapes.get(node.input[0])
            if shape is None or len(shape) != 3 or shape[1] != timesteps * spatial or shape[2] != feature_dim:
                raise ValueError(f"{node.name} reads {node.input[0]} {shape}, expected [batch, {timesteps * spatial}, "
                                 f"{feature_dim}] tokens")
            token = int(index) % (timesteps * spatial)
            if token // spatial != timesteps - 1:
                raise ValueError(f"{node.name} reads token {token}, not one of the last step")
            # Tokens [batch, T * S, D] -> [batch * T, S, D], then the same spatial token of every step.
            dense_tokens = f"_dense_{node.output[0]}_tokens"
            nodes.append(
                helper.make_node("Reshape", [node.input[0], tokens_shape], [dense_tokens], name=f"dense_{node.name}")
            )
            node.input[0] = dense_tokens
            node.input[1] = _add_int64(graph, f"_dense_{node.output[0]}_index", token % spatial)
            token_selects = token_selects + 1
        nodes.append(node)
    if token_selects == 0:
        raise ValueError("no Gather of the last token after the temporal transformers")
    del graph.node[:]
    graph.node.extend(nodes)

    output = graph.output[0]
    output_size = output.type.tensor_type.shape.dim[-1].dim_value
    flat_output = f"_{output.name}_flat"
    for node in graph.node:
        for index, name in enumerate(node.output):
            if name == output.name:
                node.output[index] = flat_output
    output_shape = _add_int64(graph, "_dense_output_shape", [-1, timesteps, output_size])
    graph.node.append(helper.make_node("Reshape", [flat_output, output_shape], [output.name], name="dense_outputs"))
    output.CopyFrom(
        helper.make_tensor_value_info(output.name, output.type.tensor_type.elem_type, ["batch", timesteps, output_size])
    )
    _prune_initializers(graph)

    # Policies contract, the one ORTSupercomboForInference reads: features, desire_pulse (5 Hz,
    # T + W - 1 steps), traffic_convention and action_t of the current step.
    input_dims = {}
    for value in graph.input:
        input_dims[value.name] = _value_dims(value)
    expected_inputs = ["action_t", "desire_pulse", "features", "traffic_convention"]
    if sorted(input_dims) != expected_inputs:
        raise ValueError(f"policies inputs {sorted(input_dims)}, expected {expected_inputs}")
    desire_dims = input_dims["desire_pulse"]
    if len(desire_dims) != 3 or not isinstance(desire_dims[1], int) or desire_dims[1] < timesteps:
        raise ValueError(f"desire_pulse has shape {desire_dims}")
    for name in ("traffic_convention", "action_t"):
        if len(input_dims[name]) != 2:
            raise ValueError(f"{name} has shape {input_dims[name]}, expected [batch, n]")
    return policies


__all__ = [
    "find_queue_boundaries",
    "is_queue_supercombo",
    "make_batch_dynamic",
    "make_dense",
    "make_dense_queue",
    "split_queue_supercombo",
    "split_supercombo",
]
