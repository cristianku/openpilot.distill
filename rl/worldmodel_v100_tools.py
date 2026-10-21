"""Inspection and dense eager prototype for the commaai worldmodel package (Tesla V100 port).

The inference and conversion paths require the package's compatible torchao.
This adapter was derived from the inspected package, but has not run on a GPU.
Only load pickle files from a trusted source.

Not done yet: CFG combination and Euler update still run in the generate() dtype (fp16).
Moving them to fp32 needs the extracted model_for_inference.py (see `inspect --sources`).
"""

import argparse
import inspect
import json
from pathlib import Path
import time
import types
import zipfile


def load_quantized_state(path):
    import torch

    with zipfile.ZipFile(path) as archive:
        with archive.open("archive/assets/state_dict.pt") as stream:
            state = torch.load(stream, map_location="cpu", weights_only=False)
    return state


def describe_tensor(value):
    return {
        "class": type(value).__module__ + "." + type(value).__name__,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def inspect_package(args):
    import torch
    from torch.package import PackageImporter

    with zipfile.ZipFile(args.package) as archive:
        if args.sources is not None:
            root = args.sources.resolve()
            root.mkdir(parents=True, exist_ok=True)
            for name in archive.namelist():
                if not name.endswith(".py"):
                    continue
                target = (root / name).resolve()
                if not target.is_relative_to(root):
                    raise ValueError("Unsafe archive path: " + name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    output.write(archive.read(name))

    importer = PackageImporter(str(args.package))
    metadata = importer.load_pickle("meta", "meta.pkl", map_location="cpu")
    print("METADATA", metadata)
    state = load_quantized_state(args.package)
    counts = {}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError((name, type(value)))
        description = describe_tensor(value)
        kind = description["class"]
        counts[kind] = counts.get(kind, 0) + 1
        for attribute in (
            "qdata", "scale", "per_tensor_scale", "act_per_tensor_scale",
            "block_size", "orig_dtype", "is_swizzled_scales", "use_triton_kernel",
        ):
            if not hasattr(value, attribute):
                continue
            item = getattr(value, attribute)
            if isinstance(item, torch.Tensor):
                description[attribute] = describe_tensor(item)
            else:
                description[attribute] = str(item)
        print(name, json.dumps(description))
    print("COUNTS", counts)
    seen = set()
    for value in state.values():
        cls = type(value)
        if cls is torch.Tensor or cls in seen:
            continue
        seen.add(cls)
        print("CLASS SOURCE", inspect.getfile(cls))
        if hasattr(cls, "dequantize"):
            print(inspect.getsource(cls.dequantize))
        if hasattr(cls, "get_hp_scales"):
            print(inspect.getsource(cls.get_hp_scales))


def convert(args):
    import torch
    from torchao.quantization import Float8Tensor
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    if args.output.exists():
        raise FileExistsError(args.output)
    state = load_quantized_state(args.package)
    dense = {}
    total = 0
    for name in list(state.keys()):
        value = state.pop(name)
        if isinstance(value, NVFP4Tensor):
            # In torchao 0.17 get_hp_scales consults orig_dtype.
            # This changes wrapper metadata, then dequantizes in FP32.
            working = value.to(dtype=torch.float32)
            result = working.dequantize(torch.float32)
        elif isinstance(value, Float8Tensor):
            result = value.dequantize(torch.float32)
        elif type(value) is torch.Tensor:
            if value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                raise TypeError("Raw FP8 needs an explicit scale mapping: " + name)
            result = value
        else:
            raise TypeError("Unsupported state entry: " + name + " " + str(type(value)))

        if type(result) is not torch.Tensor:
            raise TypeError("Dequantization retained a tensor subclass: " + name)
        if result.is_floating_point():
            result = result.float()
            if not torch.isfinite(result).all().item():
                raise ValueError("Nonfinite source: " + name)
            if args.dtype == "fp16":
                result = result.to(torch.float16)
            if not torch.isfinite(result).all().item():
                raise ValueError("Overflow after conversion: " + name)
        result = result.contiguous()
        dense[name] = result
        total += result.numel() * result.element_size()
        del value, result
    with args.output.open("xb") as stream:
        torch.save(dense, stream)
    print("SAVED", args.output, "tensor GiB", total / (1024 ** 3))


def resolve_dense_cache_dtype(name):
    import torch

    if name == "float16":
        return torch.float16
    elif name == "float32":
        return torch.float32
    else:
        raise ValueError("Dense prototype requires float16 or float32 KV cache")


def rmsnorm_fp32(self, x):
    import torch.nn.functional as functional

    weight = None
    if self.weight is not None:
        weight = self.weight.float()
    result = functional.rms_norm(x.float(), self.normalized_shape, weight, self.eps)
    return result.to(dtype=x.dtype)


def load_dense_model(package, checkpoint, device, dtype):
    import torch
    from torch.package import PackageImporter

    importer = PackageImporter(str(package))
    inference = importer.import_module("torchtitan.experiments.worldmodel.model_for_inference")
    architecture = importer.import_module("torchtitan.experiments.worldmodel.model")
    model = importer.load_pickle("model", "model.pkl", map_location="cpu")
    metadata = importer.load_pickle("meta", "meta.pkl", map_location="cpu")

    # Adapter confined to this PackageImporter's private module namespace.
    # A maintained fork should implement this resolver directly in source.
    inference._resolve_kv_cache_dtype = resolve_dense_cache_dtype
    model.cleanup_caches()
    model.clear_inference_masks()
    model.mask = None
    model.config.transformer.attention_impl = "SDPA"
    for module in model.modules():
        if isinstance(module, architecture.SelfAttention):
            module.config.attention_impl = "SDPA"
            module.flex_attention = None
            if module.sdpa is None:
                module.sdpa = architecture.ScaledDotProductAttention.Config().build()
        if isinstance(module, torch.nn.Linear):
            if type(module).forward is not torch.nn.Linear.forward:
                raise TypeError("Review custom Linear.forward: " + str(type(module)))

    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    for name, value in state.items():
        if type(value) is not torch.Tensor:
            raise TypeError("Dense checkpoint contains a subclass: " + name)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.eval()
    # Only the parameters take the run dtype. Buffers (timestep/rope frequencies, masks) keep the
    # dtype they were packaged with, as in the original runtime: model.to(dtype=...) would push them to fp16.
    model.to(device=device)
    for value in model.parameters():
        if value.is_floating_point():
            value.data = value.data.to(dtype=dtype)
    for name, value in model.named_buffers():
        print("BUFFER", name, str(value.dtype), list(value.shape))

    patched_norms = 0
    unpatched_norms = {}
    for module in model.modules():
        if isinstance(module, torch.nn.RMSNorm):
            module.forward = types.MethodType(rmsnorm_fp32, module)
            patched_norms = patched_norms + 1
        elif "Norm" in type(module).__name__:
            class_name = type(module).__module__ + "." + type(module).__name__
            if class_name in unpatched_norms:
                unpatched_norms[class_name] = unpatched_norms[class_name] + 1
            else:
                unpatched_norms[class_name] = 1
    print("RMSNORM PATCHED TO FP32", patched_norms)
    # A custom (adaptive) norm class is not covered by the fp32 patch: check its forward
    # in the extracted sources before trusting an fp16 run.
    for class_name in unpatched_norms:
        print("WARNING NORM NOT PATCHED", class_name, unpatched_norms[class_name])
    for name, value in model.named_parameters():
        if value.is_meta:
            raise ValueError("Unmaterialized parameter: " + name)
    model.requires_grad_(False)
    # Deliberately no compile_for_inference() and no quantize_for_inference().
    return model, metadata


def make_inputs(args):
    import torch
    from torch.package import PackageImporter

    if args.output.exists():
        raise FileExistsError(args.output)
    importer = PackageImporter(str(args.package))
    metadata = importer.load_pickle("meta", "meta.pkl", map_location="cpu")
    shapes = metadata["model_io"]["in_shape"]
    torch.manual_seed(1234)
    inputs = {}
    for name, shape in shapes.items():
        if name == "latents":
            inputs[name] = torch.randn(tuple(shape), dtype=torch.float32)
        elif name in ("augments_pos_ref_augment", "ref_augment_from_augments_euler"):
            inputs[name] = torch.zeros(tuple(shape), dtype=torch.float32)
        elif name == "pose_mask":
            inputs[name] = torch.ones(tuple(shape), dtype=torch.int64)
            inputs[name][:, -1] = 0
        elif name == "fidxs":
            inputs[name] = torch.arange(shape[1], dtype=torch.int64).unsqueeze(0).expand(shape[0], -1).clone()
        else:
            raise ValueError("Unknown input: " + name)
    with args.output.open("xb") as stream:
        torch.save(inputs, stream)


def run(args):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    if args.dtype == "fp16":
        dtype = torch.float16
        cache_name = "float16"
    else:
        dtype = torch.float32
        cache_name = "float32"
    if device.type == "cuda":
        print("GPU", torch.cuda.get_device_name(device))
        print("CAPABILITY", torch.cuda.get_device_capability(device))
        print("ARCH LIST", torch.cuda.get_arch_list())
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    model, metadata = load_dense_model(args.package, args.checkpoint, device, dtype)
    print("TRANSFORMER", model.config.transformer)
    source = torch.load(args.inputs, map_location="cpu", weights_only=True)
    inputs = {}
    for name, value in source.items():
        if value.is_floating_point():
            inputs[name] = value.to(device=device, dtype=dtype)
        else:
            inputs[name] = value.to(device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        outputs = model.generate(
            **inputs,
            steps=args.steps,
            num_prefill_frames=inputs["latents"].shape[1] - 1,
            dtype=dtype,
            inference_schedule="linear",
            cfg=2.0,
            kv_cache_dtype=cache_name,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        print("PEAK TORCH GiB", torch.cuda.max_memory_allocated(device) / (1024 ** 3))
    print("COLD GENERATE SECONDS", time.perf_counter() - started)
    saved = {}
    for name, value in outputs.items():
        value = value.detach().float().cpu()
        if not torch.isfinite(value).all().item():
            raise ValueError("Nonfinite output: " + name)
        saved[name] = value
    with args.output.open("xb") as stream:
        torch.save(saved, stream)


def compare(args):
    import torch

    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=True)
    if reference.keys() != candidate.keys():
        raise ValueError("Output keys differ")
    for name in reference:
        a = reference[name].double()
        b = candidate[name].double()
        if a.shape != b.shape:
            raise ValueError("Shape mismatch: " + name)
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("Nonfinite output: " + name)
        error = b - a
        rmse = error.square().mean().sqrt()
        reference_rms = a.square().mean().sqrt().clamp_min(1e-12)
        print(name, "max_abs", error.abs().max().item(), "rmse", rmse.item(),
              "relative_l2", (rmse / reference_rms).item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("package", type=Path)
    inspect_parser.add_argument("--sources", type=Path)
    inspect_parser.set_defaults(function=inspect_package)
    convert_parser = commands.add_parser("convert")
    convert_parser.add_argument("package", type=Path)
    convert_parser.add_argument("output", type=Path)
    convert_parser.add_argument("--dtype", choices=("fp16", "fp32"), required=True)
    convert_parser.set_defaults(function=convert)
    inputs_parser = commands.add_parser("inputs")
    inputs_parser.add_argument("package", type=Path)
    inputs_parser.add_argument("output", type=Path)
    inputs_parser.set_defaults(function=make_inputs)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("package", type=Path)
    run_parser.add_argument("checkpoint", type=Path)
    run_parser.add_argument("inputs", type=Path)
    run_parser.add_argument("output", type=Path)
    run_parser.add_argument("--device", required=True)
    run_parser.add_argument("--dtype", choices=("fp16", "fp32"), required=True)
    run_parser.add_argument("--steps", type=int, default=1)
    run_parser.set_defaults(function=run)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("reference", type=Path)
    compare_parser.add_argument("candidate", type=Path)
    compare_parser.set_defaults(function=compare)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
