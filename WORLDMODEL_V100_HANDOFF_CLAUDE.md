# openpilot.distill on NVIDIA Tesla V100 — handoff for Claude

Date: 17 October 2026.

This document contains Codex's full answer and, in the appendix, the complete Python script the answer refers to. The links to the temporary script have been adapted to point to the appendix; the text of the answer is otherwise preserved.

Verification status: the sources and metadata of the public torchpackage and the local checkout of openpilot.distill were inspected. The script has passed only the syntax and CLI check. Weight conversion, CPU/GPU forward, numerical comparison and benchmarks are still to be run. The estimates are not measurements.

Handing over this document does not authorize live changes, installations or restarts on the comma, nor powering on Serverone-ai.

## Full answer

**The most solid route is to dequantize on the CPU, use dense FP16 weights and an FP16 KV cache, and run the model in eager mode with SDPA.** Before that I would also build a dense FP32 reference.

I inspected the sources and metadata of the public torchpackage through partial reads, without downloading all the weights. I prepared [a complete Python script](#appendix-complete-python-script) for inspection, conversion, loading and comparison. **I verified syntax and CLI; conversion and forward have not been run.** No changes to the repository or the devices.

1. **Dequantization: in the package the format can be identified precisely.**

   The state dict index does contain:

   | Part | Representation |
   |---|---|
   | Attention | `torchao.quantization.Float8Tensor` |
   | MLP | `torchao.prototype.mx_formats.nvfp4_tensor.NVFP4Tensor` |
   | FP4 data | `uint8`, two values per byte |
   | NVFP4 blocks | 16 elements |
   | NVFP4 scales | FP8 E4M3, with a **swizzled** layout |
   | Global scale | `per_tensor_scale` |
   | Declared original dtype | BF16 |

   So **`weight.half()` is not enough**: on tensor subclasses it can change the declared dtype while leaving the data and the behavior quantized.

   The core of the conversion is:

   ```python
   from torchao.quantization import Float8Tensor
   from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

   if isinstance(weight, NVFP4Tensor):
       working = weight.to(dtype=torch.float32)
       dense = working.dequantize(torch.float32)
   elif isinstance(weight, Float8Tensor):
       dense = weight.dequantize(torch.float32)
   else:
       raise TypeError(type(weight))

   if type(dense) is not torch.Tensor:
       raise TypeError("Dequantization retained a tensor subclass")

   if not torch.isfinite(dense).all():
       raise ValueError("Nonfinite dequantized weights")

   dense_fp16 = dense.to(torch.float16)

   if not torch.isfinite(dense_fp16).all():
       raise ValueError("Overflow in the FP16 conversion")
   ```

   The intermediate NVFP4 step is intentional: in torchao 0.17, `get_hp_scales()` consults `orig_dtype`. Moving it to FP32 before dequantization avoids passing the scales through BF16. The dequantizer also handles transpositions and the reordering of the scales. [NVFP4 implementation in torchao 0.17](https://github.com/pytorch/ao/blob/v0.17.0/torchao/prototype/mx_formats/nvfp4_tensor.py).

   Conceptually:

   ```text
   weight = decoded_FP4_value × block_scale × global_scale
   ```

   But nibble order, orientation and swizzle must follow that class. **I would not rebuild the format by hand when the dequantizer is available.**

   The conversion keeps the approximation of the quantized weights: it does not recover the BF16 weights from before quantization.

2. **In the inspected model there is no need to replace the Linear layers wholesale.**

   The class `torchtitan.models.common.nn_modules.Linear` inherits from `nn.Linear` without redefining its forward. The quantization lives in the tensor-subclass weights.

   For this package it is better to:

   - keep the module names and structure;
   - load ordinary tensors with `strict=True, assign=True`;
   - select SDPA;
   - change the cache dtype resolver;
   - drop the caches and masks built earlier.

   The current resolver recognizes FP8 and otherwise returns BF16: **simply passing `"float16"` today does not produce an FP16 cache**. In the extracted source I would change it like this:

   ```python
   KVCacheDType = Literal[
       "bfloat16",
       "float8_e4m3fn",
       "float16",
       "float32",
   ]

   def _resolve_kv_cache_dtype(requested_dtype):
       if requested_dtype == "float16":
           return torch.float16
       elif requested_dtype == "float32":
           return torch.float32
       elif requested_dtype == "bfloat16":
           return torch.bfloat16
       elif requested_dtype == "float8_e4m3fn":
           return torch.float8_e4m3fn
       else:
           raise ValueError(requested_dtype)
   ```

   Then:

   ```python
   output = model.generate(
       **inputs,
       dtype=torch.float16,
       kv_cache_dtype="float16",
       steps=15,
       num_prefill_frames=14,
       cfg=2.0,
   )
   ```

   The K/V caches are already registered with `persistent=False`: **they add no keys to the state dict and do not break `strict=True`**.

   For the first attempt I would keep the original torchpackage as the container of the code, loading the dense state dict separately. The attached script uses this approach and applies the adaptation in the private namespace of the `PackageImporter`.

   After validation I would make the changes permanent in the extracted sources. A new torchpackage would also need updated metadata: the original metadata declares BF16, `fp8_nvfp4` and a minimum compute capability of `(10, 0)`. Changing only this metadata does not make the model compatible.

3. **`compile_for_inference()` should be skipped at first, together with the FLEX path.**

   In the actual code the function does:

   ```python
   for block in self.blocks:
       block.compile(mode="max-autotune-no-cudagraphs")
   ```

   It does not convert the weights and is not needed for sampling.

   There is also a separate compilation to build the FLEX masks. For the first V100 version I would set SDPA on the configuration and on the attention modules, invalidating the existing masks.

   **Skipping only `compile_for_inference()` while leaving FLEX and the quantized weights is not enough.** I would start without compilation; any optimizations come after the numerical comparison.

4. **FP16 must be validated: protecting only the softmax is not enough.**

   I would use this initial policy:

   | Operation | Precision |
   |---|---|
   | Projection and MLP weights | FP16 |
   | KV cache | FP16 |
   | RMSNorm, including Q/K norm | FP32 computation, output in the requested dtype |
   | Reference attention | SDPA `MATH` |
   | Frequencies and sinusoidal timestep functions | FP32 |
   | CFG and Euler update | Preferably FP32 |
   | Residual and adaptive modulation | Monitor; promote to FP32 if needed |

   The model already keeps part of the timestep computation in FP32. In the math SDPA backend PyTorch normally keeps the intermediates in FP32 for FP16/BF16 inputs. This improves stability, but increases the memory and cost of attention. [PyTorch numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html).

   The CFG combination must be promoted **before** the operations:

   ```python
   unconditional = unconditional.float()
   conditional = conditional.float()

   velocity = unconditional + cfg * (conditional - unconditional)
   next_x = scheduler.step(velocity, step_idx, x.float())
   x = next_x.to(dtype=x.dtype)
   ```

   Likewise, an FP32 residual must do the sum and the gating in FP32. Calling `.float()` after a value has become `inf` does not recover it.

   The provided script protects RMSNorm and uses math SDPA; **it is a baseline to be validated**, not a guarantee that all FP16 residuals are stable.

   Full FP32 makes sense as a reference and to diagnose overflows. On a 32 GB V100 it is plausible for the world model alone; on 16 GB, weights and working memory make the configuration impractical. GradScaler does not fix forward overflows in inference.

   For the autoencoder I would start from FP32. Its roughly 128 million parameters in total take about 0.48 GiB in FP32. [Autoencoder model card](https://huggingface.co/commaai/vit-ae-2x-f8c32).

   Beware of exports: `.float()` on the parameters does not remove BF16 casts that may be embedded in the graph. To find them:

   ```python
   import torch

   exported = torch.export.load("encoder.pt2")
   print(exported.graph_module.code)
   ```

   If it contains explicit casts or specialized operations, the graph must be adapted or re-exported from source; I would not assume that `.to(torch.float16)` is enough.

5. **Memory: the 32 GB card is the reasonable target; the 16 GB one needs measured margins.**

   From the structure metadata I derived **56 blocks, dimension 2304, 36 heads**, hence a head dimension of 64.

   This sampling shares the prefix between the two CFG branches:

   ```text
   prefix:            14 × 128 = 1792 tokens
   suffix with CFG:    2 × 128 =  256 tokens
   cache capacity:                 2048 tokens
   ```

   The FP16 cache is therefore:

   ```text
   2 × 56 × 2048 × 2304 × 2 bytes = 0.984375 GiB
   ```

   **CFG does not double the whole cache.** The prefill/decode operation is also described in the [world model's model card](https://huggingface.co/commaai/worldmodel-4B).

   Planning estimates, not measurements:

   | Component | Order of magnitude |
   |---|---:|
   | 4B FP16 weights | 7.45 GiB |
   | FP16 KV with CFG | 0.98 GiB |
   | One FP32 score matrix of the prefill | 0.49 GiB |
   | FP16 world model with math SDPA, total | about 11–15+ GiB |
   | World model entirely in FP32 | about 19–24+ GiB |

   The attention intermediates live for the current block: their peak must not be multiplied by 56. The allocator, workspaces and the other models must be added, though. **These estimates do not include the learner with gradients and optimizer.**

   Math SDPA is enough for a first correct implementation. I would not bring in xformers right away. Afterwards I would try the `EFFICIENT_ATTENTION` backend available in the build, checking support for the GPU, the shapes and the mask. SDPA allows selecting the backends explicitly. [SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).

   As for speed, the projections alone count about:

   ```text
   2 × 4B × (1792 + 15 × 256) ≈ 45 TFLOP
   ```

   plus attention and other computations. I would budget **seconds or tens of seconds per prediction**, not real time, until a benchmark exists. PCIe/SXM, attention backend and precision change the result a lot.

6. **The reproducible procedure is in the attached script.**

   Copy [worldmodel_v100_tools.py](#appendix-complete-python-script) into the Linux environment. The following commands assume that the torchpackage is already available and that the environment contains compatible versions of PyTorch and torchao. In the local lockfile I found torchao **0.17.0**: I would start from that version, checking that it imports.

   ```bash
   uv run --no-sync python worldmodel_v100_tools.py inspect \
     models/model.fp8_nvfp4.torchpackage \
     --sources worldmodel-inspected \
     > worldmodel-inspection.txt
   ```

   This prints classes, shapes, dtypes and quantization attributes, plus the code of the installed dequantizers.

   Build the FP32 reference first:

   ```bash
   uv run --no-sync python worldmodel_v100_tools.py convert \
     models/model.fp8_nvfp4.torchpackage \
     worldmodel-dense-fp32.pt \
     --dtype fp32
   ```

   Then the FP16 version:

   ```bash
   uv run --no-sync python worldmodel_v100_tools.py convert \
     models/model.fp8_nvfp4.torchpackage \
     worldmodel-dense-fp16.pt \
     --dtype fp16
   ```

   Plan space for both outputs, about 24 decimal GB on top of the original. For the FP32 conversion I would plan at least 32 GB of free RAM, preferably with more margin.

   Generate a shared synthetic input:

   ```bash
   uv run --no-sync python worldmodel_v100_tools.py inputs \
     models/model.fp8_nvfp4.torchpackage \
     smoke-inputs.pt
   ```

   First comparison with a single step:

   ```bash
   uv run --no-sync python worldmodel_v100_tools.py run \
     models/model.fp8_nvfp4.torchpackage \
     worldmodel-dense-fp32.pt smoke-inputs.pt cpu-output.pt \
     --device cpu --dtype fp32 --steps 1

   uv run --no-sync python worldmodel_v100_tools.py run \
     models/model.fp8_nvfp4.torchpackage \
     worldmodel-dense-fp16.pt smoke-inputs.pt gpu-output.pt \
     --device cuda:0 --dtype fp16 --steps 1

   uv run --no-sync python worldmodel_v100_tools.py compare \
     cpu-output.pt gpu-output.pt
   ```

   Loading uses `strict=True, assign=True`; the comparison reports maximum absolute error, RMSE and relative L2 error.

   After the first result: repeat with 15 steps, then with real inputs and rollout. The time printed by the script is a **cold** call, not a stabilized benchmark. The reported memory covers PyTorch allocations, not ORT's.

   **The FP32 CPU reference verifies the dense port. It does not prove equivalence to the original quantized model**, which also quantizes the activations and uses an FP8 cache. That check needs the original on supported hardware or a complete emulation of the quantizations. The simple synthetic test does not validate the quality of the simulator.

7. **`onnxruntime-gpu 1.29` from PyPI is not the right build for the V100.**

   Here the documentation is explicit:

   | ORT 1.29 Linux x64 build | CUDA | `sm_70` |
   |---|---|---|
   | Default PyPI | 13.x | Absent |
   | CUDA 12 variant | 12.8 | Present |

   Both the [CUDA EP matrix](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) and the [1.29 release notes](https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0) state this.

   **Torch `+cu126` and ORT CUDA 12.8 are not automatically incompatible, but it is not enough that both say “CUDA 12”.** ORT needs libraries matching its own build; the ones loaded by Torch cu126 may not be enough.

   To check whether the 1.29 variant is available in the Microsoft feed, without installing:

   ```bash
   uv pip install --dry-run --no-deps \
     --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
     "onnxruntime-gpu==1.29.0"
   ```

   To keep an environment based on the Torch cu126 libraries, **ORT 1.20.1 CUDA 12/cuDNN 9 is a conservative candidate**, but only if the model is compatible: ORT 1.20 supports ONNX IR up to 10 and standard opsets up to 21. [ORT compatibility](https://onnxruntime.ai/docs/reference/compatibility.html).

   Check first:

   ```python
   import onnx

   model = onnx.load(
       "models/big_driving_supercombo.onnx",
       load_external_data=False,
   )

   print("IR:", model.ir_version)
   for entry in model.opset_import:
       print("DOMAIN:", entry.domain, "OPSET:", entry.version)
   ```

   In the project the graphs produced by `split_supercombo()` and `make_dense()` must be checked too, because those are the ones actually passed to ORT.

   If a recent ORT is needed, I would use the CUDA 12.8 variant with verified dependencies, or a targeted build from source. **I would not present a 1.29 build with CUDA 12.6 as surely supported without checking its build requirements.**

   Also check the actual cuDNN: the [NVIDIA support matrix of cuDNN 9.10.2](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/reference/support-matrix.html) explicitly includes Volta.

   Finally, `get_available_providers()` does not prove execution on the GPU. A real inference and per-node profiling are needed; the CPU fallback can hide a wrong configuration.

8. **For the teacher I would start from the original ONNX; FP16 comes after the reference.**

   Keep the original precision at first to verify the runtime and the results. Then try FP16 or mixed precision: on the V100 it can reduce memory and speed up the compatible parts.

   In your project I would convert the graphs **after split and densification**, keeping the I/O in FP32 at first:

   ```python
   import onnx
   from onnxconverter_common import float16

   from supervised.onnx_helpers import make_dense, split_supercombo

   original = onnx.load("models/big_driving_supercombo.onnx")
   vision, policies = split_supercombo(original)
   policies = make_dense(policies)

   vision_fp16 = float16.convert_float_to_float16(
       vision,
       keep_io_types=True,
   )

   policies_fp16 = float16.convert_float_to_float16(
       policies,
       keep_io_types=True,
   )
   ```

   `keep_io_types=True` preserves the external interface. The converter also supports exclusions for operators or nodes when some parts must stay in FP32. [ORT FP16 and mixed-precision conversion](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html).

   Compare FP32 and FP16 on the **same recorded sequence**, starting from the same queue state: a single frame does not reveal temporal drift.

   The six Hugging Face variants are not interchangeable just because they have the same filename. Before choosing one, their I/O, metadata and compatibility with the project's split must be compared; an empty README does not make it possible to tell which one is better.

<oai-mem-citation>
<citation_entries>
MEMORY.md:476-476|note=[prior V100 compatibility context checked against current primary sources]
</citation_entries>
<rollout_ids>
01a1011b-f2bb-78f0-9942-96bd7781134d
</rollout_ids>
</oai-mem-citation>

## Appendix: complete Python script

Save the content of the following block as `worldmodel_v100_tools.py`. The code is that of the prototype described in the answer, without additional changes. It needs review and verification in the target Linux environment before it can be considered working.

```python
"""Inspection and dense eager prototype for the commaai worldmodel package.

The inference and conversion paths require the package's compatible torchao.
This adapter was derived from the inspected package, but has not run on a GPU.
Only load pickle files from a trusted source.
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
    model.to(device=device, dtype=dtype)
    for module in model.modules():
        if isinstance(module, torch.nn.RMSNorm):
            module.forward = types.MethodType(rmsnorm_fp32, module)
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
```
