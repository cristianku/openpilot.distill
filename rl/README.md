# Reinforcement learning

The world model is running in a FastAPI inference server.

```bash
uv run hf download commaai/worldmodel-4B model.fp8_nvfp4.torchpackage --local-dir models
```

## Interactive

Terminal 1:

```bash
uv run python -m rl.server
```

Terminal 2:

```bash
uv run python -m rl.env --actor "supercombo|wasd" --cache-dir outputs/rl/cache
```

`supercombo` uses the default teacher, `models/big_driving_supercombo_1365126a_12864.onnx` (MD v3, see
[`supervised/README.md`](../supervised/README.md)), through ONNX Runtime. `--model models/big_driving_supercombo.onnx`
runs the old 084747c7 teacher instead. Pass a supervised Torch state dict to use the Torch runtime:

```bash
uv run python -m rl.env --actor supercombo --model /path/to/supercombo.pt
```

## Training

With the server running:

```bash
uv run python -m rl.train --model /path/to/supercombo.pt --steps 10
```

Try the RL trained policy

```bash
uv run python -m rl.env --actor supercombo --model /path/to/supercombo.pt --on-policy /path/to/on_policy.pt
```
