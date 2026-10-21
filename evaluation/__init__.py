"""Offline evaluation of driving models on recorded segments.

Commands:
  python -m evaluation.compare  comparison video of two models on one segment (outputs/compare/)
  python -m evaluation.holdout  avg miss of one model on the held-out segments (outputs/eval/)

Importing the package imports nothing heavy: torch, onnxruntime, PyNvVideoCodec and openpilot are
loaded only by the functions that need them, so the metric and spec code also runs on a Mac.
"""
