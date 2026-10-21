"""CPU-only checks for the files supervised/train.py writes next to the model: metrics.jsonl and run.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

# supervised.train imports torch, wandb and onnxruntime: without them (e.g. a Mac without the
# training environment) the tests are skipped, with the reason shown.
try:
    from supervised import train
    IMPORT_ERROR = ""
except ImportError as error:
    train = None
    IMPORT_ERROR = str(error)


class FakeImage:
    """Stands for wandb.Image: not a number, so it must not reach metrics.jsonl."""


def make_args(mode: str = "fresh") -> argparse.Namespace:
    return argparse.Namespace(
        mode=mode,
        steps=5000,
        batch_size=8,
        workers=6,
        teacher="/repo/models/big_driving_supercombo.onnx",
        dataset="/repo",
        hf_repo=None,
    )


@unittest.skipIf(train is None, f"supervised.train not importable here: {IMPORT_ERROR}")
class MetricsRecordTest(unittest.TestCase):
    def test_keeps_scalars_and_drops_images(self):
        metrics = {
            "loss": 38.6,
            "gt": 5.21,
            "distill": 33.4,
            "gt/plan": 2.08,
            "distill/hidden_state": 0.5,
            "grad_norm": 244.0,
            "step_time": 0.71,
            "imgs/gt_narrow": FakeImage(),
        }
        record = train.metrics_record(12, 5000, metrics)
        self.assertEqual(record["step"], 12)
        self.assertEqual(record["total_steps"], 5000)
        self.assertIsInstance(record["time"], float)
        self.assertEqual(record["loss"], 38.6)
        self.assertEqual(record["gt/plan"], 2.08)
        self.assertEqual(record["distill/hidden_state"], 0.5)
        self.assertEqual(record["grad_norm"], 244.0)
        self.assertEqual(record["step_time"], 0.71)
        self.assertNotIn("imgs/gt_narrow", record)

    def test_non_finite_values_become_null(self):
        # An fp16 overflow step has an infinite gradient norm: the line must stay valid JSON.
        record = train.metrics_record(1, 10, {"loss": float("nan"), "grad_norm": float("inf"), "gt": 1.0})
        self.assertIsNone(record["loss"])
        self.assertIsNone(record["grad_norm"])
        self.assertEqual(record["gt"], 1.0)
        line = json.dumps(record, allow_nan=False)
        self.assertEqual(json.loads(line)["grad_norm"], None)


@unittest.skipIf(train is None, f"supervised.train not importable here: {IMPORT_ERROR}")
class RunInfoTest(unittest.TestCase):
    def test_fresh_run_writes_every_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            train.start_run_info(path, make_args("fresh"), "fp16", 0)
            saved = json.loads(path.read_text())
            self.assertEqual(saved["mode"], "fresh")
            self.assertEqual(saved["steps"], 5000)
            self.assertEqual(saved["batch_size"], 8)
            self.assertEqual(saved["workers"], 6)
            self.assertEqual(saved["precision"], "fp16")
            self.assertEqual(saved["teacher"], "/repo/models/big_driving_supercombo.onnx")
            self.assertEqual(saved["dataset"], "/repo")
            self.assertIsNone(saved["hf_repo"])
            self.assertIsInstance(saved["started"], float)
            self.assertNotIn("finished", saved)
            self.assertNotIn("resumed", saved)
            # No temporary file left behind.
            self.assertEqual(sorted(entry.name for entry in Path(directory).iterdir()), ["run.json"])

    def test_resume_keeps_the_start_and_adds_an_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            first = train.start_run_info(path, make_args("fresh"), "fp16", 0)
            train.start_run_info(path, make_args("resume"), "fp16", 1500)
            train.start_run_info(path, make_args("resume"), "fp16", 3000)
            saved = json.loads(path.read_text())
            self.assertEqual(saved["mode"], "fresh")
            self.assertEqual(saved["started"], first["started"])
            self.assertEqual([entry["step"] for entry in saved["resumed"]], [1500, 3000])

    def test_resume_without_run_json_starts_a_new_one(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text("{not json")
            train.start_run_info(path, make_args("resume"), "bf16", 500)
            saved = json.loads(path.read_text())
            self.assertEqual(saved["mode"], "resume")
            self.assertEqual(saved["precision"], "bf16")
            self.assertEqual([entry["step"] for entry in saved["resumed"]], [500])

    def test_finished_is_added_by_a_second_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            run_info = train.start_run_info(path, make_args("fresh"), "fp16", 0)
            run_info["finished"] = run_info["started"] + 60.0
            train.write_run_info(path, run_info)
            saved = json.loads(path.read_text())
            self.assertEqual(saved["finished"] - saved["started"], 60.0)


if __name__ == "__main__":
    unittest.main()
