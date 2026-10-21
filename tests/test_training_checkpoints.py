"""CPU-only regression checks for checkpoint and Hugging Face manifest handling."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from supervised import checkpoints


class CheckpointHelpersTest(unittest.TestCase):
    def test_model_weights_accepts_legacy_and_wrapped_files(self):
        self.assertEqual(checkpoints.model_weights({"layer": 3}), {"layer": 3})
        self.assertEqual(checkpoints.model_weights({"state_dict": {"layer": 5}}), {"layer": 5})
        with self.assertRaises(ValueError):
            checkpoints.model_weights({"state_dict": {}})

    def test_stable_manifest_checksum_and_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_file = root / "reader.pt"
            model_file.write_bytes(b"tensor checkpoint fixture")
            sha = hashlib.sha256(model_file.read_bytes()).hexdigest()
            manifest_file = root / "stable.json"

            def download(*, repo_id, filename, revision):
                self.assertEqual(repo_id, "example/weights")
                self.assertEqual(revision, "main")
                return str(manifest_file if filename == "training/stable.json" else model_file)

            fake_hub = types.ModuleType("huggingface_hub")
            fake_hub.hf_hub_download = download
            with patch.dict(sys.modules, {"huggingface_hub": fake_hub}), patch.object(
                checkpoints, "torch_load", return_value={"state_dict": {"x": 1}}
            ):
                manifest = {"schema_version": 1, "models": {
                    "reader": {"path": "training/releases/v1/reader.pt", "sha256": sha}
                }}
                manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
                result, source = checkpoints.stable_weights("example/weights", "reader")
                self.assertEqual(result["state_dict"]["x"], 1)
                self.assertIn("training/releases/v1/reader.pt", source)
                manifest["models"]["reader"]["sha256"] = "0" * 64
                manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    checkpoints.stable_weights("example/weights", "reader")
                manifest["models"]["reader"]["path"] = "../secrets.pt"
                manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "invalid stable weight path"):
                    checkpoints.stable_weights("example/weights", "reader")

    def test_missing_repo(self):
        with self.assertRaisesRegex(ValueError, "requires a Hugging Face repository"):
            checkpoints.stable_weights("", "reader")


if __name__ == "__main__":
    unittest.main()
