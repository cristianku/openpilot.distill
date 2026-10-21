"""CPU-only checks for staging training weights next to a published model (no torch, no network)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from helpers import publish_training_weights as publish


class StageReleaseTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.stage = self.root / "stage"
        weights = self.root / "supercombo.pt"
        weights.write_bytes(b"state dict fixture")
        self.sha = hashlib.sha256(weights.read_bytes()).hexdigest()
        self.models = {"supercombo": {
            "path": str(weights),
            "bytes": weights.stat().st_size,
            "sha256": self.sha,
            "architecture": "supercombo",
            "classes": None,
            "hub_path": "training/releases/run_1/supercombo.pt",
        }}

    def tearDown(self):
        self.directory.cleanup()

    def stable(self):
        return json.loads((self.stage / "training/stable.json").read_text(encoding="utf-8"))

    def test_new_release_without_stable(self):
        staged = publish.stage_release(self.stage, "run_1", self.models, None, None, False)
        self.assertEqual(staged, ["training/releases/run_1/supercombo.pt", "training/releases/run_1/manifest.json"])
        self.assertEqual((self.stage / staged[0]).read_bytes(), b"state dict fixture")
        manifest = json.loads((self.stage / staged[1]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["models"]["supercombo"]["sha256"], self.sha)
        self.assertFalse((self.stage / "training/stable.json").exists())

    def test_stable_keeps_other_tasks(self):
        current = {"schema_version": 1, "models": {
            "reader": {"path": "training/releases/v1/reader.pt", "sha256": "a" * 64},
        }}
        publish.stage_release(self.stage, "run_1", self.models, None, current, True)
        self.assertEqual(self.stable()["models"]["supercombo"],
                         {"path": "training/releases/run_1/supercombo.pt", "sha256": self.sha})
        self.assertEqual(self.stable()["models"]["reader"]["sha256"], "a" * 64)

    def test_first_stable_creates_the_manifest(self):
        publish.stage_release(self.stage, "run_1", self.models, None, None, True)
        self.assertEqual(self.stable()["schema_version"], 1)

    def test_same_release_again_is_not_restaged(self):
        existing = {"schema_version": 1, "models": {"supercombo": {"sha256": self.sha}}}
        staged = publish.stage_release(self.stage, "run_1", self.models, existing, None, True)
        self.assertEqual(staged, ["training/stable.json"])
        self.assertFalse((self.stage / "training/releases").exists())

    def test_same_release_with_other_weights_is_refused(self):
        existing = {"schema_version": 1, "models": {"supercombo": {"sha256": "0" * 64}}}
        with self.assertRaisesRegex(ValueError, "different weights"):
            publish.stage_release(self.stage, "run_1", self.models, existing, None, False)

    def test_invalid_stable_is_refused(self):
        with self.assertRaisesRegex(ValueError, "stable.json is invalid"):
            publish.stage_release(self.stage, "run_1", self.models, None, {"models": []}, True)


if __name__ == "__main__":
    unittest.main()
