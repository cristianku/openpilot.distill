"""CPU-only checks of the model specs of evaluation/ (standard library only, no data, no network)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evaluation import specs


class ParseSpecTest(unittest.TestCase):
    def test_valid_specs(self):
        self.assertEqual(specs.parse_model_spec("stock").kind, "stock")
        self.assertEqual(specs.parse_model_spec("teacher").kind, "teacher")
        self.assertEqual(specs.parse_model_spec("logged").kind, "logged")

        run = specs.parse_model_spec("run:2026-10-21_1544")
        self.assertEqual(run.kind, "run")
        self.assertEqual(run.value, "2026-10-21_1544")
        self.assertEqual(run.key, "run_2026-10-21_1544")

        release = specs.parse_model_spec("hf:2026_10_21_1544")
        self.assertEqual(release.kind, "hf")
        self.assertEqual(release.value, "2026_10_21_1544")
        self.assertEqual(release.key, "hf_2026_10_21_1544")

    def test_invalid_specs(self):
        invalid = [
            "",
            "Stock",
            "stock ",
            "run:",
            "run:../x",
            "run:a/b",
            "run:..",
            "run:.hidden",
            "hf:UPPER",
            "hf:with-dash",
            "hf:a.b",
            "file:/etc/passwd",
            "run:x;rm -rf /",
            "teacher\n",
        ]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                specs.parse_model_spec(text)

    def test_labels_are_set(self):
        for text in ("stock", "teacher", "logged", "run:2026-10-21_1544", "hf:abc"):
            self.assertTrue(specs.parse_model_spec(text).label)

    def test_segment_names(self):
        self.assertEqual(specs.validate_segment_name("00000126f84ad127f010"), "00000126f84ad127f010")
        for name in ("00000126f84ad127f01", "00000126F84AD127F010", "../00000126f84ad127f0", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                specs.validate_segment_name(name)


class ResolveSpecTest(unittest.TestCase):
    def test_run_prefers_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "outputs" / "supervised" / "2026-10-21_1544"
            run_dir.mkdir(parents=True)
            (run_dir / "supercombo.training.pt").write_bytes(b"in progress")
            spec = specs.parse_model_spec("run:2026-10-21_1544")

            resolved = specs.resolve_model(spec, root)
            self.assertEqual(resolved.path.name, "supercombo.training.pt")
            self.assertEqual(resolved.format, specs.FORMAT_TORCH)

            (run_dir / "supercombo.pt").write_bytes(b"final")
            resolved = specs.resolve_model(spec, root)
            self.assertEqual(resolved.path.name, "supercombo.pt")
            self.assertEqual(len(resolved.sha256), 64)

    def test_missing_run_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                specs.resolve_model(specs.parse_model_spec("run:nope"), Path(directory))

    def test_stock_with_wrong_hash_is_refused(self):
        # A student copied over the stock path must never be used as the reference.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stock = root / specs.STOCK_PATH
            stock.parent.mkdir(parents=True)
            stock.write_bytes(b"not the CD210 model")
            with self.assertRaises(ValueError):
                specs.resolve_model(specs.parse_model_spec("stock"), root)

    def test_logged_needs_no_file(self):
        with tempfile.TemporaryDirectory() as directory:
            resolved = specs.resolve_model(specs.parse_model_spec("logged"), Path(directory))
            self.assertIsNone(resolved.path)
            self.assertEqual(resolved.format, specs.FORMAT_LOGGED)


if __name__ == "__main__":
    unittest.main()
