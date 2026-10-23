"""CPU-only checks of evaluation/holdout.py around the GPU run: which kept result still counts, the same
drives for several models in one job, and one progress bar over the job (no model, no video)."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evaluation import holdout
from evaluation.metrics import EVAL_CODE_VERSION


class DrivesTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        patcher = mock.patch.object(holdout, 'ROOT', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.folder.cleanup)

    def segment(self, name, localizer=b'path'):
        directory = self.root / 'data' / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'frame_info.safetensors').write_bytes(b'frames')
        if localizer is not None:
            (directory / 'localizer.safetensors').write_bytes(localizer)

    def test_fingerprints(self):
        self.segment('a')
        self.segment('b', localizer=None)
        first = holdout.drive_fingerprints(['a', 'b'])
        self.assertEqual(len(first['a']), 16)
        # Not localized yet: no fingerprint, so a later localization changes the drives.
        self.assertIsNone(first['b'])
        self.segment('a', localizer=b'refined path')
        self.assertNotEqual(holdout.drive_fingerprints(['a'])['a'], first['a'])

    def test_kept_result_needs_the_same_drives(self):
        self.segment('a')
        settings = {'rate_hz': 5}
        drives = holdout.drive_fingerprints(['a'])
        path = self.root / 'result.json'
        saved = {'code_version': EVAL_CODE_VERSION, 'model': {'sha256': 'x'}, 'segments': ['a'],
                 'settings': settings, 'drives': drives}
        path.write_text(json.dumps(saved), encoding='utf-8')
        self.assertIsNotNone(holdout.cached_result(path, 'x', ['a'], settings, drives))
        # The segment localized again, or a result from before the drives were recorded: computed again.
        self.assertIsNone(holdout.cached_result(path, 'x', ['a'], settings, {'a': 'another'}))
        del saved['drives']
        path.write_text(json.dumps(saved), encoding='utf-8')
        self.assertIsNone(holdout.cached_result(path, 'x', ['a'], settings, drives))

    def test_pin_gives_every_model_the_same_drives(self):
        self.segment('a')
        self.segment('b', localizer=None)
        pin = self.root / 'drives.json'
        drives = holdout.drive_fingerprints(['a', 'b'])
        self.assertEqual(holdout.pinned_drives(pin, ['a', 'b'], drives), (['a', 'b'], drives, []))
        # A download added a held-out segment before the second model: it keeps the first one's drives.
        self.segment('c')
        segments, now, changed = holdout.pinned_drives(pin, ['a', 'b', 'c'], {})
        self.assertEqual((segments, now, changed), (['a', 'b'], drives, []))

    def test_pin_and_a_localization_during_the_job(self):
        self.segment('a')
        self.segment('b', localizer=None)
        pin = self.root / 'drives.json'
        drives = holdout.drive_fingerprints(['a', 'b'])
        holdout.pinned_drives(pin, ['a', 'b'], drives)
        # b localized for the first time and a localized again, between the two models.
        self.segment('b')
        self.segment('a', localizer=b'refined path')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            segments, now, changed = holdout.pinned_drives(pin, ['a', 'b'], {})
        self.assertEqual(segments, ['a', 'b'])
        self.assertEqual(changed, ['a', 'b'])
        # b stays out, as for the first model; a is measured as it is now.
        self.assertIsNone(now['b'])
        self.assertEqual(now['a'], holdout.drive_fingerprints(['a'])['a'])
        self.assertIn('WARNING', output.getvalue())

    def test_listed_segment_gone_from_data(self):
        self.segment('0000011e0c8825628a01')
        names, tag = holdout.select_segments('0000011e0c8825628a01,0000011e0c8825628a02', check_exists=False)
        self.assertEqual(len(names), 2)
        self.assertTrue(tag.startswith('lista-'))
        with self.assertRaises(SystemExit):
            holdout.select_segments('0000011e0c8825628a01,0000011e0c8825628a02')


class ProgressTest(unittest.TestCase):
    def lines(self, progress, steps):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            for frames in steps:
                progress(frames)
        return output.getvalue().split()

    def test_one_model(self):
        self.assertEqual(self.lines(holdout.Progress(200), [100, 100]), ['PROGRESS', '100/200', 'PROGRESS', '200/200'])

    def test_second_of_two_models_continues_the_bar(self):
        self.assertEqual(self.lines(holdout.Progress(200, 1, 2), [100, 100]),
                         ['PROGRESS', '300/400', 'PROGRESS', '400/400'])

    def test_part_option(self):
        self.assertEqual(holdout.progress_part('1/2'), (1, 2))
        for text in ('2/2', '0/0', 'x', '1'):
            with self.assertRaises(SystemExit):
                holdout.progress_part(text)


if __name__ == '__main__':
    unittest.main()
