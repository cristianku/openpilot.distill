"""distill-web: the drive list of the comparison (cached numbers, thumbnails), standard library only."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from web.drives import CACHE_VERSION, DriveIndex

SEGMENT = '0000011e0c8825628a08'


class DriveIndexTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.repo = Path(self.directory.name)
        self.folder = self.repo / 'data' / SEGMENT
        self.folder.mkdir(parents=True)
        (self.folder / 'localizer.safetensors').write_bytes(b'x')
        (self.repo / 'data' / 'not-a-segment').mkdir()
        self.index = DriveIndex(self.repo)
        # No real computation in the tests: the background fill does nothing.
        self.index._fill = lambda folders: None

    def tearDown(self):
        self.directory.cleanup()

    def cache(self, **values):
        cache = self.repo / 'outputs' / 'web' / 'drives'
        cache.mkdir(parents=True, exist_ok=True)
        data = {'version': CACHE_VERSION, 'source_mtime': (self.folder / 'localizer.safetensors').stat().st_mtime,
                'start': 1792500221.4, 'duration_s': 60.0, 'avg_kmh': 60.3, 'max_kmh': 70.0, 'distance_km': 1.0,
                'turn_deg': 198, 'trace': [[0, 0], [1, 1]]}
        data.update(values)
        (cache / (SEGMENT + '.json')).write_text(json.dumps(data))

    def test_pending_then_cached(self):
        answer = self.index.drives()
        self.assertEqual(answer['pending'], 1)
        self.assertEqual(answer['drives'], [{'segment': SEGMENT, 'route': '0000011e--0c8825628a', 'number': 8,
                                             'heldout': True}])
        self.cache()
        answer = self.index.drives()
        self.assertEqual(answer['pending'], 0)
        self.assertEqual(answer['drives'][0]['avg_kmh'], 60.3)
        self.assertEqual(answer['drives'][0]['trace'], [[0, 0], [1, 1]])

    def test_stale_cache_is_recomputed(self):
        self.cache(source_mtime=1.0)
        self.assertEqual(self.index.drives()['pending'], 1)
        self.cache(version=CACHE_VERSION + 1)
        self.assertEqual(self.index.drives()['pending'], 1)

    def test_thumbnail_refuses_bad_names(self):
        self.assertIsNone(self.index.thumbnail('../etc/passwd'))
        self.assertIsNone(self.index.thumbnail(SEGMENT))  # no video
        thumbs = self.repo / 'outputs' / 'web' / 'thumbs'
        thumbs.mkdir(parents=True)
        (thumbs / (SEGMENT + '.jpg')).write_bytes(b'jpeg')
        os.symlink(self.repo / 'missing.hevc', self.folder / 'fcamera.hevc')
        self.assertIsNone(self.index.thumbnail(SEGMENT))  # the video link points nowhere
        (self.repo / 'missing.hevc').write_bytes(b'video')
        self.assertEqual(self.index.thumbnail(SEGMENT), thumbs / (SEGMENT + '.jpg'))


if __name__ == '__main__':
    unittest.main()
