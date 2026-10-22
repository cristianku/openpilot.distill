"""distill-web: run parsing (train.log, metrics.jsonl), ETA, chart series, the data listing with the
hold-out rule, evaluation results, comparison videos, the comma probe and the Hugging Face list.
Standard library only, no GPU, no network."""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from datetime import UTC, datetime
from pathlib import Path

from web import comma as comma_module
from web import hf as hf_module
from web.common import comma_host, hf_repo, is_heldout, read_env_file
from web.metrics import RunReader, average_step_time, downsample, parse_metrics_jsonl, parse_train_log, time_from_name
from web.status import compare_videos, eval_results, failure_reason, segments, video_path

# The 15 held-out segments of the server on 2026-10-21 (critic report, from supervised/dataset.py).
HELDOUT = ('0000011e0c8825628a08', '0000011e0c8825628a12', '0000011f77d17f9ee308', '0000011f77d17f9ee312',
           '000001222a2060273706', '000001222a2060273710', '000001256d7bcedee206', '000001256d7bcedee210',
           '000001256d7bcedee224', '00000126f84ad127f006', '00000126f84ad127f010', '00000126f84ad127f024',
           '00000128edfa65432f02', '00000128edfa65432f16', '00000128edfa65432f20')

TRAIN_LOG = """student: 17.0M parameters
precision: fp16
wandb: Syncing run summer-bird-9
step 1/5: loss=1393.81 gt=462.596 distill=931.215 time=34.10s
step 2/5: loss=900 gt=300 distill=600 time=0.70s
step 3/5: loss=800 gt=250 distill=550 time=0.72s
resume local training from step 2/5
step 3/5: loss=700 gt=200 distill=500 time=0.80s
step 4/5: loss=nan gt=100 distill=inf time=0.70s
step 5/5: loss=600 gt=100 distill=500 time=0.70s
wandb: Run summary:
step 6/5: loss=1 gt=1 distill=1 time=0.70s (not a step line)
step 7/5: loss=1 gt=1"""


class ParseTest(unittest.TestCase):
    def test_train_log_keeps_the_last_occurrence(self):
        records, total = parse_train_log(TRAIN_LOG)
        self.assertEqual(total, 5)
        self.assertEqual(sorted(records), [1, 2, 3, 4, 5])
        self.assertEqual(records[3]['loss'], 700.0)
        self.assertIsNone(records[4]['loss'])
        self.assertIsNone(records[4]['distill'])
        self.assertEqual(records[5]['step_time'], 0.7)

    def test_metrics_jsonl(self):
        lines = []
        for step in range(1, 4):
            lines.append(json.dumps({'step': step, 'total_steps': 3, 'time': 1000.0 + step, 'loss': 10.0 - step,
                                     'gt': 1.0, 'distill': 2.0, 'grad_norm': 5.0, 'step_time': 0.5,
                                     'gt/plan': 0.25, 'distill/plan': 0.1}))
        lines.append(json.dumps({'step': 2, 'total_steps': 3, 'loss': 99.0}))
        lines.append('{"step": 4, "loss": ')
        lines.append('[1, 2]')
        lines.append(json.dumps({'step': True, 'loss': 1}))
        lines.append('{"step": 3.5, "loss": NaN}')
        records, total, last = parse_metrics_jsonl('\n'.join(lines))
        self.assertEqual(total, 3)
        self.assertEqual(sorted(records), [1, 2, 3])
        self.assertEqual(records[2]['loss'], 99.0)
        self.assertIsNone(records[2]['gt'])
        self.assertEqual(records[3]['gt/plan'], 0.25)
        self.assertEqual(last['distill/plan'], 0.1)

    def test_average_step_time(self):
        self.assertIsNone(average_step_time({}))
        records = {1: {'step_time': 34.0}, 2: {'step_time': 1.0}, 3: {'step_time': 3.0}}
        self.assertEqual(average_step_time(records), 2.0)
        # The wall clock wins: it also counts the time between the steps.
        timed = {10: {'step_time': 0.5, 'time': 100.0}, 20: {'step_time': 0.5, 'time': 120.0}}
        self.assertEqual(average_step_time(timed), 2.0)

    def test_downsample(self):
        records = {}
        for step in range(1, 11):
            records[step] = {'loss': float(step), 'gt': None, 'distill': 1.0, 'gt/plan': None, 'grad_norm': None,
                             'step_time': 0.5}
        result = downsample(records, 5, 10, 'train.log', 'x')
        self.assertEqual(result['bucket'], 2)
        self.assertEqual(result['steps'], [2, 4, 6, 8, 10])
        self.assertEqual(result['series']['loss'], [1.5, 3.5, 5.5, 7.5, 9.5])
        self.assertEqual(result['series']['gt'], [None, None, None, None, None])
        self.assertEqual(result['available'], ['loss', 'distill', 'step_time'])
        self.assertEqual(downsample(records, 100, 10, None)['bucket'], 1)
        self.assertEqual(downsample({}, 100, None, None)['steps'], [])

    def test_time_from_name(self):
        self.assertEqual(time_from_name('2026-10-21_1544'), datetime(2026, 10, 21, 15, 44, tzinfo=UTC).timestamp())
        self.assertEqual(time_from_name('2026-10-21_154412'),
                         datetime(2026, 10, 21, 15, 44, 12, tzinfo=UTC).timestamp())
        self.assertIsNone(time_from_name('x'))


class RunReaderTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.repo = Path(self.directory.name)
        self.base = self.repo / 'outputs' / 'supervised'
        self.base.mkdir(parents=True)

    def tearDown(self):
        self.directory.cleanup()

    def run_folder(self, name, log=None, metrics=None, files=()):
        folder = self.base / name
        folder.mkdir()
        if log is not None:
            (folder / 'train.log').write_text(log)
        if metrics is not None:
            (folder / 'metrics.jsonl').write_text(metrics)
        for file_name in files:
            (folder / file_name).write_bytes(b'x')
        return folder

    def test_states(self):
        self.run_folder('2026-10-21_1544', TRAIN_LOG, files=('supercombo.pt', 'driving_supercombo.onnx'))
        self.run_folder('2026-10-21_0834', 'step 109/5000: loss=651.69 gt=245.609 distill=406.082 time=0.76s\n',
                        files=('supercombo.training.pt',))
        self.run_folder('2026-10-21_1543', '')
        self.run_folder('not-a-run')
        reader = RunReader(self.repo)
        self.assertEqual(reader.run_names(), ['2026-10-21_1544', '2026-10-21_1543', '2026-10-21_0834'])
        finished = reader.summary('2026-10-21_1544', False)
        self.assertEqual(finished['status'], 'finished')
        self.assertEqual(finished['steps_done'], 5)
        self.assertEqual(finished['percent'], 100.0)
        self.assertTrue(finished['has_onnx'])
        self.assertFalse(finished['resumable'])
        self.assertIsNone(finished['eta_seconds'])
        self.assertEqual(finished['last']['loss'], 600.0)
        interrupted = reader.summary('2026-10-21_0834', False)
        self.assertEqual(interrupted['status'], 'interrupted')
        self.assertTrue(interrupted['resumable'])
        self.assertEqual(interrupted['steps_total'], 5000)
        running = reader.summary('2026-10-21_0834', True)
        self.assertEqual(running['status'], 'running')
        self.assertFalse(running['resumable'])
        self.assertEqual(running['eta_seconds'], round((5000 - 109) * 0.76))
        empty = reader.summary('2026-10-21_1543', False)
        self.assertEqual(empty['steps_done'], 0)
        self.assertIsNone(empty['source'])
        self.assertIsNone(reader.summary('../x', False))
        self.assertIsNone(reader.summary('2026-01-01_0000', False))

    def test_metrics_jsonl_is_preferred_and_run_json_is_read(self):
        metrics = ''
        for step in range(1, 101):
            metrics += json.dumps({'step': step, 'total_steps': 200, 'time': 1000.0 + 2 * step, 'loss': 5.0,
                                   'gt': 1.0, 'distill': 4.0, 'grad_norm': 3.0, 'step_time': 0.7,
                                   'gt/plan': 0.5}) + '\n'
        folder = self.run_folder('2026-10-22_101530', 'step 1/200: loss=1 gt=1 distill=1 time=1.00s\n', metrics)
        (folder / 'run.json').write_text(json.dumps({'mode': 'fresh', 'steps': 200, 'batch_size': 8, 'workers': 6,
                                                     'precision': 'fp16', 'started': 999.0}))
        reader = RunReader(self.repo)
        summary = reader.summary('2026-10-22_101530', True)
        self.assertEqual(summary['source'], 'metrics.jsonl')
        self.assertEqual(summary['steps_done'], 100)
        self.assertEqual(summary['avg_step_time'], 2.0)
        self.assertEqual(summary['eta_seconds'], 200)
        self.assertEqual(summary['last']['gt/plan'], 0.5)
        self.assertEqual(summary['config']['batch_size'], 8)
        self.assertEqual(summary['started'], 999.0)
        series = reader.series('2026-10-22_101530', 50)
        self.assertEqual(series['bucket'], 2)
        self.assertIn('grad_norm', series['available'])
        # A run.json with "finished" makes the run finished even without supercombo.pt.
        (folder / 'run.json').write_text(json.dumps({'started': 999.0, 'finished': 2000.0}))
        self.assertEqual(reader.summary('2026-10-22_101530', False)['status'], 'finished')

    def test_detail(self):
        log = 'Teacher md-v3: ok\n  10%\r  50%\r 100%\nstep 1/9: loss=3 gt=1 distill=2 time=0.70s\n'
        folder = self.run_folder('2026-10-22_101530', log)
        (folder / 'run.json').write_text(json.dumps({'mode': 'resume', 'gpus': '0', 'world_size': 1,
                                                     'resumed': [{'time': 5.0, 'step': 500}, 'junk']}))
        reader = RunReader(self.repo)
        detail = reader.detail('2026-10-22_101530', False)
        self.assertEqual(detail['resumed'], [{'time': 5.0, 'step': 500}])
        self.assertEqual(detail['config']['gpus'], '0')
        # A progress bar rewritten with \r keeps only its last state.
        self.assertEqual(detail['log_tail'].splitlines()[:2], ['Teacher md-v3: ok', ' 100%'])
        # Only the end of a long log, from a whole line.
        tail = reader.detail('2026-10-22_101530', False, log_bytes=60)['log_tail']
        self.assertTrue(tail.startswith('step 1/9'))
        self.assertIsNone(reader.detail('../x', False))

    def test_cache_follows_the_file(self):
        folder = self.run_folder('2026-10-21_1544', 'step 1/9: loss=3 gt=1 distill=2 time=0.70s\n')
        reader = RunReader(self.repo)
        self.assertEqual(reader.summary('2026-10-21_1544', True)['steps_done'], 1)
        with (folder / 'train.log').open('a') as handle:
            handle.write('step 2/9: loss=3 gt=1 distill=2 time=0.70s\n')
        stamp = time.time() + 5
        os.utime(folder / 'train.log', (stamp, stamp))
        self.assertEqual(reader.summary('2026-10-21_1544', True)['steps_done'], 2)


class HeldoutTest(unittest.TestCase):
    def test_rule_matches_the_training(self):
        for name in HELDOUT:
            self.assertTrue(is_heldout(name), name)
        for name in ('0000011e0c8825628a01', '0000011e0c8825628a02', '0000011e0c8825628a03', '00000128edfa65432f03'):
            self.assertFalse(is_heldout(name), name)
        self.assertFalse(is_heldout('not-hex'))

    def test_same_expression_as_supervised_dataset(self):
        # supervised/dataset.py: (hash(int(segment, 16)) % 10 == 0) == val
        for name in HELDOUT + ('0000011e0c8825628a01',):
            self.assertEqual(is_heldout(name), hash(int(name, 16)) % 10 == 0)


class DataTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.repo = Path(self.directory.name)
        data = self.repo / 'data'
        localized = data / '0000011e0c8825628a08'
        localized.mkdir(parents=True)
        (localized / 'localizer.safetensors').write_bytes(b'x')
        (localized / 'frame_info.safetensors').write_bytes(b'x')
        (localized / 'localizer.ionex').write_text('2026/293 RAP\n')
        (localized / 'fcamera.hevc').write_bytes(b'x')
        failed = data / '0000011e0c8825628a01'
        failed.mkdir()
        (failed / 'localizer.failed').write_text('2026-10-21 15:42 RuntimeError: no network\nTraceback\n')
        os.symlink(self.repo / 'gone' / 'fcamera.hevc', failed / 'fcamera.hevc')
        (data / '0000011f77d17f9ee301').mkdir()
        (data / 'README').write_text('not a segment')
        discarded = self.repo / 'data_scartati'
        for name, reason in (('000001222a2060273700', 'UnusableSegment: only comma four'),
                             ('000001222a2060273701', 'UnusableSegment: only comma four'),
                             ('0000011e0c8825628a14', 'UnusableSegment: fewer than four GPS satellites')):
            (discarded / name).mkdir(parents=True)
            (discarded / name / 'localizer.failed').write_text('2026-10-21 15:10 ' + reason + '\n' + reason + '\n')
        (discarded / '0000011e0c8825628a15').mkdir()

    def tearDown(self):
        self.directory.cleanup()

    def test_segments(self):
        result = segments(self.repo)
        counts = result['counts']
        self.assertEqual(counts['total'], 3)
        self.assertEqual(counts['localized'], 1)
        self.assertEqual(counts['failed'], 1)
        self.assertEqual(counts['todo'], 1)
        self.assertEqual(counts['heldout'], 1)
        self.assertEqual(counts['heldout_localized'], 1)
        self.assertEqual(counts['broken_video'], 1)
        self.assertEqual(counts['discarded'], 4)
        rows = {}
        for row in result['segments']:
            rows[row['name']] = row
        self.assertEqual(rows['0000011e0c8825628a08']['map'], '2026/293 RAP')
        self.assertEqual(rows['0000011e0c8825628a08']['route'], '0000011e--0c8825628a')
        self.assertEqual(rows['0000011e0c8825628a08']['number'], 8)
        self.assertEqual(rows['0000011e0c8825628a01']['reason'], 'RuntimeError: no network')
        self.assertTrue(rows['0000011e0c8825628a01']['broken_video'])
        self.assertEqual(result['routes'][0], {'route': '0000011e--0c8825628a', 'segments': 2, 'localized': 1,
                                               'failed': 1, 'todo': 0, 'heldout': 1,
                                               'map': {'final': 0, 'rapid': 1, 'prediction': 0, 'unknown': 0}})
        self.assertEqual(rows['0000011e0c8825628a08']['map_quality'], 'rapid')
        self.assertEqual(result['counts']['map']['rapid'], 1)
        self.assertEqual(result['discarded_reasons'][0], {'reason': 'UnusableSegment: only comma four', 'count': 2})
        reasons = []
        for item in result['discarded_reasons']:
            reasons.append(item['reason'])
        self.assertIn('reason not recorded', reasons)

    def test_map_quality(self):
        from datetime import datetime

        from web.status import map_of
        folder = Path(self.directory.name) / 'segment'
        folder.mkdir()
        self.assertEqual(map_of(folder), ('unknown', False))
        today = datetime(2026, 10, 22, 20, 0)
        (folder / 'localizer.ionex').write_text('2026/294 P1D\n')
        self.assertEqual(map_of(folder, today), ('prediction', True))
        self.assertEqual(map_of(folder, datetime(2026, 10, 21, 20, 0)), ('prediction', False))
        # A drive across midnight: the worst day counts.
        (folder / 'localizer.ionex').write_text('2026/270 FIN\n2026/271 RAP\n')
        self.assertEqual(map_of(folder, today), ('rapid', True))
        (folder / 'localizer.ionex').write_text('2026/280 FIN\n')
        self.assertEqual(map_of(folder, today), ('final', False))

    def test_proc_stat(self):
        from web.status import parse_proc_stat
        text = ('cpu  100 0 50 800 50 0 0 0 0 0\n'
                'cpu0 60 0 20 400 20 0 0 0 0 0\n'
                'cpu1 40 0 30 400 30 0 0 0 0 0\n'
                'intr 12345\n')
        # busy = total - idle - iowait
        self.assertEqual(parse_proc_stat(text), [(150, 1000), (80, 500), (70, 500)])

    def test_failure_reason_without_date(self):
        path = self.repo / 'x.failed'
        path.write_text('2026-10-20 17:32 RuntimeError: global yaw\n')
        self.assertEqual(failure_reason(path), 'RuntimeError: global yaw')
        self.assertEqual(failure_reason(self.repo / 'missing'), '')

    def test_empty_repo(self):
        empty = self.repo / 'empty'
        empty.mkdir()
        self.assertEqual(segments(empty)['counts']['total'], 0)
        self.assertEqual(eval_results(empty), [])
        self.assertEqual(compare_videos(empty), [])


class EvalAndVideoTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.repo = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_leaderboard(self):
        folder = self.repo / 'outputs' / 'eval'
        folder.mkdir(parents=True)
        # Schema 1 of evaluation/holdout.py.
        (folder / 'stock.json').write_text(json.dumps({
            'schema': 1, 'kind': 'holdout', 'created': '2026-10-22T10:00:00Z',
            'model': {'spec': 'stock', 'key': 'stock', 'label': 'comma CD210 (stock)', 'sha256': '65a08adc31d5c456aa'},
            'selection': 'heldout', 'segments': ['a', 'b', 'c'], 'settings': {'rate_hz': 5, 'horizon_s': 10.0},
            'per_segment': {}, 'skipped': ['c'],
            'overall': {'segments': 2, 'frames': 400, 'avg_miss_m': 1.5, 'avg_lateral_miss_m': 0.4}}))
        (folder / 'run_2026-10-21_1544.json').write_text(json.dumps({
            'model': {'spec': 'run:2026-10-21_1544', 'label': 'student run 2026-10-21_1544'},
            'selection': 'heldout', 'settings': {'rate_hz': 5},
            'overall': {'segments': 15, 'frames': 3000, 'avg_miss_m': 0.9, 'avg_lateral_miss_m': None}}))
        (folder / 'teacher__all_r20_h10.json').write_text(json.dumps({
            'model': {'spec': 'teacher'}, 'selection': 'all', 'settings': {'rate_hz': 20, 'horizon_s': 10},
            'overall': {'segments': 130, 'avg_miss_m': 1.2}}))
        (folder / 'broken.json').write_text('{')
        (folder / 'weird.json').write_text('{"overall": {"avg_miss_m": 1e999}}')
        (folder / 'stock.json.tmp').write_text('{}')
        (folder / 'notes.txt').write_text('x')
        results = eval_results(self.repo)
        models = []
        for row in results:
            models.append(row['model'])
        self.assertEqual(models, ['run:2026-10-21_1544', 'teacher', 'stock', 'weird'])
        self.assertEqual(results[0]['segments'], 15)
        self.assertIsNone(results[0]['lateral_miss_m'])
        self.assertEqual(results[1]['selection'], 'all')
        self.assertEqual(results[1]['rate_hz'], 20.0)
        stock = results[2]
        self.assertEqual(stock['label'], 'comma CD210 (stock)')
        self.assertEqual(stock['segments'], 2)
        self.assertEqual(stock['frames'], 400)
        self.assertEqual(stock['skipped'], 1)
        self.assertEqual(stock['lateral_miss_m'], 0.4)
        self.assertEqual(stock['date'], '2026-10-22T10:00:00Z')
        self.assertTrue(stock['sha256'].startswith('65a08adc31d5c456'))
        self.assertIsNone(results[3]['avg_miss_m'])
        json.dumps(results, allow_nan=False)

    def test_videos(self):
        folder = self.repo / 'outputs' / 'compare'
        folder.mkdir(parents=True)
        name = '20261022-101500_0000011e0c8825628a08_stock_vs_run_2026-10-21_1544'
        (folder / (name + '.mp4')).write_bytes(b'v')
        # Schema 1 of evaluation/compare.py.
        (folder / (name + '.json')).write_text(json.dumps({
            'schema': 1, 'kind': 'compare', 'created': '2026-10-22T10:15:00Z', 'segment': '0000011e0c8825628a08',
            'video': str(folder / (name + '.mp4')),
            'args': {'reference': 'stock', 'student': 'run:2026-10-21_1544', 'start_s': 2.0, 'duration_s': 15.0,
                     'speed': 0.5, 'inset': True, 'show_gt': False},
            'clip': {'fps': 20, 'start_s': 5.2, 'end_s': 17.0},
            'models': {'reference': {'spec': 'stock', 'label': 'comma CD210 (stock)', 'avg_miss_m': 1.25,
                                     'avg_lateral_miss_m': 0.3},
                       'student': {'spec': 'run:2026-10-21_1544', 'label': 'student run 2026-10-21_1544',
                                   'avg_miss_m': 0.75, 'avg_lateral_miss_m': None}},
            'per_frame': []}))
        (folder / 'old.mp4').write_bytes(b'v')
        (folder / 'next.partial.mp4').write_bytes(b'v')
        os.symlink(folder / (name + '.mp4'), folder / 'link.mp4')
        (folder / 'bad name.mp4').write_bytes(b'v')
        videos = {}
        for video in compare_videos(self.repo):
            videos[video['name']] = video
        self.assertEqual(sorted(videos), [name + '.mp4', 'old.mp4'])
        video = videos[name + '.mp4']
        self.assertTrue(video['has_json'])
        self.assertEqual(video['segment'], '0000011e0c8825628a08')
        self.assertEqual(video['start'], 5.2)
        self.assertEqual(video['duration'], 11.8)
        self.assertEqual(video['speed'], 0.5)
        self.assertEqual(video['reference'], {'model': 'stock', 'label': 'comma CD210 (stock)', 'avg_miss_m': 1.25,
                                              'lateral_miss_m': 0.3})
        self.assertEqual(video['student']['avg_miss_m'], 0.75)
        self.assertIsNone(video['student']['lateral_miss_m'])
        self.assertFalse(videos['old.mp4']['has_json'])
        self.assertIsNone(videos['old.mp4']['student']['model'])
        self.assertIsNotNone(video_path(self.repo, name + '.mp4'))
        self.assertIsNone(video_path(self.repo, 'link.mp4'))
        self.assertIsNone(video_path(self.repo, 'next.partial.mp4'))
        self.assertIsNone(video_path(self.repo, '../old.mp4'))
        self.assertIsNone(video_path(self.repo, 'missing.mp4'))


class SettingsTest(unittest.TestCase):
    def test_env_file_and_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / '.env').write_text('# comment\nDISTILL_WEB_TOKEN=secret\nexport HF_REPO="someone/models"\n'
                                       'COMMA_HOST=-oProxyCommand=x\nbroken line\n')
            self.assertEqual(read_env_file(repo / '.env', ('HF_REPO',)), {'HF_REPO': 'someone/models'})
            saved = {}
            for key in ('HF_REPO', 'COMMA_HOST'):
                saved[key] = os.environ.pop(key, None)
            try:
                self.assertEqual(hf_repo(repo), 'someone/models')
                # A value that would become an ssh option falls back to the default.
                self.assertEqual(comma_host(repo), 'comma')
                os.environ['HF_REPO'] = 'other/repo'
                self.assertEqual(hf_repo(repo), 'other/repo')
                os.environ['HF_REPO'] = 'bad repo'
                self.assertEqual(hf_repo(repo), 'cristianku/openpilot-distill-models')
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


class CommaTest(unittest.TestCase):
    def test_parse_probe(self):
        result = comma_module.parse_probe(0, 'onroad=0\nbundle={"displayName": "Distill 1544", "index": 1000}\n'
                                             'backup=1\n')
        self.assertEqual(result, {'reachable': True, 'onroad': False, 'bundle': 'Distill 1544',
                                  'bundle_default': False, 'backup': True, 'error': ''})
        # The param cut in the middle of its JSON: the name is still found.
        cut = comma_module.parse_probe(0, 'bundle={"index": 1002, "internalName": "2026-10-21_1544", "displayN\n')
        self.assertEqual(cut['bundle'], '2026-10-21_1544')
        self.assertTrue(comma_module.parse_probe(0, 'bundle=\n')['bundle_default'])
        self.assertTrue(comma_module.parse_probe(0, 'onroad=1\nbundle=\nbackup=0\n')['onroad'])
        self.assertEqual(comma_module.parse_probe(0, 'onroad=\nbundle={}\n')['bundle'], 'Default')
        self.assertFalse(comma_module.parse_probe(255, '')['reachable'])
        self.assertIn('timeout', comma_module.parse_probe(124, '')['error'])

    def test_monitor_uses_the_injected_probe(self):
        calls = []

        def probe(host):
            calls.append(host)
            return 0, 'onroad=0\nbundle=\nbackup=0\n'

        monitor = comma_module.CommaMonitor('comma', probe=probe)
        self.assertIsNone(monitor.snapshot()['reachable'])
        state = monitor.probe_now()
        self.assertTrue(state['reachable'])
        self.assertEqual(calls, ['comma'])
        self.assertTrue(monitor.snapshot()['reachable'])


class HfTest(unittest.TestCase):
    def test_catalog(self):
        def fetch(repo, path):
            if path == 'models':
                return [{'type': 'directory', 'path': 'models/cd210_native'},
                        {'type': 'directory', 'path': 'models/2026_10_21_1544'},
                        {'type': 'file', 'path': 'models/readme.md'}]
            if path == 'training':
                return [{'type': 'directory', 'path': 'training/releases'},
                        {'type': 'file', 'path': 'training/stable.json'}]
            return [{'type': 'directory', 'path': 'training/releases/2026_10_21_1544'}]

        catalog = hf_module.HfCatalog('cristianku/openpilot-distill-models', fetch=fetch)
        snapshot = catalog.snapshot()
        self.assertTrue(snapshot['ok'])
        self.assertEqual(snapshot['models'], ['2026_10_21_1544', 'cd210_native'])
        self.assertEqual(snapshot['releases'], ['2026_10_21_1544'])
        self.assertTrue(snapshot['stable'])

    def test_unreachable(self):
        def fetch(repo, path):
            raise OSError('no network')

        snapshot = hf_module.HfCatalog('x/y', fetch=fetch).snapshot()
        self.assertFalse(snapshot['ok'])
        self.assertIn('Hugging Face', snapshot['error'])


if __name__ == '__main__':
    unittest.main()
