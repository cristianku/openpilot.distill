"""tasks.fetch and tasks.localize: arguments, segment selection and the folders on disk, with a fake
comma, the log reader and the localizer mocked, in temporary folders (no ssh, no network)."""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tasks.fetch as fetch
import tasks.localize as localize
from tasks.common import TaskError

ROUTE = '0000011c--8152ba83cf'
OTHER = '0000011d--aaaabbbbcc'


class FakeComma:
    """The comma: `files` maps remote paths to contents; `listing` is what the stat command prints."""

    def __init__(self, files: dict[str, bytes], listing: str = ''):
        self.files = files
        self.listing = listing
        self.pulls = []

    def require(self):
        pass

    def run(self, command, **kwargs):
        assert command == fetch.LISTING_COMMAND
        return subprocess.CompletedProcess([], 0, stdout=self.listing, stderr='')

    def script(self, script, *arguments, check=True, timeout=None):
        if script == fetch.ROUTE_SEGMENTS_SCRIPT:
            prefix = fetch.REMOTE_DIR + '/' + arguments[0] + '--'
            folders = sorted({path.rsplit('/', 1)[0] for path in self.files if path.startswith(prefix)})
            return subprocess.CompletedProcess([], 0, stdout=''.join(f + '\n' for f in folders), stderr='')
        if script == fetch.FILE_EXISTS_SCRIPT:
            return subprocess.CompletedProcess([], 0 if arguments[0] in self.files else 1, stdout='', stderr='')
        raise AssertionError('unexpected script')

    def pull(self, remote_path, local_path, progress=False):
        self.pulls.append(remote_path)
        if remote_path not in self.files:
            raise TaskError('copy with the comma failed (rsync, code 23)')
        target = Path(local_path)
        if target.is_dir():
            target = target / remote_path.rsplit('/', 1)[1]
        target.write_bytes(self.files[remote_path])


def segment_files(route: str, segments, names=('rlog.zst', 'fcamera.hevc', 'ecamera.hevc', 'qlog.zst')):
    files = {}
    for segment in segments:
        for name in names:
            files[fetch.REMOTE_DIR + '/' + route + '--' + str(segment) + '/' + name] = (
                (route + str(segment) + name).encode())
    return files


class FetchArgumentsTest(unittest.TestCase):
    def test_automatic(self):
        options = fetch.parse_args([])
        self.assertTrue(options.auto)
        self.assertEqual((options.since, options.localizer, options.host), (fetch.DEFAULT_SINCE, False, None))
        self.assertEqual(fetch.parse_args(['--since=2026-10-25']).since, '2026-10-25')
        self.assertEqual(fetch.parse_args(['--since', '2026-10-25']).since, '2026-10-25')

    def test_route_defaults_to_strip_ends(self):
        options = fetch.parse_args(['6616faac453a3064/' + ROUTE])
        self.assertEqual((options.route, options.auto, options.skip_mode), (ROUTE, False, True))
        self.assertEqual(options.skip, list(fetch.STRIP_ENDS))

    def test_manual_segments(self):
        self.assertEqual(fetch.parse_args([ROUTE, '--segments=1,2,5-7']).segments, [1, 2, 5, 6, 7])
        self.assertEqual(fetch.parse_args([ROUTE, '--segments', '3']).segments, [3])
        self.assertEqual(fetch.parse_args([ROUTE, '0', '2-4']).segments, [0, 2, 3, 4])
        options = fetch.parse_args(['--host', '192.168.88.29', '6616faac453a3064/' + ROUTE + '/1', '--localizer'])
        self.assertEqual((options.route, options.segments, options.host, options.localizer, options.skip_mode),
                         (ROUTE, [1], '192.168.88.29', True, False))
        self.assertFalse(fetch.parse_args([ROUTE, '--localizer', '--no-localizer']).localizer)

    def test_skip(self):
        options = fetch.parse_args([ROUTE, '--skip=0,1,2-3,last,last-2,second_last'])
        self.assertEqual(options.skip, [0, 1, 2, 3, 'last', 'last-2', 'second_last'])
        self.assertEqual(fetch.parse_args([ROUTE, '--skip', '0,1,']).skip, [0, 1])

    def test_help(self):
        self.assertTrue(fetch.parse_args(['--help']).help)

    def test_refused(self):
        for arguments in (['--since=20261020'], ['--since'], ['--segments=1'], ['--skip=0'], ['--strip-ends'],
                          [ROUTE, '--skip=0', '--segments=1'], [ROUTE, '--strip-ends', '3'],
                          ['not-a-route'], ['3'], [ROUTE, 'x'], [ROUTE, '--segments='], [ROUTE, '--segments=1,,2'],
                          [ROUTE, '--skip=foo'], ['--host='], ['--host'], [ROUTE, ROUTE + '/1']):
            with self.assertRaises(TaskError, msg=arguments):
                fetch.parse_args(arguments)


class SelectionTest(unittest.TestCase):
    def test_strip_ends_with_gaps(self):
        found = [0, 1, 3, 4, 7, 9]
        with mock.patch('sys.stdout'):
            self.assertEqual(fetch.choose_segments(ROUTE, found, list(fetch.STRIP_ENDS)), [3, 4])

    def test_skip_words(self):
        found = [0, 1, 2, 3, 4, 5]
        self.assertEqual(fetch.resolve_skip(found, ['last', 'last-2', 'first', 2]), [5, 3, 0, 2])
        self.assertEqual(fetch.resolve_skip([4], list(fetch.STRIP_ENDS)), [4, 4])
        self.assertEqual(fetch.resolve_skip([0, 1], list(fetch.STRIP_ENDS)), [0, 1, 0, 1])

    def test_nothing_left(self):
        with mock.patch('sys.stderr'):
            self.assertIsNone(fetch.choose_segments(ROUTE, [0, 1, 2, 3], list(fetch.STRIP_ENDS)))
            self.assertIsNone(fetch.choose_segments(ROUTE, [], list(fetch.STRIP_ENDS)))

    def test_remote_segments_sorted_numerically(self):
        comma = FakeComma(segment_files(ROUTE, [10, 2, 1]))
        self.assertEqual(fetch.remote_segments(comma, ROUTE), [1, 2, 10])

    def test_listing(self):
        listing = ('2026-10-18 10:00:00.0 +0000 ' + ROUTE + '--0\n'
                   '2026-10-21 10:00:00.0 +0000 ' + ROUTE + '--1\n'
                   '2026-10-19 09:00:00.0 +0000 ' + OTHER + '--3\n'
                   '2026-10-22 09:00:00.0 +0000 ' + OTHER + '--4\n'
                   '2026-10-22 09:00:00.0 +0000 not_hex--zz--1\n'
                   '2026-10-01 09:00:00.0 +0000 0000aaaa--bbbb--2\n')
        recent, first = fetch.parse_listing(listing, '2026-10-20')
        self.assertEqual(recent, [ROUTE, OTHER])
        self.assertEqual((first[ROUTE], first[OTHER]), (0, 3))

    def test_dashcam_routes(self):
        listing = ('2026-10-21 10:00:00.0 +0000 ' + ROUTE + '--0\n'
                   '2026-10-21 10:00:00.0 +0000 ' + OTHER + '--0\n'
                   '2026-10-21 10:00:00.0 +0000 0000000a--0000000b--0\n')
        files = {**segment_files(ROUTE, [0]), **segment_files(OTHER, [0])}
        comma = FakeComma(files, listing)
        states = {ROUTE: 'dashcam', OTHER: 'normal'}

        def inspector(mode, path):
            self.assertEqual(mode, 'dashcam')
            return states[path.name.split('.')[0]]

        with mock.patch.object(fetch, 'inspect_log', side_effect=inspector), mock.patch('sys.stdout'), \
                mock.patch('sys.stderr'):
            self.assertEqual(fetch.find_dashcam_routes(comma, '2026-10-20'), [ROUTE])


class LengthAndReverseTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.videos = Path(self.directory.name) / 'videos'
        self.videos.mkdir()
        for name in ('stdout', 'stderr'):
            patcher = mock.patch('sys.' + name)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_route_length_uses_first_and_last_qlog(self):
        comma = FakeComma(segment_files(ROUTE, range(6)))
        for seconds, expected in (('299.99', False), ('300.0', True), ('unknown', False), ('1e+03', False)):
            with mock.patch.object(fetch, 'inspect_log', return_value=seconds) as inspector:
                self.assertEqual(fetch.route_is_long_enough(comma, ROUTE, [0, 1, 2, 3, 4, 5]), expected)
            self.assertEqual(inspector.call_args[0][0], 'duration')
        self.assertEqual(comma.pulls[-2:], [fetch.REMOTE_DIR + '/' + ROUTE + '--0/qlog.zst',
                                            fetch.REMOTE_DIR + '/' + ROUTE + '--5/qlog.zst'])

    def test_route_length_without_qlog(self):
        comma = FakeComma(segment_files(ROUTE, [0]))
        with mock.patch.object(fetch, 'inspect_log', return_value='900.0'):
            self.assertFalse(fetch.route_is_long_enough(comma, ROUTE, [0, 5]))

    def test_reverse_excluded(self):
        comma = FakeComma(segment_files(ROUTE, [2, 3, 4, 5]))
        states = {2: 'forward', 3: 'reverse', 4: 'unknown'}

        def inspector(mode, path):
            self.assertEqual(mode, 'reverse')
            return states[int(path.parent.name.rpartition('--')[2])]

        # 5: its rlog is not on the comma; 6: a link in the shared folder.
        comma.files.pop(fetch.REMOTE_DIR + '/' + ROUTE + '--5/rlog.zst')
        os.symlink(self.videos / 'elsewhere', self.videos / (ROUTE + '--6'))
        with mock.patch.object(fetch, 'inspect_log', side_effect=inspector):
            self.assertEqual(fetch.filter_reverse_segments(comma, ROUTE, [2, 3, 4, 5, 6], self.videos), [2])
        # The rlog stays in the shared folder: the next run does not copy it again.
        self.assertTrue((self.videos / (ROUTE + '--2') / 'rlog.zst').is_file())
        comma.pulls.clear()
        with mock.patch.object(fetch, 'inspect_log', return_value='forward'):
            self.assertEqual(fetch.filter_reverse_segments(comma, ROUTE, [2], self.videos), [2])
        self.assertEqual(comma.pulls, [])


class LayoutTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.videos = root / 'comma-videos'
        self.videos.mkdir()
        self.data = root / 'data'
        self.discard = root / 'data_scartati'
        self.data.mkdir()
        for patcher in (mock.patch.object(fetch, 'DATA_DIR', self.data),
                        mock.patch.object(fetch, 'DISCARD_DIR', self.discard),
                        mock.patch('sys.stdout'), mock.patch('sys.stderr')):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_links_into_shared_folder(self):
        comma = FakeComma(segment_files(ROUTE, [3]))
        fetch.download_segments(comma, ROUTE, [3], self.videos, False)
        shared = self.videos / (ROUTE + '--3')
        folder = self.data / '0000011c8152ba83cf03'
        for name in fetch.FILES:
            self.assertTrue((shared / name).is_file())
            self.assertTrue((folder / name).is_symlink())
            self.assertEqual(os.readlink(folder / name), str(shared / name))
            self.assertTrue(os.path.isabs(os.readlink(folder / name)))
        self.assertFalse((shared / 'qlog.zst').exists())
        # Again: nothing changes, the links stay.
        fetch.download_segments(comma, ROUTE, [3], self.videos, False)
        self.assertEqual(os.readlink(folder / 'rlog.zst'), str(shared / 'rlog.zst'))

    def test_old_layout_moved(self):
        comma = FakeComma(segment_files(ROUTE, [3]))
        folder = self.data / '0000011c8152ba83cf03'
        folder.mkdir()
        (folder / 'fcamera.hevc').write_bytes(b'old copy')
        (folder / 'localizer.safetensors').write_bytes(b'outputs')
        shared = self.videos / (ROUTE + '--3')
        shared.mkdir()
        (shared / 'ecamera.hevc').write_bytes(comma.files[fetch.REMOTE_DIR + '/' + ROUTE + '--3/ecamera.hevc'])
        (folder / 'ecamera.hevc').write_bytes((shared / 'ecamera.hevc').read_bytes())
        fetch.download_segments(comma, ROUTE, [3], self.videos, False)
        self.assertTrue((folder / 'fcamera.hevc').is_symlink())
        self.assertTrue((folder / 'ecamera.hevc').is_symlink())
        self.assertEqual((folder / 'localizer.safetensors').read_bytes(), b'outputs')

    def test_different_copies_stop(self):
        folder = self.data / 'x'
        shared = self.videos / 'x'
        folder.mkdir()
        shared.mkdir()
        (folder / 'rlog.zst').write_bytes(b'one')
        (shared / 'rlog.zst').write_bytes(b'two')
        with self.assertRaisesRegex(TaskError, 'differ'):
            fetch.move_old_file(folder / 'rlog.zst', shared / 'rlog.zst')
        self.assertEqual((folder / 'rlog.zst').read_bytes(), b'one')
        os.remove(shared / 'rlog.zst')
        os.link(folder / 'rlog.zst', shared / 'rlog.zst')
        with self.assertRaisesRegex(TaskError, 'same file'):
            fetch.move_old_file(folder / 'rlog.zst', shared / 'rlog.zst')

    def test_folder_links_stop(self):
        comma = FakeComma(segment_files(ROUTE, [3]))
        (self.videos / 'real').mkdir()
        os.symlink(self.videos / 'real', self.data / '0000011c8152ba83cf03')
        with self.assertRaisesRegex(TaskError, 'real folder'):
            fetch.download_segments(comma, ROUTE, [3], self.videos, False)

    def test_missing_on_comma(self):
        comma = FakeComma(segment_files(ROUTE, [3], names=('rlog.zst', 'fcamera.hevc')))
        with self.assertRaisesRegex(TaskError, 'ecamera.hevc does not exist on the comma'):
            fetch.download_segments(comma, ROUTE, [3], self.videos, False)

    def test_discarded_not_again(self):
        comma = FakeComma(segment_files(ROUTE, [3, 4]))
        (self.discard / '0000011c8152ba83cf03').mkdir(parents=True)
        fetch.download_segments(comma, ROUTE, [3, 4], self.videos, False)
        self.assertFalse((self.data / '0000011c8152ba83cf03').exists())
        self.assertTrue((self.data / '0000011c8152ba83cf04').is_dir())

    def test_localizer_after_download(self):
        comma = FakeComma(segment_files(ROUTE, [3]))
        with mock.patch.object(fetch, 'run_localizer') as localizer:
            fetch.download_segments(comma, ROUTE, [3], self.videos, True)
        localizer.assert_called_once_with('0000011c8152ba83cf03')

    def test_automatic_fetch(self):
        listing = ''.join('2026-10-21 10:00:00.0 +0000 ' + ROUTE + '--' + str(s) + '\n' for s in range(7))
        comma = FakeComma(segment_files(ROUTE, range(7)), listing)

        def inspector(mode, *paths):
            if mode == 'dashcam':
                return 'dashcam'
            if mode == 'duration':
                return '420.5'
            return 'reverse' if paths[0].parent.name.endswith('--3') else 'forward'

        with mock.patch.object(fetch, 'Comma', return_value=comma), \
                mock.patch.object(fetch, 'inspect_log', side_effect=inspector), \
                mock.patch.dict(os.environ, {'COMMA_VIDEOS_DIR': str(self.videos)}):
            self.assertEqual(fetch.fetch(fetch.parse_args([])), 0)
        self.assertEqual(sorted(path.name for path in self.data.iterdir()),
                         ['0000011c8152ba83cf02', '0000011c8152ba83cf04'])

    def test_manual_skip_nothing_left(self):
        comma = FakeComma(segment_files(ROUTE, range(3)))
        with mock.patch.object(fetch, 'Comma', return_value=comma), \
                mock.patch.dict(os.environ, {'COMMA_VIDEOS_DIR': str(self.videos)}):
            self.assertEqual(fetch.fetch(fetch.parse_args([ROUTE])), 1)
        self.assertEqual(list(self.data.iterdir()), [])


class LocalizeArgumentsTest(unittest.TestCase):
    def test_names(self):
        names, status, show_help = localize.parse_args(
            ['0000011c8152ba83cf01', '6616faac453a3064/' + ROUTE + '/1', ROUTE, '2-3', '12', '--status'])
        self.assertEqual(names, ['0000011c8152ba83cf01', '0000011c8152ba83cf01', '0000011c8152ba83cf02',
                                 '0000011c8152ba83cf03', '0000011c8152ba83cf12'])
        self.assertEqual((status, show_help), (True, False))
        # Without a route a number is a folder name.
        self.assertEqual(localize.parse_args(['12'])[0], ['12'])
        self.assertTrue(localize.parse_args(['--help'])[2])

    def test_refused(self):
        for argument in ('../data', 'ABC', '1-2', '--force'):
            with self.assertRaises(TaskError, msg=argument):
                localize.parse_args([argument])


class LocalizeTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.data = root / 'data'
        self.discard = root / 'data_scartati'
        self.data.mkdir()
        self.output = []
        for patcher in (mock.patch.object(localize, 'DATA_DIR', self.data),
                        mock.patch.object(localize, 'DISCARD_DIR', self.discard),
                        mock.patch.object(localize, 'say', side_effect=self.output.append)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def segment(self, name, state='todo', ionex=None):
        folder = self.data / name
        folder.mkdir()
        (folder / 'rlog.zst').write_bytes(b'x')
        if state == 'done':
            for output in localize.OUTPUTS:
                (folder / output).write_bytes(b'x')
        elif state == 'failed':
            (folder / localize.FAILED_FILE).write_text('2026-10-21 10:00 boom\n')
        if ionex:
            (folder / localize.IONEX_FILE).write_text('2026/293 ' + ionex + '\n')
        return folder

    def ionex_answer(self, lines):
        def run(command, **kwargs):
            self.assertEqual(command[1:3], ['-m', 'localizer.ionex'])
            self.asked = sorted(Path(folder).name for folder in command[3:])
            return subprocess.CompletedProcess(command, 0, stdout=lines, stderr='')
        return mock.patch.object(localize.subprocess, 'run', side_effect=run)

    def test_picks_todo_failed_and_refine(self):
        self.segment('a1')
        self.segment('a2', 'failed')
        self.segment('a3', 'done', 'P1D')
        self.segment('a4', 'done', 'FIN')
        self.segment('a5', 'done')
        (self.data / 'a6').mkdir()
        lines = str(self.data / 'a3') + ' P1D RAP\n' + str(self.data / 'a4') + ' FIN -\n'
        with self.ionex_answer(lines):
            self.assertEqual(localize.segments_to_do(), ['a1', 'a2', 'a3'])
        self.assertEqual(self.asked, ['a3', 'a4'])
        self.assertIn('a3: redoing with map RAP (was P1D)', self.output)

    def test_nothing_done_no_ionex_call(self):
        self.segment('a1')
        with mock.patch.object(localize.subprocess, 'run') as run:
            self.assertEqual(localize.segments_to_do(), ['a1'])
        run.assert_not_called()

    def test_status(self):
        self.segment('a1')
        self.segment('a2', 'failed')
        self.segment('a3', 'done', 'P1D')
        self.segment('a4', 'done', 'FIN')
        (self.discard / 'b1').mkdir(parents=True)
        (self.discard / 'b1' / localize.FAILED_FILE).write_text('2026-10-20 09:00 no fix\nmore\n')
        lines = str(self.data / 'a3') + ' P1D RAP\n' + str(self.data / 'a4') + ' FIN -\n'
        with self.ionex_answer(lines):
            localize.print_status()
        self.assertEqual(self.output, ['todo  a1', 'failed  a2  (2026-10-21 10:00 boom)',
                                       'refine  a3  (map P1D, out now RAP)', 'done  a4  (map FIN)',
                                       'discarded  b1  (2026-10-20 09:00 no fix)'])

    def fake_localizer(self, codes):
        def run(name, log_path):
            log_path.write_bytes(b''.join(b'line %d\n' % i for i in range(30)) + b'error of ' + name.encode() + b'\n')
            return codes[name]
        return mock.patch.object(localize, 'run_localizer', side_effect=run)

    def test_outcomes(self):
        self.segment('a1', 'failed')
        self.segment('a2')
        refine = self.segment('a3', 'done', 'P1D')
        self.segment('a4')
        (self.discard / 'a4').mkdir(parents=True)
        self.segment('a5')
        with self.fake_localizer({'a1': 0, 'a2': 3, 'a3': 3, 'a4': 3, 'a5': 1}):
            self.assertEqual(localize.localize(['a1', 'a2', 'a3', 'a4', 'a5', 'a9']), 1)
        # Success: localizer.failed removed.
        self.assertFalse((self.data / 'a1' / localize.FAILED_FILE).exists())
        # Unusable: moved, with its reason.
        self.assertFalse((self.data / 'a2').exists())
        failed = (self.discard / 'a2' / localize.FAILED_FILE).read_text()
        self.assertRegex(failed.splitlines()[0], r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2} error of a2$')
        self.assertEqual(len(failed.splitlines()), 21)
        # Unusable but with old outputs (failed refine): stays.
        self.assertTrue(refine.is_dir())
        self.assertTrue((refine / localize.FAILED_FILE).is_file())
        # Already discarded once: both kept.
        self.assertTrue((self.discard / 'a4').is_dir())
        self.assertEqual(len([path for path in self.discard.iterdir() if path.name.startswith('a4.')]), 1)
        # Other failure: stays in data/.
        self.assertTrue((self.data / 'a5' / localize.FAILED_FILE).is_file())
        self.assertIn('missing ' + str(self.data / 'a9' / 'rlog.zst'), self.output)
        self.assertEqual(self.output[-3:], ['OK: a1', 'Discarded (in data_scartati/): a2 a4', 'Failed: a3 a5 a9'])
        self.assertIn('PROGRESS 6/6', self.output)

    def test_runs_in_parallel(self):
        import threading
        import time as clock
        for name in ('a1', 'a2', 'a3', 'a4'):
            self.segment(name)
        running = []
        peak = []
        lock = threading.Lock()

        def run(name, log_path):
            with lock:
                running.append(name)
                peak.append(len(running))
            clock.sleep(0.2)
            log_path.write_bytes(b'done\n')
            with lock:
                running.remove(name)
            return 0
        with mock.patch.object(localize, 'run_localizer', side_effect=run):
            self.assertEqual(localize.localize(['a1', 'a2', 'a3', 'a4'], jobs=2), 0)
        self.assertEqual(max(peak), 2)
        self.assertEqual(self.output[-1], 'OK: a1 a2 a3 a4')

    def test_segments_at_a_time_change_while_running(self):
        import json
        import threading
        import time as clock
        for index in range(8):
            self.segment('b' + str(index))
        control = self.data.parent / 'job.control.json'
        running = []
        peak_after = []
        lock = threading.Lock()
        started = []

        def run(name, log_path):
            with lock:
                running.append(name)
                started.append(name)
                if len(started) > 2:
                    peak_after.append(len(running))
                # After the first two, the site asks for 4 at a time.
                if len(started) == 2:
                    control.write_text(json.dumps({'jobs': 4}))
            clock.sleep(0.3)
            log_path.write_bytes(b'done\n')
            with lock:
                running.remove(name)
            return 0
        with mock.patch.dict(localize.os.environ, {'DISTILL_JOB_CONTROL': str(control)}), \
                mock.patch.object(localize, 'CONTROL_PERIOD', 0.05), \
                mock.patch.object(localize, 'run_localizer', side_effect=run):
            names = ['b' + str(index) for index in range(8)]
            self.assertEqual(localize.localize(names, jobs=2), 0)
        self.assertEqual(max(peak_after), 4)
        self.assertIn('Segments at a time: 2 -> 4.', self.output)
        state = json.loads((self.data.parent / 'job.state.json').read_text())
        self.assertEqual(state, {'jobs': 4, 'auto': localize.default_jobs(), 'running': 0})
        # The job never writes the site's file.
        self.assertEqual(json.loads(control.read_text()), {'jobs': 4})

    def test_live_value_capped_by_free_memory(self):
        import json
        control = self.data.parent / 'job.control.json'
        control.write_text(json.dumps({'jobs': 12}))
        with mock.patch.dict(localize.os.environ, {'DISTILL_JOB_CONTROL': str(control)}), \
                mock.patch.object(localize, 'memory_jobs', return_value=3):
            live = localize.Control(2, 2)
            self.assertEqual(live.read(), 3)
            self.assertEqual(live.read(), 3)
        self.assertIn('Segments at a time: 12 asked, 3 allowed by the free memory.', self.output)
        self.assertEqual(self.output.count('Segments at a time: 2 -> 3.'), 1)

    def test_jobs_option(self):
        self.assertEqual(localize.parse_args(['--jobs', '3', 'a1'])[0], ['a1'])
        self.assertEqual(localize.jobs_option['value'], 3)
        localize.parse_args(['--jobs=5'])
        self.assertEqual(localize.jobs_option['value'], 5)
        localize.parse_args([])
        self.assertIsNone(localize.jobs_option['value'])
        for bad in (['--jobs'], ['--jobs', '0'], ['--jobs=x'], ['--jobs', '17'], ['--jobs', '99']):
            with self.assertRaises(TaskError):
                localize.parse_args(bad)
        self.assertTrue(1 <= localize.default_jobs() <= localize.MAX_JOBS)

    def test_all_ok(self):
        self.segment('a1')
        with self.fake_localizer({'a1': 0}):
            self.assertEqual(localize.localize(['a1']), 0)
        self.assertEqual(self.output[-1], 'OK: a1')

    def test_tail(self):
        self.assertEqual(localize.tail_lines(b'a\nb\nc\n', 2), b'b\nc\n')
        self.assertEqual(localize.tail_lines(b'a\nb\nc', 1), b'c')
        self.assertEqual(localize.tail_lines(b'', 20), b'')


if __name__ == '__main__':
    unittest.main()
