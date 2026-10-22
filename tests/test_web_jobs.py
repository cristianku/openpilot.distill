"""distill-web: job lifecycle with fake jobs (sh, sleep, false), stop, groups and recovery after a
restart of the site. Standard library only, runs on the Mac and on Linux."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from web.actions import JobSpec
from web.common import WebError
from web.jobs import JobManager, parse_progress, parse_run_dir, parse_video


def spec(argv, group='data', action='fetch', needs_gpu=False, params=None):
    return JobSpec(action=action, label='test ' + action, group=group, argv=argv, needs_gpu=needs_gpu,
                   params=params or {}, summary='test')


class JobTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.repo = Path(self.directory.name).resolve() / 'repo'
        (self.repo / 'outputs' / 'supervised').mkdir(parents=True)
        self.managers = []

    def tearDown(self):
        # No job may outlive the test.
        for manager in self.managers:
            for job in manager.list_jobs():
                if job['state'] == 'running':
                    record = json.loads(manager.job_file(job['id']).read_text())
                    try:
                        os.killpg(record['pgid'], 9)
                    except (ProcessLookupError, TypeError):
                        pass
        self.directory.cleanup()

    def manager(self, **options):
        made = JobManager(self.repo, **options)
        self.managers.append(made)
        return made

    def wait_state(self, manager, job_id, states, timeout=15.0):
        deadline = time.monotonic() + timeout
        view = manager.get(job_id)
        while time.monotonic() < deadline:
            view = manager.get(job_id)
            if view['state'] in states:
                return view
            time.sleep(0.1)
        self.fail('job ' + job_id + ' still ' + view['state'] + ', expected ' + str(states))
        return view


class ProgressTest(unittest.TestCase):
    def test_rsync_speed(self):
        from web.jobs import parse_progress
        text = ('PROGRESS 3/10\n  1,234,567  45%   12.34MB/s    0:00:12\r  2,000,000 80%  0.00kB/s 0:00:00\n'
                'PROGRESS 4/10\n')
        self.assertEqual(parse_progress(text), {'done': 4, 'total': 10, 'percent': 40.0, 'rate_mb_s': 12.3})
        self.assertEqual(parse_progress('  1,000 100%  512.00kB/s  0:00:00\nPROGRESS 1/2\n')['rate_mb_s'], 0.5)


class LifecycleTest(JobTestCase):
    def test_job_runs_and_records_its_exit_code(self):
        manager = self.manager()
        job = manager.start(spec(['sh', '-c', 'echo hello; echo PROGRESS 3/10; sleep 1']))
        self.assertEqual(job['state'], 'running')
        self.assertTrue(job['stoppable'])
        self.assertRegex(job['id'], r'^[0-9]{8}-[0-9]{6}-fetch$')
        time.sleep(0.5)
        running = manager.get(job['id'])
        self.assertEqual(running['progress'], {'done': 3, 'total': 10, 'percent': 30.0, 'rate_mb_s': None})
        done = self.wait_state(manager, job['id'], ('ok', 'failed'))
        self.assertEqual(done['state'], 'ok')
        self.assertEqual(done['exit_code'], 0)
        self.assertIsNotNone(done['finished'])
        text = manager.log(job['id'], 0)['text']
        # The header says what the job does, never its command line.
        self.assertIn('== test\n', text)
        self.assertNotIn('sh -c', text)
        self.assertIn('hello', text)
        self.assertIn('== finished with code 0', text)
        record = json.loads(manager.job_file(job['id']).read_text())
        for key in ('id', 'action', 'argv', 'pid', 'pgid', 'started', 'finished', 'exit_code', 'log'):
            self.assertIn(key, record)
        self.assertEqual(record['pid'], record['pgid'])
        self.assertEqual(record['log'], 'outputs/web/jobs/' + job['id'] + '.log')

    def test_failing_job(self):
        manager = self.manager()
        job = manager.start(spec(['false']))
        done = self.wait_state(manager, job['id'], ('ok', 'failed'))
        self.assertEqual(done['state'], 'failed')
        self.assertEqual(done['exit_code'], 1)
        self.assertEqual(done['state_label'], 'failed')

    def test_program_that_does_not_exist(self):
        manager = self.manager()
        job = manager.start(spec([str(self.repo / 'helpers' / 'missing.sh')]))
        done = self.wait_state(manager, job['id'], ('ok', 'failed'))
        self.assertEqual(done['exit_code'], 127)
        self.assertIn('cannot start', manager.log(job['id'], 0)['text'])

    def test_job_runs_in_the_repo_without_the_site_token(self):
        os.environ['DISTILL_WEB_TOKEN'] = 'secret-token-1234567890'
        try:
            manager = self.manager()
            job = manager.start(spec(['sh', '-c', 'pwd; echo "token=${DISTILL_WEB_TOKEN:-none}"; '
                                                  'echo "unbuffered=$PYTHONUNBUFFERED"']))
            self.wait_state(manager, job['id'], ('ok', 'failed'))
        finally:
            del os.environ['DISTILL_WEB_TOKEN']
        text = manager.log(job['id'], 0)['text']
        self.assertIn('token=none', text)
        self.assertIn('unbuffered=1', text)
        self.assertIn(str(self.repo), text)
        self.assertNotIn('secret-token', text)


class GroupTest(JobTestCase):
    def test_one_job_per_group(self):
        manager = self.manager()
        first = manager.start(spec(['sleep', '2']))
        with self.assertRaises(WebError) as caught:
            manager.start(spec(['true'], action='localize'))
        self.assertEqual(caught.exception.status, 409)
        self.assertIn('already', caught.exception.message)
        other = manager.start(spec(['true'], group='cpu', action='export'))
        self.wait_state(manager, other['id'], ('ok',))
        self.assertEqual(manager.running_groups(), {'data'})
        self.wait_state(manager, first['id'], ('ok',))
        self.assertEqual(manager.running_groups(), set())

    def test_update_runs_alone(self):
        manager = self.manager()
        data_job = manager.start(spec(['sleep', '2']))
        with self.assertRaises(WebError) as caught:
            manager.start(spec(['true'], group='system', action='update'))
        self.assertEqual(caught.exception.status, 409)
        self.wait_state(manager, data_job['id'], ('ok',))
        update = manager.start(spec(['sleep', '2'], group='system', action='update'))
        with self.assertRaises(WebError):
            manager.start(spec(['true'], group='cpu', action='export'))
        self.wait_state(manager, update['id'], ('ok',))

    def test_gpu_check_refuses(self):
        manager = self.manager(gpu_check=lambda: 'The GPU is busy with another job (lock taken).')
        with self.assertRaises(WebError) as caught:
            manager.start(spec(['true'], group='gpu', action='train_test', needs_gpu=True))
        self.assertEqual(caught.exception.status, 409)
        self.assertEqual(caught.exception.reason, 'gpu_busy')
        self.assertIn('GPU', caught.exception.message)
        job = manager.start(spec(['true'], group='cpu', action='export'))
        self.wait_state(manager, job['id'], ('ok',))

    def test_gpu_check_runs_only_for_gpu_jobs_and_passes_when_free(self):
        calls = []

        def check():
            calls.append(1)
            return ''

        manager = self.manager(gpu_check=check)
        job = manager.start(spec(['true'], group='gpu', action='train_test', needs_gpu=True))
        self.wait_state(manager, job['id'], ('ok',))
        manager.start(spec(['true'], group='cpu', action='export'))
        self.assertEqual(len(calls), 1)

    def test_external_groups(self):
        manager = self.manager(external_groups=lambda: {'gpu', 'other'})
        with self.assertRaises(WebError):
            manager.start(spec(['true'], group='gpu', action='eval'))
        self.assertEqual(manager.running_groups(), {'gpu'})


class StopTest(JobTestCase):
    def test_stop_with_sigterm(self):
        manager = self.manager()
        job = manager.start(spec(['sleep', '30']))
        stopped = manager.stop(job['id'])
        self.assertIn(stopped['state'], ('running', 'stopped'))
        done = self.wait_state(manager, job['id'], ('stopped', 'failed', 'ok'), timeout=10)
        self.assertEqual(done['state'], 'stopped')
        self.assertEqual(done['exit_code'], -15)
        text = manager.log(job['id'], 0)['text']
        self.assertIn('stopped from the site', text)
        self.assertIn('== finished with code -15', text)
        with self.assertRaises(WebError):
            manager.stop(job['id'])

    def test_stop_escalates_to_sigkill(self):
        manager = self.manager(stop_grace=1.0)
        # sh and its sleep ignore SIGTERM: only SIGKILL ends them.
        job = manager.start(spec(['sh', '-c', 'trap "" TERM; sleep 30; echo never']))
        time.sleep(0.5)
        manager.stop(job['id'])
        done = self.wait_state(manager, job['id'], ('stopped', 'failed', 'ok', 'unknown'), timeout=10)
        done = self.wait_state(manager, job['id'], ('stopped',), timeout=5)
        self.assertEqual(done['exit_code'], -9)
        record = json.loads(manager.job_file(job['id']).read_text())
        self.assertTrue(record['killed'])
        self.assertIn('SIGKILL', manager.log(job['id'], 0)['text'])

    def test_stop_unknown_job(self):
        manager = self.manager()
        with self.assertRaises(WebError) as caught:
            manager.stop('20260101-000000-fetch')
        self.assertEqual(caught.exception.status, 404)
        with self.assertRaises(WebError):
            manager.stop('../../etc/passwd')


class RestartTest(JobTestCase):
    def test_a_new_site_process_finds_the_running_job_and_its_result(self):
        first = self.manager()
        job = first.start(spec(['sh', '-c', 'sleep 1.5; exit 3'], group='gpu', action='train'))
        # A restarted site: a new manager without the Popen objects of the old one.
        second = self.manager()
        view = second.get(job['id'])
        self.assertEqual(view['state'], 'running')
        self.assertEqual(second.running_groups(), {'gpu'})
        done = self.wait_state(second, job['id'], ('ok', 'failed'))
        self.assertEqual(done['state'], 'failed')
        self.assertEqual(done['exit_code'], 3)

    def test_a_new_site_process_can_stop_an_old_job(self):
        first = self.manager()
        job = first.start(spec(['sleep', '30']))
        second = self.manager()
        second.stop(job['id'])
        done = self.wait_state(second, job['id'], ('stopped',), timeout=10)
        self.assertEqual(done['exit_code'], -15)

    def test_job_without_exit_code_and_without_process(self):
        manager = self.manager()
        manager.jobs_dir.mkdir(parents=True)
        # A pid that is surely gone: a process that already ended and was reaped.
        ended = subprocess.Popen(['true'])
        ended.wait()
        job_id = '20260101-000000-localize'
        record = {'id': job_id, 'action': 'localize', 'label': 'Localize', 'group': 'data', 'argv': ['x'],
                  'cwd': str(self.repo), 'started': 1767225600.0, 'pid': ended.pid, 'pgid': ended.pid,
                  'finished': None, 'exit_code': None, 'log': 'outputs/web/jobs/' + job_id + '.log'}
        manager.job_file(job_id).write_text(json.dumps(record))
        manager.log_file(job_id).write_text('== x\n')
        view = manager.get(job_id)
        self.assertEqual(view['state'], 'unknown')
        self.assertEqual(view['state_label'], 'ended (unknown result)')
        self.assertFalse(view['stoppable'])
        self.assertEqual(manager.running_groups(), set())
        # The group is free again.
        job = manager.start(spec(['true']))
        self.wait_state(manager, job['id'], ('ok',))

    def test_pid_reused_by_another_process_is_not_the_job(self):
        manager = self.manager()
        manager.jobs_dir.mkdir(parents=True)
        other = subprocess.Popen(['sleep', '5'])
        try:
            job_id = '20260101-000001-fetch'
            record = {'id': job_id, 'action': 'fetch', 'label': 'x', 'group': 'data', 'argv': ['x'],
                      'cwd': str(self.repo), 'started': 1.0, 'pid': other.pid, 'pgid': other.pid,
                      'finished': None, 'exit_code': None}
            manager.job_file(job_id).write_text(json.dumps(record))
            self.assertEqual(manager.get(job_id)['state'], 'unknown')
            with self.assertRaises(WebError):
                manager.stop(job_id)
            self.assertIsNone(other.poll())
        finally:
            other.kill()
            other.wait()


class ListAndLogTest(JobTestCase):
    def test_list_and_log_offsets(self):
        manager = self.manager()
        job = manager.start(spec(['sh', '-c', 'echo uno; echo due']))
        self.wait_state(manager, job['id'], ('ok',))
        listed = manager.list_jobs()
        self.assertEqual(listed[0]['id'], job['id'])
        chunk = manager.log(job['id'], 0)
        self.assertEqual(chunk['offset'], chunk['size'])
        self.assertEqual(manager.log(job['id'], chunk['offset'])['text'], '')
        self.assertIn('uno', manager.log(job['id'], 10 ** 9)['text'])
        for bad in ('x', '../jobs', '20260101-000000-FETCH'):
            with self.assertRaises(WebError):
                manager.log(bad, 0)

    def test_last_successful_publish_dry(self):
        manager = self.manager()
        params = {'run': '2026-10-21_1544', 'name': 'prova', 'stable': False}
        job = manager.start(spec(['true'], group='cpu', action='publish_dry', params=params))
        self.wait_state(manager, job['id'], ('ok',))
        self.assertIsNotNone(manager.last_successful('publish_dry', params))
        other = dict(params)
        other['stable'] = True
        self.assertIsNone(manager.last_successful('publish_dry', other))
        # Job ids have a resolution of one second.
        time.sleep(1.1)
        failed = manager.start(spec(['false'], group='cpu', action='publish_dry', params=other))
        self.wait_state(manager, failed['id'], ('failed',))
        self.assertIsNone(manager.last_successful('publish_dry', other))

    def test_run_dir_of_a_training_job(self):
        manager = self.manager()
        run = self.repo / 'outputs' / 'supervised' / '2026-10-22_101530'
        run.mkdir()
        job = manager.start(spec(['sh', '-c', 'echo Teacher ok; echo RUN_DIR=' + str(run) + '; echo step 3/10: x'],
                                 group='gpu', action='train'))
        done = self.wait_state(manager, job['id'], ('ok',))
        self.assertEqual(done['run'], '2026-10-22_101530')


class ParseTest(unittest.TestCase):
    def test_progress(self):
        self.assertIsNone(parse_progress('nothing here\n'))
        self.assertEqual(parse_progress('PROGRESS 1/4\nPROGRESS 2/4\n'),
                         {'done': 2, 'total': 4, 'percent': 50.0, 'rate_mb_s': None})
        self.assertEqual(parse_progress('a\rPROGRESS 5/5\r')['percent'], 100.0)
        text = 'step 10/5000: loss=1 gt=1 distill=1 time=0.70s\nstep 11/5000: loss=1 gt=1 distill=1 time=0.70s\n'
        self.assertEqual(parse_progress(text)['done'], 11)
        self.assertEqual(parse_progress('PROGRESS 3/0\n')['percent'], None)

    def test_run_dir(self):
        repo = Path('/srv/repo')
        self.assertIsNone(parse_run_dir('RUN_DIR=/etc\n', repo))
        self.assertIsNone(parse_run_dir('no line\n', repo))

    def test_video(self):
        text = ('PROGRESS 10/10\navg miss: ...\n'
                'OUTPUT_MP4=/root/x/outputs/compare/20261022-101500_seg_stock_vs_run_1.mp4\n'
                'OUTPUT_JSON=/root/x/outputs/compare/20261022-101500_seg_stock_vs_run_1.json\n')
        self.assertEqual(parse_video(text), '20261022-101500_seg_stock_vs_run_1.mp4')
        self.assertIsNone(parse_video('scritto outputs/compare/abc_1.mp4\n'))
        self.assertIsNone(parse_video('OUTPUT_MP4=/tmp/elsewhere/x.mp4\n'))
        self.assertIsNone(parse_video('OUTPUT_MP4=/r/outputs/compare/../x.mp4\n'))
        self.assertIsNone(parse_video('OUTPUT_MP4=/r/outputs/compare/x.partial.mp4\n'))


if __name__ == '__main__':
    unittest.main()
