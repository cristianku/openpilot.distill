"""tasks/: the checks the training does before it starts, without GPU, network or a real training."""
from __future__ import annotations

import io
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tasks.evaluation as evaluation_task
import tasks.train as train_task
from tasks.common import Gpu, TaskError, say, select_gpus
from tasks.evaluation import model_runs, uses_gpu
from tasks.teacher import TEACHERS, teacher_of_file

GPUS = [Gpu(0, 'Tesla V100', 32768.0), Gpu(1, 'Tesla V100', 32768.0)]


class GpuChoiceTest(unittest.TestCase):
    def test_all_and_single(self):
        self.assertEqual(select_gpus('all', GPUS), [0, 1])
        self.assertEqual(select_gpus('1', GPUS), [1])
        self.assertEqual(select_gpus('1,0,1', GPUS), [1, 0])

    def test_refused(self):
        with self.assertRaises(TaskError):
            select_gpus('2', GPUS)
        with self.assertRaises(TaskError):
            select_gpus('all', [])


class ArgumentsTest(unittest.TestCase):
    def test_defaults(self):
        args = train_task.parse_args([])
        self.assertEqual((args.mode, args.steps, args.batch_size, args.workers, args.gpus),
                         ('fresh', None, 8, 6, 'all'))

    def test_test_run(self):
        self.assertEqual(train_task.parse_args(['--test']).steps, train_task.TEST_STEPS)

    def test_refused(self):
        for arguments in (['--mode', 'resume'], ['--mode', 'resume', '--run', 'x', '--teacher', 'md-v3'],
                          ['--run', 'x'], ['--gpus', '0;1'], ['--steps', '0'], ['--batch-size', '-1']):
            with self.assertRaises(SystemExit, msg=arguments), mock.patch('sys.stderr'):
                train_task.parse_args(arguments)


class TeacherTest(unittest.TestCase):
    def test_teacher_of_file(self):
        self.assertEqual(teacher_of_file('/x/models/big_driving_supercombo_1365126a_12864.onnx').name, 'md-v3')
        self.assertEqual(teacher_of_file('models/big_driving_supercombo.onnx').name, '084747c7')
        self.assertIsNone(teacher_of_file('models/other.onnx'))

    def test_pinned(self):
        for teacher in TEACHERS.values():
            self.assertRegex(teacher.sha256, r'^[0-9a-f]{64}$')
            self.assertTrue(teacher.url.startswith('https://'))


class SegmentCheckTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.data = Path(self.directory.name) / 'data'
        self.data.mkdir()
        patcher = mock.patch.object(train_task, 'DATA_DIR', self.data)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.directory.cleanup()

    def segment(self, name, localized=True):
        folder = self.data / name
        folder.mkdir()
        (folder / 'fcamera.hevc').write_bytes(b'x')
        if localized:
            (folder / 'localizer.safetensors').write_bytes(b'x')
            (folder / 'frame_info.safetensors').write_bytes(b'x')
        return folder

    def test_counts_localized(self):
        self.segment('a')
        self.segment('b')
        (self.data / 'not-a-segment').mkdir()
        self.assertEqual(train_task.check_segments(), 2)

    def test_empty(self):
        with self.assertRaises(TaskError):
            train_task.check_segments()

    def test_not_localized(self):
        self.segment('a')
        self.segment('b', localized=False)
        with self.assertRaisesRegex(TaskError, 'would stop the training: b'):
            train_task.check_segments()

    def test_broken_link(self):
        folder = self.segment('a')
        os.symlink(self.data / 'missing.hevc', folder / 'ecamera.hevc')
        with self.assertRaisesRegex(TaskError, 'videos missing'):
            train_task.check_segments()


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.runs = Path(self.directory.name)
        patcher = mock.patch.object(train_task, 'RUNS_DIR', self.runs)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.directory.cleanup()

    def test_refused(self):
        with self.assertRaisesRegex(TaskError, 'does not exist'):
            train_task.resume_info('2026-10-22_101530')
        (self.runs / '2026-10-22_101530').mkdir()
        with self.assertRaisesRegex(TaskError, 'checkpoint'):
            train_task.resume_info('2026-10-22_101530')
        (self.runs / '2026-10-22_101530' / 'supercombo.training.pt').write_bytes(b'x')
        (self.runs / '2026-10-22_101530' / 'supercombo.pt').write_bytes(b'x')
        with self.assertRaisesRegex(TaskError, 'already finished'):
            train_task.resume_info('2026-10-22_101530')
        with self.assertRaisesRegex(TaskError, 'does not exist'):
            train_task.resume_info('../2026-10-22_101530')


class WandbTest(unittest.TestCase):
    def test_modes(self):
        with tempfile.TemporaryDirectory() as home, mock.patch.dict(os.environ, {'HOME': home}, clear=False):
            os.environ.pop('WANDB_API_KEY', None)
            os.environ.pop('WANDB_MODE', None)
            self.assertEqual(train_task.wandb_mode(None), 'offline')
            self.assertEqual(train_task.wandb_mode('disabled'), 'disabled')
            with self.assertRaises(TaskError):
                train_task.wandb_mode('online')
            os.environ['WANDB_API_KEY'] = 'key'
            self.assertEqual(train_task.wandb_mode(None), 'online')


class EvaluationTest(unittest.TestCase):
    def test_uses_gpu(self):
        self.assertTrue(uses_gpu(['--model=stock']))
        self.assertFalse(uses_gpu(['--model=stock', '--device=cpu']))
        self.assertFalse(uses_gpu(['--device', 'cpu']))

    def test_one_run_per_model(self):
        self.assertEqual(model_runs(['--model=stock', '--rate', '5']), [['--model=stock', '--rate', '5']])
        self.assertEqual(model_runs(['--model=run:a', '--model=stock', '--device=cpu']),
                         [['--model=run:a', '--device=cpu'], ['--model=stock', '--device=cpu']])

    def run_job(self, arguments, kept=(), on_wait=None):
        """tasks.evaluation main() with a stand-in for each process: (exit code, arguments of each model run,
        arguments of each --check-cache probe). Models in `kept` have their result kept from before."""
        runs = []
        probes = []

        class Child:
            def __init__(self, argv, **options):
                self.arguments = argv[3:]
                self.probe = '--check-cache' in self.arguments
                (probes if self.probe else runs).append(self.arguments)
                self.returncode = (0 if self.arguments[0] in kept else 10) if self.probe else 0

            def terminate(self):
                self.returncode = -15

            def wait(self):
                if on_wait and not self.probe:
                    on_wait(len(runs))
                return self.returncode

        previous = signal.getsignal(signal.SIGTERM)
        self.addCleanup(signal.signal, signal.SIGTERM, previous)
        with mock.patch.object(sys, 'argv', ['tasks.evaluation', 'holdout', *arguments]), \
                mock.patch.object(evaluation_task, 'load_env'), \
                mock.patch.object(evaluation_task.subprocess, 'Popen', Child):
            code = evaluation_task.main()
        return code, runs, probes

    def test_two_models_share_drives_and_progress(self):
        code, runs, probes = self.run_job(['--model=run:a', '--model=stock', '--device=cpu'])
        self.assertEqual(code, 0)
        self.assertEqual([run[0] for run in runs], ['--model=run:a', '--model=stock'])
        pins = {argument for run in runs + probes for argument in run if argument.startswith('--pin=')}
        self.assertEqual(len(pins), 1)
        self.assertEqual(len(probes), 2)
        self.assertEqual([run[-1] for run in runs], ['--progress-part=0/2', '--progress-part=1/2'])
        # One model: the arguments go through as they are, no probe.
        self.assertEqual(self.run_job(['--model=stock', '--device=cpu'])[1:], ([['--model=stock', '--device=cpu']], []))

    def test_kept_result_leaves_the_progress_bar_to_the_other(self):
        code, runs, probes = self.run_job(['--model=run:a', '--model=stock', '--device=cpu'], kept=('--model=stock',))
        self.assertEqual(code, 0)
        self.assertEqual(len(runs), 2)
        self.assertFalse([argument for run in runs for argument in run if argument.startswith('--progress-part')])

    def test_stop_between_two_models_starts_no_other(self):
        # The site's stop lands just as the first model ends with 0.
        def stop(count):
            if count == 1:
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

        code, runs, probes = self.run_job(['--model=run:a', '--model=stock', '--device=cpu'], on_wait=stop)
        self.assertEqual(code, 143)
        self.assertEqual(len(runs), 1)


class RunLogTest(unittest.TestCase):
    def test_say_through_the_tee_reaches_stdout_and_the_log(self):
        # The checks before a training print with sys.stdout swapped for a TeeText over the run log.
        with tempfile.TemporaryDirectory() as folder:
            stdout = io.TextIOWrapper(io.BytesIO(), encoding='utf-8')
            with mock.patch.object(sys, 'stdout', stdout):
                log = train_task.RunLog(Path(folder) / 'train.log')
                sys.stdout = train_task.TeeText(log)
                say('Teacher md-v3: ok (1 bytes).')
                sys.stdout = stdout
                log.say('Training started')
                log.file.close()
            self.assertEqual(stdout.buffer.getvalue(), b'Teacher md-v3: ok (1 bytes).\nTraining started\n')
            self.assertEqual((Path(folder) / 'train.log').read_bytes(), stdout.buffer.getvalue())


if __name__ == '__main__':
    unittest.main()
