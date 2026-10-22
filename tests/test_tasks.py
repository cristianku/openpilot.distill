"""tasks/: the checks the training does before it starts, without GPU, network or a real training."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tasks.train as train_task
from tasks.common import Gpu, TaskError, select_gpus
from tasks.evaluation import uses_gpu
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


if __name__ == '__main__':
    unittest.main()
