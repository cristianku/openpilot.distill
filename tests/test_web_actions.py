"""distill-web: the fixed action table and the strict checks of every parameter (standard library only)."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from web.actions import ACTIONS, build_job, script_missing, slug_of
from web.common import WebError

PY = sys.executable
RUN = '2026-10-21_1544'
RUN_SECONDS = '2026-10-22_101530'
UNFINISHED = '2026-10-21_0834'
SEGMENT = '0000011e0c8825628a08'
NOT_LOCALIZED = '0000011e0c8825628a09'


class ActionTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.repo = Path(self.directory.name) / 'repo'
        runs = self.repo / 'outputs' / 'supervised'
        for name in (RUN, RUN_SECONDS):
            (runs / name).mkdir(parents=True)
            (runs / name / 'supercombo.pt').write_bytes(b'weights')
        (runs / UNFINISHED).mkdir(parents=True)
        (runs / UNFINISHED / 'supercombo.training.pt').write_bytes(b'checkpoint')
        segment = self.repo / 'data' / SEGMENT
        segment.mkdir(parents=True)
        (segment / 'localizer.safetensors').write_bytes(b'x')
        (segment / 'frame_info.safetensors').write_bytes(b'x')
        (self.repo / 'data' / NOT_LOCALIZED).mkdir()
        # Something outside the repo that a link could point to.
        self.outside = Path(self.directory.name) / 'outside'
        (self.outside / 'run').mkdir(parents=True)
        (self.outside / 'run' / 'supercombo.pt').write_bytes(b'not ours')

    def tearDown(self):
        self.directory.cleanup()

    def build(self, action, params):
        return build_job(self.repo, action, params)

    def refused(self, action, params, status=None):
        with self.assertRaises(WebError) as caught:
            self.build(action, params)
        if status is not None:
            self.assertEqual(caught.exception.status, status)
        return caught.exception


class ArgvTest(ActionTestCase):
    def test_fetch(self):
        self.assertEqual(self.build('fetch', {}).argv, [PY, '-m', 'tasks.fetch'])
        self.assertEqual(self.build('fetch', {'since': ''}).argv, [PY, '-m', 'tasks.fetch'])
        self.assertEqual(self.build('fetch', {'since': '2026-10-20'}).argv,
                         [PY, '-m', 'tasks.fetch', '--since=2026-10-20'])

    def test_localize(self):
        self.assertEqual(self.build('localize', {}).argv, [PY, '-m', 'tasks.localize'])
        self.assertEqual(self.build('localize', {'jobs': 'auto'}).argv, [PY, '-m', 'tasks.localize'])
        spec = self.build('localize', {'jobs': 4})
        self.assertEqual(spec.argv, [PY, '-m', 'tasks.localize', '--jobs', '4'])
        self.assertEqual(spec.params, {'jobs': 4})
        for bad in (0, 17, 'many', '4;rm', True):
            self.refused('localize', {'jobs': bad}, 400)

    def test_train_fresh(self):
        spec = self.build('train', {'steps': 5000, 'batch_size': 8, 'workers': 6, 'mode': 'fresh', 'wandb': ''})
        self.assertEqual(spec.argv, [PY, '-m', 'tasks.train', '--mode', 'fresh', '--steps', '5000', '--teacher',
                                     'md-v3', '--batch-size', '8', '--workers', '6', '--gpus', 'all'])
        self.assertEqual(spec.group, 'gpu')
        self.assertTrue(spec.needs_gpu)

    def test_train_gpus_and_teacher(self):
        spec = self.build('train', {'steps': 100, 'mode': 'fresh', 'gpus': '0,1', 'teacher': '084747c7'})
        self.assertEqual(spec.argv[spec.argv.index('--gpus') + 1], '0,1')
        self.assertEqual(spec.argv[spec.argv.index('--teacher') + 1], '084747c7')
        self.assertEqual(spec.params['gpus'], '0,1')
        for bad_gpus in ('tutte', '0;rm', '-1', 'all,0', 1):
            self.refused('train', {'steps': 100, 'gpus': bad_gpus}, 400)
        self.refused('train', {'steps': 100, 'teacher': 'nuovo'}, 400)
        self.refused('train', {'mode': 'resume', 'run': UNFINISHED, 'teacher': 'md-v3'}, 400)

    def test_train_continue_with_wandb(self):
        spec = self.build('train', {'steps': '200', 'batch_size': 4, 'workers': 0, 'mode': 'continue',
                                    'wandb': 'offline'})
        self.assertEqual(spec.argv, [PY, '-m', 'tasks.train', '--mode', 'continue', '--steps', '200', '--teacher',
                                     'md-v3', '--batch-size', '4', '--workers', '0', '--gpus', 'all',
                                     '--wandb', 'offline'])

    def test_train_resume(self):
        spec = self.build('train', {'mode': 'resume', 'run': UNFINISHED, 'batch_size': 8, 'workers': 6})
        self.assertEqual(spec.argv, [PY, '-m', 'tasks.train', '--mode', 'resume', '--run', UNFINISHED,
                                     '--batch-size', '8', '--workers', '6', '--gpus', 'all'])

    def test_train_test(self):
        self.assertEqual(self.build('train_test', {}).argv, [PY, '-m', 'tasks.train', '--test', '--gpus', 'all'])
        self.assertEqual(self.build('train_test', {'gpus': '0'}).argv[-1], '0')

    def test_eval(self):
        for spec in ('stock', 'teacher', 'logged', 'run:' + RUN, 'run:' + UNFINISHED, 'hf:2026_10_21_1544',
                     'hf:cd210_native'):
            self.assertEqual(self.build('eval', {'model': spec}).argv,
                             [PY, '-m', 'tasks.evaluation', 'holdout', '--model=' + spec])

    def test_compare(self):
        spec = self.build('compare', {'segment': SEGMENT, 'reference': 'stock', 'student': 'run:' + RUN,
                                      'start': 10, 'duration': 20.5, 'speed': 0.5, 'inset': True, 'show_gt': False})
        self.assertEqual(spec.argv, [PY, '-m', 'tasks.evaluation', 'compare', '--segment=' + SEGMENT,
                                     '--reference=stock',
                                     '--student=run:' + RUN, '--start=10', '--duration=20.5', '--speed=0.5',
                                     '--inset'])
        spec = self.build('compare', {'segment': SEGMENT, 'reference': 'teacher', 'student': 'hf:cd210_native',
                                      'start': '0', 'duration': '60', 'speed': 1, 'inset': False, 'show_gt': True})
        self.assertEqual(spec.argv, [PY, '-m', 'tasks.evaluation', 'compare', '--segment=' + SEGMENT,
                                     '--reference=teacher',
                                     '--student=hf:cd210_native', '--start=0', '--duration=60', '--speed=1',
                                     '--no-inset', '--show-gt'])
        spec = self.build('compare', {'segment': SEGMENT, 'reference': 'logged', 'student': 'run:' + RUN,
                                      'speed': 0.25})
        self.assertIn('--speed=0.25', spec.argv)
        # compare.py draws the inset by default: an unchecked box must say --no-inset.
        self.assertIn('--no-inset', spec.argv)

    def test_export_publish_install_restore_update(self):
        self.assertEqual(self.build('export', {'run': RUN}).argv, [PY, '-m', 'tasks.export', '--run', RUN])
        self.assertEqual(self.build('publish_dry', {'run': RUN, 'name': 'prova.1', 'stable': False}).argv,
                         [PY, '-m', 'tasks.publish', '--name=prova.1', '--run', RUN])
        self.assertEqual(self.build('publish', {'run': RUN, 'name': 'prova', 'stable': True, 'confirm': 'prova'}).argv,
                         [PY, '-m', 'tasks.publish', '--upload', '--name=prova', '--stable', '--run', RUN])
        self.assertEqual(self.build('install', {'run': RUN, 'confirm': 'INSTALL'}).argv,
                         [PY, '-m', 'tasks.install', '--run', RUN])
        self.assertEqual(self.build('restore', {'confirm': 'RESTORE'}).argv,
                         [PY, '-m', 'tasks.install', '--restore'])
        self.assertEqual(self.build('update', {}).argv, [PY, '-m', 'tasks.update'])
        self.assertEqual(self.build('export', {'run': RUN_SECONDS}).argv[-1], RUN_SECONDS)

    def test_publish_params_are_kept_for_the_dry_run_match(self):
        dry = self.build('publish_dry', {'run': RUN, 'name': 'x', 'stable': True})
        real = self.build('publish', {'run': RUN, 'name': 'x', 'stable': True, 'confirm': 'x'})
        self.assertEqual(dry.params, real.params)
        self.assertNotIn('confirm', real.params)

    def test_every_action_has_a_known_group(self):
        for action in ACTIONS.values():
            self.assertIn(action.group, ('data', 'gpu', 'cpu', 'comma', 'system'))

    def test_argv_is_a_list_of_strings(self):
        spec = self.build('train', {'steps': 1, 'mode': 'fresh'})
        for item in spec.argv:
            self.assertIsInstance(item, str)


class ValidationTest(ActionTestCase):
    def test_unknown_action_and_parameter(self):
        self.refused('rm', {}, 404)
        self.refused('selftest', {}, 404)
        self.refused('localize', {'extra': 1}, 400)
        self.refused('fetch', {'since': '2026-10-20', 'host': 'evil'}, 400)
        self.refused('fetch', ['since'], 400)

    def test_steps_batch_workers_ranges(self):
        good = {'mode': 'fresh', 'batch_size': 8, 'workers': 6}
        for steps in (0, -1, 200001, True, 1.5, '1e3', '5 ', '', None, '12;rm', [5]):
            params = dict(good)
            params['steps'] = steps
            self.refused('train', params, 400)
        for steps in (1, 200000, '42'):
            params = dict(good)
            params['steps'] = steps
            self.build('train', params)
        for batch in (0, 65, False, '8.0'):
            self.refused('train', {'mode': 'fresh', 'steps': 10, 'batch_size': batch}, 400)
        for workers in (-1, 17, 'six'):
            self.refused('train', {'mode': 'fresh', 'steps': 10, 'workers': workers}, 400)
        self.build('train', {'mode': 'fresh', 'steps': 10, 'batch_size': 64, 'workers': 0})

    def test_mode_and_wandb(self):
        self.refused('train', {'steps': 10, 'mode': 'scratch'}, 400)
        self.refused('train', {'steps': 10, 'mode': 'fresh', 'wandb': 'yes'}, 400)
        self.refused('train', {'steps': 10, 'mode': 'fresh', 'run': RUN}, 400)
        self.refused('train', {'mode': 'resume', 'run': UNFINISHED, 'steps': 10}, 400)

    def test_resume_needs_an_unfinished_run_with_checkpoint(self):
        self.refused('train', {'mode': 'resume', 'run': RUN}, 409)
        self.refused('train', {'mode': 'resume', 'run': '2026-01-01_0000'}, 404)
        self.refused('train', {'mode': 'resume'}, 400)

    def test_since_dates(self):
        for since in ('2026-13-01', '2026-02-30', '2026-1-1', '20261020', '2026-10-20; rm -rf /', '2026-10-20\n',
                      '--since=x', 20261020):
            self.refused('fetch', {'since': since}, 400)

    def test_run_names_and_path_traversal(self):
        for run in ('../../etc', RUN + '/..', '2026-10-21_1544/../../x', '/etc/passwd', '2026-10-21', '1544',
                    '2026-10-21_15440000', '', None, 7, '2026-10-21_1544 ', '.'):
            self.refused('export', {'run': run})
        self.refused('export', {'run': UNFINISHED}, 409)
        self.refused('export', {'run': '2026-01-01_0000'}, 404)

    def test_run_folder_that_is_a_link_outside_the_repo(self):
        link = self.repo / 'outputs' / 'supervised' / '2026-10-23_0000'
        os.symlink(self.outside / 'run', link)
        self.refused('export', {'run': '2026-10-23_0000'}, 404)
        self.refused('eval', {'model': 'run:2026-10-23_0000'}, 404)

    def test_final_model_that_is_a_link_outside_the_run(self):
        folder = self.repo / 'outputs' / 'supervised' / '2026-10-24_0000'
        folder.mkdir()
        os.symlink(self.outside / 'run' / 'supercombo.pt', folder / 'supercombo.pt')
        self.refused('export', {'run': '2026-10-24_0000'}, 409)

    def test_segments(self):
        base = {'reference': 'stock', 'student': 'run:' + RUN}
        for segment in ('../data', '0000011e0c8825628a0', '0000011E0C8825628A08', SEGMENT + '/', '', None,
                        '0000011e0c8825628a08 ', 'g000011e0c8825628a08'):
            params = dict(base)
            params['segment'] = segment
            self.refused('compare', params, 400)
        params = dict(base)
        params['segment'] = 'ffffffffffffffffffff'
        self.refused('compare', params, 404)
        params['segment'] = NOT_LOCALIZED
        self.refused('compare', params, 409)

    def test_segment_folder_that_is_a_link_outside(self):
        os.symlink(self.outside, self.repo / 'data' / 'aaaaaaaaaaaaaaaaaaaa')
        self.refused('compare', {'segment': 'aaaaaaaaaaaaaaaaaaaa', 'reference': 'stock', 'student': 'run:' + RUN},
                     404)

    def test_model_specs(self):
        for model in ('run:../x', 'run:' + RUN + '/../..', 'hf:ABC', 'hf:a/b', 'hf:../x', 'stock ', 'Stock', 'run:',
                      'hf:', 'file:/etc/passwd', '', None, 5, 'run:2026-10-21_1544;ls'):
            self.refused('eval', {'model': model})
        self.refused('eval', {'model': 'run:2026-01-01_0000'}, 404)
        # Reference must be stock/teacher/logged, student a run or an hf model.
        self.refused('compare', {'segment': SEGMENT, 'reference': 'run:' + RUN, 'student': 'run:' + RUN}, 400)
        self.refused('compare', {'segment': SEGMENT, 'reference': 'stock', 'student': 'teacher'}, 400)

    def test_compare_numbers(self):
        base = {'segment': SEGMENT, 'reference': 'stock', 'student': 'run:' + RUN}
        bad_values = {'start': (-1, 59.5, 'nan', 'inf', '1e1', True, '10s', float('nan'), float('inf')),
                      'duration': (0, 0.5, 61, '-5', None),
                      'speed': (0.3, 2, 0, '0.75', True)}
        for key, values in bad_values.items():
            for value in values:
                params = dict(base)
                params[key] = value
                if key == 'duration' and value is None:
                    continue
                self.refused('compare', params, 400)
        for key in ('inset', 'show_gt'):
            params = dict(base)
            params[key] = 'true'
            self.refused('compare', params, 400)

    def test_names(self):
        for name in ('', 'a b', 'x;rm', 'a/b', '../x', 'é', 'a' * 65, None, 5, 'x\n', '$(id)'):
            self.refused('publish_dry', {'run': RUN, 'name': name}, 400)
        self.build('publish_dry', {'run': RUN, 'name': 'A' * 64})
        # The spec regex allows a leading "-": it stays glued to --name=, one argv item, never an option.
        spec = self.build('publish_dry', {'run': RUN, 'name': '--stable'})
        self.assertIn('--name=--stable', spec.argv)
        self.assertNotIn('--stable', spec.argv)

    def test_typed_confirmations(self):
        self.refused('publish', {'run': RUN, 'name': 'prova', 'confirm': 'Prova'}, 400)
        self.refused('publish', {'run': RUN, 'name': 'prova'}, 400)
        self.refused('install', {'run': RUN, 'confirm': 'installa'}, 400)
        self.refused('install', {'run': RUN}, 400)
        self.refused('restore', {'confirm': 'RESTORE '}, 400)
        self.refused('restore', {}, 400)

    def test_slug_matches_publish_model_sh(self):
        self.assertEqual(slug_of('2026-10-21_1544'), '2026_10_21_1544')
        self.assertEqual(slug_of('Prova.V2'), 'prova_v2')
        self.assertEqual(slug_of('cd210-native'), 'cd210_native')

    def test_missing_scripts(self):
        self.assertIn('this version of the code', script_missing(self.repo, 'eval'))
        (self.repo / 'tasks').mkdir()
        (self.repo / 'tasks' / 'evaluation.py').write_text('')
        self.assertEqual(script_missing(self.repo, 'eval'), '')


if __name__ == '__main__':
    unittest.main()
