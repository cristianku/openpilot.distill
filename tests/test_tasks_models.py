"""tasks/export.py, tasks/install.py, tasks/publish.py: arguments, the reference model, the publish
dry run against the upload, the install refusals. A fake comma, fake requests/huggingface_hub modules
and temporary folders: no ssh, no network, no export."""
from __future__ import annotations

import contextlib
import hashlib
import io
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import tasks.export as export_task
import tasks.install as install_task
import tasks.publish as publish_task
from tasks.comma import BACKUP, NOT_REACHABLE, REMOTE_OPENPILOT
from tasks.common import DEFAULT_HF_REPO, TaskError

RUN = '2026-10-22_101530'
OLDER_RUN = '2026-10-21_090000'
MODEL_PATH = 'selfdrive/modeld/models/driving_supercombo.onnx'
IN_PLACE = REMOTE_OPENPILOT + '/' + MODEL_PATH
ORIGINAL_BYTES = b'original model'
ORIGINAL = hashlib.sha256(ORIGINAL_BYTES).hexdigest()
TRAINED = hashlib.sha256(b'trained model').hexdigest()


class FakeComma:
    """The calls of tasks.comma.Comma, recorded; pull writes the bytes of `files` (remote path -> bytes)."""

    def __init__(self, info=None, *, reachable=True, files=None, compile_code=0, compile_output='12.5 ms\n'):
        self.info = dict(info or {})
        self.is_reachable = reachable
        self.files = dict(files or {})
        self.compile_code = compile_code
        self.compile_output = compile_output
        self.calls = []

    def reachable(self):
        return self.is_reachable

    def require(self):
        if not self.is_reachable:
            raise TaskError(NOT_REACHABLE)

    def key_values(self, script, *arguments):
        self.calls.append(('info', arguments))
        return dict(self.info)

    def script(self, script, *arguments, check=True, timeout=None):
        self.calls.append(('script', script, arguments))
        if 'compile' in script:
            return subprocess.CompletedProcess([], self.compile_code, stdout=self.compile_output, stderr='')
        return subprocess.CompletedProcess([], 0, stdout='', stderr='')

    def run(self, command, *, check=True, capture=True, timeout=None):
        self.calls.append(('run', command))
        return subprocess.CompletedProcess([], 0, stdout='', stderr='')

    def pull(self, remote_path, local_path, *, progress=False):
        self.calls.append(('pull', remote_path, Path(local_path)))
        Path(local_path).write_bytes(self.files.get(remote_path, b'compiled model'))

    def push(self, local_path, remote_path, *, progress=False):
        self.calls.append(('push', Path(local_path), remote_path))

    def kinds(self):
        return [call[0] for call in self.calls]

    def scripts(self, script):
        return [call[2] for call in self.calls if call[0] == 'script' and call[1] == script]


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


class TempRepoTest(unittest.TestCase):
    """A temporary outputs/supervised/, models/ and outputs/publish/ in place of the repo ones."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.runs = self.root / 'outputs' / 'supervised'
        self.runs.mkdir(parents=True)
        self.reference = self.root / 'models' / 'comma_driving_supercombo.onnx'
        for target, name, value in ((export_task, 'RUNS_DIR', self.runs),
                                    (export_task, 'REFERENCE', self.reference),
                                    (publish_task, 'PUBLISH_DIR', self.root / 'outputs' / 'publish'),
                                    (publish_task, 'STOCK_DIR', self.root / 'outputs' / 'stock')):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_run(self, name=RUN, finished=True, mtime=None):
        folder = self.runs / name
        folder.mkdir()
        if finished:
            (folder / 'supercombo.pt').write_bytes(b'weights')
            if mtime is not None:
                os.utime(folder / 'supercombo.pt', (mtime, mtime))
        return folder

    def keep_reference(self, content=ORIGINAL_BYTES):
        self.reference.parent.mkdir(parents=True, exist_ok=True)
        self.reference.write_bytes(content)


class ArgumentsTest(unittest.TestCase):
    def refused(self, parse, arguments):
        with self.assertRaises(SystemExit, msg=arguments), mock.patch('sys.stderr'):
            parse(arguments)

    def test_export(self):
        args = export_task.parse_args([])
        self.assertEqual((args.run, args.model, args.host), (None, None, None))
        self.assertEqual(export_task.parse_args(['--run', RUN]).run, RUN)
        self.assertEqual(export_task.parse_args(['outputs/supervised/' + RUN + '/supercombo.pt']).run, None)
        for arguments in (['--run', '../' + RUN], ['--run', RUN, 'x.pt'], ['--host='], ['a.pt', 'b.pt']):
            self.refused(export_task.parse_args, arguments)

    def test_install(self):
        args = install_task.parse_args(['--run', RUN])
        self.assertEqual((args.run, args.restore, args.no_reboot), (RUN, False, False))
        args = install_task.parse_args(['--restore', '--no-reboot'])
        self.assertEqual((args.restore, args.no_reboot), (True, True))
        for arguments in (['--restore', '--run', RUN], ['--restore', 'x.pt'], ['--run', 'latest']):
            self.refused(install_task.parse_args, arguments)

    def test_publish(self):
        args = publish_task.parse_args(['--name=prova', '--stable', '--run', RUN])
        self.assertEqual((args.name, args.stable, args.upload, args.run), ('prova', True, False, RUN))
        self.assertTrue(publish_task.parse_args(['--upload', '--run', RUN]).upload)
        self.assertTrue(publish_task.parse_args(['--stock']).stock)
        for arguments in (['--no-upload'], ['--name=a/b'], ['--name='], ['--stock', '--stable'],
                          ['--stock', '--run', RUN], ['--stock', 'x.onnx']):
            self.refused(publish_task.parse_args, arguments)

    def test_names(self):
        self.assertEqual(publish_task.slug_of('2026-10-20_1910'), '2026_10_20_1910')
        self.assertEqual(publish_task.slug_of('Prova.V2'), 'prova_v2')
        self.assertEqual(publish_task.stock_name('CD210'), 'cd210-native')
        self.assertEqual(publish_task.stock_name('North Dakota v2'), 'north-dakota-v2-native')


class ResolveModelTest(TempRepoTest):
    def test_latest(self):
        self.make_run(OLDER_RUN, mtime=1000)
        self.make_run(RUN, mtime=2000)
        self.make_run('2026-10-22_120000', finished=False)
        self.assertEqual(export_task.resolve_model(None, None), self.runs / RUN / 'supercombo.pt')

    def test_run_and_file(self):
        self.make_run(RUN)
        self.assertEqual(export_task.resolve_model(RUN, None), self.runs / RUN / 'supercombo.pt')
        path = export_task.resolve_model(None, str(self.runs / RUN / '..' / RUN / 'supercombo.pt'))
        self.assertEqual(path, self.runs / RUN / 'supercombo.pt')

    def test_refused(self):
        with self.assertRaisesRegex(TaskError, 'no finished run'):
            export_task.resolve_model(None, None)
        self.make_run(RUN, finished=False)
        with self.assertRaisesRegex(TaskError, 'has no supercombo.pt'):
            export_task.resolve_model(RUN, None)
        with self.assertRaisesRegex(TaskError, 'does not exist'):
            export_task.resolve_model(None, str(self.root / 'missing.pt'))


class ReferenceTest(TempRepoTest):
    def info(self, current=ORIGINAL, backup=''):
        return {'reference': MODEL_PATH, 'original': ORIGINAL, 'current': current, 'backup': backup}

    def update(self, comma, info, strict=False):
        with quiet():
            return export_task.update_reference(comma, info, strict=strict)

    def test_kept_copy_already_original(self):
        self.keep_reference()
        comma = FakeComma()
        self.assertEqual(self.update(comma, self.info(current=TRAINED)), self.reference)
        self.assertEqual(comma.calls, [])

    def test_copied_from_its_place(self):
        self.keep_reference(b'older openpilot version')
        comma = FakeComma(files={IN_PLACE: ORIGINAL_BYTES})
        self.update(comma, self.info())
        self.assertEqual(comma.calls, [('pull', IN_PLACE, self.reference.with_name(self.reference.name + '.new'))])
        self.assertEqual(self.reference.read_bytes(), ORIGINAL_BYTES)
        self.assertFalse(self.reference.with_name(self.reference.name + '.new').exists())

    def test_copied_from_backup(self):
        comma = FakeComma(files={BACKUP: ORIGINAL_BYTES})
        self.update(comma, self.info(current=TRAINED, backup=ORIGINAL), strict=True)
        self.assertEqual(comma.calls[0][1], BACKUP)
        self.assertEqual(self.reference.read_bytes(), ORIGINAL_BYTES)

    def test_wrong_copy_never_kept(self):
        self.keep_reference(b'older openpilot version')
        comma = FakeComma(files={IN_PLACE: b'trained model'})
        with self.assertRaisesRegex(TaskError, 'left untouched'):
            self.update(comma, self.info())
        self.assertEqual(self.reference.read_bytes(), b'older openpilot version')
        self.assertFalse(self.reference.with_name(self.reference.name + '.new').exists())

    def test_original_missing_on_the_comma(self):
        info = self.info(current=TRAINED, backup=TRAINED)
        with self.assertRaisesRegex(TaskError, 'there is no'):
            self.update(FakeComma(), info)
        self.keep_reference(b'older openpilot version')
        self.assertEqual(self.update(FakeComma(), info), self.reference)
        with self.assertRaisesRegex(TaskError, 'original model of this version'):
            self.update(FakeComma(), info, strict=True)

    def test_comma_off(self):
        with self.assertRaisesRegex(TaskError, 'comma not reachable'), quiet():
            export_task.reference_for_export(FakeComma(reachable=False))
        self.keep_reference(b'older openpilot version')
        with quiet():
            self.assertEqual(export_task.reference_for_export(FakeComma(reachable=False)), self.reference)

    def test_comma_on(self):
        comma = FakeComma(self.info(), files={IN_PLACE: ORIGINAL_BYTES})
        with quiet():
            self.assertEqual(export_task.reference_for_export(comma), self.reference)
        self.assertEqual(comma.kinds(), ['info', 'pull'])
        self.assertEqual(comma.calls[0][1], (BACKUP,))


def fake_requests(status_code=404, error=False):
    module = types.ModuleType('requests')
    module.RequestException = type('RequestException', (Exception,), {})
    if error:
        module.get = mock.Mock(side_effect=module.RequestException('offline'))
    else:
        module.get = mock.Mock(return_value=types.SimpleNamespace(status_code=status_code))
    return module


def fake_huggingface_hub(role='write', login=True):
    module = types.ModuleType('huggingface_hub')
    module.HfApi = mock.Mock()
    api = module.HfApi.return_value
    if login:
        api.whoami.return_value = {'name': 'cristianku', 'auth': {'accessToken': {'role': role}}}
    else:
        api.whoami.side_effect = OSError('Token is required')
    return module


PUBLISH_INFO = {'onroad': '0', 'compiler': 'sunnypilot/modeld_v2/compile_modeld_v2.py', 'reference': MODEL_PATH,
                'selector_version': '14', 'default_model': 'CD210', 'original': ORIGINAL, 'current': ORIGINAL,
                'backup': '', 'tinygrad_ref': 'abc123', 'python': '/usr/local/venv/bin/python3'}


class PublishTest(TempRepoTest):
    def setUp(self):
        super().setUp()
        self.make_run(RUN)
        self.keep_reference()
        self.modules = []
        self.exports = []
        patcher = mock.patch.object(publish_task, 'export_onnx', side_effect=self.fake_export)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(publish_task, 'run_module', side_effect=self.fake_module)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop('HF_REPO', None)

    def fake_export(self, checkpoint, reference, output):
        self.exports.append((checkpoint, reference, output))
        output.write_bytes(b'exported onnx')

    def fake_module(self, module, *arguments, failure):
        self.modules.append((module, arguments))

    def publish(self, arguments, *, comma=None, requests_module=None, hub=None):
        args = publish_task.parse_args(arguments)
        comma = comma or FakeComma(PUBLISH_INFO)
        modules = {'requests': requests_module or fake_requests(), 'huggingface_hub': hub or fake_huggingface_hub()}
        with mock.patch.dict(sys.modules, modules), quiet():
            return publish_task.publish(args, comma)

    def stage(self, slug='prova'):
        return self.root / 'outputs' / 'publish' / slug

    def test_dry_run_uploads_nothing(self):
        hub = fake_huggingface_hub()
        comma = FakeComma(PUBLISH_INFO)
        self.assertEqual(self.publish(['--name=prova', '--run', RUN], comma=comma, hub=hub), 0)
        hub.HfApi.assert_not_called()
        stage = self.stage()
        self.assertEqual((stage / 'models' / 'prova' / 'driving_supercombo.onnx').read_bytes(), b'exported onnx')
        self.assertTrue((stage / 'models' / 'prova' / 'driving_distill_prova_tinygrad.pkl').is_file())
        self.assertEqual([module for module, _ in self.modules],
                         ['helpers.publish_training_weights', 'helpers.model_catalog'])
        self.assertNotIn('--stable', self.modules[0][1])
        self.assertEqual(self.exports[0][1], self.reference)
        # The build folder on the comma is removed at the end.
        self.assertEqual(comma.calls[-1], ('run', 'rm -rf ' + publish_task.BUILD_DIR))

    def test_upload(self):
        hub = fake_huggingface_hub()
        self.assertEqual(self.publish(['--upload', '--stable', '--name=prova', '--run', RUN], hub=hub), 0)
        api = hub.HfApi.return_value
        api.whoami.assert_called_once()
        api.upload_folder.assert_called_once()
        options = api.upload_folder.call_args.kwargs
        self.assertEqual((options['repo_id'], options['folder_path']), (DEFAULT_HF_REPO, str(self.stage())))
        self.assertIn('--stable', self.modules[0][1])

    def test_name_taken(self):
        comma = FakeComma(PUBLISH_INFO)
        with self.assertRaisesRegex(TaskError, 'already published'):
            self.publish(['--upload', '--name=prova', '--run', RUN], comma=comma,
                         requests_module=fake_requests(200))
        self.assertNotIn('push', comma.kinds())
        self.assertEqual(self.exports, [])
        # The dry run only warns.
        self.assertEqual(self.publish(['--name=prova', '--run', RUN], requests_module=fake_requests(200)), 0)

    def test_name_not_checked(self):
        with self.assertRaisesRegex(TaskError, 'cannot check'):
            self.publish(['--upload', '--run', RUN], requests_module=fake_requests(error=True))
        self.assertEqual(self.publish(['--run', RUN], requests_module=fake_requests(error=True)), 0)

    def test_login_refused(self):
        for hub in (fake_huggingface_hub(login=False), fake_huggingface_hub(role='read')):
            comma = FakeComma(PUBLISH_INFO)
            with self.assertRaisesRegex(TaskError, 'Hugging Face'):
                self.publish(['--upload', '--run', RUN], comma=comma, hub=hub)
            hub.HfApi.return_value.upload_folder.assert_not_called()
            self.assertNotIn('push', comma.kinds())

    def test_upload_failure(self):
        hub = fake_huggingface_hub()
        hub.HfApi.return_value.upload_folder.side_effect = OSError('403 Forbidden')
        with self.assertRaisesRegex(TaskError, 'upload to Hugging Face failed: 403'):
            self.publish(['--upload', '--run', RUN], hub=hub)

    def test_compile_failure_uploads_nothing(self):
        hub = fake_huggingface_hub()
        with self.assertRaisesRegex(TaskError, 'compile failed'):
            self.publish(['--upload', '--run', RUN], comma=FakeComma(PUBLISH_INFO, compile_code=1), hub=hub)
        hub.HfApi.return_value.upload_folder.assert_not_called()

    def test_comma_refusals(self):
        for key, value, message in (('onroad', '1', 'onroad'), ('compiler', '', 'sunnypilot'),
                                    ('selector_version', '', 'REQUIRED_JSON_VERSION'), ('python', '', 'python3')):
            info = dict(PUBLISH_INFO, **{key: value})
            with self.assertRaisesRegex(TaskError, message, msg=key):
                self.publish(['--run', RUN], comma=FakeComma(info))
        with self.assertRaisesRegex(TaskError, 'not reachable'):
            self.publish(['--run', RUN], comma=FakeComma(PUBLISH_INFO, reachable=False))

    def test_onnx_given(self):
        onnx = self.runs / RUN / 'driving_supercombo.onnx'
        onnx.write_bytes(b'old export')
        self.assertEqual(self.publish(['--stable', str(onnx)]), 0)
        self.assertEqual(self.exports, [])
        weights = [argument for argument in self.modules[0][1] if argument.startswith('--model=')]
        self.assertEqual(weights, ['--model=supercombo=' + str(self.runs / RUN / 'supercombo.pt')])
        # --stable without weights next to the .onnx.
        (self.runs / RUN / 'supercombo.pt').unlink()
        with self.assertRaisesRegex(TaskError, 'PyTorch weights'):
            self.publish(['--stable', str(onnx)])

    def test_stock(self):
        self.assertEqual(self.publish(['--stock']), 0)
        model = self.root / 'outputs' / 'stock' / 'cd210-native' / 'driving_supercombo.onnx'
        self.assertEqual(model.read_bytes(), ORIGINAL_BYTES)
        self.assertEqual(self.exports, [])
        self.assertEqual([module for module, _ in self.modules], ['helpers.model_catalog'])


INSTALL_INFO = {'onroad': '0', 'active_bundle': '', 'reference': MODEL_PATH,
                'tg_flags': 'DEV=QCOM FLOAT16=1 NOLOCALS=1', 'original': ORIGINAL, 'current': ORIGINAL, 'backup': '',
                'python': '/usr/local/venv/bin/python3'}


class InstallTest(TempRepoTest):
    def setUp(self):
        super().setUp()
        self.make_run(RUN)
        self.keep_reference()
        self.exports = []
        patcher = mock.patch.object(install_task, 'export_onnx', side_effect=self.fake_export)
        patcher.start()
        self.addCleanup(patcher.stop)

    def fake_export(self, checkpoint, reference, output):
        self.exports.append((checkpoint, reference, output))
        output.write_bytes(b'exported onnx')

    def install(self, arguments, comma):
        with quiet():
            return install_task.install(install_task.parse_args(arguments), comma)

    def assert_untouched(self, comma):
        self.assertEqual(comma.scripts(install_task.PUT_MODEL_SCRIPT), [])
        self.assertNotIn(('run', 'sudo reboot'), comma.calls)

    def test_install(self):
        comma = FakeComma(INSTALL_INFO)
        self.assertEqual(self.install(['--run', RUN], comma), 0)
        self.assertEqual(comma.scripts(install_task.COPY_SCRIPT), [(IN_PLACE, BACKUP)])
        build_onnx = install_task.BUILD_DIR + '/driving_supercombo.onnx'
        self.assertEqual(comma.scripts(install_task.PUT_MODEL_SCRIPT), [(build_onnx, IN_PLACE)])
        self.assertEqual(comma.scripts(install_task.COMPILE_CHECK_SCRIPT)[0][2], INSTALL_INFO['tg_flags'])
        self.assertEqual(comma.calls[-1], ('run', 'sudo reboot'))
        self.assertEqual(self.exports[0][2], self.runs / RUN / 'driving_supercombo.onnx')
        self.assertEqual((self.runs / RUN / 'comma_compile.log').read_text(), '12.5 ms\n')

    def test_no_reboot_and_backup_kept(self):
        comma = FakeComma(dict(INSTALL_INFO, current=TRAINED, backup=ORIGINAL))
        self.assertEqual(self.install(['--no-reboot', '--run', RUN], comma), 0)
        self.assertEqual(comma.scripts(install_task.COPY_SCRIPT), [])
        self.assertNotIn(('run', 'sudo reboot'), comma.calls)

    def test_refusals(self):
        for changes, message in (({'onroad': '1'}, 'onroad'), ({'reference': ''}, 'driving_supercombo.onnx'),
                                 ({'original': ''}, 'sha256'), ({'active_bundle': '{"ref": "x"}'}, 'Models'),
                                 ({'tg_flags': ''}, 'tg_flags'), ({'python': ''}, 'python3'),
                                 ({'current': TRAINED, 'backup': TRAINED}, 'valid backup')):
            comma = FakeComma(dict(INSTALL_INFO, **changes))
            with self.assertRaisesRegex(TaskError, message, msg=changes):
                self.install(['--run', RUN], comma)
            self.assert_untouched(comma)
            self.assertEqual(comma.scripts(install_task.COPY_SCRIPT), [])
        self.assertEqual(self.exports, [])
        # An empty Models selection is the Default.
        self.assertEqual(self.install(['--run', RUN], FakeComma(dict(INSTALL_INFO, active_bundle='{}'))), 0)

    def test_comma_off(self):
        comma = FakeComma(INSTALL_INFO, reachable=False)
        with self.assertRaisesRegex(TaskError, 'not reachable'):
            self.install(['--run', RUN], comma)
        self.assertEqual(comma.calls, [])

    def test_compile_failure_changes_nothing(self):
        comma = FakeComma(INSTALL_INFO, compile_code=1, compile_output='error\n')
        with self.assertRaisesRegex(TaskError, 'nothing was changed'):
            self.install(['--run', RUN], comma)
        self.assert_untouched(comma)
        self.assertEqual(comma.calls[-1], ('run', 'rm -rf ' + install_task.BUILD_DIR))

    def test_unfinished_run(self):
        self.make_run(OLDER_RUN, finished=False)
        comma = FakeComma(INSTALL_INFO)
        with self.assertRaisesRegex(TaskError, 'has no supercombo.pt'):
            self.install(['--run', OLDER_RUN], comma)
        self.assert_untouched(comma)

    def test_restore(self):
        comma = FakeComma(dict(INSTALL_INFO, current=TRAINED, backup=ORIGINAL))
        self.assertEqual(self.install(['--restore'], comma), 0)
        self.assertEqual(comma.scripts(install_task.PUT_MODEL_SCRIPT), [(BACKUP, IN_PLACE)])
        self.assertEqual(comma.calls[-1], ('run', 'sudo reboot'))

    def test_restore_refusals(self):
        comma = FakeComma(INSTALL_INFO)
        self.assertEqual(self.install(['--restore'], comma), 0)
        self.assert_untouched(comma)
        for changes, message in (({'current': TRAINED, 'backup': ''}, 'backup'),
                                 ({'current': TRAINED, 'backup': TRAINED}, 'backup'),
                                 ({'current': TRAINED, 'backup': ORIGINAL, 'onroad': '1'}, 'onroad')):
            comma = FakeComma(dict(INSTALL_INFO, **changes))
            with self.assertRaisesRegex(TaskError, message, msg=changes):
                self.install(['--restore'], comma)
            self.assert_untouched(comma)


class LazyImportTest(unittest.TestCase):
    def test_no_third_party_at_import(self):
        code = ('import sys, tasks.export, tasks.install, tasks.publish; '
                'print(",".join(m for m in ("requests", "huggingface_hub", "torch") if m in sys.modules))')
        output = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, check=True).stdout
        self.assertEqual(output.strip(), '')


if __name__ == '__main__':
    unittest.main()
