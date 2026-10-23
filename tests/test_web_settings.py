"""distill-web, Settings: where the Hugging Face token and the W&B key are saved (temporary HOME and
HF_HOME, never the real ones), the netrc and .env editing, the checks with the services (the network
is mocked: nothing leaves the machine) and the settings API of the server."""
from __future__ import annotations

import base64
import importlib.util
import io
import json
import netrc
import os
import stat
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import web.settings as settings
from web.common import WebError, read_env_file
from web.settings import (ATTEMPTS_PER_WINDOW, CHECK_SECONDS, UNREACHABLE_SECONDS, Accounts, CheckFailed, Rejected,
                          check_hf_token, check_wandb_key, clean_site_changes, hf_token_path, hf_write_access,
                          netrc_password, netrc_path, netrc_with, netrc_without, update_env_file)

from test_web_server import HF_TOKEN, RUN, WANDB_KEY, ServerTestCase, isolated_home, write_task

CODE_ROOT = Path(__file__).resolve().parents[1]


def mode_of(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


class Answer:
    """What urlopen returns, as a context manager."""

    def __init__(self, payload):
        self.raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def read(self, limit=-1):
        return self.raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code):
    return urllib.error.HTTPError('https://example.invalid', code, 'error', {}, io.BytesIO(b'{}'))


class IsolatedTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name).resolve()
        self.home = isolated_home(self, self.base)


class PathTest(IsolatedTestCase):
    def test_the_token_file_huggingface_hub_reads(self):
        self.assertEqual(hf_token_path(), self.home / 'hf' / 'token')
        with mock.patch.dict(os.environ, {'HF_TOKEN_PATH': str(self.base / 'elsewhere' / 'tok')}):
            self.assertEqual(hf_token_path(), self.base / 'elsewhere' / 'tok')
        del os.environ['HF_HOME']
        self.assertEqual(hf_token_path(), self.home / '.cache' / 'huggingface' / 'token')
        with mock.patch.dict(os.environ, {'XDG_CACHE_HOME': str(self.base / 'xdg')}):
            self.assertEqual(hf_token_path(), self.base / 'xdg' / 'huggingface' / 'token')

    def test_the_netrc_wandb_reads(self):
        self.assertEqual(netrc_path(), self.home / '.netrc')
        with mock.patch.dict(os.environ, {'NETRC': str(self.base / 'n')}):
            self.assertEqual(netrc_path(), self.base / 'n')


class NetrcTest(unittest.TestCase):
    OTHER = ('# my servers\nmachine github.com\n  login me\n  password "with space"\n\n'
             'macdef init\nmachine api.wandb.ai login x password y\n\n'
             'machine example.org login a password b\n')

    def test_add_to_an_empty_file(self):
        text = netrc_with('', 'api.wandb.ai', WANDB_KEY)
        self.assertEqual(text, 'machine api.wandb.ai\n  login user\n  password ' + WANDB_KEY + '\n')
        self.assertEqual(netrc_password(text, 'api.wandb.ai'), WANDB_KEY)

    def test_other_entries_are_kept_as_they_are(self):
        text = netrc_with(self.OTHER, 'api.wandb.ai', WANDB_KEY)
        self.assertTrue(text.startswith(self.OTHER))
        # The "machine api.wandb.ai" inside the macro is the macro's text, not an entry.
        self.assertEqual(netrc_password(self.OTHER, 'api.wandb.ai'), '')
        self.assertEqual(netrc_password(text, 'api.wandb.ai'), WANDB_KEY)
        self.assertEqual(netrc_password(text, 'github.com'), 'with space')
        self.assertEqual(netrc_without(text, 'api.wandb.ai'), self.OTHER)

    def test_replace_an_entry_in_the_middle(self):
        old = ('machine a.example login u password p\nmachine api.wandb.ai\n  login user\n  password '
               + 'c' * 40 + '\nmachine b.example login v password q\n')
        text = netrc_with(old, 'api.wandb.ai', WANDB_KEY)
        self.assertEqual(text.count('api.wandb.ai'), 1)
        self.assertNotIn('c' * 40, text)
        self.assertTrue(text.startswith('machine a.example login u password p\nmachine b.example login v password q\n'))
        self.assertEqual(netrc_without('machine api.wandb.ai login user password k', 'api.wandb.ai'), '')
        # One line with several machines.
        line = ('machine a login u password p machine api.wandb.ai login user password k '
                'machine b login v password q\n')
        self.assertEqual(netrc_without(line, 'api.wandb.ai'),
                         'machine a login u password p machine b login v password q\n')

    def test_python_netrc_reads_the_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'netrc'
            path.write_text(netrc_with(self.OTHER, 'api.wandb.ai', WANDB_KEY))
            parsed = netrc.netrc(str(path))
            self.assertEqual(parsed.authenticators('api.wandb.ai'), ('user', '', WANDB_KEY))
            self.assertEqual(parsed.authenticators('github.com'), ('me', '', 'with space'))
            self.assertEqual(parsed.authenticators('example.org'), ('a', '', 'b'))


class EnvFileTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / '.env'

    def test_update_keeps_every_other_line(self):
        self.path.write_text('# site\nDISTILL_WEB_TOKEN=secret_value_1234567\nexport HF_REPO="old/repo"\n'
                             'DISTILL_UV_EXTRA=v100\nHF_REPO=dup/repo\n')
        self.path.chmod(0o644)
        update_env_file(self.path, {'HF_REPO': 'me/models', 'COMMA_HOST': 'root@10.0.0.5'})
        self.assertEqual(self.path.read_text(), '# site\nDISTILL_WEB_TOKEN=secret_value_1234567\nHF_REPO=me/models\n'
                         'DISTILL_UV_EXTRA=v100\nCOMMA_HOST=root@10.0.0.5\n')
        self.assertEqual(mode_of(self.path), 0o600)
        self.assertEqual(read_env_file(self.path, ('HF_REPO', 'COMMA_HOST')),
                         {'HF_REPO': 'me/models', 'COMMA_HOST': 'root@10.0.0.5'})
        update_env_file(self.path, {'HF_REPO': None})
        self.assertNotIn('HF_REPO', self.path.read_text())
        self.assertIn('DISTILL_WEB_TOKEN=secret_value_1234567\n', self.path.read_text())
        # No temporary file left behind.
        self.assertEqual(sorted(item.name for item in self.path.parent.iterdir()), ['.env'])

    def test_new_file_and_link(self):
        update_env_file(self.path, {'COMMA_HOST': 'comma'})
        self.assertEqual(self.path.read_text(), 'COMMA_HOST=comma\n')
        self.assertEqual(mode_of(self.path), 0o600)
        target = Path(self.directory.name) / 'target'
        target.write_text('X=1\n')
        link = Path(self.directory.name) / 'link'
        link.symlink_to(target)
        with self.assertRaises(WebError) as caught:
            update_env_file(link, {'COMMA_HOST': 'comma'})
        self.assertEqual(caught.exception.reason, 'file_is_a_link')
        self.assertEqual(target.read_text(), 'X=1\n')

    def test_clean_site_changes(self):
        self.assertEqual(clean_site_changes({'hf_repo': ' me/models ', 'comma_host': ''}),
                         {'HF_REPO': 'me/models', 'COMMA_HOST': None})
        for body in ({}, {'other': 'x'}, {'hf_repo': 1}, {'hf_repo': 'no-slash'}, {'hf_repo': 'a/b\nX=1'},
                     {'comma_host': '-oProxyCommand=x'}, {'comma_host': 'a b'}, {'comma_host': 'x' * 300}, []):
            with self.assertRaises(WebError, msg=repr(body)):
                clean_site_changes(body)


class ServiceCheckTest(IsolatedTestCase):
    def test_hugging_face_whoami(self):
        seen = []

        def answer(request, timeout):
            seen.append(request)
            return Answer({'name': 'someone', 'auth': {'accessToken': {'role': 'write', 'displayName': 'x'}}})

        with mock.patch.object(settings, '_open', answer):
            self.assertEqual(check_hf_token(HF_TOKEN), {'user': 'someone', 'role': 'write', 'fine_grained': None})
        self.assertEqual(seen[0].full_url, 'https://huggingface.co/api/whoami-v2')
        self.assertEqual(seen[0].get_header('Authorization'), 'Bearer ' + HF_TOKEN)

    def test_weights_and_biases_viewer(self):
        seen = []

        def answer(request, timeout):
            seen.append(request)
            return Answer({'data': {'viewer': {'username': 'someone', 'entity': 'team'}}})

        with mock.patch.object(settings, '_open', answer):
            self.assertEqual(check_wandb_key(WANDB_KEY), {'user': 'someone', 'entity': 'team'})
        self.assertEqual(seen[0].full_url, 'https://api.wandb.ai/graphql')
        self.assertEqual(seen[0].get_method(), 'POST')
        expected = base64.b64encode(('api:' + WANDB_KEY).encode()).decode()
        self.assertEqual(seen[0].get_header('Authorization'), 'Basic ' + expected)
        with mock.patch.object(settings, '_open', lambda request, timeout: Answer({'data': {'viewer': None}})):
            with self.assertRaises(Rejected):
                check_wandb_key(WANDB_KEY)

    def test_failures_never_quote_the_secret(self):
        def refuse(request, timeout):
            raise http_error(401)

        with mock.patch.object(settings, '_open', refuse):
            with self.assertRaises(Rejected):
                check_hf_token(HF_TOKEN)
            with self.assertRaises(Rejected):
                check_wandb_key(WANDB_KEY)
        failures = [http_error(500), urllib.error.URLError('down ' + HF_TOKEN), TimeoutError(HF_TOKEN),
                    ValueError('Invalid header value Bearer ' + HF_TOKEN)]
        for failure in failures:
            def fail(request, timeout, failure=failure):
                raise failure

            with mock.patch.object(settings, '_open', fail):
                with self.assertRaises(CheckFailed) as caught:
                    check_hf_token(HF_TOKEN)
            self.assertNotIn(HF_TOKEN, str(caught.exception))
            self.assertIsNone(caught.exception.__cause__)
        with mock.patch.object(settings, '_open', lambda request, timeout: Answer(b'<html>')):
            with self.assertRaises(CheckFailed):
                check_hf_token(HF_TOKEN)

    def test_no_redirects_with_a_secret(self):
        handler = settings._NoRedirect()
        self.assertIsNone(handler.redirect_request(None, None, 302, 'Found', {}, 'https://elsewhere.example/'))

    def test_write_access(self):
        repo = 'owner/models'
        self.assertEqual(hf_write_access({'role': 'write'}, repo), 'write')
        self.assertEqual(hf_write_access({'role': 'read'}, repo), 'read')
        self.assertEqual(hf_write_access({'role': None}, repo), 'unknown')
        scoped = {'scoped': [{'entity': {'type': 'user', 'name': 'owner'}, 'permissions': ['repo.content.read']},
                             {'entity': {'type': 'org', 'name': 'owner'}, 'permissions': ['repo.write']}]}
        self.assertEqual(hf_write_access({'role': 'fineGrained', 'fine_grained': scoped}, repo), 'write')
        scoped = {'scoped': [{'entity': {'type': 'model', 'name': repo}, 'permissions': ['repo.write']}]}
        self.assertEqual(hf_write_access({'role': 'fineGrained', 'fine_grained': scoped}, repo), 'write')
        scoped = {'scoped': [{'entity': {'type': 'user', 'name': 'other'}, 'permissions': ['repo.write']}]}
        self.assertEqual(hf_write_access({'role': 'fineGrained', 'fine_grained': scoped}, repo), 'read')


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class AccountsTest(IsolatedTestCase):
    def setUp(self):
        super().setUp()
        self.calls = {'hf': 0, 'wandb': 0}
        self.clock = Clock()
        self.hf_answer = None
        self.accounts = Accounts(check_hf=self.check_hf, check_wandb=self.check_wandb, clock=self.clock)

    def check_hf(self, token):
        self.calls['hf'] += 1
        if self.hf_answer is not None:
            raise self.hf_answer
        if token != HF_TOKEN:
            raise Rejected()
        return {'user': 'tester', 'role': 'write', 'fine_grained': None}

    def check_wandb(self, key):
        self.calls['wandb'] += 1
        if key != WANDB_KEY:
            raise Rejected()
        return {'user': 'tester', 'entity': 'team'}

    def test_hugging_face_connect_and_disconnect(self):
        state = self.accounts.state('hf', 'owner/models')
        self.assertEqual(state['state'], 'none')
        self.assertEqual(self.calls['hf'], 0)
        for bad in ('', 'not a token', 'hf_short', 'api_org_' + 'x' * 30, 'hf_' + 'x' * 30 + '\nX', 42, None):
            with self.assertRaises(WebError) as caught:
                self.accounts.connect('hf', bad, 'owner/models')
            self.assertEqual(caught.exception.reason, 'invalid_secret')
        self.assertEqual(self.calls['hf'], 0)
        with self.assertRaises(WebError) as caught:
            self.accounts.connect('hf', 'hf_' + 'B' * 34, 'owner/models')
        self.assertEqual(caught.exception.reason, 'rejected')
        self.assertFalse(hf_token_path().exists())
        self.hf_answer = CheckFailed('Hugging Face does not answer (TimeoutError)')
        with self.assertRaises(WebError) as caught:
            self.accounts.connect('hf', HF_TOKEN, 'owner/models')
        self.assertEqual(caught.exception.status, 503)
        self.assertFalse(hf_token_path().exists())
        self.hf_answer = None

        state = self.accounts.connect('hf', ' ' + HF_TOKEN + ' ', 'owner/models')
        self.assertEqual((state['state'], state['user'], state['access'], state['source']),
                         ('connected', 'tester', 'write', 'saved'))
        self.assertNotIn(HF_TOKEN, json.dumps(state))
        path = hf_token_path()
        self.assertEqual(path.read_text(), HF_TOKEN)
        self.assertEqual(mode_of(path), 0o600)
        self.assertEqual(mode_of(path.parent), 0o700)
        # The check made by connect is reused: no second request.
        calls = self.calls['hf']
        self.assertEqual(self.accounts.state('hf', 'owner/models')['user'], 'tester')
        self.assertEqual(self.calls['hf'], calls)

        # "hf auth login" kept a copy of the same token and another one: only the copy goes.
        stored = path.parent / 'stored_tokens'
        stored.write_text('[mine]\nhf_token = ' + HF_TOKEN + '\n\n[other]\nhf_token = hf_' + 'C' * 34 + '\n')
        state = self.accounts.disconnect('hf', 'owner/models')
        self.assertEqual(state['state'], 'none')
        self.assertFalse(path.exists())
        self.assertNotIn(HF_TOKEN, stored.read_text())
        self.assertIn('[other]', stored.read_text())
        # Disconnecting twice is fine.
        self.assertEqual(self.accounts.disconnect('hf', 'owner/models')['state'], 'none')

    def test_weights_and_biases_connect_and_disconnect(self):
        other = 'machine github.com\n  login me\n  password p\n'
        netrc_file = self.home / '.netrc'
        netrc_file.write_text(other)
        netrc_file.chmod(0o644)
        with self.assertRaises(WebError):
            self.accounts.connect('wandb', 'short', 'owner/models')
        state = self.accounts.connect('wandb', WANDB_KEY, 'owner/models')
        self.assertEqual((state['state'], state['user'], state['entity']), ('connected', 'tester', 'team'))
        self.assertNotIn(WANDB_KEY, json.dumps(state))
        self.assertTrue(netrc_file.read_text().startswith(other))
        self.assertEqual(mode_of(netrc_file), 0o600)
        self.assertEqual(netrc.netrc(str(netrc_file)).authenticators('api.wandb.ai'), ('user', '', WANDB_KEY))
        # tasks/train.py decides online when the netrc has W&B credentials.
        from tasks.train import wandb_mode
        self.assertEqual(wandb_mode(None), 'online')
        self.accounts.disconnect('wandb', 'owner/models')
        self.assertEqual(netrc_file.read_text(), other)
        self.assertEqual(wandb_mode(None), 'offline')
        # A netrc with only the W&B entry goes away with it.
        netrc_file.unlink()
        self.accounts.connect('wandb', WANDB_KEY, 'owner/models')
        self.accounts.disconnect('wandb', 'owner/models')
        self.assertFalse(netrc_file.exists())

    def test_credentials_from_the_environment_are_left_alone(self):
        with mock.patch.dict(os.environ, {'HF_TOKEN': HF_TOKEN, 'WANDB_API_KEY': WANDB_KEY}):
            state = self.accounts.state('hf', 'owner/models')
            self.assertEqual((state['state'], state['source'], state['can_change']),
                             ('connected', 'environment', False))
            for kind, secret in (('hf', HF_TOKEN), ('wandb', WANDB_KEY)):
                with self.assertRaises(WebError) as caught:
                    self.accounts.connect(kind, secret, 'owner/models')
                self.assertEqual(caught.exception.reason, 'set_by_environment')
                with self.assertRaises(WebError):
                    self.accounts.disconnect(kind, 'owner/models')
        self.assertFalse(hf_token_path().exists())
        self.assertFalse((self.home / '.netrc').exists())

    def test_checks_are_kept_for_a_while(self):
        hf_token_path().parent.mkdir(parents=True)
        hf_token_path().write_text(HF_TOKEN + '\n')
        # The status never asks the service.
        presence = self.accounts.presence('owner/models')
        self.assertEqual(presence['hf'], {'connected': True, 'user': None, 'problem': ''})
        self.assertEqual(presence['wandb']['connected'], False)
        self.assertIn('Settings', presence['wandb']['problem'])
        self.assertEqual(self.calls['hf'], 0)
        self.assertEqual(self.accounts.state('hf', 'owner/models')['user'], 'tester')
        self.assertEqual(self.accounts.presence('owner/models')['hf']['user'], 'tester')
        self.accounts.state('hf', 'owner/models')
        self.assertEqual(self.calls['hf'], 1)
        self.clock.now += CHECK_SECONDS + 1
        self.accounts.state('hf', 'owner/models')
        self.assertEqual(self.calls['hf'], 2)
        # Hugging Face down: asked again sooner.
        self.clock.now += CHECK_SECONDS + 1
        self.hf_answer = CheckFailed('Hugging Face does not answer (URLError)')
        self.assertEqual(self.accounts.state('hf', 'owner/models')['state'], 'unreachable')
        self.clock.now += UNREACHABLE_SECONDS + 1
        self.hf_answer = Rejected()
        self.assertEqual(self.accounts.state('hf', 'owner/models')['state'], 'invalid')
        self.assertEqual(self.accounts.presence('owner/models')['hf']['connected'], False)
        self.assertIn('no longer works', self.accounts.problem('hf', 'owner/models'))
        # Another token saved by hand: checked again at once.
        self.hf_answer = None
        hf_token_path().write_text('hf_' + 'D' * 34)
        self.assertEqual(self.accounts.state('hf', 'owner/models')['state'], 'invalid')

    def test_read_only_token(self):
        def read_only(token):
            return {'user': 'tester', 'role': 'read', 'fine_grained': None}

        accounts = Accounts(check_hf=read_only, check_wandb=self.check_wandb, clock=self.clock)
        state = accounts.connect('hf', HF_TOKEN, 'owner/models')
        self.assertEqual(state['access'], 'read')
        self.assertIn('cannot write owner/models', accounts.problem('hf', 'owner/models'))

    def test_attempts_are_limited_and_one_at_a_time(self):
        for _ in range(ATTEMPTS_PER_WINDOW):
            with self.assertRaises(WebError) as caught:
                self.accounts.connect('hf', 'hf_' + 'B' * 34, 'owner/models')
            self.assertEqual(caught.exception.reason, 'rejected')
        with self.assertRaises(WebError) as caught:
            self.accounts.connect('hf', HF_TOKEN, 'owner/models')
        self.assertEqual(caught.exception.status, 429)
        # The other account has its own count.
        self.accounts.connect('wandb', WANDB_KEY, 'owner/models')
        self.clock.now += 61
        self.accounts.connect('hf', HF_TOKEN, 'owner/models')
        self.accounts._change_lock.acquire()
        try:
            with self.assertRaises(WebError) as caught:
                self.accounts.connect('hf', HF_TOKEN, 'owner/models')
            self.assertEqual(caught.exception.reason, 'busy')
        finally:
            self.accounts._change_lock.release()


@unittest.skipUnless(importlib.util.find_spec('huggingface_hub'), 'huggingface_hub is not installed')
class LibraryCompatibilityTest(IsolatedTestCase):
    def run_python(self, code):
        environment = dict(os.environ)
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=120,
                                env=environment, cwd=str(CODE_ROOT))
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        return result.stdout.strip()

    def test_huggingface_hub_uses_the_saved_token(self):
        accounts = Accounts(check_hf=lambda token: {'user': 'tester', 'role': 'write', 'fine_grained': None})
        accounts.connect('hf', HF_TOKEN, 'owner/models')
        self.assertEqual(self.run_python('from huggingface_hub import get_token; print(get_token())'), HF_TOKEN)
        accounts.disconnect('hf', 'owner/models')
        self.assertEqual(self.run_python('from huggingface_hub import get_token; print(get_token())'), 'None')

    @unittest.skipUnless(importlib.util.find_spec('wandb'), 'wandb is not installed')
    def test_wandb_uses_the_saved_key(self):
        accounts = Accounts(check_wandb=lambda key: {'user': 'tester', 'entity': None})
        accounts.connect('wandb', WANDB_KEY, 'owner/models')
        code = ('from wandb.sdk.lib.wbauth import read_netrc_auth; '
                'print(read_netrc_auth(host="https://api.wandb.ai"))')
        self.assertEqual(self.run_python(code), WANDB_KEY)


class SettingsApiTest(ServerTestCase):
    def test_settings_without_accounts(self):
        response, body = self.request('GET', '/api/settings')
        self.assertEqual(response.status, 200)
        data = json.loads(body)
        self.assertEqual(data['accounts']['hf']['state'], 'none')
        self.assertEqual(data['accounts']['wandb']['state'], 'none')
        self.assertEqual(data['site']['hf_repo'], {'value': 'cristianku/openpilot-distill-models', 'saved': '',
                                                   'default': 'cristianku/openpilot-distill-models',
                                                   'from_environment': False})
        self.assertEqual(data['site']['comma_host']['value'], 'comma')
        for path in ('/api/settings/hf', '/api/settings/hf/logout', '/api/settings/wandb', '/api/settings/site'):
            response, body = self.request('GET', path)
            self.assertEqual(response.status, 405, path)

    def test_connect_hugging_face(self):
        response, body = self.post('/api/settings/hf', {'token': 'hf_' + 'B' * 34})
        self.assertEqual(response.status, 400)
        self.assertEqual(json.loads(body)['error'], 'rejected')
        self.assertNotIn(b'hf_BBBB', body)
        for payload in ({}, {'token': HF_TOKEN, 'x': 1}, {'key': HF_TOKEN}):
            response, body = self.post('/api/settings/hf', payload)
            self.assertEqual(response.status, 400, payload)
        response, body = self.post('/api/settings/hf', {'token': HF_TOKEN}, {'Origin': 'http://evil.example'})
        self.assertEqual(response.status, 403)
        response, body = self.request('POST', '/api/settings/hf', json.dumps({'token': HF_TOKEN}).encode(),
                                      {'Content-Type': 'text/plain'})
        self.assertEqual(response.status, 415)
        self.assertFalse((self.home / 'hf' / 'token').exists())

        response, body = self.post('/api/settings/hf', {'token': HF_TOKEN})
        self.assertEqual(response.status, 200, body)
        self.assertNotIn(HF_TOKEN.encode(), body)
        account = json.loads(body)['account']
        self.assertEqual((account['state'], account['user'], account['access']), ('connected', 'tester', 'write'))
        self.assertEqual((self.home / 'hf' / 'token').read_text(), HF_TOKEN)
        response, body = self.request('GET', '/api/settings')
        self.assertNotIn(HF_TOKEN.encode(), body)
        self.assertEqual(json.loads(body)['accounts']['hf']['user'], 'tester')
        response, body = self.request('GET', '/api/status')
        status = json.loads(body)
        self.assertNotIn(HF_TOKEN.encode(), body)
        self.assertEqual(status['accounts']['hf'], {'connected': True, 'user': 'tester', 'problem': ''})
        self.assertNotIn('Settings', status['actions']['publish']['reason'])

        response, body = self.post('/api/settings/hf/logout', {'x': 1})
        self.assertEqual(response.status, 400)
        response, body = self.post('/api/settings/hf/logout', {})
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(body)['account']['state'], 'none')
        self.assertFalse((self.home / 'hf' / 'token').exists())

    def test_wandb_online_needs_an_account(self):
        write_task(self.repo, 'train.py', 'print("train", *sys.argv[1:])')
        params = {'mode': 'fresh', 'steps': 10, 'wandb': 'online', 'gpus': 'all'}
        response, body = self.post('/api/run/train', params)
        self.assertEqual(response.status, 409)
        self.assertEqual(json.loads(body)['error'], 'wandb_not_connected')
        self.assertIn('Settings', json.loads(body)['message'])
        response, body = self.post('/api/settings/wandb', {'key': WANDB_KEY})
        self.assertEqual(response.status, 200, body)
        self.assertNotIn(WANDB_KEY.encode(), body)
        self.assertEqual(json.loads(body)['account']['entity'], 'tester-team')
        self.assertIn('machine api.wandb.ai', (self.home / '.netrc').read_text())
        response, body = self.post('/api/run/train', params)
        self.assertEqual(response.status, 200, body)
        response, body = self.post('/api/settings/wandb/logout', {})
        self.assertEqual(json.loads(body)['account']['state'], 'none')
        self.assertFalse((self.home / '.netrc').exists())

    def test_site_settings_apply_at_once(self):
        env_file = self.repo / '.env'
        env_file.write_text('DISTILL_WEB_TOKEN=' + 'T' * 24 + '\nDISTILL_UV_EXTRA=v100\n')
        response, body = self.post('/api/settings/site', {'hf_repo': 'someone/models', 'comma_host': 'root@10.1.2.3'})
        self.assertEqual(response.status, 200, body)
        answer = json.loads(body)
        self.assertTrue(answer['applied'])
        self.assertEqual(answer['site']['hf_repo']['value'], 'someone/models')
        self.assertNotIn(b'TTTT', body)
        self.assertEqual(env_file.read_text(), 'DISTILL_WEB_TOKEN=' + 'T' * 24 + '\nDISTILL_UV_EXTRA=v100\n'
                         'HF_REPO=someone/models\nCOMMA_HOST=root@10.1.2.3\n')
        self.assertEqual(mode_of(env_file), 0o600)
        self.assertEqual(self.app.hf.repo, 'someone/models')
        self.assertEqual(self.app.comma.host, 'root@10.1.2.3')
        response, body = self.request('GET', '/api/status')
        self.assertEqual(json.loads(body)['hf_repo'], 'someone/models')
        for payload in ({'hf_repo': 'bad'}, {'comma_host': '-oProxyCommand=x'}, {'nope': 'x'}, {}):
            response, body = self.post('/api/settings/site', payload)
            self.assertEqual(response.status, 400, payload)
        # Empty: back to the default.
        response, body = self.post('/api/settings/site', {'hf_repo': ''})
        self.assertEqual(json.loads(body)['site']['hf_repo']['value'], 'cristianku/openpilot-distill-models')
        self.assertNotIn('HF_REPO', env_file.read_text())
        self.assertEqual(self.app.hf.repo, 'cristianku/openpilot-distill-models')
        # The environment of the server wins over the .env, and the page is told so.
        with mock.patch.dict(os.environ, {'COMMA_HOST': 'from-env'}):
            response, body = self.post('/api/settings/site', {'comma_host': 'other'})
            site = json.loads(body)['site']
            self.assertEqual(site['comma_host']['value'], 'from-env')
            self.assertEqual(site['comma_host']['saved'], 'other')
            self.assertTrue(site['comma_host']['from_environment'])

    def test_publish_with_a_read_only_token(self):
        write_task(self.repo, 'publish.py', 'print("publish")')
        self.accounts._checkers['hf'] = lambda token: {'user': 'tester', 'role': 'read', 'fine_grained': None}
        response, body = self.post('/api/settings/hf', {'token': HF_TOKEN})
        self.assertEqual(json.loads(body)['account']['access'], 'read')
        response, body = self.request('GET', '/api/status')
        self.assertIn('write access in Settings', json.loads(body)['actions']['publish']['reason'])
        response, body = self.post('/api/run/publish', {'run': RUN, 'name': 'nuovo', 'confirm': 'nuovo'})
        self.assertEqual(json.loads(body)['error'], 'hf_not_connected')


if __name__ == '__main__':
    unittest.main()
