"""distill-web: the HTTP server (administrator sign-in and first-run setup, session cookie, same-origin
POST, body limits, CSP, Range for the videos, the publish and install pre-checks). Standard library
only, no network."""
from __future__ import annotations

import http.client
import json
import os
import re
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import web.auth
import web.settings
from web.auth import AdminAuth
from web.jobs import JobManager
from web.server import App, create_server, parse_range
from web.settings import Accounts, Rejected

TOKEN = 'TestToken_1234567890abcdef'
ADMIN = 'admin'
PASSWORD = 'correct horse battery'
RUN = '2026-10-21_1544'
HF_TOKEN = 'hf_' + 'A' * 34
WANDB_KEY = 'b' * 40
# The variables that point the settings at the real credentials of this machine: every test runs
# with them unset and HOME/HF_HOME in its own temporary folder.
CREDENTIAL_VARIABLES = ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HF_TOKEN_PATH', 'HF_HOME', 'XDG_CACHE_HOME',
                        'HF_ENDPOINT', 'WANDB_API_KEY', 'NETRC', 'HF_REPO', 'COMMA_HOST')


def isolated_home(test: unittest.TestCase, base: Path) -> Path:
    """HOME and HF_HOME in `base` for the whole test, and no request leaves the machine."""
    home = base / 'home'
    home.mkdir()
    environment = mock.patch.dict(os.environ, {'HOME': str(home), 'HF_HOME': str(home / 'hf')})
    environment.start()
    test.addCleanup(environment.stop)
    for name in CREDENTIAL_VARIABLES:
        if name != 'HF_HOME':
            os.environ.pop(name, None)

    def no_network(request, timeout):
        raise AssertionError('a test tried to reach ' + request.full_url)

    network = mock.patch.object(web.settings, '_open', no_network)
    network.start()
    test.addCleanup(network.stop)
    return home


def fake_hf_check(token):
    if token != HF_TOKEN:
        raise Rejected()
    return {'user': 'tester', 'role': 'write', 'fine_grained': None}


def fake_wandb_check(key):
    if key != WANDB_KEY:
        raise Rejected()
    return {'user': 'tester', 'entity': 'tester-team'}


class FakeComma:
    host = 'comma'

    def __init__(self):
        self.state = {'host': 'comma', 'checked': 1.0, 'reachable': True, 'onroad': False, 'bundle': 'Default',
                      'bundle_default': True, 'backup': False, 'error': ''}

    def set_host(self, host):
        self.host = host
        self.state['host'] = host

    def snapshot(self):
        return dict(self.state)

    def probe_now(self):
        return dict(self.state)

    def start(self):
        pass


class FakeHf:
    repo = 'cristianku/openpilot-distill-models'

    def __init__(self):
        self.models = ['2026_10_21_1544', 'cd210_native']

    def set_repo(self, repo):
        self.repo = repo

    def snapshot(self, force=False):
        return {'repo': self.repo, 'url': 'https://huggingface.co/' + self.repo, 'ok': True, 'error': '',
                'checked': 1.0, 'models': list(self.models), 'releases': ['2026_10_21_1544'], 'stable': False}


def write_task(repo: Path, name: str, body: str) -> None:
    """A fake tasks/<name>.py in the test repo: the jobs run "python -m tasks.<name>" from the repo."""
    path = repo / 'tasks' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / '__init__.py').touch()
    path.write_text('import sys\n' + body + '\n')


def fast_scrypt(test: unittest.TestCase) -> None:
    """A cheap scrypt cost for the tests (the real one takes ~0.1 s per check)."""
    patcher = mock.patch.object(web.auth, 'SCRYPT_N', 2 ** 10)
    patcher.start()
    test.addCleanup(patcher.stop)


class ServerTestCase(unittest.TestCase):
    local_only = False
    # False: the administrator does not exist yet (first visit).
    with_admin = True

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.home = isolated_home(self, Path(self.directory.name).resolve())
        self.repo = Path(self.directory.name).resolve() / 'repo'
        run = self.repo / 'outputs' / 'supervised' / RUN
        run.mkdir(parents=True)
        (run / 'supercombo.pt').write_bytes(b'weights')
        lines = []
        for step in range(1, 21):
            lines.append('step ' + str(step) + '/20: loss=' + str(100 - step) + ' gt=1 distill=2 time=0.70s')
        (run / 'train.log').write_text('\n'.join(lines) + '\n')
        self.comma = FakeComma()
        self.hf = FakeHf()
        jobs = JobManager(self.repo, gpu_check=lambda: '', external_groups=set)
        self.accounts = Accounts(check_hf=fake_hf_check, check_wandb=fake_wandb_check)
        fast_scrypt(self)
        self.auth = AdminAuth(self.repo / 'outputs' / 'web' / 'auth', TOKEN)
        self.session = None
        if self.with_admin:
            self.session = self.auth.setup(TOKEN, ADMIN, PASSWORD, 'test')
        app = App(self.repo, comma=self.comma, hf=self.hf, jobs=jobs, accounts=self.accounts, auth=self.auth)
        self.server, self.app = create_server(self.repo, host='127.0.0.1', port=0, setup_code=TOKEN,
                                              local_only=self.local_only, app=app)
        self.port = self.server.server_port
        self.cookie = 'distill_web_' + str(self.port) + '=' + str(self.session)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        for job in self.app.jobs.list_jobs():
            if job['state'] == 'running':
                try:
                    self.app.jobs.stop(job['id'])
                except Exception:
                    pass
        self.server.shutdown()
        self.server.server_close()
        self.directory.cleanup()

    def request(self, method, path, body=None, headers=None, auth=True):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        all_headers = {}
        if auth:
            all_headers['Cookie'] = self.cookie
        if headers:
            for name, value in headers.items():
                all_headers[name] = value
        connection.request(method, path, body=body, headers=all_headers)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response, data

    def post(self, path, payload, headers=None):
        all_headers = {'Content-Type': 'application/json'}
        if headers:
            for name, value in headers.items():
                all_headers[name] = value
        return self.request('POST', path, json.dumps(payload).encode(), all_headers)

    def raw(self, text: bytes) -> bytes:
        with socket.create_connection(('127.0.0.1', self.port), timeout=10) as connection:
            connection.sendall(text)
            chunks = []
            while True:
                try:
                    chunk = connection.recv(65536)
                except TimeoutError:
                    break
                if not chunk:
                    break
                chunks.append(chunk)
        return b''.join(chunks)


class SignInTest(ServerTestCase):
    def cookie_of(self, response):
        header = response.getheader('Set-Cookie')
        self.assertIsNotNone(header)
        return header.split(';')[0]

    def test_without_a_session_only_the_sign_in_page(self):
        response, body = self.request('GET', '/', auth=False)
        self.assertEqual(response.status, 200)
        self.assertIn('/static/login.js', body.decode())
        self.assertNotIn('/static/app.js', body.decode())
        self.assertIn("script-src 'nonce-", response.getheader('Content-Security-Policy'))
        for name in ('login.js', 'app.css', 'favicon.svg'):
            response, body = self.request('GET', '/static/' + name, auth=False)
            self.assertEqual(response.status, 200, name)
        response, body = self.request('GET', '/static/app.js', auth=False)
        self.assertEqual(response.status, 401)
        for path in ('/api/status', '/api/settings', '/api/jobs', '/video/clip.mp4', '/thumb/' + '0' * 20 + '.jpg',
                     '/download/android'):
            response, body = self.request('GET', path, auth=False)
            self.assertEqual(response.status, 401, path)
        response, body = self.request('GET', '/api/status', auth=False)
        self.assertEqual(json.loads(body)['error'], 'login_required')
        response, body = self.post('/api/run/selftest', {})
        self.assertNotEqual(response.status, 401)
        response, body = self.request('POST', '/api/run/selftest', b'{}', {'Content-Type': 'application/json'},
                                      auth=False)
        self.assertEqual(response.status, 401)
        response, body = self.request('GET', '/api/auth/state', auth=False)
        self.assertEqual(json.loads(body), {'configured': True, 'signed_in': False, 'api': 1})

    def test_sign_in_sets_a_strict_cookie(self):
        response, body = self.request('POST', '/api/auth/login',
                                      json.dumps({'user': ADMIN, 'password': PASSWORD}).encode(),
                                      {'Content-Type': 'application/json'}, auth=False)
        self.assertEqual(response.status, 200)
        header = response.getheader('Set-Cookie')
        self.assertIn('; HttpOnly', header)
        self.assertIn('; SameSite=Strict', header)
        self.assertIn('; Path=/', header)
        self.assertNotIn('Secure', header)
        self.assertNotIn(PASSWORD.encode(), body)
        cookie = self.cookie_of(response)
        response, body = self.request('GET', '/api/status', headers={'Cookie': cookie}, auth=False)
        self.assertEqual(response.status, 200)
        response, body = self.request('GET', '/api/auth/state', headers={'Cookie': cookie}, auth=False)
        state = json.loads(body)
        self.assertEqual((state['signed_in'], state['user'], state['sessions']), (True, ADMIN, 2))
        # The server keeps only a hash of the session.
        stored = (self.repo / 'outputs' / 'web' / 'auth' / 'sessions.json').read_text()
        self.assertNotIn(cookie.split('=', 1)[1], stored)
        admin = (self.repo / 'outputs' / 'web' / 'auth' / 'admin.json').read_text()
        self.assertNotIn(PASSWORD, admin)
        self.assertEqual(os.stat(self.repo / 'outputs' / 'web' / 'auth' / 'admin.json').st_mode & 0o777, 0o600)

    def test_secure_cookie_behind_https(self):
        response, body = self.request('POST', '/api/auth/login',
                                      json.dumps({'user': ADMIN, 'password': PASSWORD}).encode(),
                                      {'Content-Type': 'application/json', 'X-Forwarded-Proto': 'https'}, auth=False)
        self.assertEqual(response.status, 200)
        self.assertIn('; Secure', response.getheader('Set-Cookie'))

    def test_wrong_name_or_password_and_lockout(self):
        for user, password in ((ADMIN, 'wrong password 1'), ('nobody', PASSWORD), (ADMIN, ''), (ADMIN, None)):
            response, body = self.request('POST', '/api/auth/login',
                                          json.dumps({'user': user, 'password': password}).encode(),
                                          {'Content-Type': 'application/json'}, auth=False)
            self.assertEqual(response.status, 401, (user, password))
            self.assertEqual(json.loads(body)['message'], web.auth.WRONG_LOGIN)
            self.assertIsNone(response.getheader('Set-Cookie'))
        response, body = self.request('POST', '/api/auth/login',
                                      json.dumps({'user': ADMIN, 'password': 'wrong again'}).encode(),
                                      {'Content-Type': 'application/json'}, auth=False)
        self.assertEqual(response.status, 401)
        # Five wrong answers: even the right password is refused for a while.
        response, body = self.request('POST', '/api/auth/login',
                                      json.dumps({'user': ADMIN, 'password': PASSWORD}).encode(),
                                      {'Content-Type': 'application/json'}, auth=False)
        self.assertEqual(response.status, 429)
        self.assertIsNone(response.getheader('Set-Cookie'))
        # A browser already signed in keeps working.
        response, body = self.request('GET', '/api/status')
        self.assertEqual(response.status, 200)

    def test_sign_in_from_another_origin_is_refused(self):
        response, body = self.request('POST', '/api/auth/login',
                                      json.dumps({'user': ADMIN, 'password': PASSWORD}).encode(),
                                      {'Content-Type': 'application/json', 'Origin': 'http://evil.example'},
                                      auth=False)
        self.assertEqual(response.status, 403)
        response, body = self.request('POST', '/api/auth/login', b'user=admin&password=x',
                                      {'Content-Type': 'application/x-www-form-urlencoded'}, auth=False)
        self.assertEqual(response.status, 415)

    def test_bad_or_foreign_cookies(self):
        headers = {'Cookie': 'nnslr_review_8765=zzz; other; =x; ' + self.cookie}
        response, body = self.request('GET', '/api/status', headers=headers, auth=False)
        self.assertEqual(response.status, 200)
        # The cookie of another port is not ours.
        headers = {'Cookie': 'distill_web_1=' + self.session}
        response, body = self.request('GET', '/api/status', headers=headers, auth=False)
        self.assertEqual(response.status, 401)
        for value in (self.session[:-1], TOKEN, '', 'x' * 500):
            headers = {'Cookie': 'distill_web_' + str(self.port) + '=' + value}
            response, body = self.request('GET', '/api/status', headers=headers, auth=False)
            self.assertEqual(response.status, 401, value)
        # The old access-token header opens nothing.
        response, body = self.request('GET', '/api/status', headers={'X-Distill-Token': TOKEN}, auth=False)
        self.assertEqual(response.status, 401)

    def test_old_token_links_open_nothing(self):
        response, body = self.request('GET', '/?token=' + TOKEN, auth=False)
        self.assertEqual(response.status, 303)
        self.assertEqual(response.getheader('Location'), '/')
        self.assertIsNone(response.getheader('Set-Cookie'))
        response, body = self.request('GET', '/api/status?token=' + TOKEN, auth=False)
        self.assertEqual(response.getheader('Location'), '/api/status')
        for path in ('/a\\b?token=x', '/%2F%2Fevil?token=x', '/x%20y?token=x', '/%5C%5Cevil?token=x'):
            response, body = self.request('GET', path, auth=False)
            self.assertEqual(response.status, 303, path)
            self.assertEqual(response.getheader('Location'), '/', path)
        # http.server itself turns a leading "//" into "/": the target is a path of this site.
        response, body = self.request('GET', '//evil.example/?token=x', auth=False)
        location = response.getheader('Location')
        self.assertTrue(location.startswith('/') and not location.startswith('//'), location)

    def test_sign_out(self):
        response, body = self.post('/api/auth/logout', {})
        self.assertEqual(response.status, 200)
        self.assertIn('Max-Age=0', response.getheader('Set-Cookie'))
        response, body = self.request('GET', '/api/status')
        self.assertEqual(response.status, 401)
        response, body = self.request('GET', '/api/auth/logout')
        self.assertEqual(response.status, 401)

    def test_sign_out_everywhere(self):
        other = self.auth.login(ADMIN, PASSWORD, 'test')
        response, body = self.post('/api/auth/logout-all', {})
        self.assertEqual(response.status, 200)
        for value in (self.session, other):
            headers = {'Cookie': 'distill_web_' + str(self.port) + '=' + value}
            response, body = self.request('GET', '/api/status', headers=headers, auth=False)
            self.assertEqual(response.status, 401)

    def test_change_password(self):
        other = self.auth.login(ADMIN, PASSWORD, 'test')
        response, body = self.post('/api/auth/password', {'current': 'not it', 'password': 'a brand new pass'})
        self.assertEqual(response.status, 403)
        response, body = self.post('/api/auth/password', {'current': PASSWORD, 'password': 'short'})
        self.assertEqual(response.status, 400)
        response, body = self.post('/api/auth/password', {'current': PASSWORD, 'password': 'a brand new pass'})
        self.assertEqual(response.status, 200)
        fresh = self.cookie_of(response)
        # Every other browser, and the old cookie of this one, are signed out.
        for value in (self.session, other):
            headers = {'Cookie': 'distill_web_' + str(self.port) + '=' + value}
            response, body = self.request('GET', '/api/status', headers=headers, auth=False)
            self.assertEqual(response.status, 401)
        response, body = self.request('GET', '/api/status', headers={'Cookie': fresh}, auth=False)
        self.assertEqual(response.status, 200)
        with self.assertRaises(web.settings.WebError):
            self.auth.login(ADMIN, PASSWORD, 'test2')
        self.assertTrue(self.auth.login(ADMIN, 'a brand new pass', 'test2'))

    def test_sessions_survive_a_restart(self):
        again = AdminAuth(self.repo / 'outputs' / 'web' / 'auth', None)
        self.assertTrue(again.session_valid(self.session))
        self.assertTrue(again.configured())


class FirstVisitTest(ServerTestCase):
    with_admin = False

    def setup_call(self, path, payload, headers=None):
        all_headers = {'Content-Type': 'application/json'}
        all_headers.update(headers or {})
        return self.request('POST', path, json.dumps(payload).encode(), all_headers, auth=False)

    def test_state_and_nothing_open(self):
        response, body = self.request('GET', '/api/auth/state', auth=False)
        self.assertEqual(json.loads(body), {'configured': False, 'signed_in': False, 'api': 1})
        response, body = self.request('GET', '/api/status', auth=False)
        self.assertEqual(response.status, 401)
        response, body = self.setup_call('/api/auth/login', {'user': ADMIN, 'password': PASSWORD})
        self.assertEqual(response.status, 409)

    def test_setup_needs_the_code(self):
        response, body = self.setup_call('/api/auth/setup/check', {'code': 'wrong'})
        self.assertEqual(response.status, 403)
        response, body = self.setup_call('/api/auth/setup', {'code': 'wrong', 'user': ADMIN, 'password': PASSWORD})
        self.assertEqual(response.status, 403)
        self.assertFalse(self.auth.configured())
        response, body = self.setup_call('/api/auth/setup/check', {'code': TOKEN})
        self.assertEqual(response.status, 200)
        self.assertIsNone(response.getheader('Set-Cookie'))
        for user, password in (('bad name', PASSWORD), ('', PASSWORD), (ADMIN, 'short'), (ADMIN, 'a' * 20),
                               ('longpassword', 'LongPassword')):
            response, body = self.setup_call('/api/auth/setup', {'code': TOKEN, 'user': user, 'password': password})
            self.assertEqual(response.status, 400, (user, password))
        response, body = self.setup_call('/api/auth/setup', {'code': TOKEN, 'user': ADMIN, 'password': PASSWORD})
        self.assertEqual(response.status, 200)
        cookie = response.getheader('Set-Cookie').split(';')[0]
        response, body = self.request('GET', '/api/status', headers={'Cookie': cookie}, auth=False)
        self.assertEqual(response.status, 200)
        # Once created, the code opens nothing and cannot create it again.
        response, body = self.setup_call('/api/auth/setup', {'code': TOKEN, 'user': 'other', 'password': PASSWORD})
        self.assertEqual(response.status, 409)
        self.assertEqual(self.auth.username(), ADMIN)

    def test_setup_code_attempts_are_limited(self):
        for _ in range(web.auth.FAILS_PER_ADDRESS):
            response, body = self.setup_call('/api/auth/setup/check', {'code': 'wrong'})
            self.assertEqual(response.status, 403)
        response, body = self.setup_call('/api/auth/setup/check', {'code': TOKEN})
        self.assertEqual(response.status, 429)


class PageTest(ServerTestCase):
    def test_page_has_a_fresh_nonce_and_strict_headers(self):
        nonces = []
        for _ in range(2):
            response, body = self.request('GET', '/')
            self.assertEqual(response.status, 200)
            csp = response.getheader('Content-Security-Policy')
            match = re.search(r"script-src 'nonce-([A-Za-z0-9_-]+)'", csp)
            self.assertIsNotNone(match)
            nonce = match.group(1)
            self.assertIn('<script nonce="' + nonce + '" src="/static/app.js">', body.decode())
            self.assertNotIn('__DISTILL_NONCE__', body.decode())
            self.assertIn("media-src 'self'", csp)
            self.assertIn("img-src 'self' data:", csp)
            self.assertIn("default-src 'none'", csp)
            self.assertIn("frame-ancestors 'none'", csp)
            self.assertEqual(response.getheader('X-Frame-Options'), 'DENY')
            self.assertEqual(response.getheader('Cache-Control'), 'no-store')
            self.assertEqual(response.getheader('X-Content-Type-Options'), 'nosniff')
            nonces.append(nonce)
        self.assertNotEqual(nonces[0], nonces[1])

    def test_sections_of_the_page(self):
        """The tabs of the page and of app.js match; the Comma tab is gone: Publish installs on the comma.
        Settings is not in the bar: a gear at the bottom of the screen opens it."""
        response, body = self.request('GET', '/')
        page = body.decode()
        tabs = re.findall(r'data-tab="([a-z]+)"', page)
        self.assertEqual(tabs, ['status', 'preprocessing', 'training', 'evaluation', 'compare', 'publish', 'settings'])
        for tab in tabs:
            self.assertIn('id="panel-' + tab + '"', page)
        bar = page[page.index('<nav class="tabs"'):page.index('</nav>')]
        self.assertEqual(re.findall(r'data-tab="([a-z]+)"', bar), tabs[:-1])
        self.assertIn('<button class="gear" type="button" id="tab-settings" data-tab="settings"', page)
        # Every section that starts a job is a wizard, as Training.
        for wizard in ('train', 'prep', 'eval', 'cmp', 'publish'):
            self.assertIn('id="' + wizard + '-step-1"', page)
            self.assertIn('id="' + wizard + '-back"', page)
            self.assertIn('id="' + wizard + '-dots"', page)
        self.assertNotIn('id="panel-comma"', page)
        self.assertNotIn('data-tab="comma"', page)
        response, body = self.request('GET', '/static/app.js')
        script = body.decode()
        match = re.search(r'const TABS = \[([^\]]*)\]', script)
        self.assertIsNotNone(match)
        self.assertEqual(re.findall(r"'([a-z]+)'", match.group(1)), tabs)
        # Old bookmarks of the Comma tab land on Publish, "On your comma".
        self.assertIn("if (name === 'comma')", script)
        # Publish holds everything the Comma tab did, next to the Hugging Face publish.
        for element in ('publish-run-picker', 'dest-comma', 'dest-hf', 'btn-comma-check', 'btn-install', 'btn-restore',
                        'btn-publish-dry', 'btn-publish', 'hf-models'):
            self.assertEqual(page.count('id="' + element + '"'), 1, element)
        # No file path of the comma on the page.
        self.assertNotIn('/data/driving', page + script)

    def test_static_files(self):
        response, body = self.request('GET', '/static/app.js')
        self.assertEqual(response.status, 200)
        self.assertIn('javascript', response.getheader('Content-Type'))
        response, body = self.request('GET', '/static/app.css')
        self.assertEqual(response.status, 200)
        for path in ('/static/index.html', '/static/../server.py', '/static/evil.js', '/web/server.py'):
            response, body = self.request('GET', path)
            self.assertEqual(response.status, 404, path)

    def test_status_shape(self):
        response, body = self.request('GET', '/api/status')
        status = json.loads(body)
        for key in ('system', 'activities', 'jobs', 'comma', 'training', 'actions', 'code', 'limits'):
            self.assertIn(key, status)
        self.assertEqual(status['actions']['localize']['allowed'], False)
        self.assertIn('this version of the code', status['actions']['localize']['reason'])
        self.assertNotIn('selftest', status['actions'])

    def test_runs_and_metrics(self):
        response, body = self.request('GET', '/api/runs')
        runs = json.loads(body)['runs']
        self.assertEqual(runs[0]['name'], RUN)
        self.assertEqual(runs[0]['status'], 'finished')
        self.assertEqual(runs[0]['steps_done'], 20)
        response, body = self.request('GET', '/api/runs/' + RUN + '/metrics?points=10')
        series = json.loads(body)
        self.assertEqual(series['bucket'], 2)
        self.assertEqual(len(series['steps']), 10)
        self.assertEqual(series['source'], 'train.log')
        for path in ('/api/runs/..%2F..%2Fetc/metrics', '/api/runs/2026-01-01_0000/metrics', '/api/runs/x/metrics'):
            response, body = self.request('GET', path)
            self.assertEqual(response.status, 404, path)

    def test_other_gets(self):
        for path in ('/api/data', '/api/eval', '/api/compare', '/api/hf', '/api/jobs'):
            response, body = self.request('GET', path)
            self.assertEqual(response.status, 200, path)
            json.loads(body)
        response, body = self.request('GET', '/api/run/localize')
        self.assertEqual(response.status, 405)
        response, body = self.request('GET', '/api/jobs/../../etc/log')
        self.assertEqual(response.status, 404)
        response, body = self.request('GET', '/nope')
        self.assertEqual(response.status, 404)


class PostTest(ServerTestCase):
    def test_json_only(self):
        response, body = self.request('POST', '/api/run/localize', b'{}')
        self.assertEqual(response.status, 415)
        response, body = self.request('POST', '/api/run/localize', b'{}', {'Content-Type': 'text/plain'})
        self.assertEqual(response.status, 415)
        response, body = self.request('POST', '/api/run/localize', b'a=1',
                                      {'Content-Type': 'application/x-www-form-urlencoded'})
        self.assertEqual(response.status, 415)

    def test_other_origins_are_refused(self):
        response, body = self.post('/api/run/localize', {}, {'Origin': 'http://evil.example'})
        self.assertEqual(response.status, 403)
        self.assertEqual(json.loads(body)['error'], 'cross_origin')
        response, body = self.post('/api/run/localize', {}, {'Sec-Fetch-Site': 'cross-site'})
        self.assertEqual(response.status, 403)
        response, body = self.post('/api/run/localize', {}, {'Sec-Fetch-Site': 'same-site'})
        self.assertEqual(response.status, 403)
        # Same origin passes the check (and then stops on the missing helper of this test repo).
        response, body = self.post('/api/run/localize', {}, {'Origin': 'http://127.0.0.1:' + str(self.port),
                                                             'Sec-Fetch-Site': 'same-origin'})
        self.assertEqual(response.status, 409)
        self.assertEqual(json.loads(body)['error'], 'script_missing')

    def test_bad_bodies(self):
        for body in (b'{', b'[]', b'"x"', b'{"a": NaN}', b'{"a": 1, "a": 2}', b'\xff\xfe'):
            response, data = self.request('POST', '/api/run/localize', body, {'Content-Type': 'application/json'})
            self.assertEqual(response.status, 400, body)
        response, data = self.request('POST', '/api/run/localize', b'{"x": "' + b'a' * 70000 + b'"}',
                                      {'Content-Type': 'application/json'})
        self.assertEqual(response.status, 413)

    def test_length_required(self):
        head = ('POST /api/run/localize HTTP/1.1\r\nHost: 127.0.0.1:' + str(self.port) + '\r\nCookie: ' + self.cookie
                + '\r\nContent-Type: application/json\r\nConnection: close\r\n')
        answer = self.raw((head + '\r\n').encode())
        self.assertIn(b' 411 ', answer.split(b'\r\n')[0])
        answer = self.raw((head + 'Transfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\n\r\n').encode())
        self.assertIn(b' 411 ', answer.split(b'\r\n')[0])

    def test_unknown_action_and_parameter(self):
        response, body = self.post('/api/run/format_disk', {})
        self.assertEqual(response.status, 404)
        write_task(self.repo, 'localize.py', 'print("ok")')
        response, body = self.post('/api/run/localize', {'segment': '../../etc'})
        self.assertEqual(response.status, 400)
        response, body = self.post('/api/run/selftest', {})
        self.assertEqual(response.status, 404)

    def test_change_segments_at_a_time_of_a_running_localization(self):
        # A fake localization: reports its state, then waits for the site to ask for 5 at a time.
        write_task(self.repo, 'localize.py', '''
import json, os, time
from pathlib import Path
control = Path(os.environ['DISTILL_JOB_CONTROL'])
state = control.with_name(control.name.replace('.control.json', '.state.json'))
state.write_text(json.dumps({'jobs': 8, 'auto': 8, 'running': 8}))
print('PROGRESS 1/10', flush=True)
for _ in range(200):
    try:
        wanted = json.loads(control.read_text())['jobs']
    except (OSError, ValueError, KeyError):
        wanted = None
    if wanted == 5:
        print('now 5 at a time', flush=True)
        break
    time.sleep(0.05)
''')
        response, body = self.post('/api/run/localize', {})
        self.assertEqual(response.status, 200, body)
        job_id = json.loads(body)['job']['id']
        deadline = time.monotonic() + 10
        parallel = None
        while time.monotonic() < deadline and parallel is None:
            response, body = self.request('GET', '/api/jobs/' + job_id)
            parallel = json.loads(body).get('parallel')
            time.sleep(0.05)
        self.assertEqual(parallel, {'jobs': 8, 'auto': 8, 'running': 8})
        for bad in ({'jobs': 0}, {'jobs': 17}, {'jobs': '5'}, {'jobs': True}, {'jobs': 5, 'x': 1}, {}):
            response, body = self.post('/api/jobs/' + job_id + '/parallel', bad)
            self.assertEqual(response.status, 400, bad)
        response, body = self.post('/api/jobs/' + job_id + '/parallel', {'jobs': 5})
        self.assertEqual(response.status, 200, body)
        # The answer already shows the new number, before the job has taken it.
        self.assertEqual(json.loads(body)['job']['parallel']['jobs'], 5)
        response, body = self.request('GET', '/api/jobs/' + job_id + '/parallel')
        self.assertEqual(response.status, 405)
        text = ''
        while time.monotonic() < deadline and 'finished with code' not in text:
            response, body = self.request('GET', '/api/jobs/' + job_id + '/log?offset=0')
            text = json.loads(body)['text']
            time.sleep(0.05)
        self.assertIn('segments at a time set to 5 from the site', text)
        self.assertIn('now 5 at a time', text)
        # Finished: no more changes.
        response, body = self.post('/api/jobs/' + job_id + '/parallel', {'jobs': 6})
        self.assertEqual(response.status, 409)
        response, body = self.post('/api/jobs/nonsense/parallel', {'jobs': 6})
        self.assertEqual(response.status, 404)

    def test_start_a_job_and_read_its_log(self):
        write_task(self.repo, 'localize.py', 'print("localizzo", len(sys.argv) - 1)\nprint("PROGRESS 1/1")')
        response, body = self.post('/api/run/localize', {})
        self.assertEqual(response.status, 200, body)
        job = json.loads(body)['job']
        self.assertEqual(job['action'], 'localize')
        deadline = time.monotonic() + 10
        text = ''
        while time.monotonic() < deadline:
            response, body = self.request('GET', '/api/jobs/' + job['id'] + '/log?offset=0')
            text = json.loads(body)['text']
            if 'finished with code' in text:
                break
            time.sleep(0.1)
        self.assertIn('localizzo 0', text)
        self.assertIn('== finished with code 0', text)
        response, body = self.request('GET', '/api/jobs/' + job['id'])
        self.assertEqual(json.loads(body)['state'], 'ok')
        response, body = self.post('/api/jobs/' + job['id'] + '/stop', {})
        self.assertEqual(response.status, 409)
        response, body = self.post('/api/jobs/x/stop', {})
        self.assertEqual(response.status, 404)

    def test_publish_needs_a_free_name_and_a_good_dry_run(self):
        write_task(self.repo, 'publish.py', 'print("publish", *sys.argv[1:])')
        params = {'run': RUN, 'name': 'nuovo', 'stable': False, 'confirm': 'nuovo'}
        # Without a Hugging Face login the publish is refused before anything else.
        response, body = self.post('/api/run/publish', params)
        self.assertEqual(response.status, 409)
        self.assertEqual(json.loads(body)['error'], 'hf_not_connected')
        self.assertIn('Settings', json.loads(body)['message'])
        response, body = self.request('GET', '/api/status')
        self.assertIn('connect it in Settings', json.loads(body)['actions']['publish']['reason'])
        (self.home / 'hf').mkdir()
        (self.home / 'hf' / 'token').write_text(HF_TOKEN)
        response, body = self.post('/api/run/publish', {'run': RUN, 'name': '2026_10_21_1544', 'stable': False,
                                                        'confirm': '2026_10_21_1544'})
        self.assertEqual(response.status, 409)
        self.assertEqual(json.loads(body)['error'], 'name_taken')
        # Different spelling, same slug on Hugging Face.
        response, body = self.post('/api/run/publish', {'run': RUN, 'name': '2026-10-21_1544',
                                                        'confirm': '2026-10-21_1544'})
        self.assertEqual(json.loads(body)['error'], 'name_taken')
        response, body = self.post('/api/run/publish', params)
        self.assertEqual(response.status, 409)
        self.assertEqual(json.loads(body)['error'], 'dry_run_missing')
        response, body = self.post('/api/run/publish_dry', {'run': RUN, 'name': 'nuovo', 'stable': False})
        self.assertEqual(response.status, 200, body)
        dry = json.loads(body)['job']
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.app.jobs.get(dry['id'])['state'] != 'running':
                break
            time.sleep(0.1)
        self.assertEqual(self.app.jobs.get(dry['id'])['state'], 'ok')
        time.sleep(1.1)
        response, body = self.post('/api/run/publish', {'run': RUN, 'name': 'nuovo', 'stable': True,
                                                        'confirm': 'nuovo'})
        self.assertEqual(json.loads(body)['error'], 'dry_run_missing')
        response, body = self.post('/api/run/publish', params)
        self.assertEqual(response.status, 200, body)
        job = json.loads(body)['job']
        self.assertEqual(job['params'], {'run': RUN, 'name': 'nuovo', 'stable': False})

    def test_install_needs_the_comma_offroad(self):
        write_task(self.repo, 'install.py', 'print("install")')
        self.comma.state['reachable'] = False
        self.comma.state['error'] = 'comma not reachable over ssh'
        response, body = self.post('/api/run/install', {'run': RUN, 'confirm': 'INSTALL'})
        self.assertEqual(json.loads(body)['error'], 'comma_unreachable')
        self.comma.state['reachable'] = True
        self.comma.state['onroad'] = True
        response, body = self.post('/api/run/restore', {'confirm': 'RESTORE'})
        self.assertEqual(json.loads(body)['error'], 'comma_onroad')
        self.comma.state['onroad'] = False
        self.comma.state['bundle_default'] = False
        response, body = self.post('/api/run/install', {'run': RUN, 'confirm': 'INSTALL'})
        self.assertEqual(json.loads(body)['error'], 'comma_bundle')
        response, body = self.request('GET', '/api/status')
        reason = json.loads(body)['actions']['install']['reason']
        self.assertIn('Models menu', reason)
        # The comma's own Settings, not the Settings tab of the site.
        self.assertIn('on the comma screen', reason)
        self.comma.state['bundle_default'] = True
        response, body = self.post('/api/run/install', {'run': RUN, 'confirm': 'installa'})
        self.assertEqual(response.status, 400)
        response, body = self.post('/api/run/install', {'run': RUN, 'confirm': 'INSTALL'})
        self.assertEqual(response.status, 200, body)

    def test_comma_check(self):
        response, body = self.post('/api/comma/check', {})
        self.assertEqual(response.status, 200)
        self.assertTrue(json.loads(body)['comma']['reachable'])


class AppApiTest(ServerTestCase):
    """The API of Distill for Android (web/API.md): plain HTTP requests from the phone, with no Origin
    or Sec-Fetch-Site header, a session cookie it keeps itself, and the fields it reads."""

    def app_call(self, method, path, payload=None, cookie=None):
        headers = {'Accept': 'application/json'}
        body = None
        if payload is not None:
            headers['Content-Type'] = 'application/json'
            body = json.dumps(payload).encode()
        if cookie is not None:
            headers['Cookie'] = cookie
        response, raw = self.request(method, path, body, headers, auth=False)
        return response, json.loads(raw)

    def test_sign_in_work_and_sign_out_as_the_app(self):
        response, state = self.app_call('GET', '/api/auth/state')
        self.assertEqual((state['configured'], state['signed_in'], state['api']), (True, False, 1))
        response, answer = self.app_call('GET', '/api/status')
        self.assertEqual((response.status, answer['error']), (401, 'login_required'))
        response, answer = self.app_call('POST', '/api/auth/login', {'user': ADMIN, 'password': 'wrong password'})
        self.assertEqual((response.status, answer['message']), (401, web.auth.WRONG_LOGIN))
        response, answer = self.app_call('POST', '/api/auth/login', {'user': ADMIN, 'password': PASSWORD})
        self.assertEqual(response.status, 200)
        cookie = response.getheader('Set-Cookie').split(';')[0]
        response, state = self.app_call('GET', '/api/auth/state', cookie=cookie)
        self.assertEqual((state['signed_in'], state['user'], state['api']), (True, ADMIN, 1))

        # What the screens of the app read (DistillApi.kt).
        response, status = self.app_call('GET', '/api/status', cookie=cookie)
        self.assertEqual(response.status, 200)
        for key in ('system', 'training', 'jobs', 'comma', 'actions', 'code'):
            self.assertIn(key, status)
        for key in ('cpu', 'memory', 'disk', 'gpus'):
            self.assertIn(key, status['system'])
        for action in ('train', 'eval', 'install', 'restore'):
            self.assertEqual(set(status['actions'][action]) >= {'allowed', 'reason'}, True, action)
        for key in ('reachable', 'onroad', 'bundle', 'bundle_default', 'backup', 'host', 'error', 'checked'):
            self.assertIn(key, status['comma'])
        response, runs = self.app_call('GET', '/api/runs', cookie=cookie)
        for key in ('name', 'status', 'steps_done', 'steps_total', 'has_pt', 'has_training_pt', 'resumable', 'started',
                    'config', 'last', 'avg_step_time'):
            self.assertIn(key, runs['runs'][0])
        response, results = self.app_call('GET', '/api/eval', cookie=cookie)
        self.assertIsInstance(results['results'], list)
        response, hf = self.app_call('GET', '/api/hf', cookie=cookie)
        for key in ('ok', 'stable', 'models', 'error'):
            self.assertIn(key, hf)

        # Start, follow and stop a job; ask the comma.
        write_task(self.repo, 'localize.py', 'import time\ntime.sleep(30)')
        response, answer = self.app_call('POST', '/api/run/localize', {}, cookie=cookie)
        self.assertEqual(response.status, 200, answer)
        job = answer['job']
        self.assertTrue(job['label'])
        response, status = self.app_call('GET', '/api/status', cookie=cookie)
        running = [item for item in status['jobs'] if item['id'] == job['id']]
        self.assertEqual(running[0]['state'], 'running')
        for key in ('label', 'summary', 'progress', 'stoppable'):
            self.assertIn(key, running[0])
        response, answer = self.app_call('POST', '/api/jobs/' + job['id'] + '/stop', {}, cookie=cookie)
        self.assertEqual(response.status, 200, answer)
        response, answer = self.app_call('POST', '/api/comma/check', {}, cookie=cookie)
        self.assertTrue(answer['comma']['reachable'])

        response, answer = self.app_call('POST', '/api/auth/logout', {}, cookie=cookie)
        self.assertEqual(response.status, 200)
        response, answer = self.app_call('GET', '/api/status', cookie=cookie)
        self.assertEqual(response.status, 401)


class VideoTest(ServerTestCase):
    def setUp(self):
        super().setUp()
        folder = self.repo / 'outputs' / 'compare'
        folder.mkdir(parents=True)
        self.content = bytes(range(250)) * 4
        (folder / 'clip_1.mp4').write_bytes(self.content)
        secret = self.repo / 'secret.mp4'
        secret.write_bytes(b'secret')
        os.symlink(secret, folder / 'link.mp4')

    def test_whole_file(self):
        response, body = self.request('GET', '/video/clip_1.mp4')
        self.assertEqual(response.status, 200)
        self.assertEqual(body, self.content)
        self.assertEqual(response.getheader('Content-Type'), 'video/mp4')
        self.assertEqual(response.getheader('Accept-Ranges'), 'bytes')
        self.assertEqual(response.getheader('Content-Length'), '1000')

    def test_ranges(self):
        cases = {'bytes=0-99': (0, 99), 'bytes=900-': (900, 999), 'bytes=-50': (950, 999), 'bytes=990-2000': (990, 999),
                 'bytes=0-0': (0, 0), 'bytes=999-999': (999, 999), 'bytes = 10 - 19': (10, 19)}
        for header, (first, last) in cases.items():
            response, body = self.request('GET', '/video/clip_1.mp4', headers={'Range': header})
            self.assertEqual(response.status, 206, header)
            self.assertEqual(response.getheader('Content-Range'), 'bytes ' + str(first) + '-' + str(last) + '/1000')
            self.assertEqual(body, self.content[first:last + 1], header)
            self.assertEqual(response.getheader('Content-Length'), str(last - first + 1))

    def test_bad_ranges(self):
        for header in ('bytes=1000-', 'bytes=5000-6000', 'bytes=-0'):
            response, body = self.request('GET', '/video/clip_1.mp4', headers={'Range': header})
            self.assertEqual(response.status, 416, header)
            self.assertEqual(response.getheader('Content-Range'), 'bytes */1000')
            self.assertEqual(body, b'')
        # Malformed or several ranges: ignored, the whole file.
        for header in ('bytes=5-2', 'bytes=0-1,5-6', 'items=0-5', 'bytes=a-b', 'bytes=-', 'bytes=1.5-3', 'bytes=²-3'):
            response, body = self.request('GET', '/video/clip_1.mp4', headers={'Range': header})
            self.assertEqual(response.status, 200, header)
            self.assertEqual(body, self.content)

    def test_head(self):
        response, body = self.request('HEAD', '/video/clip_1.mp4', headers={'Range': 'bytes=0-9'})
        self.assertEqual(response.status, 206)
        self.assertEqual(body, b'')

    def test_only_plain_names_inside_outputs_compare(self):
        for path in ('/video/link.mp4', '/video/..%2Fsecret.mp4', '/video/../secret.mp4', '/video/%2e%2e%2fsecret.mp4',
                     '/video/clip_1.MP4', '/video/clip_1.mp4.json', '/video/.hidden.mp4', '/video/'):
            response, body = self.request('GET', path)
            self.assertEqual(response.status, 404, path)
            self.assertNotEqual(body, b'secret')
        response, body = self.request('GET', '/video/clip_1.mp4', auth=False)
        self.assertEqual(response.status, 401)


class AndroidDownloadTest(ServerTestCase):
    """The app built on the server by helpers/build_android.sh, offered in Settings and at /download/android."""

    def build(self, info, name='distill-android-0.3.120.apk', content=b'PK apk'):
        folder = self.repo / 'outputs' / 'android'
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(content)
        (folder / 'latest.json').write_text(json.dumps(info))
        return folder

    def test_no_build(self):
        response, body = self.request('GET', '/api/settings')
        self.assertIsNone(json.loads(body)['android'])
        response, body = self.request('GET', '/download/android')
        self.assertEqual(response.status, 404)

    def test_the_last_build(self):
        self.build({'file': 'distill-android-0.3.120.apk', 'version': '0.3.120', 'version_code': 1120,
                    'commit': 'abc1234', 'signed_with': 'release', 'sha256': 'f' * 64, 'built': 1.5e9})
        response, body = self.request('GET', '/api/settings')
        android = json.loads(body)['android']
        self.assertEqual(android, {'file': 'distill-android-0.3.120.apk', 'size': 6, 'version': '0.3.120',
                                   'version_code': 1120, 'commit': 'abc1234', 'signed_with': 'release',
                                   'sha256': 'f' * 64, 'built': 1.5e9})
        response, body = self.request('GET', '/download/android')
        self.assertEqual(response.status, 200)
        self.assertEqual(body, b'PK apk')
        self.assertEqual(response.getheader('Content-Type'), 'application/vnd.android.package-archive')
        self.assertEqual(response.getheader('Content-Disposition'),
                         'attachment; filename="distill-android-0.3.120.apk"')

    def test_only_a_plain_apk_inside_outputs_android(self):
        secret = self.repo / 'secret.apk'
        secret.write_bytes(b'secret')
        folder = self.build({'file': 'distill-android-0.3.1.apk'}, name='other.apk')
        os.symlink(secret, folder / 'distill-android-0.3.1.apk')
        for name in ('distill-android-0.3.1.apk', '../secret.apk', 'other.apk', 'distill-android-0.3.1.apk/../x'):
            (folder / 'latest.json').write_text(json.dumps({'file': name, 'version': '1'}))
            response, body = self.request('GET', '/download/android')
            self.assertEqual(response.status, 404, name)
            self.assertNotEqual(body, b'secret')
        (folder / 'latest.json').write_text('not json')
        response, body = self.request('GET', '/api/settings')
        self.assertIsNone(json.loads(body)['android'])

    def test_wrong_field_types_are_dropped(self):
        self.build({'file': 'distill-android-0.3.120.apk', 'version': 3, 'version_code': True, 'built': 'yesterday'})
        response, body = self.request('GET', '/api/settings')
        android = json.loads(body)['android']
        self.assertEqual((android['version'], android['version_code'], android['built']), (None, None, None))


class RangeParseTest(unittest.TestCase):
    def test_parse_range(self):
        self.assertIsNone(parse_range(None, 10))
        self.assertEqual(parse_range('bytes=0-', 10), (0, 9))
        self.assertEqual(parse_range('bytes=3-5', 10), (3, 5))
        self.assertEqual(parse_range('bytes=-3', 10), (7, 9))
        self.assertEqual(parse_range('bytes=-30', 10), (0, 9))
        self.assertEqual(parse_range('bytes=8-100', 10), (8, 9))
        self.assertEqual(parse_range('bytes=10-', 10), 'unsatisfiable')
        self.assertEqual(parse_range('bytes=0-', 0), 'unsatisfiable')
        self.assertEqual(parse_range('bytes=-1', 0), 'unsatisfiable')
        self.assertIsNone(parse_range('bytes=5-3', 10))
        self.assertIsNone(parse_range('bytes=1-2, 4-5', 10))
        self.assertIsNone(parse_range('bytes=', 10))
        self.assertIsNone(parse_range('bits=1-2', 10))
        self.assertIsNone(parse_range('bytes=0x1-2', 10))


class LoopbackTest(ServerTestCase):
    local_only = True

    def test_sign_in_and_loopback_host_only(self):
        response, body = self.request('GET', '/api/status', auth=False)
        self.assertEqual(response.status, 401)
        response, body = self.request('GET', '/api/status')
        self.assertEqual(response.status, 200)
        response, body = self.request('GET', '/api/status', headers={'Host': 'localhost:' + str(self.port)})
        self.assertEqual(response.status, 200)
        response, body = self.request('GET', '/api/status', headers={'Host': 'evil.example'})
        self.assertEqual(response.status, 403)
        self.assertEqual(json.loads(body)['error'], 'host_not_allowed')


if __name__ == '__main__':
    unittest.main()
