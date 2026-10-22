"""distill-web: the HTTP server (token, cookie, redirect, same-origin POST, body limits, CSP, Range
for the videos, the publish and install pre-checks). Standard library only, no network."""
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

from web.jobs import JobManager
from web.server import App, create_server, parse_range

TOKEN = 'TestToken_1234567890abcdef'
RUN = '2026-10-21_1544'


class FakeComma:
    host = 'comma'

    def __init__(self):
        self.state = {'host': 'comma', 'checked': 1.0, 'reachable': True, 'onroad': False, 'bundle': 'Default',
                      'bundle_default': True, 'backup': False, 'error': ''}

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

    def snapshot(self, force=False):
        return {'repo': self.repo, 'url': 'https://huggingface.co/' + self.repo, 'ok': True, 'error': '',
                'checked': 1.0, 'models': list(self.models), 'releases': ['2026_10_21_1544'], 'stable': False}


def write_task(repo: Path, name: str, body: str) -> None:
    """A fake tasks/<name>.py in the test repo: the jobs run "python -m tasks.<name>" from the repo."""
    path = repo / 'tasks' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / '__init__.py').touch()
    path.write_text('import sys\n' + body + '\n')


class ServerTestCase(unittest.TestCase):
    require_token = True

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
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
        app = App(self.repo, token=TOKEN, require_token=self.require_token, comma=self.comma, hf=self.hf, jobs=jobs)
        token = TOKEN
        if not self.require_token:
            token = None
        self.server, self.app = create_server(self.repo, host='127.0.0.1', port=0, token=token,
                                              require_token=self.require_token, app=app)
        self.port = self.server.server_port
        self.cookie = 'distill_web_' + str(self.port) + '=' + TOKEN
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


class TokenTest(ServerTestCase):
    def test_pages_and_api_need_the_token(self):
        response, body = self.request('GET', '/', auth=False)
        self.assertEqual(response.status, 403)
        self.assertIn('text/html', response.getheader('Content-Type'))
        self.assertIn('Access denied', body.decode())
        response, body = self.request('GET', '/api/status', auth=False)
        self.assertEqual(response.status, 403)
        self.assertEqual(json.loads(body)['error'], 'token_required')
        response, body = self.request('GET', '/static/app.js', auth=False)
        self.assertEqual(response.status, 403)

    def test_token_in_the_query_becomes_a_cookie(self):
        response, body = self.request('GET', '/?token=' + TOKEN, auth=False)
        self.assertEqual(response.status, 303)
        self.assertEqual(response.getheader('Location'), '/')
        self.assertEqual(response.getheader('Set-Cookie'),
                         'distill_web_' + str(self.port) + '=' + TOKEN + '; Path=/; HttpOnly; SameSite=Strict')
        self.assertEqual(response.getheader('Referrer-Policy'), 'no-referrer')
        response, body = self.request('GET', '/api/status?token=' + TOKEN, auth=False)
        self.assertEqual(response.getheader('Location'), '/api/status')

    def test_redirect_target_stays_on_this_site(self):
        for path in ('/a\\b?token=' + TOKEN, '/%2F%2Fevil?token=' + TOKEN, '/x y?token=' + TOKEN,
                     '/%5C%5Cevil?token=' + TOKEN):
            response, body = self.request('GET', path.replace(' ', '%20'), auth=False)
            self.assertEqual(response.status, 303, path)
            self.assertEqual(response.getheader('Location'), '/', path)
        # http.server itself turns a leading "//" into "/": the target is a path of this site.
        response, body = self.request('GET', '//evil.example/?token=' + TOKEN, auth=False)
        self.assertEqual(response.status, 303)
        location = response.getheader('Location')
        self.assertTrue(location.startswith('/') and not location.startswith('//'), location)

    def test_wrong_token(self):
        response, body = self.request('GET', '/?token=wrong', auth=False)
        self.assertEqual(response.status, 403)
        self.assertIsNone(response.getheader('Set-Cookie'))
        response, body = self.request('GET', '/?token=' + TOKEN + 'x', auth=False)
        self.assertEqual(response.status, 403)
        response, body = self.request('GET', '/?token=', auth=False)
        self.assertEqual(response.status, 403)

    def test_token_in_the_query_of_a_post_is_refused(self):
        response, body = self.request('POST', '/api/run/localize?token=' + TOKEN, b'{}',
                                      {'Content-Type': 'application/json'}, auth=False)
        self.assertEqual(response.status, 403)
        self.assertEqual(json.loads(body)['error'], 'invalid_token')

    def test_cookie_among_other_cookies(self):
        headers = {'Cookie': 'nnslr_review_8765=zzz; other; =x; distill_web_' + str(self.port) + '=' + TOKEN}
        response, body = self.request('GET', '/api/status', headers=headers, auth=False)
        self.assertEqual(response.status, 200)
        # The cookie of another port is not ours.
        headers = {'Cookie': 'distill_web_1=' + TOKEN}
        response, body = self.request('GET', '/api/status', headers=headers, auth=False)
        self.assertEqual(response.status, 403)
        headers = {'Cookie': 'distill_web_' + str(self.port) + '=' + TOKEN[:-1]}
        response, body = self.request('GET', '/api/status', headers=headers, auth=False)
        self.assertEqual(response.status, 403)

    def test_header_token(self):
        response, body = self.request('GET', '/api/status', headers={'X-Distill-Token': TOKEN}, auth=False)
        self.assertEqual(response.status, 200)
        response, body = self.request('GET', '/api/status', headers={'X-Distill-Token': 'nope'}, auth=False)
        self.assertEqual(response.status, 403)


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
        self.assertIn('Models menu', json.loads(body)['actions']['install']['reason'])
        self.comma.state['bundle_default'] = True
        response, body = self.post('/api/run/install', {'run': RUN, 'confirm': 'installa'})
        self.assertEqual(response.status, 400)
        response, body = self.post('/api/run/install', {'run': RUN, 'confirm': 'INSTALL'})
        self.assertEqual(response.status, 200, body)

    def test_comma_check(self):
        response, body = self.post('/api/comma/check', {})
        self.assertEqual(response.status, 200)
        self.assertTrue(json.loads(body)['comma']['reachable'])


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
        self.assertEqual(response.status, 403)


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
    require_token = False

    def test_no_token_but_loopback_host_only(self):
        response, body = self.request('GET', '/api/status', auth=False)
        self.assertEqual(response.status, 200)
        response, body = self.request('GET', '/api/status', headers={'Host': 'localhost:' + str(self.port)},
                                      auth=False)
        self.assertEqual(response.status, 200)
        response, body = self.request('GET', '/api/status', headers={'Host': 'evil.example'}, auth=False)
        self.assertEqual(response.status, 403)
        self.assertEqual(json.loads(body)['error'], 'host_not_allowed')
        response, body = self.request('GET', '/?token=anything', auth=False)
        self.assertEqual(response.status, 303)
        self.assertIsNone(response.getheader('Set-Cookie'))


if __name__ == '__main__':
    unittest.main()
