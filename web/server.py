"""distill-web: the control site of openpilot.distill (standard library HTTP server).

    python -m web.server [--host 127.0.0.1] [--port 8766]

On 127.0.0.1 (the default) it needs no token: only this machine reaches it (ssh -L for the PC).
On any other address every request needs the access token DISTILL_WEB_TOKEN (repo .env, or
/etc/distill-web.env for the systemd service): open http://<ip>:8766/?token=<token> once, the
server answers with a redirect to the same page without the token and an HttpOnly SameSite=Strict
cookie (distill_web_<port>). Scripts can send the header X-Distill-Token instead.
POSTs are accepted only from the same origin, as JSON (an HTML form cannot send it), with a
Content-Length and a small body. Pages carry a strict Content-Security-Policy with a nonce.
The server never prints the token from the environment, and never logs request URLs.
"""
from __future__ import annotations

import argparse
import html
import ipaddress
import json
import os
import re
import secrets
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from web import actions as actions_module
from web.actions import ACTIONS, TEST_ACTIONS, build_job, script_missing, slug_of
from web.comma import CommaMonitor
from web.common import CODE_ROOT, RUN_RE, WebError, comma_host, hf_repo, short_text
from web.hf import HfCatalog
from web.jobs import JOB_ID_RE, JobManager, describe
from web.metrics import RunReader, running_runs_from_processes
from web.status import (Cache, activities, compare_videos, eval_results, gpu_busy_reason, segments, system_info,
                        video_path)

TOKEN_ENV = 'DISTILL_WEB_TOKEN'
TOKEN_RE = re.compile(r'^[A-Za-z0-9_-]{16,}$')
TOKEN_HEADER = 'X-Distill-Token'
DEFAULT_PORT = 8766
MAX_BODY_BYTES = 64 * 1024
# A connection that sends nothing for this many seconds is closed, and at most MAX_CONNECTIONS
# are open at once, so idle clients cannot use up threads.
REQUEST_TIMEOUT_SECONDS = 30
MAX_CONNECTIONS = 64
VIDEO_CHUNK_BYTES = 64 * 1024
STATIC_DIR = Path(__file__).resolve().parent / 'static'
# The only files served from web/static/, with their type.
STATIC_FILES = {
    'app.js': 'text/javascript; charset=utf-8',
    'app.css': 'text/css; charset=utf-8',
    'favicon.svg': 'image/svg+xml',
}
BUNDLE_REASON = ('sul comma e\' scelto un modello nel menu Modelli: rimetti Default (Impostazioni > Modelli) '
                 'e controlla di nuovo')
TOKEN_MESSAGE = 'Accesso negato: apri l\'indirizzo con il token (helpers/web.sh lo stampa).'
DEFAULT_CSP = "default-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
PROBLEM_CSP = ("default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; "
               "frame-ancestors 'none'")


def page_csp(nonce: str) -> str:
    # media-src for the comparison videos, img-src data: for the charts.
    return ("default-src 'none'; script-src 'nonce-" + nonce + "'; style-src 'self'; img-src 'self' data:; "
            "media-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")


def is_loopback(host: str) -> bool:
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host.strip('[]')).is_loopback
    except ValueError:
        return False


def access_token() -> str:
    """DISTILL_WEB_TOKEN when set (must look like a token), otherwise a new random one."""
    configured = os.environ.get(TOKEN_ENV, '')
    if configured:
        if not TOKEN_RE.fullmatch(configured):
            raise SystemExit(TOKEN_ENV + ' deve avere almeno 16 caratteri tra lettere, cifre, "-" e "_"')
        return configured
    return secrets.token_urlsafe(24)


def lan_addresses() -> list[str]:
    """IPv4 addresses of this machine that other computers can use (best effort, no traffic sent)."""
    found = []
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        infos = []
    for info in infos:
        address = info[4][0]
        if not is_loopback(address) and address not in found:
            found.append(address)
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # A UDP connect only selects the outgoing interface; nothing is sent.
        probe.connect(('10.255.255.255', 1))
        address = probe.getsockname()[0]
        if not is_loopback(address) and address not in found:
            found.append(address)
    except OSError:
        pass
    finally:
        probe.close()
    return found


def parse_range(header, size: int):
    """The byte range of a Range header for a file of `size` bytes.
    Returns None (send the whole file: no header, a malformed one, or several ranges, which
    RFC 9110 lets a server ignore), (first, last) inclusive, or 'unsatisfiable' (416)."""
    if header is None:
        return None
    match = re.fullmatch(r'\s*bytes\s*=\s*([0-9]*)\s*-\s*([0-9]*)\s*', header)
    if match is None:
        return None
    first_text = match.group(1)
    last_text = match.group(2)
    if first_text == '' and last_text == '':
        return None
    if first_text == '':
        # "bytes=-N": the last N bytes.
        length = int(last_text)
        if length == 0 or size == 0:
            return 'unsatisfiable'
        return (max(0, size - length), size - 1)
    first = int(first_text)
    if last_text == '':
        last = size - 1
    else:
        last = int(last_text)
        if last < first:
            return None
    if first >= size:
        return 'unsatisfiable'
    if last > size - 1:
        last = size - 1
    return (first, last)


def strict_json_loads(text: str):
    """json.loads without NaN/Infinity and without repeated keys."""
    def reject_constant(name):
        raise ValueError('constant ' + name + ' not allowed')

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('repeated key')
            result[key] = value
        return result

    return json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_object)


def code_version(repo: Path) -> str:
    """Short commit and branch of the checkout, read from .git without running git."""
    git = repo / '.git'
    try:
        head = (git / 'HEAD').read_text().strip()
    except OSError:
        return ''
    if not head.startswith('ref: '):
        return head[:7]
    ref = head[len('ref: '):]
    branch = ref.rsplit('/', 1)[-1]
    sha = ''
    try:
        sha = (git / ref).read_text().strip()
    except OSError:
        try:
            for line in (git / 'packed-refs').read_text().splitlines():
                parts = line.split(' ')
                if len(parts) == 2 and parts[1] == ref:
                    sha = parts[0]
        except OSError:
            pass
    if not sha:
        return branch
    return sha[:7] + ' (' + branch + ')'


class App:
    """State shared by the request threads."""

    def __init__(self, repo: Path, *, token: str | None, require_token: bool, test_actions: bool = False,
                 jobs: JobManager | None = None, comma: CommaMonitor | None = None, hf: HfCatalog | None = None):
        self.repo = repo
        self.token = token
        self.require_token = require_token
        self.test_actions = test_actions
        self.cookie_name = 'distill_web'
        if jobs is None:
            jobs = JobManager(repo, gpu_check=self._gpu_check, external_groups=self._external_groups)
        self.jobs = jobs
        if comma is None:
            comma = CommaMonitor(comma_host(repo))
        self.comma = comma
        if hf is None:
            hf = HfCatalog(hf_repo(repo))
        self.hf = hf
        self.runs = RunReader(repo)
        self._machine = Cache(3.0)

    # -- access ------------------------------------------------------------------------------------
    def token_matches(self, supplied) -> bool:
        if self.token is None or not isinstance(supplied, str):
            return False
        return secrets.compare_digest(supplied.encode('utf-8'), self.token.encode('utf-8'))

    def host_allowed(self, host_header) -> bool:
        # With a token the cookie protects every request. Without one (loopback only) the Host
        # header must be a loopback name, against DNS rebinding.
        if self.require_token:
            return True
        if not isinstance(host_header, str) or not host_header:
            return False
        name = host_header.strip().lower()
        if name.startswith('['):
            end = name.find(']')
            if end < 0:
                return False
            name = name[1:end]
        elif ':' in name:
            name = name.rsplit(':', 1)[0]
        return is_loopback(name)

    # -- data for the page -------------------------------------------------------------------------
    def machine(self) -> dict:
        def compute():
            return {'system': system_info(self.repo), 'activities': activities(self.repo)}
        return self._machine.get(compute)

    def _gpu_check(self) -> str:
        return gpu_busy_reason(self.repo)

    def _external_groups(self) -> set:
        groups = set()
        for activity in activities(self.repo):
            groups.add(activity['group'])
        return groups

    def all_actions(self) -> dict:
        result = dict(ACTIONS)
        if self.test_actions:
            for name, action in TEST_ACTIONS.items():
                result[name] = action
        return result

    def running_run_names(self, jobs: list[dict]) -> set:
        names = running_runs_from_processes(self.repo)
        for job in jobs:
            if job['state'] == 'running' and job.get('run'):
                names.add(job['run'])
        return names

    def runs_payload(self, jobs: list[dict] | None = None) -> list[dict]:
        if jobs is None:
            jobs = self.jobs.list_jobs()
        running = self.running_run_names(jobs)
        result = []
        for name in self.runs.run_names():
            summary = self.runs.summary(name, name in running)
            if summary is not None:
                result.append(summary)
        return result

    def action_states(self, machine: dict, jobs: list[dict], comma_state: dict, runs: list[dict]) -> dict:
        running = set()
        for job in jobs:
            if job['state'] == 'running':
                running.add(job['group'])
        for activity in machine['activities']:
            if activity['group'] != 'other':
                running.add(activity['group'])
        gpu_reasons = []
        gpu = machine['system']['gpu']
        if gpu is not None and gpu['processes']:
            gpu_reasons.append(str(len(gpu['processes'])) + ' processi sulla GPU')
        for activity in machine['activities']:
            if activity['uses_gpu']:
                gpu_reasons.append(activity['label'])
        finished_runs = 0
        for run in runs:
            if run['has_pt']:
                finished_runs = finished_runs + 1

        states = {}
        for action_id, action in self.all_actions().items():
            reason = script_missing(self.repo, action_id, self.test_actions)
            if not reason:
                reason = self.jobs.conflict_reason(action.group, running)
            if not reason and action.needs_gpu and gpu_reasons:
                reason = 'la GPU e\' occupata (' + ', '.join(gpu_reasons) + ')'
            if not reason and action.needs_comma_offroad:
                if comma_state.get('reachable') is not True:
                    reason = 'il comma non e\' raggiungibile'
                elif comma_state.get('onroad') is True:
                    reason = 'il comma e\' onroad: spegni l\'auto'
                elif action_id == 'install' and comma_state.get('bundle_default') is False:
                    reason = BUNDLE_REASON
            if not reason and action_id in ('export', 'publish_dry', 'publish', 'install') and not finished_runs:
                reason = 'nessun run finito (con supercombo.pt)'
            states[action_id] = {'label': action.label, 'group': action.group, 'allowed': not reason,
                                 'reason': reason, 'needs_gpu': action.needs_gpu}
        return states

    def status_payload(self) -> dict:
        machine = self.machine()
        jobs = self.jobs.list_jobs()
        comma_state = self.comma.snapshot()
        runs = self.runs_payload(jobs)
        training = []
        for run in runs:
            if run['status'] == 'running':
                training.append(run)
        return {'time': time.time(), 'code': code_version(self.repo), 'system': machine['system'],
                'activities': machine['activities'], 'jobs': jobs, 'comma': comma_state, 'training': training,
                'actions': self.action_states(machine, jobs, comma_state, runs), 'hf_repo': self.hf.repo,
                'test_actions': self.test_actions,
                'limits': {'steps': [actions_module.STEPS_MIN, actions_module.STEPS_MAX],
                           'batch_size': [actions_module.BATCH_MIN, actions_module.BATCH_MAX],
                           'workers': [actions_module.WORKERS_MIN, actions_module.WORKERS_MAX],
                           'start': [actions_module.START_MIN, actions_module.START_MAX],
                           'duration': [actions_module.DURATION_MIN, actions_module.DURATION_MAX]}}

    def hf_payload(self) -> dict:
        snapshot = dict(self.hf.snapshot())
        return snapshot

    # -- actions -----------------------------------------------------------------------------------
    def run_action(self, action_id: str, params) -> dict:
        spec = build_job(self.repo, action_id, params, test_actions=self.test_actions)
        missing = script_missing(self.repo, action_id, self.test_actions)
        if missing:
            raise WebError(409, 'script_missing', 'Non posso: ' + missing + '.')
        if spec.needs_comma_offroad:
            state = self.comma.probe_now()
            if state.get('reachable') is not True:
                raise WebError(409, 'comma_unreachable',
                               'Il comma non e\' raggiungibile: ' + state.get('error', '') + '.')
            if state.get('onroad') is True:
                raise WebError(409, 'comma_onroad', 'Il comma e\' onroad: spegni l\'auto e riprova.')
            # helpers/install_model_on_comma.sh refuses it too: a model chosen in the Models menu runs
            # in place of openpilot's one, so the installed model would not be used.
            if action_id == 'install' and state.get('bundle_default') is False:
                raise WebError(409, 'comma_bundle', BUNDLE_REASON[0].upper() + BUNDLE_REASON[1:] + '.')
        if action_id == 'publish':
            snapshot = self.hf.snapshot(force=True)
            if not snapshot['ok']:
                raise WebError(503, 'hf_unreachable',
                               'Non riesco a controllare i nomi su Hugging Face (' + snapshot['error'] + '): riprova.')
            slug = slug_of(spec.params['name'])
            if slug in snapshot['models'] or slug in snapshot['releases']:
                raise WebError(409, 'name_taken', 'Su Hugging Face c\'e\' gia\' un modello ' + slug
                               + ': scegli un altro nome.')
            if self.jobs.last_successful('publish_dry', spec.params) is None:
                raise WebError(409, 'dry_run_missing', 'Prima fai la prova (dry run) con lo stesso run, nome e '
                               'stable, e controlla che finisca bene.')
        job = self.jobs.start(spec)
        print('Job avviato: ' + describe(job), flush=True)
        return job


def render_problem(status: int, message: str) -> str:
    return ('<!doctype html><html lang="it"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<meta name="color-scheme" content="light dark"><title>Distill</title>'
            '<style>body{margin:0;padding:24px 16px;'
            'font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;background:#e9ecef;color:#16191d}'
            'main{max-width:640px;margin:0 auto;background:#fff;'
            'border:1px solid #cfd4da;border-radius:12px;padding:20px}p{margin:0 0 8px}'
            '@media (prefers-color-scheme:dark){body{background:#111317;color:#e8ebee}'
            'main{background:#191c21;border-color:#2e333a}}</style></head><body><main>'
            '<p>' + html.escape(message) + '</p><p>Codice HTTP ' + str(status) + '.</p></main></body></html>')


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'distill-web'
        sys_version = ''
        # Socket timeout: a client that stops sending is disconnected, even before the token check.
        timeout = REQUEST_TIMEOUT_SECONDS
        head_only = False

        def do_GET(self):
            self.head_only = False
            self._dispatch('GET')

        def do_HEAD(self):
            self.head_only = True
            self._dispatch('GET')

        def do_POST(self):
            self.head_only = False
            self._dispatch('POST')

        def log_message(self, *args):
            # Request lines carry ?token= on the first visit: nothing is logged per request.
            pass

        def _dispatch(self, method):
            try:
                self._handle(method)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                self.close_connection = True
            except WebError as exc:
                self._problem(exc.status, exc.reason, exc.message, self._is_api())
            except Exception as exc:
                print('Errore interno: ' + type(exc).__name__ + ': ' + short_text(exc, 200), flush=True)
                self._problem(500, 'server_error', 'Errore interno del server: ' + type(exc).__name__ + '.',
                              self._is_api())

        def _is_api(self):
            return urlsplit(self.path).path.startswith('/api/')

        def _handle(self, method):
            parts = urlsplit(self.path)
            path = unquote(parts.path)
            api = path.startswith('/api/')
            if not app.host_allowed(self.headers.get('Host')):
                self._problem(403, 'host_not_allowed', 'Nome host non consentito per questo server.', api)
                return
            query = parse_qs(parts.query, keep_blank_values=True)
            if 'token' in query:
                self._accept_token(method, path, query['token'][0], api)
                return
            if not self._authorized():
                self._problem(403, 'token_required', TOKEN_MESSAGE, api)
                return
            if method == 'POST':
                self._post(path)
            else:
                self._get(path, query)

        def _authorized(self):
            if not app.require_token:
                return True
            if app.token_matches(self.headers.get(TOKEN_HEADER)):
                return True
            # Plain "name=value; name=value" split: the browser also sends the cookies of other web
            # apps on this host (any port, e.g. the speedlimits site), and a value a stricter parser
            # would not like must not hide ours.
            for raw in self.headers.get_all('Cookie', []):
                for part in raw.split(';'):
                    name, separator, value = part.strip().partition('=')
                    if not separator:
                        continue
                    if name.strip() == app.cookie_name and app.token_matches(value.strip()):
                        return True
            return False

        def _accept_token(self, method, path, supplied, api):
            # Clean target: our own paths only, never "//host" or a backslash.
            target = path
            if not re.fullmatch(r'/[A-Za-z0-9._/-]*', target) or target.startswith('//'):
                target = '/'
            if not app.require_token:
                self._redirect(target, None)
                return
            if method != 'GET' or not app.token_matches(supplied):
                self._problem(403, 'invalid_token', 'Token non valido: usa l\'indirizzo stampato da helpers/web.sh.',
                              api)
                return
            cookie = app.cookie_name + '=' + app.token + '; Path=/; HttpOnly; SameSite=Strict'
            self._redirect(target, cookie)

        # -- GET -----------------------------------------------------------------------------------
        def _get(self, path, query):
            if path == '/' or path == '/index.html':
                nonce = secrets.token_urlsafe(18)
                page = (STATIC_DIR / 'index.html').read_text(encoding='utf-8').replace('__DISTILL_NONCE__', nonce)
                self._send(200, 'text/html; charset=utf-8', page.encode('utf-8'), page_csp(nonce))
                return
            match = re.fullmatch(r'/static/([a-z]+\.[a-z]+)', path)
            if match:
                name = match.group(1)
                if name not in STATIC_FILES:
                    self._problem(404, 'not_found', 'File non trovato.', False)
                    return
                self._send(200, STATIC_FILES[name], (STATIC_DIR / name).read_bytes())
                return
            if path == '/api/status':
                self._json(200, app.status_payload())
                return
            if path == '/api/data':
                self._json(200, segments(app.repo))
                return
            if path == '/api/runs':
                self._json(200, {'runs': app.runs_payload()})
                return
            match = re.fullmatch(r'/api/runs/([^/]+)/metrics', path)
            if match:
                name = match.group(1)
                if not RUN_RE.fullmatch(name):
                    raise WebError(404, 'unknown_run', 'Run sconosciuto.')
                points = 800
                raw_points = query.get('points', [''])[0]
                if re.fullmatch(r'[0-9]{1,5}', raw_points):
                    points = min(max(int(raw_points), 10), 5000)
                series = app.runs.series(name, points)
                if series is None:
                    raise WebError(404, 'unknown_run', 'Run sconosciuto.')
                self._json(200, series)
                return
            if path == '/api/eval':
                self._json(200, {'results': eval_results(app.repo)})
                return
            if path == '/api/compare':
                self._json(200, {'videos': compare_videos(app.repo)})
                return
            if path == '/api/hf':
                self._json(200, app.hf_payload())
                return
            if path == '/api/jobs':
                self._json(200, {'jobs': app.jobs.list_jobs()})
                return
            match = re.fullmatch(r'/api/jobs/([^/]+)', path)
            if match:
                self._json(200, app.jobs.get(match.group(1)))
                return
            match = re.fullmatch(r'/api/jobs/([^/]+)/log', path)
            if match:
                raw_offset = query.get('offset', ['0'])[0]
                offset = 0
                if re.fullmatch(r'[0-9]{1,15}', raw_offset):
                    offset = int(raw_offset)
                self._json(200, app.jobs.log(match.group(1), offset))
                return
            match = re.fullmatch(r'/video/([^/]+)', path)
            if match:
                self._send_video(match.group(1))
                return
            if path.startswith('/api/run/') or re.fullmatch(r'/api/jobs/[^/]+/stop', path):
                self._problem(405, 'method_not_allowed', 'Usa POST.', True)
                return
            self._problem(404, 'not_found', 'Pagina non trovata.', path.startswith('/api/'))

        def _send_video(self, name):
            path = video_path(app.repo, name)
            if path is None:
                self._problem(404, 'not_found', 'Video non trovato.', False)
                return
            size = path.stat().st_size
            wanted = parse_range(self.headers.get('Range'), size)
            if wanted == 'unsatisfiable':
                self.send_response(416)
                self.send_header('Content-Range', 'bytes */' + str(size))
                self.send_header('Content-Length', '0')
                self._security_headers(DEFAULT_CSP)
                self.end_headers()
                return
            if wanted is None:
                first = 0
                last = size - 1
                self.send_response(200)
            else:
                first = wanted[0]
                last = wanted[1]
                self.send_response(206)
                self.send_header('Content-Range', 'bytes ' + str(first) + '-' + str(last) + '/' + str(size))
            length = last - first + 1
            if size == 0:
                length = 0
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(length))
            self._security_headers(DEFAULT_CSP)
            self.end_headers()
            if self.head_only or length == 0:
                return
            with path.open('rb') as source:
                source.seek(first)
                remaining = length
                while remaining > 0:
                    block = source.read(min(VIDEO_CHUNK_BYTES, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining = remaining - len(block)

        # -- POST ----------------------------------------------------------------------------------
        def _post(self, path):
            run = re.fullmatch(r'/api/run/([a-z_]+)', path)
            stop = re.fullmatch(r'/api/jobs/([^/]+)/stop', path)
            check = path == '/api/comma/check'
            if not run and not stop and not check:
                self._problem(404, 'not_found', 'Indirizzo non valido.', True)
                return
            # Same origin and JSON only: no other site and no HTML form can start a job.
            if not self._same_origin():
                self._problem(403, 'cross_origin', 'Richiesta da un\'altra origine rifiutata.', True)
                return
            content_type = self.headers.get('Content-Type', '')
            if content_type.split(';')[0].strip().lower() != 'application/json':
                self._problem(415, 'unsupported_media_type', 'Sono accettati solo dati JSON.', True)
                return
            body = self._read_json()
            if run:
                job = app.run_action(run.group(1), body)
                self._json(200, {'ok': True, 'job': job})
            elif stop:
                job_id = stop.group(1)
                if not JOB_ID_RE.fullmatch(job_id):
                    raise WebError(404, 'unknown_job', 'Job sconosciuto.')
                job = app.jobs.stop(job_id)
                print('Job fermato dal sito: ' + describe(job), flush=True)
                self._json(200, {'ok': True, 'job': job})
            else:
                self._json(200, {'ok': True, 'comma': app.comma.probe_now()})

        def _same_origin(self):
            fetch_site = self.headers.get('Sec-Fetch-Site')
            if fetch_site is not None and fetch_site not in ('same-origin', 'none'):
                return False
            origin = self.headers.get('Origin')
            if origin is None:
                return True
            host = self.headers.get('Host')
            if not host:
                return False
            # Same host name and port as the request; https as well, for a reverse proxy that
            # terminates TLS and keeps the Host header.
            supplied = origin.strip().lower()
            if supplied == ('http://' + host.strip()).lower():
                return True
            if supplied == ('https://' + host.strip()).lower():
                return True
            return False

        def _read_json(self):
            if self.headers.get('Transfer-Encoding') is not None:
                raise WebError(411, 'length_required', 'Richiesta senza Content-Length.')
            raw_length = self.headers.get('Content-Length')
            if raw_length is None or not re.fullmatch(r'[0-9]{1,9}', raw_length.strip()):
                raise WebError(411, 'length_required', 'Richiesta senza Content-Length.')
            length = int(raw_length.strip())
            if length > MAX_BODY_BYTES:
                self.close_connection = True
                raise WebError(413, 'too_large', 'Richiesta troppo grande (massimo 64 KiB).')
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise WebError(400, 'invalid_json', 'Richiesta incompleta.')
            if length == 0:
                return {}
            try:
                data = strict_json_loads(raw.decode('utf-8'))
            except (UnicodeDecodeError, ValueError, TypeError, RecursionError) as exc:
                raise WebError(400, 'invalid_json', 'Dati JSON non validi.') from exc
            if not isinstance(data, dict):
                raise WebError(400, 'invalid_json', 'Serve un oggetto JSON.')
            return data

        # -- responses -----------------------------------------------------------------------------
        def _security_headers(self, csp):
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
            self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
            self.send_header('Content-Security-Policy', csp)

        def _send(self, status, content_type, body, csp=DEFAULT_CSP, extra=None):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self._security_headers(csp)
            if extra is not None:
                for name, value in extra:
                    self.send_header(name, value)
            self.end_headers()
            if not self.head_only:
                self.wfile.write(body)

        def _json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self._send(status, 'application/json; charset=utf-8', body)

        def _problem(self, status, reason, message, api):
            if api:
                self._json(status, {'ok': False, 'error': reason, 'message': message})
            else:
                self._send(status, 'text/html; charset=utf-8', render_problem(status, message).encode('utf-8'),
                           PROBLEM_CSP)

        def _redirect(self, location, cookie):
            extra = [('Location', location)]
            if cookie is not None:
                extra.append(('Set-Cookie', cookie))
            self._send(303, 'text/plain; charset=utf-8', b'', DEFAULT_CSP, extra)

    return Handler


class WebServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, handler_class):
        self._connection_slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        super().__init__(server_address, handler_class)

    def process_request(self, request, client_address):
        if not self._connection_slots.acquire(blocking=False):
            # Every slot is busy: close this connection now instead of starting another thread.
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            # The thread did not start, so it will never release its slot.
            self._connection_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connection_slots.release()


class WebServerIPv6(WebServer):
    address_family = socket.AF_INET6


def create_server(repo: Path, *, host: str = '127.0.0.1', port: int = DEFAULT_PORT, token: str | None = None,
                  require_token: bool | None = None, test_actions: bool = False, app: App | None = None):
    """Binds the server (not started). The token is required whenever the address is not loopback."""
    if require_token is None:
        require_token = not is_loopback(host)
    if require_token and token is None:
        token = access_token()
    if app is None:
        app = App(repo, token=token, require_token=require_token, test_actions=test_actions)
    else:
        app.token = token
        app.require_token = require_token
    if ':' in host:
        server = WebServerIPv6((host, port), make_handler(app))
    else:
        server = WebServer((host, port), make_handler(app))
    # One cookie per port, so two sites on one machine (speedlimits on 8765) never mix them up.
    app.cookie_name = 'distill_web_' + str(server.server_port)
    return server, app


def base_urls(host: str, port: int) -> list[str]:
    if host in ('0.0.0.0', '', '::'):
        names = lan_addresses()
        if not names:
            names = [socket.gethostname()]
    elif ':' in host:
        names = ['[' + host + ']']
    else:
        names = [host]
    urls = []
    for name in names:
        urls.append('http://' + name + ':' + str(port))
    return urls


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m web.server', description='distill-web, il sito di controllo')
    parser.add_argument('--host', default='127.0.0.1', help='indirizzo (default 127.0.0.1; 0.0.0.0 = rete locale, '
                        'con token)')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help='porta (default 8766)')
    parser.add_argument('--test-actions', action='store_true',
                        help='aggiunge l\'azione "selftest" (job finto, per provare il sito)')
    options = parser.parse_args(arguments)

    repo = CODE_ROOT
    token_from_env = bool(os.environ.get(TOKEN_ENV))
    try:
        server, app = create_server(repo, host=options.host, port=options.port, test_actions=options.test_actions)
    except OSError as exc:
        print('Non riesco ad avviare il sito su ' + options.host + ':' + str(options.port) + ': ' + str(exc),
              file=sys.stderr, flush=True)
        return 1
    app.comma.start()
    with server:
        port = server.server_port
        suffix = ''
        if app.require_token:
            if token_from_env:
                suffix = '?token=<' + TOKEN_ENV + '>'
            else:
                suffix = '?token=' + app.token
        print('distill-web in ascolto su ' + options.host + ':' + str(port), flush=True)
        print('Apri nel browser:', flush=True)
        for base in base_urls(options.host, port):
            print('  ' + base + '/' + suffix, flush=True)
        if app.require_token and token_from_env:
            # A token fixed in the environment (the systemd service) is never printed: the output ends
            # in a log.
            print('Il token e\' quello di ' + TOKEN_ENV + ' (servizio: /etc/distill-web.env, a mano: .env del clone) '
                  'e non viene stampato.', flush=True)
        elif app.require_token:
            print('Il token resta valido finche\' il sito e\' acceso (fissalo con ' + TOKEN_ENV + ').', flush=True)
        else:
            print('Solo da questo computer. Da un altro: ssh -L ' + str(port) + ':127.0.0.1:' + str(port)
                  + ' <server>, oppure riavvia con --host 0.0.0.0', flush=True)
        if options.test_actions:
            print('Azione di prova "selftest" attiva (--test-actions).', flush=True)
        print('Job e log in ' + str(app.jobs.jobs_dir) + '. Ctrl+C per fermare (i job continuano).', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
