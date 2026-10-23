"""The Settings tab: the Hugging Face and Weights & Biases accounts of this server, and the site
settings kept in the repo .env (HF_REPO, COMMA_HOST).

Hugging Face: a token is checked with the Hub first (GET /api/whoami-v2, the call HfApi.whoami makes;
made here with urllib and a timeout, because the default HTTP client of huggingface_hub has none and a
page request must not hang) and only then written where huggingface_hub reads it: the token file
(HF_TOKEN_PATH, default $HF_HOME/token = ~/.cache/huggingface/token), mode 600. tasks/publish.py
(HfApi()) uses it from there. Disconnect deletes that file and the copy of the same token that
"hf auth login" keeps in stored_tokens.

Weights & Biases: an API key is checked with the W&B GraphQL API (viewer { username entity }) and only
then written in the netrc file (NETRC, default ~/.netrc) as "machine api.wandb.ai", mode 600, every
other entry kept as it is: wandb and tasks/train.py (wandb_mode) read it there. Disconnect removes that
entry only.

A token in the server's environment (HF_TOKEN, WANDB_API_KEY) wins over a saved one: the site shows it
but does not replace or remove it.

Secrets never go back to the page, never into the log and never into an error message: the page only
sees account names. Every path comes from the environment at the time of the call, so tests can point
HOME and HF_HOME at temporary folders. Standard library only.
"""
from __future__ import annotations

import base64
import configparser
import hashlib
import io
import json
import os
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from web.common import (COMMA_HOST_RE, DEFAULT_COMMA_HOST, DEFAULT_HF_REPO, HF_REPO_RE, WebError, comma_host,
                        hf_repo, read_env_file)

USER_AGENT = 'distill-web'
TIMEOUT_SECONDS = 10
MAX_ANSWER_BYTES = 1024 * 1024
HF_DEFAULT_ENDPOINT = 'https://huggingface.co'
HF_ENV_KEYS = ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN')
WANDB_ENV_KEY = 'WANDB_API_KEY'
WANDB_MACHINE = 'api.wandb.ai'
WANDB_GRAPHQL_URL = 'https://api.wandb.ai/graphql'
WANDB_VIEWER_QUERY = 'query Viewer { viewer { username entity } }'
# The netrc words followed by a value.
VALUE_KEYWORDS = ('machine', 'macdef', 'login', 'user', 'account', 'password')
# User access tokens of huggingface.co start with "hf_" (organization tokens "api_org" cannot log in).
HF_TOKEN_RE = re.compile(r'^hf_[A-Za-z0-9_-]{8,250}$')
# W&B API keys: 40 characters or more (the classic key is 40 hex digits; newer and on-prem keys are
# longer or carry a "name-" prefix). Same rule as wandb's own check_api_key.
WANDB_KEY_RE = re.compile(r'^[A-Za-z0-9_-]{40,250}$')
# How long a check of the saved secret is trusted: an answer, and a failure to get one.
CHECK_SECONDS = 600
UNREACHABLE_SECONDS = 60
# Connect attempts that reach the service, per account, in a sliding window.
ATTEMPTS_PER_WINDOW = 5
ATTEMPT_WINDOW_SECONDS = 60
NAMES = {'hf': 'Hugging Face', 'wandb': 'Weights & Biases'}
SITE_FIELDS = {
    'hf_repo': ('HF_REPO', HF_REPO_RE, DEFAULT_HF_REPO,
                'The Hugging Face repository is owner/name: letters, digits, "-", "_" and ".".'),
    'comma_host': ('COMMA_HOST', COMMA_HOST_RE, DEFAULT_COMMA_HOST,
                   'The comma address is a host name, an IP address or user@host: letters, digits, ".", "-", '
                   '"_" and "@", not starting with "-".'),
}
SITE_VALUE_MAX = 200


class Rejected(Exception):
    """The service answered and refused the secret."""


class CheckFailed(Exception):
    """The service could not be asked (network, unexpected answer): it says nothing about the secret.
    The message is safe to show: it never carries the secret."""


# -- where the secrets live ----------------------------------------------------------------------------
def hf_token_path() -> Path:
    """The token file huggingface_hub reads (huggingface_hub.constants.HF_TOKEN_PATH), from the
    environment of now: HF_TOKEN_PATH, else $HF_HOME/token, else $XDG_CACHE_HOME/huggingface/token,
    else ~/.cache/huggingface/token."""
    explicit = os.environ.get('HF_TOKEN_PATH', '')
    if explicit:
        return Path(os.path.expandvars(os.path.expanduser(explicit)))
    home = os.environ.get('HF_HOME', '')
    if not home:
        cache = os.environ.get('XDG_CACHE_HOME', '') or os.path.join(os.path.expanduser('~'), '.cache')
        home = os.path.join(cache, 'huggingface')
    return Path(os.path.expandvars(os.path.expanduser(home))) / 'token'


def netrc_path() -> Path:
    """The netrc file wandb reads: NETRC, else ~/.netrc."""
    explicit = os.environ.get('NETRC', '')
    if explicit:
        return Path(explicit).expanduser()
    return Path(os.path.expanduser('~')) / '.netrc'


def _read_text(path: Path) -> str:
    """The text of a file, '' when it does not exist. Any other read error is refused, so a file
    that cannot be read is never overwritten."""
    try:
        return path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return ''
    except (OSError, UnicodeDecodeError) as exc:
        raise WebError(500, 'unreadable', 'Cannot read the saved settings on this server (' + type(exc).__name__
                       + ').') from None


def write_private(path: Path, text: str) -> None:
    """Writes a file readable by its owner only (600), all at once: a temporary file in the same
    folder, flushed to disk, then renamed over the old one. A link is never written through."""
    if path.is_symlink():
        raise WebError(409, 'file_is_a_link', 'The file that keeps this setting on the server is a link: '
                       'change it by hand there.')
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        handle, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=str(path.parent))
    except OSError as exc:
        raise WebError(500, 'cannot_save', 'Cannot save it on this server (' + type(exc).__name__ + ').') from None
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except OSError as exc:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise WebError(500, 'cannot_save', 'Cannot save it on this server (' + type(exc).__name__ + ').') from None


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise WebError(500, 'cannot_save', 'Cannot remove it on this server (' + type(exc).__name__ + ').') from None


# -- netrc ----------------------------------------------------------------------------------------------
def _netrc_tokens(text: str) -> list[tuple[str, int]]:
    """(token, offset of its first character) of a netrc text, like Python's netrc lexer: tokens are
    separated by white space, "..." quotes a token, a backslash escapes the next character, "#" starts a
    comment up to the end of the line, and the body of a macdef (up to the next empty line) is skipped."""
    tokens = []
    index = 0
    length = len(text)
    # The keyword whose value the next token is ('' when the next token is a keyword).
    expecting = ''
    while index < length:
        character = text[index]
        if character.isspace():
            index = index + 1
            continue
        if character == '#':
            end = text.find('\n', index)
            index = length if end < 0 else end + 1
            continue
        start = index
        value = []
        if character == '"':
            index = index + 1
            while index < length and text[index] != '"':
                if text[index] == '\\' and index + 1 < length:
                    index = index + 1
                value.append(text[index])
                index = index + 1
            index = index + 1
        else:
            while index < length and not text[index].isspace():
                if text[index] == '\\' and index + 1 < length:
                    index = index + 1
                value.append(text[index])
                index = index + 1
        token = ''.join(value)
        tokens.append((token, start))
        if expecting == 'macdef':
            # "macdef <name>": the rest of the line and the next lines up to an empty one are the macro.
            end = text.find('\n', index)
            if end < 0:
                break
            blank = text.find('\n\n', end)
            index = length if blank < 0 else blank + 2
            expecting = ''
        elif expecting:
            expecting = ''
        elif token in VALUE_KEYWORDS:
            expecting = token
    return tokens


def netrc_entries(text: str) -> list[dict]:
    """The entries of a netrc text: {'kind': machine/default/macdef, 'name', 'start', 'fields'}, where
    start is the offset of the entry's first word, so the text of an entry runs up to the next one."""
    tokens = _netrc_tokens(text)
    entries = []
    index = 0
    while index < len(tokens):
        word, start = tokens[index]
        if word in ('machine', 'macdef') and index + 1 < len(tokens):
            entries.append({'kind': word, 'name': tokens[index + 1][0], 'start': start, 'fields': {}})
            index = index + 2
            continue
        if word == 'default':
            entries.append({'kind': 'default', 'name': None, 'start': start, 'fields': {}})
            index = index + 1
            continue
        if word in VALUE_KEYWORDS and index + 1 < len(tokens):
            if entries and entries[-1]['kind'] != 'macdef':
                entries[-1]['fields'][word] = tokens[index + 1][0]
            index = index + 2
            continue
        index = index + 1
    return entries


def netrc_password(text: str, machine: str) -> str:
    """The password of `machine` (the last entry wins, as in Python's netrc), '' when there is none."""
    password = ''
    for entry in netrc_entries(text):
        if entry['kind'] == 'machine' and entry['name'] == machine:
            password = entry['fields'].get('password', '')
    return password


def netrc_without(text: str, machine: str) -> str:
    """The text without the entries of `machine`; everything else is kept character for character."""
    entries = netrc_entries(text)
    kept = []
    position = 0
    for index, entry in enumerate(entries):
        if entry['kind'] != 'machine' or entry['name'] != machine:
            continue
        end = entries[index + 1]['start'] if index + 1 < len(entries) else len(text)
        kept.append(text[position:entry['start']])
        position = end
    kept.append(text[position:])
    return ''.join(kept)


def netrc_with(text: str, machine: str, password: str) -> str:
    """The text with one entry for `machine` (login "user", as wandb writes it), at the end."""
    rest = netrc_without(text, machine)
    if rest.strip() and not rest.endswith('\n'):
        rest = rest + '\n'
    if not rest.strip():
        rest = ''
    return rest + 'machine ' + machine + '\n  login user\n  password ' + password + '\n'


# -- .env -----------------------------------------------------------------------------------------------
def _env_key(line: str) -> str | None:
    """The key of a KEY=value line of the .env (the rules of read_env_file), None for any other line."""
    text = line.strip()
    if not text or text.startswith('#'):
        return None
    if text.startswith('export '):
        text = text[len('export '):].strip()
    key, separator, _ = text.partition('=')
    if not separator:
        return None
    return key.strip()


def update_env_file(path: Path, updates: dict) -> None:
    """Sets (value) or removes (None) keys of the .env, keeping every other line as it is (the site
    token among them, which is never read here). Written at once, mode 600."""
    lines = _read_text(path).splitlines()
    result = []
    written = set()
    for line in lines:
        key = _env_key(line)
        if key not in updates:
            result.append(line)
            continue
        if key in written or updates[key] is None:
            continue
        result.append(key + '=' + updates[key])
        written.add(key)
    for key, value in updates.items():
        if value is not None and key not in written:
            result.append(key + '=' + value)
    text = ''
    if result:
        text = '\n'.join(result) + '\n'
    write_private(path, text)


def clean_site_changes(body) -> dict:
    """{env key: value or None (back to the default)} from {"hf_repo"?, "comma_host"?}, validated."""
    if not isinstance(body, dict) or not body or not set(body) <= set(SITE_FIELDS):
        raise WebError(400, 'invalid_param', 'Send {"hf_repo": "owner/name"} and/or {"comma_host": "host"}.')
    updates = {}
    for name, value in body.items():
        key, pattern, _default, message = SITE_FIELDS[name]
        if not isinstance(value, str) or len(value) > SITE_VALUE_MAX:
            raise WebError(400, 'invalid_' + name, message)
        value = value.strip()
        if value == '':
            updates[key] = None
        elif not pattern.fullmatch(value):
            raise WebError(400, 'invalid_' + name, message)
        else:
            updates[key] = value
    return updates


def site_payload(repo: Path) -> dict:
    """The site settings as the page shows them: the value in use, the one saved in the .env, the
    default, and whether the server's environment sets it (the environment wins over the .env)."""
    saved = read_env_file(repo / '.env', ('HF_REPO', 'COMMA_HOST'))
    current = {'HF_REPO': hf_repo(repo), 'COMMA_HOST': comma_host(repo)}
    result = {}
    for name, (key, _pattern, default, _message) in SITE_FIELDS.items():
        result[name] = {'value': current[key], 'saved': saved.get(key, ''), 'default': default,
                        'from_environment': bool(os.environ.get(key))}
    return result


# -- asking the services --------------------------------------------------------------------------------
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """urllib would send the Authorization header on to wherever a redirect points: no redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _open(request: urllib.request.Request, timeout: float):
    """The one place that talks to the network (tests replace it)."""
    return _OPENER.open(request, timeout=timeout)


def _ask(request: urllib.request.Request, service: str) -> dict:
    """The JSON answer of a request that carries a secret. Rejected for 401/403; CheckFailed for
    anything else that goes wrong, with a message that names the error type only."""
    try:
        with _open(request, TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_ANSWER_BYTES)
    except urllib.error.HTTPError as exc:
        code = exc.code
        exc.close()
        if code in (401, 403):
            raise Rejected() from None
        raise CheckFailed(service + ' answered with HTTP ' + str(code)) from None
    except Exception as exc:  # network, timeout, TLS: never its text, it could quote the request
        raise CheckFailed(service + ' does not answer (' + type(exc).__name__ + ')') from None
    try:
        data = json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, ValueError):
        raise CheckFailed('unexpected answer from ' + service) from None
    if not isinstance(data, dict):
        raise CheckFailed('unexpected answer from ' + service)
    return data


def check_hf_token(token: str) -> dict:
    """Who a Hugging Face token belongs to: {'user', 'role', 'fine_grained'} (role and fine-grained
    scopes as whoami-v2 reports them)."""
    endpoint = os.environ.get('HF_ENDPOINT', '').rstrip('/') or HF_DEFAULT_ENDPOINT
    request = urllib.request.Request(endpoint + '/api/whoami-v2', headers={
        'Authorization': 'Bearer ' + token, 'User-Agent': USER_AGENT, 'Accept': 'application/json'})
    data = _ask(request, 'Hugging Face')
    name = data.get('name')
    if not isinstance(name, str) or not name:
        raise CheckFailed('unexpected answer from Hugging Face')
    auth = data.get('auth')
    access_token = auth.get('accessToken') if isinstance(auth, dict) else None
    if not isinstance(access_token, dict):
        access_token = {}
    role = access_token.get('role')
    fine_grained = access_token.get('fineGrained')
    return {'user': name, 'role': role if isinstance(role, str) else None,
            'fine_grained': fine_grained if isinstance(fine_grained, dict) else None}


def check_wandb_key(key: str) -> dict:
    """Who a W&B API key belongs to: {'user', 'entity'}."""
    credentials = base64.b64encode(('api:' + key).encode('utf-8')).decode('ascii')
    body = json.dumps({'query': WANDB_VIEWER_QUERY}).encode('utf-8')
    request = urllib.request.Request(WANDB_GRAPHQL_URL, data=body, method='POST', headers={
        'Authorization': 'Basic ' + credentials, 'Content-Type': 'application/json', 'User-Agent': USER_AGENT,
        'Accept': 'application/json'})
    data = _ask(request, 'Weights & Biases')
    answer = data.get('data')
    viewer = answer.get('viewer') if isinstance(answer, dict) else None
    if not isinstance(viewer, dict):
        if data.get('errors') and not isinstance(answer, dict):
            raise CheckFailed('unexpected answer from Weights & Biases')
        # A key W&B does not know: no viewer.
        raise Rejected()
    user = viewer.get('username')
    if not isinstance(user, str) or not user:
        raise Rejected()
    entity = viewer.get('entity')
    return {'user': user, 'entity': entity if isinstance(entity, str) and entity else None}


def hf_write_access(check: dict, repo: str) -> str:
    """Whether the token of `check` can write `repo`: 'write', 'read' or 'unknown'.
    Classic tokens have a role (read, write); a fine-grained token can write the repos of the users
    and organizations (or the single repos) it has "repo.write" for."""
    role = check.get('role')
    if role in ('write', 'admin'):
        return 'write'
    if role == 'read':
        return 'read'
    fine = check.get('fine_grained')
    if role == 'fineGrained' and isinstance(fine, dict):
        owner = repo.split('/', 1)[0]
        for scope in fine.get('scoped') or []:
            if not isinstance(scope, dict):
                continue
            entity = scope.get('entity') or {}
            permissions = scope.get('permissions') or []
            if not isinstance(entity, dict) or not isinstance(permissions, list) or 'repo.write' not in permissions:
                continue
            if entity.get('type') in ('user', 'org') and entity.get('name') == owner:
                return 'write'
            if entity.get('type') == 'model' and entity.get('name') == repo:
                return 'write'
        return 'read'
    return 'unknown'


# -- the accounts ---------------------------------------------------------------------------------------
def hf_secret() -> tuple[str, str]:
    """(token, source) huggingface_hub would use: source 'environment' or 'saved'; ('', '') for none."""
    for name in HF_ENV_KEYS:
        value = os.environ.get(name, '').strip()
        if value:
            return value, 'environment'
    try:
        value = hf_token_path().read_text(encoding='utf-8').strip()
    except (OSError, UnicodeDecodeError):
        value = ''
    if value:
        return value, 'saved'
    return '', ''


def wandb_secret() -> tuple[str, str]:
    """(API key, source) wandb would use: WANDB_API_KEY, else the api.wandb.ai entry of the netrc."""
    value = os.environ.get(WANDB_ENV_KEY, '').strip()
    if value:
        return value, 'environment'
    try:
        text = netrc_path().read_text(encoding='utf-8', errors='replace')
    except OSError:
        text = ''
    value = netrc_password(text, WANDB_MACHINE)
    if value:
        return value, 'saved'
    return '', ''


def _fingerprint(secret: str) -> str:
    # Kept in memory to know whether the saved secret changed since it was checked; never sent.
    return hashlib.sha256(secret.encode('utf-8')).hexdigest()


def save_hf_token(token: str) -> None:
    write_private(hf_token_path(), token)


def remove_hf_token() -> None:
    """Deletes the token file and the copy of the same token in stored_tokens ("hf auth login" keeps
    one); the other stored tokens stay."""
    path = hf_token_path()
    token = ''
    if not path.is_symlink():
        try:
            token = path.read_text(encoding='utf-8').strip()
        except (OSError, UnicodeDecodeError):
            token = ''
    _remove(path)
    if not token:
        return
    stored = path.parent / 'stored_tokens'
    if stored.is_symlink() or not stored.is_file():
        return
    config = configparser.ConfigParser(interpolation=None)
    try:
        config.read_string(stored.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, configparser.Error):
        return
    matching = [name for name in config.sections() if config.get(name, 'hf_token', fallback='') == token]
    if not matching:
        return
    for name in matching:
        config.remove_section(name)
    if not config.sections():
        _remove(stored)
        return
    buffer = io.StringIO()
    config.write(buffer)
    write_private(stored, buffer.getvalue())


def save_wandb_key(key: str) -> None:
    path = netrc_path()
    write_private(path, netrc_with(_read_text(path), WANDB_MACHINE, key))


def remove_wandb_key() -> None:
    path = netrc_path()
    text = _read_text(path)
    rest = netrc_without(text, WANDB_MACHINE)
    if rest == text:
        return
    if rest.strip():
        write_private(path, rest)
    else:
        _remove(path)


class Accounts:
    """The two accounts: what is saved, who it belongs to (checked with the service, kept for a few
    minutes per secret), connect and disconnect. One connect or disconnect at a time, and a few
    attempts per minute per account."""

    def __init__(self, check_hf=check_hf_token, check_wandb=check_wandb_key, clock=time.monotonic):
        self._checkers = {'hf': check_hf, 'wandb': check_wandb}
        self._clock = clock
        self._lock = threading.Lock()
        self._check_locks = {'hf': threading.Lock(), 'wandb': threading.Lock()}
        self._change_lock = threading.Lock()
        # kind -> {'fingerprint', 'state', 'info', 'error', 'at', 'checked'}
        self._cache = {}
        self._attempts = {'hf': [], 'wandb': []}

    @staticmethod
    def secret(kind: str) -> tuple[str, str]:
        if kind == 'hf':
            return hf_secret()
        return wandb_secret()

    def _cached(self, kind: str, fingerprint: str) -> dict | None:
        with self._lock:
            entry = self._cache.get(kind)
            if entry is None or entry['fingerprint'] != fingerprint:
                return None
            return dict(entry)

    def _store(self, kind: str, fingerprint: str, state: str, info: dict | None, error: str) -> dict:
        entry = {'fingerprint': fingerprint, 'state': state, 'info': info, 'error': error, 'at': self._clock(),
                 'checked': time.time()}
        with self._lock:
            self._cache[kind] = entry
        return dict(entry)

    def _fresh(self, entry: dict | None) -> bool:
        if entry is None:
            return False
        limit = UNREACHABLE_SECONDS if entry['state'] == 'unreachable' else CHECK_SECONDS
        return self._clock() - entry['at'] < limit

    def _check(self, kind: str, secret: str, fingerprint: str) -> dict:
        """Asks the service who `secret` belongs to, once per secret and period (other requests wait
        for the answer instead of asking again)."""
        with self._check_locks[kind]:
            entry = self._cached(kind, fingerprint)
            if self._fresh(entry):
                return entry
            try:
                info = self._checkers[kind](secret)
            except Rejected:
                return self._store(kind, fingerprint, 'invalid', None, '')
            except CheckFailed as exc:
                return self._store(kind, fingerprint, 'unreachable', None, str(exc))
            except Exception as exc:  # never its text: it could quote the secret
                return self._store(kind, fingerprint, 'unreachable', None, 'check failed (' + type(exc).__name__ + ')')
            return self._store(kind, fingerprint, 'connected', info, '')

    def state(self, kind: str, repo: str, check: bool = True) -> dict:
        """What the page shows of one account (no secret in it). check=False never asks the service:
        a secret not checked yet is 'unchecked'."""
        secret, source = self.secret(kind)
        result = {'state': 'none', 'source': source, 'user': None, 'error': '', 'checked': None,
                  'can_change': source != 'environment'}
        if kind == 'hf':
            result['access'] = None
            result['fine_grained'] = False
        else:
            result['entity'] = None
        if not secret:
            return result
        fingerprint = _fingerprint(secret)
        entry = self._cached(kind, fingerprint)
        if check and not self._fresh(entry):
            entry = self._check(kind, secret, fingerprint)
        if entry is None:
            result['state'] = 'unchecked'
            return result
        result['state'] = entry['state']
        result['error'] = entry['error']
        result['checked'] = entry['checked']
        info = entry['info']
        if info:
            result['user'] = info.get('user')
            if kind == 'hf':
                result['access'] = hf_write_access(info, repo)
                result['fine_grained'] = info.get('role') == 'fineGrained'
            else:
                result['entity'] = info.get('entity')
        return result

    def payload(self, repo: str) -> dict:
        return {'hf': self.state('hf', repo), 'wandb': self.state('wandb', repo)}

    @staticmethod
    def problem_of(kind: str, state: dict, repo: str) -> str:
        """Why the account cannot be used ('' when it can, or when nothing says it cannot)."""
        name = NAMES[kind]
        if state['state'] == 'none':
            return name + ' is not connected: connect it in Settings'
        if state['state'] == 'invalid':
            secret = 'token' if kind == 'hf' else 'API key'
            return 'the saved ' + name + ' ' + secret + ' no longer works: connect again in Settings'
        if kind == 'hf' and state['access'] == 'read':
            return 'the Hugging Face token cannot write ' + repo + ': connect a token with write access in Settings'
        return ''

    def problem(self, kind: str, repo: str) -> str:
        """problem_of() from what is saved and the last check only: no network."""
        return self.problem_of(kind, self.state(kind, repo, check=False), repo)

    def presence(self, repo: str) -> dict:
        """For the status every few seconds: connected or not, the name when a check knows it, and what
        stops the account from being used. No network."""
        result = {}
        for kind in ('hf', 'wandb'):
            state = self.state(kind, repo, check=False)
            result[kind] = {'connected': state['state'] not in ('none', 'invalid'), 'user': state['user'],
                            'problem': self.problem_of(kind, state, repo)}
        return result

    # -- connect and disconnect ---------------------------------------------------------------------
    def _count_attempt(self, kind: str) -> None:
        now = self._clock()
        with self._lock:
            recent = [moment for moment in self._attempts[kind] if now - moment < ATTEMPT_WINDOW_SECONDS]
            if len(recent) >= ATTEMPTS_PER_WINDOW:
                self._attempts[kind] = recent
                raise WebError(429, 'too_many_attempts', 'Too many attempts: wait a minute and try again.')
            recent.append(now)
            self._attempts[kind] = recent

    def connect(self, kind: str, secret, repo: str) -> dict:
        name = NAMES[kind]
        if kind == 'hf':
            pattern = HF_TOKEN_RE
            invalid = 'This is not a Hugging Face access token: it starts with hf_ (huggingface.co, Settings, ' \
                      'Access Tokens).'
        else:
            pattern = WANDB_KEY_RE
            invalid = 'This is not a W&B API key: it has 40 or more letters and digits (wandb.ai/authorize).'
        if not isinstance(secret, str):
            raise WebError(400, 'invalid_secret', invalid)
        secret = secret.strip()
        if not pattern.fullmatch(secret):
            raise WebError(400, 'invalid_secret', invalid)
        if self.secret(kind)[1] == 'environment':
            raise WebError(409, 'set_by_environment', 'This server gets its ' + name + ' credentials from its '
                           'service configuration, which wins over the site: change them there.')
        if not self._change_lock.acquire(blocking=False):
            raise WebError(429, 'busy', 'Another account change is in progress: try again in a moment.')
        try:
            self._count_attempt(kind)
            try:
                info = self._checkers[kind](secret)
            except Rejected:
                raise WebError(400, 'rejected', name + ' does not accept this '
                               + ('token' if kind == 'hf' else 'API key') + ': check it and try again.') from None
            except CheckFailed as exc:
                raise WebError(503, 'unreachable', 'Cannot check it now: ' + str(exc) + '. Nothing was saved.') \
                    from None
            except Exception as exc:  # never its text: it could quote the secret
                raise WebError(503, 'unreachable', 'Cannot check it now (' + type(exc).__name__ + '). Nothing was '
                               'saved.') from None
            if kind == 'hf':
                save_hf_token(secret)
            else:
                save_wandb_key(secret)
            self._store(kind, _fingerprint(secret), 'connected', info, '')
        finally:
            self._change_lock.release()
        return self.state(kind, repo, check=False)

    def disconnect(self, kind: str, repo: str) -> dict:
        name = NAMES[kind]
        if self.secret(kind)[1] == 'environment':
            raise WebError(409, 'set_by_environment', 'This server gets its ' + name + ' credentials from its '
                           'service configuration: remove them there.')
        if not self._change_lock.acquire(blocking=False):
            raise WebError(429, 'busy', 'Another account change is in progress: try again in a moment.')
        try:
            if kind == 'hf':
                remove_hf_token()
            else:
                remove_wandb_key()
            with self._lock:
                self._cache.pop(kind, None)
        finally:
            self._change_lock.release()
        return self.state(kind, repo, check=False)
