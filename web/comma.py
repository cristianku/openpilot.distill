"""The state of the comma, read over ssh (read-only: two params files and one file test):
reachable, onroad, the model chosen in the sunnypilot Models menu (ModelManager_ActiveBundle) and
whether helpers/install_model_on_comma.sh left a copy of the original model.

A background thread asks every 30 s, but only while someone has the page open (the comma is in a
car: no ssh every 30 s all day for nobody). install/restore ask again right before starting.
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import time

from web.common import short_text

INTERVAL_SECONDS = 30
BUNDLE_NAME_RE = re.compile(r'"(?:displayName|internalName)"\s*:\s*"([^"]{1,120})"')
# Probe only when a page asked for the status in the last ... seconds.
INTEREST_SECONDS = 120
# Fixed command, nothing from a request in it. Every line is key=value.
REMOTE_COMMAND = (
    'echo "onroad=$(cat /data/params/d/IsOnroad 2>/dev/null)"; '
    'echo "bundle=$(head -c 8000 /data/params/d/ModelManager_ActiveBundle 2>/dev/null | tr -d \'\\n\')"; '
    'if [ -f /data/driving_supercombo.onnx.orig ]; then echo "backup=1"; else echo "backup=0"; fi'
)


def ssh_probe(host: str) -> tuple[int, str]:
    arguments = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', host, REMOTE_COMMAND]
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=20, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return 124, ''
    except OSError:
        return 127, ''
    return result.returncode, result.stdout


def bundle_is_default(raw: str) -> bool:
    """True when no model is chosen in the sunnypilot Models menu (the rule of
    helpers/install_model_on_comma.sh: the param is empty or "{}")."""
    text = raw.strip()
    return text == '' or text == '{}'


def bundle_name(raw: str) -> str:
    """A readable name of the active Models entry; 'Default' when none is chosen."""
    text = raw.strip()
    if bundle_is_default(text) or text == 'null':
        return 'Default'
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if isinstance(data, dict):
        for key in ('displayName', 'display_name', 'internalName', 'name', 'ref'):
            value = data.get(key)
            if isinstance(value, str) and value:
                return short_text(value, 80)
    # Cut or odd JSON: the name is still readable in the text.
    match = BUNDLE_NAME_RE.search(text)
    if match is not None:
        return short_text(match.group(1), 80)
    return short_text(text, 80)


def parse_probe(code: int, output: str) -> dict:
    result = {'reachable': False, 'onroad': None, 'bundle': None, 'bundle_default': None, 'backup': None,
              'error': ''}
    if code != 0:
        if code == 124:
            result['error'] = 'ssh does not answer (timeout)'
        elif code == 127:
            result['error'] = 'ssh is not available on this server'
        else:
            result['error'] = 'comma not reachable over ssh (off or out of the network)'
        return result
    result['reachable'] = True
    for line in output.splitlines():
        key, separator, value = line.partition('=')
        if not separator:
            continue
        if key == 'onroad':
            # Same rule as the helpers: onroad only when the param says "1".
            result['onroad'] = value.strip() == '1'
        elif key == 'bundle':
            result['bundle'] = bundle_name(value)
            result['bundle_default'] = bundle_is_default(value)
        elif key == 'backup':
            result['backup'] = value.strip() == '1'
    return result


class CommaMonitor:
    def __init__(self, host: str, probe=ssh_probe):
        self.host = host
        self._probe = probe
        self._lock = threading.Lock()
        self._probe_lock = threading.Lock()
        self._state = {'host': host, 'checked': None, 'reachable': None, 'onroad': None, 'bundle': None,
                       'bundle_default': None, 'backup': None, 'error': 'not checked yet'}
        self._last_interest = 0.0
        self._thread = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name='comma-monitor', daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            interested = time.monotonic() - self._last_interest < INTEREST_SECONDS
            if interested or self._state['checked'] is None:
                self.probe_now()
            time.sleep(INTERVAL_SECONDS)

    def probe_now(self) -> dict:
        # One ssh at a time: a page and the thread asking together get the same answer.
        with self._probe_lock:
            code, output = self._probe(self.host)
            result = parse_probe(code, output)
            result['host'] = self.host
            result['checked'] = time.time()
            with self._lock:
                self._state = result
                return dict(self._state)

    def snapshot(self) -> dict:
        with self._lock:
            self._last_interest = time.monotonic()
            return dict(self._state)
