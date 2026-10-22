"""Talking to the comma over ssh: the commands and copies of the tasks that use the device.

Every ssh never asks anything (BatchMode: a job started from the site has no terminal), gives up when
the comma does not answer (ConnectTimeout) and notices a comma that goes away in the middle of a copy
(ServerAlive). The comma only accepts the keys registered on GitHub: this machine needs one of them.

Remote commands are fixed texts written in the tasks; a value that comes from outside (a path, a
sha256) goes to a remote script as a positional argument ("bash -s -- <args>"), quoted with
shlex.quote, never pasted into the script itself.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

from tasks.common import TaskError

DEFAULT_HOST = 'comma'
# An ssh host name or user@host; never starting with "-" (it would become an ssh option).
HOST_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.@-]*$')
SSH_OPTIONS = ('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15',
               '-o', 'ServerAliveCountMax=4')
REMOTE_OPENPILOT = '/data/openpilot'
# The original model, saved by the installation of a trained one.
BACKUP = '/data/driving_supercombo.onnx.orig'
NOT_REACHABLE = ('comma not reachable over ssh: is it on and online? (it needs an ssh key registered on its '
                 'GitHub account)')


def comma_host(explicit: str | None = None) -> str:
    """The ssh host of the comma: the explicit one ("comma@" added when it has no user), else
    COMMA_HOST (environment or repo .env, see tasks.common.load_env), else "comma" (~/.ssh/config)."""
    if explicit:
        host = explicit if '@' in explicit else 'comma@' + explicit
    else:
        host = os.environ.get('COMMA_HOST') or DEFAULT_HOST
    if not HOST_RE.fullmatch(host):
        raise TaskError('invalid comma name: ' + host)
    return host


class Comma:
    def __init__(self, host: str):
        self.host = host

    def _ssh(self, remote: list[str] | str, *, input_text: str | None = None, check: bool = True,
             capture: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess:
        command = ['ssh', *SSH_OPTIONS, self.host]
        command += [remote] if isinstance(remote, str) else remote
        try:
            result = subprocess.run(command, input=input_text, text=True, capture_output=capture,
                                    stdin=None if input_text is not None else subprocess.DEVNULL, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise TaskError('the comma does not answer (timeout)') from exc
        except OSError as exc:
            raise TaskError('ssh not available on this server: ' + str(exc)) from exc
        if check and result.returncode != 0:
            detail = (result.stderr or '').strip().splitlines()[-1:] if capture else []
            raise TaskError('command on the comma failed (code ' + str(result.returncode) + ')'
                            + (': ' + detail[0] if detail else ''))
        return result

    def reachable(self) -> bool:
        try:
            return self._ssh('true', check=False, timeout=30).returncode == 0
        except TaskError:
            return False

    def require(self) -> None:
        if not self.reachable():
            raise TaskError(NOT_REACHABLE)

    def run(self, command: str, *, check: bool = True, capture: bool = True,
            timeout: float | None = None) -> subprocess.CompletedProcess:
        """A fixed command on the comma (a text written in the task, run by the comma's shell)."""
        return self._ssh(command, check=check, capture=capture, timeout=timeout)

    def script(self, script: str, *arguments: str, check: bool = True,
               timeout: float | None = None) -> subprocess.CompletedProcess:
        """A fixed bash script on the comma, read from stdin; arguments are its $1, $2, ..."""
        remote = 'bash -s --' + ''.join(' ' + shlex.quote(argument) for argument in arguments)
        return self._ssh(remote, input_text=script, check=check, timeout=timeout)

    def key_values(self, script: str, *arguments: str) -> dict:
        """The "key=value" lines printed by a remote script."""
        values = {}
        for line in self.script(script, *arguments).stdout.splitlines():
            key, separator, value = line.partition('=')
            if separator:
                values[key.strip()] = value.strip()
        return values

    def pull(self, remote_path: str, local_path: Path, *, progress: bool = False) -> None:
        """Copies a file or folder of the comma here (rsync over the same ssh options)."""
        self._rsync([self.host + ':' + remote_path, str(local_path)], progress)

    def push(self, local_path: Path, remote_path: str, *, progress: bool = False) -> None:
        self._rsync([str(local_path), self.host + ':' + remote_path], progress)

    def _rsync(self, paths: list[str], progress: bool) -> None:
        # --partial: an interrupted copy of a big video goes on from where it stopped.
        command = ['rsync', '-a', '--partial', '-e', 'ssh ' + ' '.join(SSH_OPTIONS)]
        if progress:
            command.append('--info=progress2')
        try:
            code = subprocess.run([*command, '--', *paths], stdin=subprocess.DEVNULL).returncode
        except OSError as exc:
            raise TaskError('rsync not available on this server: ' + str(exc)) from exc
        if code != 0:
            raise TaskError('copy with the comma failed (rsync, code ' + str(code) + ')')


# Where this openpilot keeps driving_supercombo.onnx (asked to git: the layout changes between
# versions), the sha256 of the original model of this version (from git, the onnx is in LFS), of the
# current one and of the backup. $1 = the backup path.
MODEL_INFO_SCRIPT = r'''
backup="$1"
cd /data/openpilot || exit 1
reference="$(git ls-files '*selfdrive/modeld/models/driving_supercombo.onnx' | head -1)"
echo "reference=$reference"
if [ -n "$reference" ]; then
  if [ "$(git cat-file -s "HEAD:$reference")" -lt 1024 ]; then
    echo "original=$(git cat-file -p "HEAD:$reference" | sed -n 's/^oid sha256://p')"
  else
    echo "original=$(git cat-file -p "HEAD:$reference" | sha256sum | cut -d' ' -f1)"
  fi
  echo "current=$(sha256sum "$reference" | cut -d' ' -f1)"
fi
if [ -f "$backup" ]; then
  echo "backup=$(sha256sum "$backup" | cut -d' ' -f1)"
fi
'''


def model_info(comma: Comma) -> dict:
    """reference (path in the openpilot checkout), original, current and backup sha256 ('' when missing)."""
    info = comma.key_values(MODEL_INFO_SCRIPT, BACKUP)
    for key in ('reference', 'original', 'current', 'backup'):
        info.setdefault(key, '')
    if not info['reference']:
        raise TaskError('driving_supercombo.onnx not found in the openpilot checkout on the comma')
    return info


def original_model_source(info: dict) -> str | None:
    """Where the original model of this openpilot version is on the comma: in its place, or the
    backup while a trained model is installed in its place. None when it is in neither."""
    if not info['original']:
        return None
    if info['current'] == info['original']:
        return REMOTE_OPENPILOT + '/' + info['reference']
    if info['backup'] == info['original']:
        return BACKUP
    return None
