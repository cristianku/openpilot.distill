"""Update this clone from GitHub and put the new code to work, always the same way.

    python -m tasks.update

  0. refuses while a job uses this clone (a training, an evaluation, a comparison video, a download,
     a localization, an export, a publication, an installation): the package sync would change the
     packages under it. The GPU lock is held until the update ends, so no GPU job starts meanwhile.
  1. git pull --ff-only of the current branch. Refuses when files of the repo are modified locally
     (it lists them) or the branch has diverged: nothing is changed then. Then the submodules
     (perception_repo, openpilot.perception) go to the commit the new code records.
  2. uv sync --locked, with --extra DISTILL_UV_EXTRA when the repo .env sets it (v100 on the Tesla
     V100 server). --locked: uv.lock is used as it is and never rewritten.
  3. the Android app (helpers/build_android.sh), when the update changed app/android or its build, or
     the server has none yet: the site then offers the new APK, and a server with the key of the Google
     Play service account publishes it on the internal track. A failure is only a warning.
  4. with the systemd service distill-web installed: restarts it, so the site runs the new code. The
     jobs started from the site keep running (KillMode=process in the unit).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tasks.common import REPO, GpuLock, TaskError, load_env, run_main, say

SERVICE = 'distill-web'
# Jobs that run from .venv, by a fragment of their command line: updating the packages under them
# would break them. The helper scripts too, while they exist and can be started by hand.
BUSY_JOBS = (
    (re.compile(r'supervised[.]train|tasks[.]train\b|helpers/train_supervised[.]sh'), 'a training'),
    (re.compile(r'evaluation[.](compare|holdout)|tasks[.]evaluation\b|helpers/(compare_video|eval_holdout)[.]sh'),
     'an evaluation or a comparison video'),
    (re.compile(r'tasks[.]fetch\b|helpers/fetch_comma_segments[.]sh'), 'a download from the comma'),
    (re.compile(r'localizer[.]run|tasks[.]localize\b|helpers/localize_segments[.]sh'), 'a localization'),
    (re.compile(r'helpers[.]export_onnx|tasks[.]export\b|helpers/export_onnx[.]sh'), 'an ONNX export'),
    (re.compile(r'tasks[.]publish\b|helpers/publish_model[.]sh'), 'a publication'),
    (re.compile(r'tasks[.]install\b|helpers/install_model_on_comma[.]sh'), 'an installation on the comma'),
    (re.compile(r'helpers/build_android[.]sh'), 'a build of the Android app'),
)
# What the Android app is built from: a change there means a new APK.
ANDROID_SOURCES = ('app/android/', 'helpers/build_android.sh')


def busy_jobs() -> list[str]:
    """The jobs that use this clone, in words, from the command lines in /proc."""
    found = []
    proc = Path('/proc')
    if not proc.is_dir():
        return found
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace')
        except OSError:
            continue
        for pattern, label in BUSY_JOBS:
            if pattern.search(command) and label not in found:
                found.append(label)
    return found


def git(*arguments: str, check: bool = True) -> str:
    environment = dict(os.environ)
    # A remote that wants a password fails instead of waiting for it.
    environment['GIT_TERMINAL_PROMPT'] = '0'
    result = subprocess.run(['git', *arguments], cwd=REPO, env=environment, capture_output=True, text=True,
                            stdin=subprocess.DEVNULL)
    if check and result.returncode != 0:
        raise TaskError('git ' + arguments[0] + ' failed: ' + (result.stderr.strip() or 'unknown error'))
    return result.stdout.strip()


def uv_executable() -> str:
    for candidate in (shutil.which('uv'), str(Path.home() / '.local' / 'bin' / 'uv')):
        if candidate and Path(candidate).is_file():
            return candidate
    raise TaskError('uv not found on this server')


def service_installed() -> bool:
    if shutil.which('systemctl') is None:
        return False
    result = subprocess.run(['systemctl', 'list-unit-files', SERVICE + '.service'], capture_output=True, text=True,
                            stdin=subprocess.DEVNULL)
    return any(line.startswith(SERVICE + '.service') for line in result.stdout.splitlines())


def main() -> int:
    if sys.argv[1:] in (['-h'], ['--help']):
        print(__doc__)
        return 0
    if len(sys.argv) > 1:
        raise TaskError('invalid option: ' + sys.argv[1])
    load_env()
    uv = uv_executable()

    # 0. Jobs that use this clone.
    busy = busy_jobs()
    if busy:
        raise TaskError('running now: ' + ', '.join(busy) + ': update when it has finished')
    lock = GpuLock()
    lock.acquire()
    if git('rev-parse', '--is-inside-work-tree', check=False) != 'true':
        raise TaskError('this folder is not a git clone: nothing to update')

    # 1. git pull.
    modified = git('status', '--porcelain', '--untracked-files=no')
    if modified:
        say('Repo files modified locally:')
        say(modified)
        raise TaskError('the repo files have local changes (above): not overwriting them')
    before = git('rev-parse', 'HEAD')
    say('== 1. downloading the new code (branch ' + git('branch', '--show-current') + ')')
    try:
        git('pull', '--ff-only')
    except TaskError as exc:
        raise TaskError(str(exc) + ' (diverged branch or network?): nothing changed') from None
    after = git('rev-parse', 'HEAD')
    if before == after:
        say('Already up to date: ' + git('log', '--oneline', '-1'))
    else:
        say(git('log', '--oneline', before + '..' + after))
    try:
        git('submodule', 'update', '--init', '--recursive')
    except TaskError as exc:
        raise TaskError(str(exc) + ' (network?): the code is updated, its submodules are not') from None

    # 2. Packages of .venv.
    command = [uv, 'sync', '--locked']
    extra = os.environ.get('DISTILL_UV_EXTRA', '')
    if extra:
        command += ['--extra', extra]
    say('== 2. updating the packages' + (' (extra ' + extra + ')' if extra else ''))
    if subprocess.run(command, cwd=REPO, stdin=subprocess.DEVNULL).returncode != 0:
        raise TaskError('package update failed: the code is updated but the packages are not '
                        '(uv.lock out of sync with pyproject.toml?)')

    # 3. The Android app.
    changed = git('diff', '--name-only', before, after).splitlines() if before != after else []
    never_built = not (REPO / 'outputs' / 'android' / 'latest.json').is_file()
    if any(name.startswith(ANDROID_SOURCES) for name in changed) or never_built:
        say('== 3. building the Android app' + (' (none on this server yet)' if never_built else ''))
        build = subprocess.run([str(REPO / 'helpers' / 'build_android.sh')], cwd=REPO, stdin=subprocess.DEVNULL)
        if build.returncode != 0:
            say('WARNING: the Android app was not built (above); the site works without it')
    else:
        say('== 3. the Android app did not change')

    # 4. The web site.
    if not service_installed():
        say('== 4. the site does not run as a service: restart it to use the new code')
        say('Done.')
        return 0
    if before != after and any(name.startswith('deploy/') for name in git('diff', '--name-only', before, after)
                               .splitlines()):
        say('WARNING: the service file changed (deploy/): reinstall it to use it')
    say('== 4. restarting the site')
    subprocess.run(['systemctl', 'restart', SERVICE], stdin=subprocess.DEVNULL, check=False)
    time.sleep(2)
    active = subprocess.run(['systemctl', 'is-active', '--quiet', SERVICE], stdin=subprocess.DEVNULL).returncode == 0
    if not active:
        raise TaskError('the site does not start again after the update: check its log')
    say('Site restarted with the new code.')
    say('Done.')
    return 0


if __name__ == '__main__':
    run_main(main)
