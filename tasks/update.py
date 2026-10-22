"""Update this clone from GitHub and put the new code to work, always the same way.

    python -m tasks.update

  0. refuses while a job uses this clone (a training, an evaluation, a comparison video, a download,
     a localization, an export, a publication, an installation): the package sync would change the
     packages under it. The GPU lock is held until the update ends, so no GPU job starts meanwhile.
  1. git pull --ff-only of the current branch. Refuses when files of the repo are modified locally
     (it lists them) or the branch has diverged: nothing is changed then.
  2. uv sync --locked, with --extra DISTILL_UV_EXTRA when the repo .env sets it (v100 on the Tesla
     V100 server). --locked: uv.lock is used as it is and never rewritten.
  3. with the systemd service distill-web installed: restarts it, so the site runs the new code. The
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
    (re.compile(r'supervised[.]train|tasks[.]train\b|helpers/train_supervised[.]sh'), 'un training'),
    (re.compile(r'evaluation[.](compare|holdout)|tasks[.]evaluation\b|helpers/(compare_video|eval_holdout)[.]sh'),
     'una valutazione o un video di confronto'),
    (re.compile(r'tasks[.]fetch\b|helpers/fetch_comma_segments[.]sh'), 'uno scaricamento dal comma'),
    (re.compile(r'localizer[.]run|tasks[.]localize\b|helpers/localize_segments[.]sh'), 'una localizzazione'),
    (re.compile(r'helpers[.]export_onnx|tasks[.]export\b|helpers/export_onnx[.]sh'), 'un export ONNX'),
    (re.compile(r'tasks[.]publish\b|helpers/publish_model[.]sh'), 'una pubblicazione'),
    (re.compile(r'tasks[.]install\b|helpers/install_model_on_comma[.]sh'), 'un\'installazione sul comma'),
)


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
        raise TaskError('git ' + arguments[0] + ' non riuscito: ' + (result.stderr.strip() or 'errore sconosciuto'))
    return result.stdout.strip()


def uv_executable() -> str:
    for candidate in (shutil.which('uv'), str(Path.home() / '.local' / 'bin' / 'uv')):
        if candidate and Path(candidate).is_file():
            return candidate
    raise TaskError('uv non trovato su questo server')


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
        raise TaskError('opzione non valida: ' + sys.argv[1])
    load_env()
    uv = uv_executable()

    # 0. Jobs that use this clone.
    busy = busy_jobs()
    if busy:
        raise TaskError('c\'e\' in corso ' + ', '.join(busy) + ': aggiorna quando ha finito')
    lock = GpuLock()
    lock.acquire()
    if git('rev-parse', '--is-inside-work-tree', check=False) != 'true':
        raise TaskError('questa cartella non e\' un clone git: niente da aggiornare')

    # 1. git pull.
    modified = git('status', '--porcelain', '--untracked-files=no')
    if modified:
        say('File del repo modificati in locale:')
        say(modified)
        raise TaskError('ci sono modifiche locali ai file del repo (sopra): non le sovrascrivo')
    before = git('rev-parse', 'HEAD')
    say('== 1. scarico il codice nuovo (branch ' + git('branch', '--show-current') + ')')
    try:
        git('pull', '--ff-only')
    except TaskError as exc:
        raise TaskError(str(exc) + ' (branch divergente o rete?): niente e\' cambiato') from None
    after = git('rev-parse', 'HEAD')
    if before == after:
        say('Gia\' aggiornato: ' + git('log', '--oneline', '-1'))
    else:
        say(git('log', '--oneline', before + '..' + after))

    # 2. Packages of .venv.
    command = [uv, 'sync', '--locked']
    extra = os.environ.get('DISTILL_UV_EXTRA', '')
    if extra:
        command += ['--extra', extra]
    say('== 2. aggiorno le librerie' + (' (extra ' + extra + ')' if extra else ''))
    if subprocess.run(command, cwd=REPO, stdin=subprocess.DEVNULL).returncode != 0:
        raise TaskError('aggiornamento delle librerie non riuscito: il codice e\' aggiornato ma le librerie no '
                        '(uv.lock non allineato a pyproject.toml?)')

    # 3. The web site.
    if not service_installed():
        say('== 3. il sito non gira come servizio: riavvialo per usare il codice nuovo')
        say('Fatto.')
        return 0
    if before != after and any(name.startswith('deploy/') for name in git('diff', '--name-only', before, after)
                               .splitlines()):
        say('ATTENZIONE: e\' cambiato il file del servizio (deploy/): va reinstallato per usarlo')
    say('== 3. riavvio il sito')
    subprocess.run(['systemctl', 'restart', SERVICE], stdin=subprocess.DEVNULL, check=False)
    time.sleep(2)
    active = subprocess.run(['systemctl', 'is-active', '--quiet', SERVICE], stdin=subprocess.DEVNULL).returncode == 0
    if not active:
        raise TaskError('il sito non riparte dopo l\'aggiornamento: controlla il suo log')
    say('Sito riavviato con il codice nuovo.')
    say('Fatto.')
    return 0


if __name__ == '__main__':
    run_main(main)
