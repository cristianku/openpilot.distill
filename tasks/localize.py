"""Run only the localizer on segments already in data/ (writes localizer.safetensors and
frame_info.safetensors next to rlog.zst).

    python -m tasks.localize [<segment> ...]
  python -m tasks.localize                                            (todo + to refine)
  python -m tasks.localize --status                                   (state only, no localizer)
  python -m tasks.localize 0000011c8152ba83cf01                       (folder name in data/)
  python -m tasks.localize 6616faac453a3064/0000011c--8152ba83cf/1    (connect format)
  python -m tasks.localize 0000011c--8152ba83cf 1 2 3
  python -m tasks.localize 0000011c--8152ba83cf 2-9

State of a segment folder:
  done     localizer.safetensors and frame_info.safetensors both present (written only on success)
  refine   done, but a better ionosphere map is out now than the one used (see below)
  failed   localizer.failed present (date + last lines of the error); removed on the next success
  todo     neither
Every run also leaves the full output in localizer.log.

Ionosphere map: the localizer writes the one it used in localizer.ionex ("2026/293 P1D"). P1D
(prediction) is the only one for a drive of today, RAP comes out about one day later, FIN about two
weeks later (details in localizer/ionex.py). Every run without arguments redoes the segments for
which a better map came out: P1D -> RAP -> FIN, one step at a time. Segments done before
localizer.ionex existed have no record and are left alone.

Unusable segments: when the localizer fails because of the data itself (car never above 18 km/h, no
GPS fix, fewer than four raw GPS satellites in every epoch, no calibration yet, incomplete segment:
exit code 3, see localizer/errors.py) the folder is moved to data_scartati/, so the training never
sees it. The reason stays in its localizer.failed. Network or GNSS map problems stay in data/ as
"failed" and are retried. To take a segment back, move its folder from data_scartati/ to data/.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tasks.common import REPO, TaskError, load_env, run_main, say

DATA_DIR = REPO / 'data'
DISCARD_DIR = REPO / 'data_scartati'
# Exit code of localizer.run for a segment unusable because of its data (localizer/errors.py; not
# imported from there: that package needs the project dependencies).
UNUSABLE_SEGMENT_EXIT_CODE = 3
OUTPUTS = ('localizer.safetensors', 'frame_info.safetensors')
FAILED_FILE = 'localizer.failed'
LOG_FILE = 'localizer.log'
IONEX_FILE = 'localizer.ionex'
FAILED_TAIL_LINES = 20

CONNECT_SEGMENT_RE = re.compile(r'^([0-9a-f]+/)?([0-9a-f]+--[0-9a-f]+)/([0-9]+)$')
ROUTE_RE = re.compile(r'^[0-9a-f]+--[0-9a-f]+$')
RANGE_RE = re.compile(r'^[0-9]+-[0-9]+$')
NUMBER_RE = re.compile(r'^[0-9]+$')
NAME_RE = re.compile(r'^[0-9a-f]+$')


def parse_args(arguments: list[str]) -> tuple[list[str], bool, bool]:
    """(segment folder names, --status, --help). Names are read as they come: a route
    (0000011c--8152ba83cf) applies to the numbers and ranges that follow it."""
    names = []
    route = ''
    status_only = False
    for argument in arguments:
        if argument in ('-h', '--help'):
            return names, status_only, True
        if argument == '--status':
            status_only = True
        elif CONNECT_SEGMENT_RE.fullmatch(argument):
            match = CONNECT_SEGMENT_RE.fullmatch(argument)
            names.append(match.group(2).replace('-', '') + f'{int(match.group(3)):02d}')
        elif ROUTE_RE.fullmatch(argument):
            route = argument.replace('-', '')
        elif route and RANGE_RE.fullmatch(argument):
            first, last = argument.split('-')
            names += [route + f'{segment:02d}' for segment in range(int(first), int(last) + 1)]
        elif route and NUMBER_RE.fullmatch(argument):
            names.append(route + f'{int(argument):02d}')
        elif NAME_RE.fullmatch(argument):
            names.append(argument)
        else:
            raise TaskError('segmento non valido: ' + argument)
    return names, status_only, False


def segment_state(name: str) -> str:
    folder = DATA_DIR / name
    if all((folder / output).is_file() for output in OUTPUTS):
        return 'done'
    if (folder / FAILED_FILE).is_file():
        return 'failed'
    return 'todo'


def all_segments() -> list[str]:
    """The folders of data/ with an rlog.zst, in name order."""
    if not DATA_DIR.is_dir():
        return []
    return [folder.name for folder in sorted(DATA_DIR.iterdir())
            if not folder.name.startswith('.') and folder.is_dir() and (folder / 'rlog.zst').is_file()]


def read_ionex_status() -> dict[str, tuple[str, str]]:
    """{folder name: (map used, better map out now or '-')} of the done segments with a
    localizer.ionex (one localizer.ionex run for all of them: it asks the map server)."""
    folders = [str(DATA_DIR / name) for name in all_segments()
               if segment_state(name) == 'done' and (DATA_DIR / name / IONEX_FILE).is_file()]
    if not folders:
        return {}
    result = subprocess.run([sys.executable, '-m', 'localizer.ionex', *folders], cwd=REPO,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True, errors='replace',
                            check=False)
    if result.returncode != 0:
        raise TaskError('controllo delle mappe della ionosfera fallito (codice ' + str(result.returncode) + ')')
    return parse_ionex_lines(result.stdout)


def parse_ionex_lines(text: str) -> dict[str, tuple[str, str]]:
    """'<folder> <map used> <better map or ->' lines, by folder name (the last line of a name wins)."""
    status = {}
    for line in text.splitlines():
        parts = line.split(None, 2)
        if not parts:
            continue
        used = parts[1] if len(parts) > 1 else ''
        better = parts[2] if len(parts) > 2 else ''
        status[parts[0].rpartition('/')[2]] = (used, better)
    return status


def ionex_of(status: dict[str, tuple[str, str]], name: str) -> tuple[str, str]:
    return status.get(name, ('?', '-'))


def _first_line(path: Path) -> str:
    try:
        with path.open('rb') as handle:
            return handle.readline().decode('utf-8', 'replace').rstrip('\n')
    except OSError:
        return ''


def print_status() -> None:
    status = read_ionex_status()
    for name in all_segments():
        state = segment_state(name)
        if state == 'failed':
            say(state + '  ' + name + '  (' + _first_line(DATA_DIR / name / FAILED_FILE) + ')')
        elif state == 'done':
            used, better = ionex_of(status, name)
            if better != '-':
                say('refine  ' + name + '  (mappa ' + used + ', uscita ' + better + ')')
            else:
                say('done  ' + name + '  (mappa ' + used + ')')
        else:
            say(state + '  ' + name)
    if DISCARD_DIR.is_dir():
        for folder in sorted(DISCARD_DIR.iterdir()):
            if folder.name.startswith('.') or not folder.is_dir():
                continue
            say('scartato  ' + folder.name + '  (' + _first_line(folder / FAILED_FILE) + ')')


def segments_to_do() -> list[str]:
    """Every folder of data/ not done yet (todo or failed), plus the done ones for which a better
    ionosphere map came out (refine)."""
    status = read_ionex_status()
    names = []
    for name in all_segments():
        if segment_state(name) != 'done':
            names.append(name)
            continue
        used, better = ionex_of(status, name)
        if better != '-':
            say(name + ': rifaccio con la mappa ' + better + ' (prima ' + used + ')')
            names.append(name)
    return names


def tail_lines(data: bytes, count: int) -> bytes:
    """The last `count` lines of `data`, as tail -n prints them."""
    lines = re.findall(rb'[^\n]*\n|[^\n]+$', data)
    return b''.join(lines[-count:])


def run_localizer(name: str, log_path: Path) -> int:
    """localizer.run on one segment: its output (stdout and stderr) goes here AND to localizer.log.
    The exit code, 128 + signal when a signal killed it."""
    environment = dict(os.environ)
    environment['PYTHONUNBUFFERED'] = '1'
    with log_path.open('wb') as log:
        child = subprocess.Popen([sys.executable, '-m', 'localizer.run', '--segment', name], cwd=REPO,
                                 env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
        while True:
            chunk = child.stdout.read1(65536)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log.write(chunk)
            log.flush()
        code = child.wait()
    return code if code >= 0 else 128 - code


def write_failed(folder: Path) -> None:
    """localizer.failed: date and last line of the log, then its last 20 lines."""
    try:
        log = (folder / LOG_FILE).read_bytes()
    except OSError:
        log = b''
    last = tail_lines(log, 1).rstrip(b'\n')
    header = time.strftime('%Y-%m-%d %H:%M').encode() + b' ' + last + b'\n'
    (folder / FAILED_FILE).write_bytes(header + tail_lines(log, FAILED_TAIL_LINES))


def discard(name: str) -> None:
    """Moves data/<name> to data_scartati/; one already discarded once (downloaded again) keeps both,
    never overwritten."""
    DISCARD_DIR.mkdir(parents=True, exist_ok=True)
    target = DISCARD_DIR / name
    if os.path.lexists(target):
        target = DISCARD_DIR / (name + '.' + time.strftime('%Y%m%d%H%M%S'))
    shutil.move(str(DATA_DIR / name), str(target))


def localize(names: list[str]) -> int:
    ok = []
    failed = []
    discarded = []
    say('PROGRESS 0/' + str(len(names)))
    for done, name in enumerate(names, start=1):
        folder = DATA_DIR / name
        say('== ' + name)
        if not (folder / 'rlog.zst').is_file():
            say('manca ' + str(folder / 'rlog.zst'))
            failed.append(name)
        else:
            code = run_localizer(name, folder / LOG_FILE)
            if code == 0:
                (folder / FAILED_FILE).unlink(missing_ok=True)
                ok.append(name)
            else:
                write_failed(folder)
                # Unusable because of its data: out of data/. A segment that already has outputs
                # (failed refine) is never moved: its old outputs are still good.
                if code == UNUSABLE_SEGMENT_EXIT_CODE and not (folder / 'localizer.safetensors').is_file():
                    discard(name)
                    say(name + ': dati inutilizzabili, spostato in data_scartati/')
                    discarded.append(name)
                else:
                    failed.append(name)
        say('PROGRESS ' + str(done) + '/' + str(len(names)))

    say('OK:' + (''.join(' ' + name for name in ok) or ' nessuno'))
    if discarded:
        say('Scartati (in data_scartati/):' + ''.join(' ' + name for name in discarded))
    if failed:
        say('Falliti:' + ''.join(' ' + name for name in failed))
        return 1
    return 0


def main() -> int:
    names, status_only, show_help = parse_args(sys.argv[1:])
    if show_help:
        print(__doc__)
        return 0
    load_env()
    try:
        if status_only:
            print_status()
            return 0
        if not names:
            names = segments_to_do()
            if not names:
                say('Niente da fare: tutti i segmenti sono gia\' localizzati con la mappa migliore uscita.')
                return 0
        return localize(names)
    except OSError as exc:
        raise TaskError('errore sul disco: ' + str(exc)) from exc


if __name__ == '__main__':
    run_main(main)
