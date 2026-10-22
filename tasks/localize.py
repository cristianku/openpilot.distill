"""Run only the localizer on segments already in data/ (writes localizer.safetensors and
frame_info.safetensors next to rlog.zst).

    python -m tasks.localize [<segment> ...]
  python -m tasks.localize                                            (todo + to refine)
  python -m tasks.localize --status                                   (state only, no localizer)
  python -m tasks.localize 0000011c8152ba83cf01                       (folder name in data/)
  python -m tasks.localize 6616faac453a3064/0000011c--8152ba83cf/1    (connect format)
  python -m tasks.localize 0000011c--8152ba83cf 1 2 3
  python -m tasks.localize 0000011c--8152ba83cf 2-9
  python -m tasks.localize --jobs 4                                   (4 segments at a time)

State of a segment folder:
  done     localizer.safetensors and frame_info.safetensors both present (written only on success)
  refine   done, but a better ionosphere map is out now than the one used (see below)
  failed   localizer.failed present (date + last lines of the error); removed on the next success
  todo     neither
Every run also leaves the full output in localizer.log.

Several segments run at the same time (--jobs, by default up to 8, fewer with few cores or little free
memory): one localizer.run uses about one core. Each one gets at most 2 numeric threads, so they do
not fight over the cores. The job log gets one line per segment when it ends; the full output of a
segment is in its localizer.log.

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
from concurrent.futures import ThreadPoolExecutor, as_completed
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


# Segments localized at the same time: at most this many, and never more than half the cores or one
# per MEMORY_PER_JOB_GB of free memory.
MAX_JOBS = 8
MEMORY_PER_JOB_GB = 1.5
# Numeric threads (BLAS, OpenMP) of every localizer.run.
THREADS_PER_JOB = '2'
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
# --jobs N or --jobs=N, read by parse_args into this (None = default_jobs()).
jobs_option = {'value': None}


def default_jobs() -> int:
    jobs = min(MAX_JOBS, max(1, (os.cpu_count() or 1) // 2))
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                available_gb = int(line.split()[1]) / 1024 / 1024
                jobs = min(jobs, max(1, int(available_gb / MEMORY_PER_JOB_GB)))
    except (OSError, ValueError, IndexError):
        pass
    return jobs


def parse_args(arguments: list[str]) -> tuple[list[str], bool, bool]:
    """(segment folder names, --status, --help). Names are read as they come: a route
    (0000011c--8152ba83cf) applies to the numbers and ranges that follow it. --jobs goes to jobs_option."""
    names = []
    route = ''
    status_only = False
    jobs_option['value'] = None
    arguments = list(arguments)
    while arguments:
        argument = arguments.pop(0)
        if argument in ('-h', '--help'):
            return names, status_only, True
        if argument == '--jobs' or argument.startswith('--jobs='):
            value = argument.split('=', 1)[1] if '=' in argument else (arguments.pop(0) if arguments else '')
            if not value.isdigit() or not 1 <= int(value) <= 64:
                raise TaskError('--jobs wants a number from 1 to 64')
            jobs_option['value'] = int(value)
            continue
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
            raise TaskError('invalid segment: ' + argument)
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
        raise TaskError('ionosphere map check failed (code ' + str(result.returncode) + ')')
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
                say('refine  ' + name + '  (map ' + used + ', out now ' + better + ')')
            else:
                say('done  ' + name + '  (map ' + used + ')')
        else:
            say(state + '  ' + name)
    if DISCARD_DIR.is_dir():
        for folder in sorted(DISCARD_DIR.iterdir()):
            if folder.name.startswith('.') or not folder.is_dir():
                continue
            say('discarded  ' + folder.name + '  (' + _first_line(folder / FAILED_FILE) + ')')


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
            say(name + ': redoing with map ' + better + ' (was ' + used + ')')
            names.append(name)
    return names


def tail_lines(data: bytes, count: int) -> bytes:
    """The last `count` lines of `data`, as tail -n prints them."""
    lines = re.findall(rb'[^\n]*\n|[^\n]+$', data)
    return b''.join(lines[-count:])


def run_localizer(name: str, log_path: Path) -> int:
    """localizer.run on one segment, its output (stdout and stderr) into localizer.log. The exit code,
    128 + signal when a signal killed it. Runs in a worker thread: it touches only its own folder."""
    environment = dict(os.environ)
    environment['PYTHONUNBUFFERED'] = '1'
    for variable in THREAD_VARIABLES:
        environment[variable] = THREADS_PER_JOB
    with log_path.open('wb') as log:
        code = subprocess.run([sys.executable, '-m', 'localizer.run', '--segment', name], cwd=REPO,
                              env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                              check=False).returncode
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


def localize(names: list[str], jobs: int | None = None) -> int:
    """Localizes the segments, `jobs` at a time; the outcome of each one (localizer.failed, moving an
    unusable one to data_scartati/) is handled here, one at a time, as they end."""
    if jobs is None:
        jobs = jobs_option['value'] or default_jobs()
    ok = []
    failed = []
    discarded = []
    done = 0
    say('PROGRESS 0/' + str(len(names)))
    runnable = []
    for name in names:
        if (DATA_DIR / name / 'rlog.zst').is_file():
            runnable.append(name)
        else:
            say('missing ' + str(DATA_DIR / name / 'rlog.zst'))
            failed.append(name)
            done = done + 1
            say('PROGRESS ' + str(done) + '/' + str(len(names)))
    if runnable:
        say('Localizing ' + str(len(runnable)) + ' segments, ' + str(min(jobs, len(runnable))) + ' at a time.')

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        started = {}
        futures = {}
        for name in runnable:
            started[name] = time.monotonic()
            futures[pool.submit(run_localizer, name, DATA_DIR / name / LOG_FILE)] = name
        for future in as_completed(futures):
            name = futures[future]
            folder = DATA_DIR / name
            took = str(round(time.monotonic() - started[name])) + ' s'
            try:
                code = future.result()
            except OSError as exc:
                say('failed  ' + name + ': ' + str(exc))
                code = None
            if code == 0:
                (folder / FAILED_FILE).unlink(missing_ok=True)
                ok.append(name)
                say('ok      ' + name + ' (' + took + ')')
            elif code is None:
                failed.append(name)
            else:
                write_failed(folder)
                reason = tail_lines((folder / LOG_FILE).read_bytes(), 1).decode('utf-8', 'replace').strip()
                # Unusable because of its data: out of data/. A segment that already has outputs
                # (failed refine) is never moved: its old outputs are still good.
                if code == UNUSABLE_SEGMENT_EXIT_CODE and not (folder / 'localizer.safetensors').is_file():
                    discard(name)
                    say(name + ': unusable data, moved to data_scartati/ (' + reason + ')')
                    discarded.append(name)
                else:
                    say('failed  ' + name + ' (' + took + '): ' + reason)
                    failed.append(name)
            done = done + 1
            say('PROGRESS ' + str(done) + '/' + str(len(names)))

    # In the order they were asked for, whatever order they ended in.
    order = {name: index for index, name in enumerate(names)}
    for outcome in (ok, failed, discarded):
        outcome.sort(key=lambda name: order[name])
    say('OK:' + (''.join(' ' + name for name in ok) or ' none'))
    if discarded:
        say('Discarded (in data_scartati/):' + ''.join(' ' + name for name in discarded))
    if failed:
        say('Failed:' + ''.join(' ' + name for name in failed))
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
                say('Nothing to do: every segment is already localized with the best map out.')
                return 0
        return localize(names)
    except OSError as exc:
        raise TaskError('disk error: ' + str(exc)) from exc


if __name__ == '__main__':
    run_main(main)
