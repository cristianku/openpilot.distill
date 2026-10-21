"""What the site shows about the machine and the data: CPU, RAM, disk, GPU and its processes, the
jobs running on the machine (also those started from a terminal or by sunnypilot.speedlimits), the
segments in data/ and data_scartati/, the evaluation results and the comparison videos.

Everything here only reads. Nothing holds the GPU lock: gpu_lock_held() takes it for an instant to
test it and releases it at once.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from web.actions import GPU_LOCK_FILE
from web.common import SEGMENT_RE, inside, is_heldout, route_of, short_text

# Comparison videos: outputs/compare/<name>.mp4 (+ <name>.json). One plain file name, nothing else.
VIDEO_NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,150}\.mp4$')
EVAL_NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.:-]{0,150}\.json$')
# The first line of localizer.failed: "YYYY-MM-DD HH:MM <last line of the error>".
FAILED_DATE_PREFIX_RE = re.compile(r'^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} ')


@dataclass(frozen=True)
class Activity:
    # The site group it keeps busy ('other' = shown only, never blocks an action).
    group: str
    label: str
    # Matched as a substring of the process command line.
    fragment: str
    # The process uses (or is about to use) the GPU: a GPU job of the site must wait.
    uses_gpu: bool


# The same work started from a terminal, or by the other project on this server.
ACTIVITIES = (
    Activity('gpu', 'training distill', 'supervised.train', True),
    Activity('gpu', 'valutazione hold-out', 'evaluation.holdout', True),
    Activity('gpu', 'video di confronto', 'evaluation.compare', True),
    Activity('data', 'scaricamento dal comma', 'helpers/fetch_comma_segments.sh', False),
    Activity('data', 'localizzazione', 'helpers/localize_segments.sh', False),
    Activity('data', 'localizzazione', 'localizer.run', False),
    Activity('cpu', 'export ONNX', 'helpers/export_onnx.sh', False),
    Activity('cpu', 'pubblicazione', 'helpers/publish_model.sh', False),
    Activity('comma', 'installazione sul comma', 'helpers/install_model_on_comma.sh', False),
    Activity('system', 'aggiornamento del codice', 'helpers/update.sh', False),
    # sunnypilot.speedlimits (same server, same GPU). It does not take the GPU lock.
    Activity('other', 'speedlimits: training', 'run_v100_baseline.sh', True),
    Activity('other', 'speedlimits: training', 'helpers/3_train.sh', True),
    Activity('other', 'speedlimits: training', 'nnslr train', True),
    Activity('other', 'speedlimits: preparazione', 'helpers/1_prepare.sh', False),
    Activity('other', 'speedlimits: pre-annotazione', 'nnslr preannotate', False),
    Activity('other', 'speedlimits: estrazione frame', 'nnslr extract-frames', False),
)


# -- processes ---------------------------------------------------------------------------------------
def process_command_line(pid: int) -> str | None:
    """The command line of a process, None when it is gone. /proc on Linux, ps elsewhere (the Mac,
    for the tests)."""
    if Path('/proc').is_dir():
        try:
            raw = (Path('/proc') / str(pid) / 'cmdline').read_bytes()
        except OSError:
            return None
        if not raw:
            # A zombie (ended, not reaped yet) has an empty command line.
            return None
        return raw.replace(b'\0', b' ').decode('utf-8', 'replace').strip()
    try:
        # -ww: the whole command line, never cut to a terminal width.
        result = subprocess.run(['ps', '-ww', '-o', 'command=', '-p', str(pid)], capture_output=True, text=True,
                                timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    text = result.stdout.strip()
    if result.returncode != 0 or not text or '<defunct>' in text:
        return None
    return text


def process_cwd(pid: int) -> str | None:
    try:
        return os.readlink('/proc/' + str(pid) + '/cwd')
    except OSError:
        return None


def activities(repo: Path) -> list[dict]:
    """Known jobs running on this machine, from the command lines in /proc (empty without /proc).
    A match whose working directory is another folder (e.g. the same helper of
    sunnypilot.speedlimits, or the live clone seen from a test copy) is shown as 'other'."""
    found = []
    proc = Path('/proc')
    if not proc.is_dir():
        return found
    own_pid = os.getpid()
    repo_text = str(repo.resolve())
    seen = set()
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == own_pid:
            continue
        try:
            command = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace')
        except OSError:
            continue
        for activity in ACTIVITIES:
            if activity.fragment not in command:
                continue
            group = activity.group
            label = activity.label
            if group != 'other':
                cwd = process_cwd(pid)
                if cwd is not None and cwd != repo_text and not cwd.startswith(repo_text + '/'):
                    group = 'other'
                    label = label + ' (altra cartella)'
            # One entry per label and group: a job has many processes (bash, uv, python, workers).
            key = (group, label)
            if key in seen:
                break
            seen.add(key)
            found.append({'group': group, 'label': label, 'pid': pid, 'uses_gpu': activity.uses_gpu,
                          'command': short_text(command.strip(), 300)})
            break
    return found


# -- machine -----------------------------------------------------------------------------------------
def _run(arguments: list[str], timeout: float = 5) -> str | None:
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _number(text: str):
    text = text.strip()
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def memory_info() -> dict | None:
    """RAM from /proc/meminfo (inside the LXC it is the container's own memory)."""
    try:
        text = Path('/proc/meminfo').read_text()
    except OSError:
        return None
    values = {}
    for line in text.splitlines():
        name, separator, rest = line.partition(':')
        if not separator:
            continue
        parts = rest.split()
        if parts and parts[0].isdigit():
            values[name.strip()] = int(parts[0]) * 1024
    if 'MemTotal' not in values or 'MemAvailable' not in values:
        return None
    total = values['MemTotal']
    available = values['MemAvailable']
    return {'total_gb': round(total / 1e9, 1), 'available_gb': round(available / 1e9, 1),
            'used_percent': round((total - available) * 100 / total, 1)}


def gpu_info() -> dict | None:
    output = _run(['nvidia-smi',
                   '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                   '--format=csv,noheader,nounits'])
    if not output or not output.strip():
        return None
    fields = output.strip().splitlines()[0].split(',')
    if len(fields) < 6:
        return None
    info = {'name': fields[0].strip(), 'utilization': _number(fields[1]), 'memory_used_mb': _number(fields[2]),
            'memory_total_mb': _number(fields[3]), 'temperature': _number(fields[4]), 'power_w': _number(fields[5]),
            'processes': []}
    apps = _run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'])
    if apps:
        for line in apps.splitlines():
            if not line.strip():
                continue
            parts = line.split(',')
            if len(parts) < 3:
                continue
            pid_text = parts[0].strip()
            name = ','.join(parts[1:-1]).strip()
            process = {'pid': int(pid_text) if pid_text.isdigit() else None, 'name': name,
                       'memory_mb': _number(parts[-1]), 'command': None}
            # Inside the LXC nvidia-smi reports host pids and "[Not Found]" names: a pid that happens
            # to exist here is another process, so the command line is looked up only when nvidia-smi
            # could name the process.
            if process['pid'] is not None and name and 'Not Found' not in name:
                command = process_command_line(process['pid'])
                if command is not None:
                    process['command'] = short_text(command, 300)
            info['processes'].append(process)
    return info


def system_info(repo: Path) -> dict:
    info = {'cpus': os.cpu_count(), 'load': None, 'memory': memory_info(), 'disk': None, 'gpu': gpu_info()}
    try:
        load = os.getloadavg()
        info['load'] = [round(load[0], 2), round(load[1], 2), round(load[2], 2)]
    except OSError:
        pass
    try:
        usage = shutil.disk_usage(repo)
        info['disk'] = {'free_gb': round(usage.free / 1e9, 1), 'total_gb': round(usage.total / 1e9, 1)}
    except OSError:
        pass
    return info


# -- GPU lock ----------------------------------------------------------------------------------------
def gpu_lock_held(lock_file: str = GPU_LOCK_FILE) -> bool | None:
    """True when another process holds the GPU lock (flock, as helpers/common.sh takes it).
    Takes the lock for an instant and releases it: call it only right before starting a GPU job,
    never in a periodic status (a helper doing "flock -n" at that very instant would be refused)."""
    try:
        handle = os.open(lock_file, os.O_RDONLY)
    except FileNotFoundError:
        # Nobody ever took it since the last boot.
        return False
    except OSError:
        return None
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    except OSError:
        return None
    else:
        fcntl.flock(handle, fcntl.LOCK_UN)
        return False
    finally:
        os.close(handle)


def gpu_lock_holders(lock_file: str = GPU_LOCK_FILE, limit: int = 3) -> list[str]:
    """'pid N: command' of the processes that have the lock file open (the holder and the children
    that inherited its fd). /proc/locks is no use inside the LXC, so the open fds are scanned."""
    holders = []
    proc = Path('/proc')
    if not proc.is_dir():
        return holders
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            descriptors = list((entry / 'fd').iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            if target == lock_file:
                command = process_command_line(int(entry.name))
                if command is not None:
                    holders.append('pid ' + entry.name + ': ' + short_text(command, 120))
                break
        if len(holders) >= limit:
            break
    return holders


def gpu_busy_reason(repo: Path, lock_file: str = GPU_LOCK_FILE) -> str:
    """'' when a GPU job can start, else why not (Italian). Checks the lock, the processes on the GPU
    (jobs that do not use the lock, e.g. a speedlimits training) and the known GPU jobs in /proc."""
    reasons = []
    held = gpu_lock_held(lock_file)
    if held is True:
        holders = gpu_lock_holders(lock_file)
        if holders:
            reasons.append('lock ' + lock_file + ' preso da ' + '; '.join(holders))
        else:
            reasons.append('lock ' + lock_file + ' preso')
    elif held is None:
        reasons.append('non riesco a controllare il lock ' + lock_file)
    gpu = gpu_info()
    if gpu is not None and gpu['processes']:
        names = []
        for process in gpu['processes']:
            names.append('pid ' + str(process['pid']) + ' ' + process['name'])
        reasons.append('processi sulla GPU: ' + ', '.join(names))
    for activity in activities(repo):
        if activity['uses_gpu']:
            reasons.append(activity['label'] + ' (pid ' + str(activity['pid']) + ')')
    if not reasons:
        return ''
    return 'La GPU e\' occupata da un altro job (' + '; '.join(reasons) + '). Riprova quando ha finito.'


# -- data ----------------------------------------------------------------------------------------------
def _first_line(path: Path, limit: int = 300) -> str:
    try:
        with path.open('r', encoding='utf-8', errors='replace') as handle:
            line = handle.readline(4096)
    except OSError:
        return ''
    return short_text(line.strip(), limit)


def failure_reason(path: Path) -> str:
    """The error of a localizer.failed without its date: the same text groups the same problem."""
    line = _first_line(path)
    return FAILED_DATE_PREFIX_RE.sub('', line, count=1)


def segments(repo: Path) -> dict:
    """The segment folders of data/ and data_scartati/ with their localizer state."""
    data_dir = repo / 'data'
    rows = []
    routes = {}
    counts = {'total': 0, 'localized': 0, 'failed': 0, 'todo': 0, 'heldout': 0, 'heldout_localized': 0,
              'broken_video': 0, 'discarded': 0}
    entries = []
    if data_dir.is_dir():
        entries = sorted(data_dir.iterdir())
    for entry in entries:
        if not entry.is_dir():
            continue
        name = entry.name
        localized = (entry / 'localizer.safetensors').is_file() and (entry / 'frame_info.safetensors').is_file()
        failed = (entry / 'localizer.failed').is_file()
        if localized:
            state = 'localized'
        elif failed:
            state = 'failed'
        else:
            state = 'todo'
        video = entry / 'fcamera.hevc'
        # A link to a file of the shared folder that is not there: the training would stop on it.
        broken_video = video.is_symlink() and not video.exists()
        heldout = is_heldout(name)
        row = {'name': name, 'state': state, 'heldout': heldout, 'broken_video': broken_video,
               'has_video': video.is_file(), 'map': '', 'reason': ''}
        if SEGMENT_RE.fullmatch(name):
            row['route'] = route_of(name)
            row['number'] = int(name[18:20])
        else:
            row['route'] = name
            row['number'] = None
        if localized:
            row['map'] = _first_line(entry / 'localizer.ionex', 40)
        elif failed:
            row['reason'] = failure_reason(entry / 'localizer.failed')
        rows.append(row)

        counts['total'] = counts['total'] + 1
        if state == 'localized':
            counts['localized'] = counts['localized'] + 1
        elif state == 'failed':
            counts['failed'] = counts['failed'] + 1
        else:
            counts['todo'] = counts['todo'] + 1
        if heldout:
            counts['heldout'] = counts['heldout'] + 1
            if localized:
                counts['heldout_localized'] = counts['heldout_localized'] + 1
        if broken_video:
            counts['broken_video'] = counts['broken_video'] + 1

        route = routes.get(row['route'])
        if route is None:
            route = {'route': row['route'], 'segments': 0, 'localized': 0, 'failed': 0, 'todo': 0, 'heldout': 0}
            routes[row['route']] = route
        route['segments'] = route['segments'] + 1
        if state == 'localized':
            route['localized'] = route['localized'] + 1
        elif state == 'failed':
            route['failed'] = route['failed'] + 1
        else:
            route['todo'] = route['todo'] + 1
        if heldout:
            route['heldout'] = route['heldout'] + 1

    discarded = []
    reasons = {}
    discard_dir = repo / 'data_scartati'
    discard_entries = []
    if discard_dir.is_dir():
        discard_entries = sorted(discard_dir.iterdir())
    for entry in discard_entries:
        if not entry.is_dir():
            continue
        reason = failure_reason(entry / 'localizer.failed')
        if not reason:
            reason = 'motivo non registrato'
        discarded.append({'name': entry.name, 'reason': reason})
        reasons[reason] = reasons.get(reason, 0) + 1
    counts['discarded'] = len(discarded)
    reason_rows = []
    for reason, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0])):
        reason_rows.append({'reason': reason, 'count': count})

    route_rows = []
    for key in sorted(routes):
        route_rows.append(routes[key])
    return {'counts': counts, 'segments': rows, 'routes': route_rows, 'discarded': discarded,
            'discarded_reasons': reason_rows}


# -- evaluation results ------------------------------------------------------------------------------
def _first_number(data: dict, keys: tuple[str, ...]):
    for key in keys:
        value = data.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) or isinstance(value, float):
            # nan/inf cannot go into the JSON of the page.
            if math.isfinite(value):
                return float(value)
    return None


def _read_json(path: Path, limit_bytes: int = 20 * 1024 * 1024):
    try:
        if path.stat().st_size > limit_bytes:
            return None
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return data


def _is_date_value(value) -> bool:
    # A date written as text (ISO) or as unix seconds.
    if isinstance(value, bool):
        return False
    return isinstance(value, str) or isinstance(value, int) or isinstance(value, float)


AVG_MISS_KEYS = ('avg_miss_m', 'avg_miss', 'miss_m', 'mean_miss_m')
LATERAL_MISS_KEYS = ('avg_lateral_miss_m', 'lateral_miss_m', 'avg_lateral_miss', 'lateral_miss')


def eval_results(repo: Path) -> list[dict]:
    """The leaderboard: outputs/eval/*.json written by evaluation/holdout.py, best avg miss first.
    The overall numbers are taken from the top level of the file or from its "overall" object."""
    folder = repo / 'outputs' / 'eval'
    results = []
    if not folder.is_dir():
        return results
    for path in sorted(folder.iterdir()):
        if not EVAL_NAME_RE.fullmatch(path.name) or not path.is_file() or not inside(folder, path):
            continue
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        overall = data.get('overall')
        if not isinstance(overall, dict):
            overall = data
        avg_miss = _first_number(overall, AVG_MISS_KEYS)
        lateral = _first_number(overall, LATERAL_MISS_KEYS)
        segments_value = data.get('segments')
        count = None
        if isinstance(segments_value, list) or isinstance(segments_value, dict):
            count = len(segments_value)
        if count is None:
            count = _first_number(overall, ('n_segments', 'segment_count'))
        model = data.get('model')
        if not isinstance(model, str):
            model = data.get('model_spec')
        if not isinstance(model, str):
            model = path.stem
        date = data.get('date')
        if not _is_date_value(date):
            date = data.get('created')
        if not _is_date_value(date):
            date = path.stat().st_mtime
        sha = data.get('model_sha256')
        if not isinstance(sha, str):
            sha = data.get('sha256')
        if not isinstance(sha, str):
            sha = ''
        results.append({'file': path.name, 'model': short_text(model, 120), 'avg_miss_m': avg_miss,
                        'lateral_miss_m': lateral, 'segments': count, 'date': date,
                        'sha256': short_text(sha, 64)})

    def sort_key(row):
        if row['avg_miss_m'] is None:
            return (1, 0.0, row['model'])
        return (0, row['avg_miss_m'], row['model'])

    results.sort(key=sort_key)
    return results


# -- comparison videos -------------------------------------------------------------------------------
def _model_side(data: dict, side: str) -> dict:
    """Name and avg miss of 'reference' or 'student' in a compare json, from either
    {"reference": {"name"/"spec", "avg_miss_m"}} or {"avg_miss_m": {"reference": x}}."""
    result = {'model': None, 'avg_miss_m': None}
    block = data.get(side)
    if isinstance(block, dict):
        for key in ('spec', 'name', 'model'):
            if isinstance(block.get(key), str):
                result['model'] = short_text(block[key], 120)
                break
        result['avg_miss_m'] = _first_number(block, AVG_MISS_KEYS)
    elif isinstance(block, str):
        result['model'] = short_text(block, 120)
    if result['avg_miss_m'] is None:
        for key in AVG_MISS_KEYS:
            values = data.get(key)
            if isinstance(values, dict):
                result['avg_miss_m'] = _first_number(values, (side,))
                if result['avg_miss_m'] is not None:
                    break
    if result['model'] is None:
        args = data.get('args')
        if isinstance(args, dict) and isinstance(args.get(side), str):
            result['model'] = short_text(args[side], 120)
    return result


def compare_videos(repo: Path) -> list[dict]:
    """outputs/compare/*.mp4, newest first, with the numbers of the json next to each video."""
    folder = repo / 'outputs' / 'compare'
    videos = []
    if not folder.is_dir():
        return videos
    for path in folder.iterdir():
        if not VIDEO_NAME_RE.fullmatch(path.name) or path.is_symlink() or not path.is_file():
            continue
        stat = path.stat()
        item = {'name': path.name, 'size': stat.st_size, 'modified': stat.st_mtime, 'segment': None,
                'start': None, 'duration': None, 'speed': None,
                'reference': {'model': None, 'avg_miss_m': None}, 'student': {'model': None, 'avg_miss_m': None}}
        data = _read_json(path.with_suffix('.json'))
        if isinstance(data, dict):
            args = data.get('args')
            if not isinstance(args, dict):
                args = data
            if isinstance(args.get('segment'), str):
                item['segment'] = short_text(args['segment'], 40)
            item['start'] = _first_number(args, ('start',))
            item['duration'] = _first_number(args, ('duration',))
            item['speed'] = _first_number(args, ('speed',))
            item['reference'] = _model_side(data, 'reference')
            item['student'] = _model_side(data, 'student')
        videos.append(item)
    videos.sort(key=lambda video: video['modified'], reverse=True)
    return videos


def video_path(repo: Path, name: str) -> Path | None:
    """The file of a comparison video, only for a plain name inside outputs/compare (no links)."""
    if not VIDEO_NAME_RE.fullmatch(name):
        return None
    folder = repo / 'outputs' / 'compare'
    path = folder / name
    if path.is_symlink() or not path.is_file() or not inside(folder, path):
        return None
    return path


class Cache:
    """A value recomputed at most every `seconds` (nvidia-smi and the /proc scan cost tens of ms,
    and every open page asks for the status every few seconds)."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self._value = None
        self._time = 0.0
        self._lock = threading.Lock()

    def get(self, compute):
        with self._lock:
            now = time.monotonic()
            if self._value is None or now - self._time > self.seconds:
                self._value = compute()
                self._time = now
            return self._value
