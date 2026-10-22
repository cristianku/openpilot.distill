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
from datetime import UTC, datetime
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
    Activity('gpu', 'distill training', 'supervised.train', True),
    # The preparation of a training (teacher download, checks) before supervised.train starts.
    Activity('gpu', 'distill training', 'tasks.train', True),
    Activity('gpu', 'hold-out evaluation', 'evaluation.holdout', True),
    Activity('gpu', 'comparison video', 'evaluation.compare', True),
    Activity('data', 'download from the comma', 'helpers/fetch_comma_segments.sh', False),
    Activity('data', 'download from the comma', 'tasks.fetch', False),
    Activity('data', 'localization', 'helpers/localize_segments.sh', False),
    Activity('data', 'localization', 'tasks.localize', False),
    Activity('data', 'localization', 'localizer.run', False),
    Activity('cpu', 'ONNX export', 'helpers/export_onnx.sh', False),
    Activity('cpu', 'ONNX export', 'tasks.export', False),
    Activity('cpu', 'publish', 'tasks.publish', False),
    Activity('comma', 'install on the comma', 'tasks.install', False),
    Activity('cpu', 'publish', 'helpers/publish_model.sh', False),
    Activity('comma', 'install on the comma', 'helpers/install_model_on_comma.sh', False),
    Activity('system', 'code update', 'helpers/update.sh', False),
    Activity('system', 'code update', 'tasks.update', False),
    # sunnypilot.speedlimits (same server, same GPU). It does not take the GPU lock.
    Activity('other', 'speedlimits: training', 'run_v100_baseline.sh', True),
    Activity('other', 'speedlimits: training', 'helpers/3_train.sh', True),
    Activity('other', 'speedlimits: training', 'nnslr train', True),
    Activity('other', 'speedlimits: preparation', 'helpers/1_prepare.sh', False),
    Activity('other', 'speedlimits: pre-annotation', 'nnslr preannotate', False),
    Activity('other', 'speedlimits: frame extraction', 'nnslr extract-frames', False),
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
                    label = label + ' (other folder)'
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


def gpu_list() -> list[dict]:
    """Every NVIDIA GPU of the machine: index, name, total memory (the choices of a training)."""
    output = _run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader,nounits'])
    gpus = []
    if not output:
        return gpus
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        gpus.append({'index': int(parts[0]), 'name': ','.join(parts[1:-1]), 'memory_total_mb': _number(parts[-1])})
    return gpus


def system_info(repo: Path) -> dict:
    info = {'cpus': os.cpu_count(), 'load': None, 'memory': memory_info(), 'disk': None, 'gpu': gpu_info(),
            'gpus': gpu_list()}
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


def gpu_busy_reason(repo: Path, lock_file: str = GPU_LOCK_FILE) -> str:
    """'' when a GPU job can start, else why not. Checks the lock, the processes on the GPU
    (jobs that do not use the lock, e.g. a speedlimits training) and the known GPU jobs in /proc."""
    # Said in words for the page: which job, never command lines or file paths.
    reasons = []
    for activity in activities(repo):
        if activity['uses_gpu'] and activity['label'] not in reasons:
            reasons.append(activity['label'])
    held = gpu_lock_held(lock_file)
    if held is True and not reasons:
        reasons.append('a job that holds the GPU lock')
    elif held is None:
        reasons.append('cannot check whether the GPU is free')
    gpu = gpu_info()
    if gpu is not None and gpu['processes'] and not reasons:
        reasons.append(str(len(gpu['processes'])) + ' processes on the GPU')
    if not reasons:
        return ''
    return 'The GPU is busy with another job (' + '; '.join(reasons) + '). Try again when it has finished.'


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


# The ionosphere map a localized segment used: localizer.ionex has one "YYYY/DDD PRODUCT" line per day
# of the drive and the worst day counts. P1D (P2D) is a prediction made before the drive, RAP comes
# about one day later, FIN about two weeks later (localizer/ionex.py); the localization re-runs a
# segment when a better map is out.
MAP_QUALITY = {'FIN': 'final', 'RAP': 'rapid', 'P1D': 'prediction', 'P2D': 'prediction'}
MAP_RANK = {'prediction': 0, 'rapid': 1, 'final': 2}
# Days after the drive when the next map is normally out: a guess, so the page says "probably".
RAPID_AFTER_DAYS = 1
FINAL_AFTER_DAYS = 15


def map_of(segment_dir: Path, today: datetime | None = None) -> tuple[str, bool]:
    """('final' | 'rapid' | 'prediction' | 'unknown', whether a better map is probably out now)."""
    worst = None
    worst_day = None
    try:
        lines = (segment_dir / 'localizer.ionex').read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError:
        return 'unknown', False
    for line in lines[:10]:
        parts = line.split()
        if len(parts) != 2 or parts[1] not in MAP_QUALITY:
            continue
        try:
            day = datetime.strptime(parts[0], '%Y/%j')
        except ValueError:
            continue
        quality = MAP_QUALITY[parts[1]]
        if worst is None or MAP_RANK[quality] < MAP_RANK[worst]:
            worst = quality
            worst_day = day
    if worst is None:
        return 'unknown', False
    age = ((today or datetime.now(UTC).replace(tzinfo=None)) - worst_day).days
    better = (worst == 'prediction' and age >= RAPID_AFTER_DAYS) or (worst != 'final' and age >= FINAL_AFTER_DAYS)
    return worst, better


def segments(repo: Path) -> dict:
    """The segment folders of data/ and data_scartati/ with their localizer state."""
    data_dir = repo / 'data'
    rows = []
    routes = {}
    counts = {'total': 0, 'localized': 0, 'failed': 0, 'todo': 0, 'heldout': 0, 'heldout_localized': 0,
              'broken_video': 0, 'discarded': 0, 'refinable': 0,
              'map': {'final': 0, 'rapid': 0, 'prediction': 0, 'unknown': 0}}
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
        row['map_quality'] = ''
        row['refinable'] = False
        if localized:
            row['map'] = _first_line(entry / 'localizer.ionex', 40)
            row['map_quality'], row['refinable'] = map_of(entry)
        elif failed:
            row['reason'] = failure_reason(entry / 'localizer.failed')
        rows.append(row)

        counts['total'] = counts['total'] + 1
        if state == 'localized':
            counts['localized'] = counts['localized'] + 1
            counts['map'][row['map_quality']] = counts['map'][row['map_quality']] + 1
            if row['refinable']:
                counts['refinable'] = counts['refinable'] + 1
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
            route = {'route': row['route'], 'segments': 0, 'localized': 0, 'failed': 0, 'todo': 0, 'heldout': 0,
                     'map': {'final': 0, 'rapid': 0, 'prediction': 0, 'unknown': 0}}
            routes[row['route']] = route
        route['segments'] = route['segments'] + 1
        if state == 'localized':
            route['localized'] = route['localized'] + 1
            route['map'][row['map_quality']] = route['map'][row['map_quality']] + 1
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
            reason = 'reason not recorded'
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


def _text(data: dict, key: str, limit: int = 120) -> str | None:
    value = data.get(key)
    if isinstance(value, str):
        return short_text(value, limit)
    return None


AVG_MISS_KEYS = ('avg_miss_m',)
LATERAL_MISS_KEYS = ('avg_lateral_miss_m',)


def eval_results(repo: Path) -> list[dict]:
    """The leaderboard: outputs/eval/*.json written by evaluation/holdout.py, best avg miss first.

    Read from each file (schema 1 of evaluation/holdout.py):
      model: {spec, key, label, sha256, ...}   overall: {segments, frames, avg_miss_m, avg_lateral_miss_m}
      selection ("heldout" by default)          settings: {rate_hz, horizon_s}     created (ISO, UTC)
    """
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
            overall = {}
        model = data.get('model')
        if not isinstance(model, dict):
            model = {}
        settings = data.get('settings')
        if not isinstance(settings, dict):
            settings = {}
        spec = _text(model, 'spec')
        if spec is None:
            spec = path.stem
        segment_count = _first_number(overall, ('segments',))
        if segment_count is not None:
            segment_count = int(segment_count)
        frames = _first_number(overall, ('frames',))
        if frames is not None:
            frames = int(frames)
        date = data.get('created')
        if not _is_date_value(date):
            date = path.stat().st_mtime
        skipped = data.get('skipped')
        skipped_count = 0
        if isinstance(skipped, list) or isinstance(skipped, dict):
            skipped_count = len(skipped)
        results.append({'file': path.name, 'model': spec, 'label': _text(model, 'label') or spec,
                        'avg_miss_m': _first_number(overall, AVG_MISS_KEYS),
                        'lateral_miss_m': _first_number(overall, LATERAL_MISS_KEYS),
                        'segments': segment_count, 'frames': frames, 'skipped': skipped_count,
                        'selection': _text(data, 'selection', 60) or '',
                        'rate_hz': _first_number(settings, ('rate_hz',)),
                        'horizon_s': _first_number(settings, ('horizon_s',)),
                        'date': date, 'sha256': _text(model, 'sha256', 64) or ''})

    def sort_key(row):
        if row['avg_miss_m'] is None:
            return (1, 0.0, row['model'])
        return (0, row['avg_miss_m'], row['model'])

    results.sort(key=sort_key)
    return results


# -- comparison videos -------------------------------------------------------------------------------
def _compare_model(data: dict, side: str) -> dict:
    """'reference' or 'student' of a compare json (schema 1 of evaluation/compare.py):
    models.<side> = {spec, label, sha256, avg_miss_m, avg_lateral_miss_m, frames, ...}."""
    result = {'model': None, 'label': None, 'avg_miss_m': None, 'lateral_miss_m': None}
    models = data.get('models')
    block = None
    if isinstance(models, dict):
        block = models.get(side)
    if isinstance(block, dict):
        result['model'] = _text(block, 'spec')
        result['label'] = _text(block, 'label')
        result['avg_miss_m'] = _first_number(block, AVG_MISS_KEYS)
        result['lateral_miss_m'] = _first_number(block, LATERAL_MISS_KEYS)
    if result['model'] is None:
        args = data.get('args')
        if isinstance(args, dict):
            result['model'] = _text(args, side)
    return result


def compare_videos(repo: Path) -> list[dict]:
    """outputs/compare/*.mp4 (not the .partial.mp4 being encoded), newest first, with the numbers
    of the <same name>.json next to each video (schema 1 of evaluation/compare.py: segment,
    args {reference, student, start_s, duration_s, speed}, clip {start_s, end_s}, models, created)."""
    folder = repo / 'outputs' / 'compare'
    videos = []
    if not folder.is_dir():
        return videos
    for path in folder.iterdir():
        if not VIDEO_NAME_RE.fullmatch(path.name) or path.name.endswith('.partial.mp4'):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        stat = path.stat()
        item = {'name': path.name, 'size': stat.st_size, 'modified': stat.st_mtime, 'has_json': False,
                'segment': None, 'start': None, 'duration': None, 'speed': None, 'created': None,
                'reference': _compare_model({}, 'reference'), 'student': _compare_model({}, 'student')}
        data = _read_json(path.with_suffix('.json'))
        if isinstance(data, dict):
            item['has_json'] = True
            args = data.get('args')
            if not isinstance(args, dict):
                args = {}
            clip = data.get('clip')
            if not isinstance(clip, dict):
                clip = {}
            item['segment'] = _text(data, 'segment', 40)
            # The clip really shown (compare.py moves a too early start after the warm-up).
            start = _first_number(clip, ('start_s',))
            end = _first_number(clip, ('end_s',))
            if start is not None and end is not None:
                item['start'] = start
                item['duration'] = round(end - start, 3)
            else:
                item['start'] = _first_number(args, ('start_s',))
                item['duration'] = _first_number(args, ('duration_s',))
            item['speed'] = _first_number(args, ('speed',))
            item['created'] = _text(data, 'created', 40)
            item['reference'] = _compare_model(data, 'reference')
            item['student'] = _compare_model(data, 'student')
        videos.append(item)
    videos.sort(key=lambda video: video['modified'], reverse=True)
    return videos


def video_path(repo: Path, name: str) -> Path | None:
    """The file of a comparison video, only for a plain name inside outputs/compare (no links)."""
    if not VIDEO_NAME_RE.fullmatch(name) or name.endswith('.partial.mp4'):
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
