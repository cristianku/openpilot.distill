"""The training runs in outputs/supervised/: state, progress, ETA, final numbers and the series of
the charts.

Source of the numbers, per run:
  1. <run>/metrics.jsonl written by supervised/train.py (one JSON object per step: step, total_steps,
     time, loss, gt, distill, grad_norm, step_time, gt/plan, ...);
  2. otherwise <run>/train.log (runs made before metrics.jsonl existed), lines
     "step N/M: loss=... gt=... distill=... time=X.XXs".
A resumed run appends to both files, so a step can appear twice: the last occurrence wins.
"""
from __future__ import annotations

import json
import math
import re
import threading
from datetime import UTC, datetime
from pathlib import Path

from web.common import RUN_RE, inside

STEP_LINE_RE = re.compile(r'^step (\d+)/(\d+): loss=(\S+) gt=(\S+) distill=(\S+) time=([\d.]+)s$')
# The series of the charts, in the order the page shows them.
CHART_KEYS = ('loss', 'gt', 'distill', 'gt/plan', 'grad_norm', 'step_time')
# Shown as "final numbers" of a run.
SUMMARY_KEYS = ('loss', 'gt', 'distill', 'gt/plan', 'grad_norm')
# Steps averaged for the ETA: recent enough to follow a slowdown, long enough to smooth the
# data-loader stalls (p95 of a step is ~9 s against a median of 0.7 s).
ETA_WINDOW = 200
MAX_FILE_BYTES = 200 * 1024 * 1024
# End of train.log shown in the detail of a run.
LOG_TAIL_BYTES = 48 * 1024


def _number(text):
    """A finite float, or None (nan/inf from a diverged loss cannot go into JSON)."""
    if isinstance(text, bool):
        return None
    if isinstance(text, int) or isinstance(text, float):
        value = float(text)
    elif isinstance(text, str):
        try:
            value = float(text)
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(value):
        return None
    return value


def parse_train_log(text: str) -> tuple[dict, int | None]:
    """{step: {loss, gt, distill, step_time}} and the total steps, from a train.log."""
    records = {}
    total = None
    for line in text.splitlines():
        match = STEP_LINE_RE.match(line.strip())
        if match is None:
            continue
        step = int(match.group(1))
        total = int(match.group(2))
        records[step] = {'loss': _number(match.group(3)), 'gt': _number(match.group(4)),
                         'distill': _number(match.group(5)), 'step_time': _number(match.group(6))}
    return records, total


def parse_metrics_jsonl(text: str) -> tuple[dict, int | None, dict]:
    """{step: {chart keys + time}}, the total steps and the full last record, from a metrics.jsonl.
    A line that is not whole JSON (the one being written) is skipped."""
    records = {}
    total = None
    last_record = {}
    last_step = -1
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        step = data.get('step')
        if isinstance(step, bool) or not isinstance(step, int):
            continue
        record = {}
        for key in CHART_KEYS:
            record[key] = _number(data.get(key))
        record['time'] = _number(data.get('time'))
        records[step] = record
        total_steps = data.get('total_steps')
        if isinstance(total_steps, int) and not isinstance(total_steps, bool):
            total = total_steps
        if step >= last_step:
            last_step = step
            last_record = data
    return records, total, last_record


class RunReader:
    """Reads the runs, re-parsing a file only when its size or mtime changed."""

    def __init__(self, repo: Path):
        self.repo = repo
        self.base = repo / 'outputs' / 'supervised'
        self._cache = {}
        self._lock = threading.Lock()

    def run_dir(self, name: str) -> Path | None:
        if not RUN_RE.fullmatch(name):
            return None
        folder = self.base / name
        if not folder.is_dir() or not inside(self.base, folder):
            return None
        return folder

    def run_names(self) -> list[str]:
        names = []
        if not self.base.is_dir():
            return names
        for entry in self.base.iterdir():
            if RUN_RE.fullmatch(entry.name) and entry.is_dir():
                names.append(entry.name)
        names.sort(reverse=True)
        return names

    def _parsed(self, path: Path, kind: str):
        try:
            stat = path.stat()
        except OSError:
            return None
        if stat.st_size > MAX_FILE_BYTES:
            return None
        key = str(path)
        signature = (stat.st_size, stat.st_mtime_ns)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None and cached[0] == signature:
            return cached[1]
        try:
            text = path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            return None
        if kind == 'jsonl':
            records, total, last_record = parse_metrics_jsonl(text)
        else:
            records, total = parse_train_log(text)
            last_record = {}
        result = {'records': records, 'total': total, 'last_record': last_record}
        with self._lock:
            self._cache[key] = (signature, result)
        return result

    def steps(self, folder: Path) -> tuple[dict, int | None, dict, str | None]:
        """records, total steps, full last record, source ('metrics.jsonl' / 'train.log' / None)."""
        parsed = self._parsed(folder / 'metrics.jsonl', 'jsonl')
        if parsed is not None and parsed['records']:
            return parsed['records'], parsed['total'], parsed['last_record'], 'metrics.jsonl'
        parsed = self._parsed(folder / 'train.log', 'log')
        if parsed is not None and parsed['records']:
            return parsed['records'], parsed['total'], {}, 'train.log'
        return {}, None, {}, None

    def summary(self, name: str, running: bool) -> dict | None:
        folder = self.run_dir(name)
        if folder is None:
            return None
        records, total, last_record, source = self.steps(folder)
        config = _read_run_json(folder)
        has_pt = (folder / 'supercombo.pt').is_file()
        has_training_pt = (folder / 'supercombo.training.pt').is_file()
        has_onnx = (folder / 'driving_supercombo.onnx').is_file()
        if total is None and isinstance(config.get('steps'), int):
            total = config['steps']

        done = 0
        if records:
            done = max(records)
        finished_time = _number(config.get('finished'))
        if running:
            status = 'running'
        elif has_pt or finished_time is not None:
            status = 'finished'
        else:
            status = 'interrupted'

        percent = None
        if total:
            percent = round(min(done, total) * 100 / total, 1)
        average = average_step_time(records)
        eta = None
        if status == 'running' and total and average is not None:
            eta = round(max(0, total - done) * average)

        last = {}
        if records:
            last_values = records[done]
            for key in SUMMARY_KEYS:
                value = None
                if key in last_values:
                    value = last_values[key]
                if value is None and key in last_record:
                    value = _number(last_record.get(key))
                last[key] = value

        started = _number(config.get('started'))
        if started is None:
            started = time_from_name(name)
        if finished_time is None and has_pt:
            finished_time = (folder / 'supercombo.pt').stat().st_mtime

        run_config = {}
        for key in ('mode', 'steps', 'batch_size', 'workers', 'precision', 'teacher', 'hf_repo', 'world_size', 'gpus'):
            value = config.get(key)
            if isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool)):
                run_config[key] = value
        return {'name': name, 'status': status, 'steps_done': done, 'steps_total': total, 'percent': percent,
                'avg_step_time': average, 'eta_seconds': eta, 'last': last, 'source': source,
                'has_pt': has_pt, 'has_training_pt': has_training_pt, 'has_onnx': has_onnx,
                'resumable': has_training_pt and not has_pt and status != 'running',
                'started': started, 'finished': finished_time, 'config': run_config}

    def detail(self, name: str, running: bool, log_bytes: int = LOG_TAIL_BYTES) -> dict | None:
        """The summary of a run, plus its resumes and the end of its train.log."""
        result = self.summary(name, running)
        if result is None:
            return None
        folder = self.base / name
        resumed = []
        raw_resumed = _read_run_json(folder).get('resumed')
        if isinstance(raw_resumed, list):
            for entry in raw_resumed:
                if isinstance(entry, dict):
                    resumed.append({'time': _number(entry.get('time')), 'step': entry.get('step')
                                    if isinstance(entry.get('step'), int) else None})
        result['resumed'] = resumed
        result['log_tail'] = _log_tail(folder / 'train.log', log_bytes)
        return result

    def series(self, name: str, max_points: int = 800) -> dict | None:
        """The chart series of a run, at most max_points per series: each point is the mean of a
        bucket of consecutive steps (bucket size in the result), at the last step of the bucket."""
        folder = self.run_dir(name)
        if folder is None:
            return None
        records, total, last_record, source = self.steps(folder)
        return downsample(records, max_points, total, source, name)


def downsample(records: dict, max_points: int, total, source, name: str = '') -> dict:
    steps = sorted(records)
    bucket = 1
    if max_points < 1:
        max_points = 1
    if len(steps) > max_points:
        bucket = math.ceil(len(steps) / max_points)
    xs = []
    series = {}
    for key in CHART_KEYS:
        series[key] = []
    for start in range(0, len(steps), bucket):
        chunk = steps[start:start + bucket]
        xs.append(chunk[-1])
        for key in CHART_KEYS:
            values = []
            for step in chunk:
                value = records[step].get(key)
                if value is not None:
                    values.append(value)
            if values:
                series[key].append(sum(values) / len(values))
            else:
                series[key].append(None)
    available = []
    for key in CHART_KEYS:
        has_value = False
        for value in series[key]:
            if value is not None:
                has_value = True
                break
        if has_value:
            available.append(key)
    return {'name': name, 'source': source, 'total': total, 'bucket': bucket, 'steps': xs, 'series': series,
            'available': available}


def average_step_time(records: dict) -> float | None:
    """Seconds per step over the last ETA_WINDOW steps: from the wall clock of metrics.jsonl when
    there is one (it also counts the time between steps), else the mean of step_time."""
    if not records:
        return None
    steps = sorted(records)
    window = steps[-ETA_WINDOW:]
    first = window[0]
    last = window[-1]
    first_time = records[first].get('time')
    last_time = records[last].get('time')
    if last > first and first_time is not None and last_time is not None and last_time > first_time:
        return (last_time - first_time) / (last - first)
    values = []
    for step in window:
        # Step 1 includes the teacher and dataloader start-up (~34 s): not a normal step.
        if step == 1 and len(window) > 1:
            continue
        value = records[step].get('step_time')
        if value is not None:
            values.append(value)
    if not values:
        return None
    return sum(values) / len(values)


def _log_tail(path: Path, size: int) -> str:
    """The last `size` bytes of a log, from the first whole line; a line rewritten with \\r (progress
    bars) keeps only its last version."""
    try:
        with path.open('rb') as handle:
            handle.seek(0, 2)
            length = handle.tell()
            handle.seek(max(0, length - size))
            data = handle.read()
    except OSError:
        return ''
    text = data.decode('utf-8', 'replace')
    if length > size and '\n' in text:
        text = text.split('\n', 1)[1]
    lines = []
    for line in text.split('\n'):
        lines.append(line.rstrip('\r').rsplit('\r', 1)[-1])
    return '\n'.join(lines)


def _read_run_json(folder: Path) -> dict:
    path = folder / 'run.json'
    try:
        if path.stat().st_size > 1024 * 1024:
            return {}
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, UnicodeDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def time_from_name(name: str) -> float | None:
    """Unix time of a run folder name (the server clock is UTC, so is the name)."""
    if not RUN_RE.fullmatch(name):
        return None
    # By length: strptime would also read "1544" as 15:04:04 with %H%M%S (one-digit minutes).
    time_part = name.split('_', 1)[1]
    if len(time_part) == 6:
        pattern = '%Y-%m-%d_%H%M%S'
    elif len(time_part) == 4:
        pattern = '%Y-%m-%d_%H%M'
    else:
        return None
    try:
        moment = datetime.strptime(name, pattern)
    except ValueError:
        return None
    return moment.replace(tzinfo=UTC).timestamp()


def running_runs_from_processes(repo: Path) -> set[str]:
    """Run folder names of the trainings running on this machine: the --output argument of every
    "supervised.train" process (torchrun and its worker carry the same command line)."""
    names = set()
    proc = Path('/proc')
    if not proc.is_dir():
        return names
    base = repo / 'outputs' / 'supervised'
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            arguments = (entry / 'cmdline').read_bytes().decode('utf-8', 'replace').split('\0')
        except OSError:
            continue
        if 'supervised.train' not in arguments:
            continue
        if '--output' not in arguments:
            continue
        position = arguments.index('--output')
        if position + 1 >= len(arguments):
            continue
        output = Path(arguments[position + 1])
        folder = output.parent
        if not folder.is_absolute():
            try:
                folder = Path(Path('/proc') / entry.name / 'cwd').resolve() / folder
            except OSError:
                continue
        if RUN_RE.fullmatch(folder.name) and inside(base, folder):
            names.add(folder.name)
    return names
