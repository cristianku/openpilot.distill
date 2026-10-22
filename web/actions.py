"""The fixed action table of the site: the ONLY commands it can run, and the strict checks of their
parameters.

A request names an action and sends its parameters as a JSON object; build_job() validates every
parameter (regex, range, existence inside the repo) and returns the argv list to run. Nothing from a
request ever becomes a shell string: the argv goes to subprocess without a shell, and every value in
it passed one of the validators below.
"""
from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from web.common import RUN_RE, SEGMENT_RE, WebError, inside

GPU_LOCK_FILE = '/run/lock/v100-gpu.lock'

# Model specs of evaluation/ (python -m evaluation.compare / evaluation.holdout).
MODEL_SPEC_RE = re.compile(r'^(stock|teacher|logged|run:[A-Za-z0-9_.-]+|hf:[a-z0-9_]+)$')
# Name of a published model (tasks/publish.py --name).
NAME_RE = re.compile(r'^[A-Za-z0-9_.-]{1,64}$')
DATE_RE = re.compile(r'^[0-9]{4}-[0-9]{2}-[0-9]{2}$')
# A number typed in a form: digits with an optional decimal part, nothing else (no "1e3", no "nan").
NUMBER_TEXT_RE = re.compile(r'^[0-9]{1,9}(\.[0-9]{1,6})?$')

STEPS_MIN = 1
STEPS_MAX = 200000
BATCH_MIN = 1
BATCH_MAX = 64
WORKERS_MIN = 0
WORKERS_MAX = 16
# A segment is 60 s long: the clip must start inside it and cannot be longer.
START_MIN = 0.0
START_MAX = 59.0
DURATION_MIN = 1.0
DURATION_MAX = 60.0
# Playback speed of the comparison video -> the exact text passed to --speed.
SPEED_TEXT = {0.25: '0.25', 0.5: '0.5', 1.0: '1'}
WANDB_MODES = ('', 'online', 'offline', 'disabled')
TRAIN_MODES = ('fresh', 'continue', 'resume')
TEACHER_NAMES = ('md-v3', '084747c7')
# 'all', or the indices of the chosen GPUs (tasks/train.py --gpus).
GPUS_RE = re.compile(r'^(all|[0-9]{1,2}(,[0-9]{1,2}){0,15})$')
REFERENCE_KINDS = ('stock', 'teacher', 'logged')
CONFIRM_INSTALL = 'INSTALL'
CONFIRM_RESTORE = 'RESTORE'


@dataclass(frozen=True)
class Action:
    label: str
    # One running job per group. "system" (the code update) runs only when nothing else runs.
    group: str
    # The helper script, relative to the repo root (None for the test action).
    script: str | None
    # The script takes the GPU lock itself; the site only checks before starting that it is free.
    needs_gpu: bool = False
    # Asks the comma to be reachable and offroad before starting.
    needs_comma_offroad: bool = False


ACTIONS = {
    'fetch': Action('Download segments from the comma', 'data', 'tasks/fetch.py'),
    'localize': Action('Localize segments', 'data', 'tasks/localize.py'),
    'train': Action('Training', 'gpu', 'tasks/train.py', needs_gpu=True),
    'train_test': Action('Training test (10 steps)', 'gpu', 'tasks/train.py', needs_gpu=True),
    'eval': Action('Evaluation on the hold-out segments', 'gpu', 'tasks/evaluation.py', needs_gpu=True),
    'compare': Action('Comparison video', 'gpu', 'tasks/evaluation.py', needs_gpu=True),
    'export': Action('ONNX export', 'cpu', 'tasks/export.py'),
    'publish_dry': Action('Test publish (dry run)', 'cpu', 'tasks/publish.py'),
    'publish': Action('Publish to Hugging Face', 'cpu', 'tasks/publish.py'),
    'install': Action('Install on the comma', 'comma', 'tasks/install.py',
                      needs_comma_offroad=True),
    'restore': Action('Restore the original model', 'comma', 'tasks/install.py',
                      needs_comma_offroad=True),
    'update': Action('Code update', 'system', 'tasks/update.py'),
}

# Only with "python -m web.server --test-actions": a harmless job to test the job runner and the GPU
# lock path of a live server without training anything. Never available otherwise.
TEST_ACTIONS = {
    'selftest': Action('Site test (no real work)', 'gpu', None, needs_gpu=True),
}

GROUP_NAMES = {
    'data': 'a download or a localization',
    'gpu': 'a GPU job',
    'cpu': 'an export or a publish',
    'comma': 'a comma operation',
    'system': 'a code update',
    'test': 'a test',
}

# The selftest job: takes the real GPU lock (when /run/lock exists) the way the helper scripts do,
# prints PROGRESS lines for ~5 s, then exits 0. Run as: python -c <this> <lock file>.
SELFTEST_CODE = '''
import fcntl, os, sys, time
lock = sys.argv[1]
if os.path.isdir(os.path.dirname(lock)):
    handle = os.open(lock, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("The GPU is busy with another job (lock taken).", flush=True)
        sys.exit(1)
    print("GPU lock taken: " + lock, flush=True)
for step in range(1, 11):
    print("PROGRESS %d/10" % step, flush=True)
    time.sleep(0.5)
print("test finished", flush=True)
'''


@dataclass
class JobSpec:
    """A validated job, ready to run."""
    action: str
    label: str
    group: str
    argv: list
    needs_gpu: bool = False
    needs_comma_offroad: bool = False
    # Validated parameters, kept in the job file (shown in the job list, matched by the publish check).
    params: dict = field(default_factory=dict)
    # One line that says what the job does, for the job list.
    summary: str = ''


def bad(message: str) -> WebError:
    return WebError(400, 'invalid_param', message)


def check_keys(params: dict, allowed: tuple[str, ...]) -> None:
    """Refuses any parameter the action does not take: a typo must not be silently ignored."""
    for key in params:
        if key not in allowed:
            raise bad('Unknown parameter: ' + str(key)[:40] + '.')


def int_param(params: dict, key: str, low: int, high: int, default: int | None = None) -> int:
    value = params.get(key)
    if value is None or value == '':
        if default is None:
            raise bad('Missing parameter: ' + key + '.')
        return default
    # bool is an int in Python: true/false must not pass as 1/0.
    if isinstance(value, bool):
        raise bad(key + ' must be a whole number.')
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and re.fullmatch(r'[0-9]{1,7}', value):
        number = int(value)
    else:
        raise bad(key + ' must be a whole number.')
    if number < low or number > high:
        raise bad(key + ' must be between ' + str(low) + ' and ' + str(high) + '.')
    return number


def float_param(params: dict, key: str, low: float, high: float, default: float | None = None) -> float:
    value = params.get(key)
    if value is None or value == '':
        if default is None:
            raise bad('Missing parameter: ' + key + '.')
        return default
    if isinstance(value, bool):
        raise bad(key + ' must be a number.')
    if isinstance(value, int) or isinstance(value, float):
        number = float(value)
    elif isinstance(value, str) and NUMBER_TEXT_RE.fullmatch(value):
        number = float(value)
    else:
        raise bad(key + ' must be a number.')
    if not math.isfinite(number):
        raise bad(key + ' must be a number.')
    if number < low or number > high:
        raise bad(key + ' must be between ' + format_number(low) + ' and ' + format_number(high) + '.')
    return number


def bool_param(params: dict, key: str) -> bool:
    value = params.get(key)
    if value is None:
        return False
    if value is True or value is False:
        return value
    raise bad(key + ' must be true or false.')


def choice_param(params: dict, key: str, choices: tuple[str, ...], default: str | None = None) -> str:
    value = params.get(key)
    if value is None:
        if default is None:
            raise bad('Missing parameter: ' + key + '.')
        return default
    if not isinstance(value, str) or value not in choices:
        raise bad(key + ' is not valid.')
    return value


def text_param(params: dict, key: str) -> str:
    value = params.get(key)
    if value is None:
        return ''
    if not isinstance(value, str):
        raise bad(key + ' must be text.')
    return value


def date_param(params: dict, key: str) -> str:
    """'' or a real date written YYYY-MM-DD."""
    value = text_param(params, key)
    if value == '':
        return ''
    if not DATE_RE.fullmatch(value):
        raise bad(key + ': invalid date (format YYYY-MM-DD).')
    try:
        datetime.strptime(value, '%Y-%m-%d')
    except ValueError as exc:
        raise bad(key + ': invalid date (format YYYY-MM-DD).') from exc
    return value


def name_param(params: dict, key: str) -> str:
    value = text_param(params, key)
    if not NAME_RE.fullmatch(value):
        raise bad('Invalid name: 1 to 64 characters, letters, digits, _ . or -')
    return value


def format_number(number: float) -> str:
    """10.0 -> '10', 2.5 -> '2.5': the text passed to the scripts and shown to the user."""
    if number == int(number):
        return str(int(number))
    return repr(number)


def run_folder(repo: Path, run: str) -> Path:
    """outputs/supervised/<run> after the name and location checks (not its content)."""
    if not isinstance(run, str) or not RUN_RE.fullmatch(run):
        raise bad('Invalid run.')
    base = repo / 'outputs' / 'supervised'
    folder = base / run
    if not folder.is_dir() or not inside(base, folder):
        raise WebError(404, 'unknown_run', 'Run ' + run + ' does not exist.')
    return folder


def run_param(repo: Path, value, need: str) -> str:
    """A run name that exists. need: 'final' = with supercombo.pt (finished),
    'model' = supercombo.pt or supercombo.training.pt, 'resume' = checkpoint without the final model."""
    folder = run_folder(repo, value)
    final = folder / 'supercombo.pt'
    checkpoint = folder / 'supercombo.training.pt'
    if need == 'final':
        if not final.is_file() or not inside(folder, final):
            raise WebError(409, 'run_not_finished',
                           'Run ' + value + ' has no supercombo.pt yet (it has not finished).')
    elif need == 'model':
        has_final = final.is_file() and inside(folder, final)
        has_checkpoint = checkpoint.is_file() and inside(folder, checkpoint)
        if not has_final and not has_checkpoint:
            raise WebError(409, 'run_without_model', 'Run ' + value + ' has no model (supercombo.pt).')
    elif need == 'resume':
        if final.is_file():
            raise WebError(409, 'run_finished', 'Run ' + value + ' has already finished: it cannot be resumed.')
        if not checkpoint.is_file() or not inside(folder, checkpoint):
            raise WebError(409, 'run_without_checkpoint',
                           'Run ' + value + ' has no supercombo.training.pt: it cannot be resumed.')
    else:
        raise ValueError('unknown need: ' + need)
    return value


def segment_param(repo: Path, value) -> str:
    """A segment folder of data/ that exists and is localized."""
    if not isinstance(value, str) or not SEGMENT_RE.fullmatch(value):
        raise bad('Invalid segment (20 hexadecimal characters).')
    base = repo / 'data'
    folder = base / value
    if not folder.is_dir() or not inside(base, folder):
        raise WebError(404, 'unknown_segment', 'Segment ' + value + ' is not in data/.')
    localized = (folder / 'localizer.safetensors').is_file() and (folder / 'frame_info.safetensors').is_file()
    if not localized:
        raise WebError(409, 'segment_not_localized', 'Segment ' + value + ' is not localized.')
    return value


def model_spec_param(repo: Path, params: dict, key: str, kinds: tuple[str, ...]) -> str:
    """A model spec of evaluation/: one of `kinds` ('stock', 'teacher', 'logged', 'run', 'hf').
    run:<x> must be an existing run with a model; hf:<slug> is checked by the script itself."""
    value = params.get(key)
    if not isinstance(value, str) or not MODEL_SPEC_RE.fullmatch(value):
        raise bad(key + ': invalid model.')
    if value.startswith('run:'):
        kind = 'run'
    elif value.startswith('hf:'):
        kind = 'hf'
    else:
        kind = value
    if kind not in kinds:
        raise bad(key + ': model not allowed here.')
    if kind == 'run':
        run_param(repo, value[len('run:'):], 'model')
    return value


def gpus_param(params: dict) -> str:
    """'all' (the default) or the indices of the chosen GPUs: '0', '0,1'."""
    value = params.get('gpus')
    if value is None or value == '':
        return 'all'
    if not isinstance(value, str) or not GPUS_RE.fullmatch(value):
        raise bad('gpus must be "all" or GPU indices (e.g. 0 or 0,1).')
    return value


def gpus_text(gpus: str) -> str:
    if gpus == 'all':
        return 'all GPUs'
    return 'GPU ' + gpus


def slug_of(name: str) -> str:
    """The folder name tasks/publish.py uses on Hugging Face for --name:
    lowercase, every character other than a-z 0-9 becomes '_' (tr -c 'a-z0-9' '_')."""
    result = []
    for character in name.lower():
        if ('a' <= character <= 'z') or ('0' <= character <= '9'):
            result.append(character)
        else:
            result.append('_')
    return ''.join(result)


def find_action(action_id: str, test_actions: bool = False) -> Action:
    if action_id in ACTIONS:
        return ACTIONS[action_id]
    if test_actions and action_id in TEST_ACTIONS:
        return TEST_ACTIONS[action_id]
    raise WebError(404, 'unknown_action', 'Unknown action.')


def build_job(repo: Path, action_id: str, params, *, test_actions: bool = False) -> JobSpec:
    """Validates the parameters of an action and builds its argv. Raises WebError on any problem."""
    action = find_action(action_id, test_actions)
    if not isinstance(params, dict):
        raise bad('The parameters must be a JSON object.')
    argv = []
    clean = {}
    summary = ''

    if action_id == 'fetch':
        check_keys(params, ('since',))
        since = date_param(params, 'since')
        argv = [sys.executable, '-m', 'tasks.fetch']
        summary = 'new drives from the comma'
        if since:
            argv.append('--since=' + since)
            clean['since'] = since
            summary = 'drives since ' + since

    elif action_id == 'localize':
        check_keys(params, ())
        argv = [sys.executable, '-m', 'tasks.localize']
        summary = 'segments to do and to refine'

    elif action_id == 'train':
        check_keys(params, ('steps', 'batch_size', 'workers', 'mode', 'run', 'wandb', 'teacher', 'gpus'))
        mode = choice_param(params, 'mode', TRAIN_MODES, 'fresh')
        batch_size = int_param(params, 'batch_size', BATCH_MIN, BATCH_MAX, 8)
        workers = int_param(params, 'workers', WORKERS_MIN, WORKERS_MAX, 6)
        wandb = choice_param(params, 'wandb', WANDB_MODES, '')
        gpus = gpus_param(params)
        argv = [sys.executable, '-m', 'tasks.train', '--mode', mode]
        if mode == 'resume':
            # The steps and the teacher of a resumed run come from its checkpoint.
            if params.get('steps') not in (None, ''):
                raise bad('A resumed run keeps the steps of the interrupted run: leave the field empty.')
            if params.get('teacher') not in (None, ''):
                raise bad('A resumed run keeps the teacher of the interrupted run.')
            run = run_param(repo, params.get('run'), 'resume')
            argv += ['--run', run]
            clean['run'] = run
            summary = 'resumes ' + run
        else:
            if params.get('run') not in (None, ''):
                raise bad('A run is only needed to resume.')
            steps = int_param(params, 'steps', STEPS_MIN, STEPS_MAX)
            teacher = choice_param(params, 'teacher', TEACHER_NAMES, 'md-v3')
            argv += ['--steps', str(steps), '--teacher', teacher]
            clean['steps'] = steps
            clean['teacher'] = teacher
            if mode == 'continue':
                summary = str(steps) + ' steps from the stable weights'
            else:
                summary = str(steps) + ' steps from scratch'
            summary = summary + ', teacher ' + teacher
        argv += ['--batch-size', str(batch_size), '--workers', str(workers), '--gpus', gpus]
        if wandb:
            argv += ['--wandb', wandb]
            summary = summary + ', wandb ' + wandb
        clean['mode'] = mode
        clean['batch_size'] = batch_size
        clean['workers'] = workers
        clean['wandb'] = wandb
        clean['gpus'] = gpus
        summary = summary + ', batch ' + str(batch_size) + ', workers ' + str(workers) + ', ' + gpus_text(gpus)

    elif action_id == 'train_test':
        check_keys(params, ('gpus',))
        gpus = gpus_param(params)
        argv = [sys.executable, '-m', 'tasks.train', '--test', '--gpus', gpus]
        clean['gpus'] = gpus
        summary = '10 steps to check that everything starts, ' + gpus_text(gpus)

    elif action_id == 'eval':
        check_keys(params, ('model',))
        model = model_spec_param(repo, params, 'model', ('stock', 'teacher', 'logged', 'run', 'hf'))
        argv = [sys.executable, '-m', 'tasks.evaluation', 'holdout', '--model=' + model]
        clean['model'] = model
        summary = model

    elif action_id == 'compare':
        check_keys(params, ('segment', 'reference', 'student', 'start', 'duration', 'speed', 'inset', 'show_gt'))
        segment = segment_param(repo, params.get('segment'))
        reference = model_spec_param(repo, params, 'reference', REFERENCE_KINDS)
        student = model_spec_param(repo, params, 'student', ('run', 'hf'))
        start = float_param(params, 'start', START_MIN, START_MAX, 0.0)
        duration = float_param(params, 'duration', DURATION_MIN, DURATION_MAX, 20.0)
        speed = float_param(params, 'speed', 0.25, 1.0, 0.5)
        if speed not in SPEED_TEXT:
            raise bad('speed must be 0.25, 0.5 or 1.')
        inset = bool_param(params, 'inset')
        show_gt = bool_param(params, 'show_gt')
        argv = [sys.executable, '-m', 'tasks.evaluation', 'compare', '--segment=' + segment, '--reference=' + reference,
                '--student=' + student, '--start=' + format_number(start),
                '--duration=' + format_number(duration), '--speed=' + SPEED_TEXT[speed]]
        # evaluation.compare draws the inset by default: both ways explicit, so the checkbox counts.
        if inset:
            argv.append('--inset')
        else:
            argv.append('--no-inset')
        if show_gt:
            argv.append('--show-gt')
        clean = {'segment': segment, 'reference': reference, 'student': student, 'start': start,
                 'duration': duration, 'speed': speed, 'inset': inset, 'show_gt': show_gt}
        summary = (student + ' vs ' + reference + ' on ' + segment + ' (' + format_number(start) + ' s, '
                   + format_number(duration) + ' s)')

    elif action_id == 'export':
        check_keys(params, ('run',))
        run = run_param(repo, params.get('run'), 'final')
        argv = [sys.executable, '-m', 'tasks.export', '--run', run]
        clean['run'] = run
        summary = run

    elif action_id == 'publish_dry' or action_id == 'publish':
        allowed = ('run', 'name', 'stable')
        if action_id == 'publish':
            allowed = ('run', 'name', 'stable', 'confirm')
        check_keys(params, allowed)
        run = run_param(repo, params.get('run'), 'final')
        name = name_param(params, 'name')
        stable = bool_param(params, 'stable')
        if action_id == 'publish':
            # Typed confirmation: the exact name, so a click alone never publishes.
            if text_param(params, 'confirm') != name:
                raise WebError(400, 'confirmation_required',
                               'To publish, type the exact model name (' + name + ').')
            argv = [sys.executable, '-m', 'tasks.publish', '--upload']
        else:
            argv = [sys.executable, '-m', 'tasks.publish']
        argv.append('--name=' + name)
        if stable:
            argv.append('--stable')
        argv += ['--run', run]
        clean = {'run': run, 'name': name, 'stable': stable}
        summary = run + ' as ' + name
        if stable:
            summary = summary + ' (stable)'

    elif action_id == 'install':
        check_keys(params, ('run', 'confirm'))
        run = run_param(repo, params.get('run'), 'final')
        if text_param(params, 'confirm') != CONFIRM_INSTALL:
            raise WebError(400, 'confirmation_required', 'To install, type ' + CONFIRM_INSTALL + '.')
        argv = [sys.executable, '-m', 'tasks.install', '--run', run]
        clean['run'] = run
        summary = run + ' on the comma (it restarts)'

    elif action_id == 'restore':
        check_keys(params, ('confirm',))
        if text_param(params, 'confirm') != CONFIRM_RESTORE:
            raise WebError(400, 'confirmation_required', 'To restore, type ' + CONFIRM_RESTORE + '.')
        argv = [sys.executable, '-m', 'tasks.install', '--restore']
        summary = 'original model on the comma (it restarts)'

    elif action_id == 'update':
        check_keys(params, ())
        argv = [sys.executable, '-m', 'tasks.update']
        summary = 'latest code, libraries, site restart'

    elif action_id == 'selftest':
        check_keys(params, ())
        argv = [sys.executable, '-c', SELFTEST_CODE, GPU_LOCK_FILE]
        summary = '5-second fake job with the GPU lock'

    else:
        raise WebError(404, 'unknown_action', 'Unknown action.')

    return JobSpec(action=action_id, label=action.label, group=action.group, argv=argv,
                   needs_gpu=action.needs_gpu, needs_comma_offroad=action.needs_comma_offroad,
                   params=clean, summary=summary)


def script_missing(repo: Path, action_id: str, test_actions: bool = False) -> str:
    """'' when the helper of the action is in this checkout, else the reason. The helpers
    written for the site (eval, compare, update) may be missing on an older checkout."""
    action = find_action(action_id, test_actions)
    if action.script is None:
        return ''
    if (repo / action.script).is_file():
        return ''
    return 'this version of the code does not have "' + action.label + '" yet'
