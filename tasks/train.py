"""Train (distill) the driving model on the segments in data/ with supervised/train.py.

    python -m tasks.train [--steps N] [--batch-size N] [--workers N] [--mode fresh|continue|resume]
                          [--run <run>] [--teacher md-v3|084747c7] [--wandb online|offline|disabled]
                          [--gpus all|<index>[,<index>...]]

Start of the model (supervised/train.py --mode):
  fresh     a new model, from the pretrained vision backbone.
  continue  a new run from the weights of training/stable.json on Hugging Face (HF_REPO): only the
            weights, not the optimizer.
  resume    the run --run, interrupted: model and optimizer from its supercombo.training.pt, same
            folder, train.log and metrics.jsonl; steps (when not given) and teacher from the checkpoint.

--gpus: the GPUs of the training ('all' by default). With more than one, torchrun starts one process
per GPU (DistributedDataParallel) and every process takes --batch-size samples per step.

Order: GPU lock (held until the end), the checks that need no run folder (stable weights, run to
resume), the run folder (a new one: outputs/supervised/<YYYY-MM-DD_hhmmss>/), then everything is
printed AND appended to <run>/train.log: teacher download and check, segment checks, training.
The run folder is printed early on a line of its own: RUN_DIR=<absolute path> (the site reads it).
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from tasks.common import DEFAULT_HF_REPO, REPO, GpuLock, TaskError, list_gpus, load_env, run_main, say, select_gpus
from tasks.teacher import TEACHERS, ensure_teacher, teacher_of_file

RUNS_DIR = REPO / 'outputs' / 'supervised'
DATA_DIR = REPO / 'data'
# Run folder names: with seconds, and the old ones to the minute (runs before 2026-10-22).
RUN_NAME_RE = re.compile(r'^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{4}([0-9]{2})?$')
GPUS_RE = re.compile(r'^(all|[0-9]{1,2}(,[0-9]{1,2}){0,15})$')
TEST_STEPS = 10


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--steps', type=int, default=None, help='total steps (5000, or those of the resumed run)')
    parser.add_argument('--test', action='store_true', help='only ' + str(TEST_STEPS) + ' steps, to see that it starts')
    parser.add_argument('--batch-size', type=int, default=8)
    # Data-loader workers: with 1 the GPU waits for the videos to be decoded.
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--mode', choices=('fresh', 'continue', 'resume'), default='fresh')
    parser.add_argument('--run', default=None, help='with --mode resume: the run to resume')
    parser.add_argument('--teacher', choices=tuple(TEACHERS), default=None, help='default md-v3')
    parser.add_argument('--wandb', choices=('online', 'offline', 'disabled'), default=None)
    parser.add_argument('--gpus', default='all')
    args = parser.parse_args(arguments)
    if args.test:
        args.steps = TEST_STEPS
    for name in ('steps', 'batch_size', 'workers'):
        value = getattr(args, name)
        if value is not None and (value < 0 or (value == 0 and name != 'workers')):
            parser.error('invalid --' + name.replace('_', '-'))
    if args.mode == 'resume':
        if not args.run:
            parser.error('--mode resume needs --run <run>')
        if args.teacher is not None:
            parser.error('with --mode resume the teacher is the one of the interrupted run: remove --teacher')
    elif args.run:
        parser.error('--run only goes with --mode resume')
    if not GPUS_RE.fullmatch(args.gpus):
        parser.error('--gpus needs all or the GPU indices (e.g. 0 or 0,1)')
    return args


def wandb_mode(requested: str | None) -> str:
    """WANDB_MODE of the run: the requested one, otherwise online only when wandb is logged in."""
    logged_in = bool(os.environ.get('WANDB_API_KEY'))
    if not logged_in:
        try:
            logged_in = 'api.wandb.ai' in (Path.home() / '.netrc').read_text(encoding='utf-8', errors='replace')
        except OSError:
            pass
    if requested == 'online' and not logged_in:
        raise TaskError('wandb online but wandb has no credentials on this server: choose offline')
    if requested:
        return requested
    if os.environ.get('WANDB_MODE'):
        return os.environ['WANDB_MODE']
    return 'online' if logged_in else 'offline'


def check_stable_weights(hf_repo: str) -> None:
    import requests

    url = 'https://huggingface.co/' + hf_repo + '/resolve/main/training/stable.json'
    try:
        response = requests.get(url, timeout=30)
    except requests.RequestException as exc:
        raise TaskError('cannot read ' + url + ': ' + str(exc)) from exc
    if response.status_code == 404:
        raise TaskError(hf_repo + ' has no "stable" weights yet: publish a model as stable first')
    if response.status_code != 200:
        raise TaskError('cannot read ' + url + ' (HTTP ' + str(response.status_code) + ')')


def resume_info(run: str) -> tuple[Path, int, str]:
    """Folder, total steps and saved --teacher of an interrupted run."""
    if not RUN_NAME_RE.fullmatch(run) or not (RUNS_DIR / run).is_dir():
        raise TaskError('run ' + run + ' does not exist in outputs/supervised/')
    folder = RUNS_DIR / run
    checkpoint = folder / 'supercombo.training.pt'
    if not checkpoint.is_file():
        raise TaskError('run ' + run + ' has no checkpoint (supercombo.training.pt): it cannot be resumed')
    if (folder / 'supercombo.pt').is_file():
        raise TaskError('run ' + run + ' has already finished: to train it further publish it as stable and '
                        'start "from the stable weights"')
    import torch

    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    return folder, int(saved['total_steps']), str(saved.get('teacher', ''))


def check_segments() -> int:
    """The number of usable segments; refuses broken links and segments without localizer outputs."""
    total = 0
    missing = []
    broken = []
    for folder in sorted(DATA_DIR.iterdir()) if DATA_DIR.is_dir() else []:
        if not folder.is_dir():
            continue
        videos = [folder / 'fcamera.hevc', folder / 'ecamera.hevc']
        # A link into the shared folder (COMMA_VIDEOS_DIR) whose file is not there: the dataset
        # still lists the segment and stops when it reads it.
        if any(video.is_symlink() and not video.exists() for video in videos):
            broken.append(folder.name)
            continue
        if not videos[0].is_file():
            continue
        total += 1
        if not (folder / 'localizer.safetensors').is_file() or not (folder / 'frame_info.safetensors').is_file():
            missing.append(folder.name)
    if broken:
        raise TaskError('videos missing in the shared folder of the segments (disk not mounted or segment '
                        'removed), the training would stop: ' + ' '.join(broken))
    if total == 0:
        raise TaskError('no segment in data/: download them from the comma first (Preprocessing)')
    if missing:
        raise TaskError('segments not localized, they would stop the training: ' + ' '.join(missing)
                        + '. Localize them first (Preprocessing).')
    return total


class RunLog:
    """Everything written here goes to stdout (the site's job log) and is appended to train.log."""

    def __init__(self, path: Path):
        self.file = path.open('ab')

    def write(self, data: bytes) -> None:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
        self.file.write(data)
        self.file.flush()

    def say(self, text: str) -> None:
        self.write((text + '\n').encode('utf-8'))


def train(args: argparse.Namespace) -> int:
    load_env()
    hf_repo = os.environ.get('HF_REPO') or DEFAULT_HF_REPO
    gpus = select_gpus(args.gpus, list_gpus())

    # Lock first: two trainings started together must not both prepare a run.
    lock = GpuLock()
    lock.acquire()

    if args.mode == 'continue':
        check_stable_weights(hf_repo)
    steps = args.steps if args.steps is not None else 5000
    if args.mode == 'resume':
        run_dir, saved_steps, teacher_arg = resume_info(args.run)
        if args.steps is None:
            steps = saved_steps
        teacher = teacher_of_file(teacher_arg)
        if teacher is None:
            raise TaskError('the run uses the teacher ' + teacher_arg + ', which cannot be downloaded from here')
    else:
        teacher = TEACHERS[args.teacher or 'md-v3']
        teacher_arg = str(REPO / teacher.path)
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        run_dir = RUNS_DIR / time.strftime('%Y-%m-%d_%H%M%S')
        try:
            run_dir.mkdir()
        except FileExistsError:
            raise TaskError(run_dir.name + ' already exists (two starts in the same second?): start it again') from None
    mode = wandb_mode(args.wandb)
    say('RUN_DIR=' + str(run_dir))

    log = RunLog(run_dir / 'train.log')
    try:
        # Prints of ensure_teacher and of the checks go to train.log too.
        sys.stdout = TeeText(log)
        ensure_teacher(teacher.name)
        say('Segments: ' + str(check_segments()) + ', all localized.')
    except TaskError as exc:
        log.say('ERROR: ' + str(exc))
        raise SystemExit(1) from None
    finally:
        sys.stdout = sys.__stdout__

    start = {'fresh': 'from scratch', 'continue': 'from the stable weights of ' + hf_repo,
             'resume': 'resumed from the last checkpoint'}[args.mode]
    gpu_text = 'GPU ' + ','.join(str(index) for index in gpus)
    log.say('Training (' + start + '): ' + str(steps) + ' steps, batch ' + str(args.batch_size) + ' per GPU, workers '
            + str(args.workers) + ', teacher ' + teacher.name + ', ' + gpu_text + ', wandb ' + mode)

    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node', str(len(gpus)),
               '-m', 'supervised.train', '--mode', args.mode]
    if args.mode == 'continue':
        command += ['--hf-repo', hf_repo]
    command += ['--steps', str(steps), '--batch-size', str(args.batch_size), '--workers', str(args.workers),
                '--teacher', teacher_arg, '--output', str(run_dir / 'supercombo.pt')]
    environment = dict(os.environ)
    environment['PYTHONUNBUFFERED'] = '1'
    environment['WANDB_MODE'] = mode
    environment['CUDA_VISIBLE_DEVICES'] = ','.join(str(index) for index in gpus)
    environment['HF_REPO'] = hf_repo

    # A stop from the site (SIGTERM to the whole process group) reaches the training too: this
    # process keeps copying its output until it ends, so train.log has the end of the story.
    signal.signal(signal.SIGTERM, lambda number, frame: None)
    child = subprocess.Popen(command, cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        chunk = child.stdout.read1(65536)
        if not chunk:
            break
        log.write(chunk)
    code = child.wait()
    if code == 0:
        log.say('Done. Model: ' + str(run_dir / 'supercombo.pt'))
    else:
        log.say('Training ended with code ' + str(code) + '.')
    lock.release()
    return 0 if code == 0 else 1


class TeeText:
    """A text stdout that writes through a RunLog (for the say() of the checks)."""

    def __init__(self, log: RunLog):
        self.log = log

    def write(self, text: str) -> int:
        self.log.write(text.encode('utf-8'))
        return len(text)

    def flush(self) -> None:
        pass


def main() -> int:
    return train(parse_args())


if __name__ == '__main__':
    run_main(main)
