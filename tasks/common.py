"""What the tasks share: the repo root, the repo .env, the GPUs and the GPU lock.

GPU lock: /run/lock/v100-gpu.lock, the same file every GPU job of the server takes (also
sunnypilot.speedlimits and "flock /run/lock/v100-gpu.lock <command>"). gpu_lock() takes it with
flock and keeps it until the task ends; a child process started while it is held (torchrun) runs
under it. It also refuses when nvidia-smi lists a process on the GPU: a job that does not use the
lock (e.g. a speedlimits training) must not share the GPU either.
"""
from __future__ import annotations

import fcntl
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from web.common import read_env_file

REPO = Path(__file__).resolve().parents[1]
GPU_LOCK_FILE = '/run/lock/v100-gpu.lock'
# The keys of the repo .env that the tasks use; every other key (the site token) is ignored.
ENV_KEYS = ('HF_REPO', 'COMMA_HOST', 'COMMA_VIDEOS_DIR', 'DISTILL_UV_EXTRA')
DEFAULT_HF_REPO = 'cristianku/openpilot-distill-models'


class TaskError(Exception):
    """A refusal or a failure, explained in Italian: main() prints it and exits with code 1."""


def load_env(repo: Path = REPO) -> None:
    """Puts the keys of the repo .env into os.environ; a variable already set wins."""
    for key, value in read_env_file(repo / '.env', ENV_KEYS).items():
        if not os.environ.get(key):
            os.environ[key] = value


def say(text: str) -> None:
    print(text, flush=True)


def run_main(main) -> None:
    """Runs a task main(): a TaskError becomes "ERRORE: ..." and exit code 1, Ctrl+C exit code 130."""
    try:
        code = main()
    except TaskError as exc:
        print('ERRORE: ' + str(exc), file=sys.stderr, flush=True)
        sys.exit(1)
    except KeyboardInterrupt:
        print('\nInterrotto.', file=sys.stderr, flush=True)
        sys.exit(130)
    sys.exit(code or 0)


@dataclass(frozen=True)
class Gpu:
    index: int
    name: str
    memory_total_mb: float | None


def list_gpus() -> list[Gpu]:
    """The NVIDIA GPUs of this machine, empty when there is no nvidia-smi or it does not answer."""
    try:
        output = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        try:
            memory = float(parts[-1])
        except ValueError:
            memory = None
        gpus.append(Gpu(int(parts[0]), ','.join(parts[1:-1]), memory))
    return gpus


def select_gpus(choice: str, gpus: list[Gpu]) -> list[int]:
    """The GPU indices of a choice: 'all', or one index ('0'), or several ('0,2')."""
    if not gpus:
        raise TaskError('nessuna GPU NVIDIA su questa macchina')
    available = [gpu.index for gpu in gpus]
    if choice == 'all':
        return available
    selected = []
    for part in choice.split(','):
        if not part.isdigit() or int(part) not in available:
            raise TaskError('GPU ' + part + ' non esiste (ci sono: ' + ', '.join(str(i) for i in available) + ')')
        if int(part) not in selected:
            selected.append(int(part))
    return selected


def _gpu_processes() -> list[str]:
    output = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
        capture_output=True, text=True, timeout=10, check=True,
    ).stdout
    return [line.strip() for line in output.splitlines() if line.strip()]


class GpuLock:
    """The GPU lock, held from acquire() until the process ends (or release())."""

    def __init__(self, lock_file: str = GPU_LOCK_FILE):
        self.lock_file = lock_file
        self.handle = None

    def acquire(self) -> None:
        if not list_gpus():
            say('Nessuna GPU NVIDIA: lock GPU non necessario.')
            return
        lock_dir = os.path.dirname(self.lock_file)
        if not os.path.isdir(lock_dir) or not os.access(lock_dir, os.W_OK):
            raise TaskError('cartella del lock GPU ' + lock_dir + ' mancante o non scrivibile')
        handle = os.open(self.lock_file, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(handle)
            raise TaskError('la GPU e\' occupata da un altro job: riprova quando ha finito') from None
        try:
            processes = _gpu_processes()
        except (OSError, subprocess.SubprocessError) as exc:
            os.close(handle)
            raise TaskError('nvidia-smi non risponde, non so se la GPU e\' libera') from exc
        if processes:
            os.close(handle)
            raise TaskError('la GPU e\' occupata da processi che non usano il lock (' + '; '.join(processes)
                            + '): riprova quando hanno finito')
        self.handle = handle

    def release(self) -> None:
        if self.handle is not None:
            os.close(self.handle)
            self.handle = None
