"""Evaluation and comparison videos on the GPU, under the GPU lock.

    python -m tasks.evaluation holdout <options of evaluation.holdout>
    python -m tasks.evaluation compare <options of evaluation.compare>

The options go to evaluation.holdout / evaluation.compare as they are (they validate them); this
module only takes the GPU lock first, unless they run with --device cpu. holdout with several
--model=<spec> evaluates them one after the other on the same drives, under the same lock.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys

from tasks.common import REPO, GpuLock, TaskError, load_env, run_main

MODULES = {'holdout': 'evaluation.holdout', 'compare': 'evaluation.compare'}


def uses_gpu(arguments: list[str]) -> bool:
    for index, argument in enumerate(arguments):
        if argument == '--device=cpu':
            return False
        if argument == '--device' and index + 1 < len(arguments) and arguments[index + 1] == 'cpu':
            return False
    return True


def model_runs(arguments: list[str]) -> list[list[str]]:
    """The arguments of each holdout run: one per --model=<spec>, in order, with the other options."""
    models = [argument for argument in arguments if argument.startswith('--model=')]
    if len(models) < 2:
        return [arguments]
    others = [argument for argument in arguments if not argument.startswith('--model=')]
    return [[model, *others] for model in models]


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in MODULES:
        raise TaskError('usage: python -m tasks.evaluation ' + '|'.join(MODULES) + ' <options>')
    module = MODULES[sys.argv[1]]
    arguments = sys.argv[2:]
    runs = model_runs(arguments) if sys.argv[1] == 'holdout' else [arguments]
    load_env()
    lock = GpuLock()
    if uses_gpu(arguments):
        lock.acquire()
    environment = dict(os.environ)
    environment['PYTHONUNBUFFERED'] = '1'
    # A stop from the site reaches the child too (same process group): wait for it to end.
    signal.signal(signal.SIGTERM, lambda number, frame: None)
    code = 0
    for run in runs:
        code = subprocess.run([sys.executable, '-m', module, *run], cwd=REPO, env=environment,
                              stdin=subprocess.DEVNULL, check=False).returncode
        # A failure or a stop ends the job: the next model does not start.
        if code != 0:
            break
    lock.release()
    # Killed by a signal: 128 + signal, as a shell reports it.
    return code if code >= 0 else 128 - code


if __name__ == '__main__':
    run_main(main)
