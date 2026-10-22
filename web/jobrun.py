"""Runs one job of the site and writes its exit code into the job file.

    python -m web.jobrun <outputs/web/jobs/<job id>.json>

The site starts this wrapper in a new session (its own process group) with stdout and stderr on
the job log; the wrapper runs the job's argv (no shell) in the same group, waits for it, appends
"== finished with code N" to the log and records exit_code and finished in the job file. Because the
wrapper and not the site records the result, a restart of the site (systemd KillMode=process keeps
the jobs alive) never loses an exit code.

Stop: the site sends SIGTERM to the whole group. The wrapper ignores it and keeps waiting, so it can
still record how the job ended; if the job does not end the site sends SIGKILL to the group and
records the result itself.
"""
from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def lock_path_of(job_file: Path) -> Path:
    return job_file.with_name(job_file.name + '.lock')


def read_job_file(job_file: Path) -> dict | None:
    try:
        data = json.loads(job_file.read_text(encoding='utf-8'))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def update_job_file(job_file: Path, changes: dict) -> dict:
    """Merges `changes` into the job file. The site and the wrapper both write it: a lock on a side
    file makes every read-modify-write whole, and the new content replaces the old in one rename."""
    with open(lock_path_of(job_file), 'a') as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            data = read_job_file(job_file)
            if data is None:
                data = {}
            for key, value in changes.items():
                data[key] = value
            temporary = job_file.with_name(job_file.name + '.tmp')
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
            os.replace(temporary, job_file)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return data


def _ignore_stop_signal(signal_number, frame):
    # The job (same process group) got the same signal: wait for it to end and record its code.
    print('\n== received signal ' + str(signal_number) + ': waiting for the job to stop', flush=True)


def main(arguments: list[str] | None = None) -> int:
    if arguments is None:
        arguments = sys.argv[1:]
    if len(arguments) != 1:
        print('usage: python -m web.jobrun <job file .json>', file=sys.stderr)
        return 2
    job_file = Path(arguments[0])
    job = read_job_file(job_file)
    if job is None:
        print('== unreadable job file: ' + str(job_file), flush=True)
        return 2
    argv = job.get('argv')
    cwd = job.get('cwd')
    if not isinstance(argv, list) or not argv or not isinstance(cwd, str):
        print('== job file without argv or cwd: ' + str(job_file), flush=True)
        return 2

    signal.signal(signal.SIGTERM, _ignore_stop_signal)
    signal.signal(signal.SIGINT, _ignore_stop_signal)
    # The site may go away (restart): its end must not stop the job.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    # From here a stop (SIGTERM) can no longer kill the wrapper: the site waits for this mark.
    update_job_file(job_file, {'pid': os.getpid(), 'pgid': os.getpgid(0), 'wrapper_ready': time.time()})
    code = 0
    try:
        # Same stdout/stderr (the log) and same process group as this wrapper.
        child = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL)
    except OSError as exc:
        print('== cannot start ' + str(argv[0]) + ': ' + str(exc), flush=True)
        code = 127
    else:
        update_job_file(job_file, {'child_pid': child.pid})
        # wait() is retried after a signal handler runs (PEP 475), so a SIGTERM does not end it.
        code = child.wait()

    finished = time.time()
    if code < 0:
        print('\n== finished with code ' + str(code) + ' (stopped by signal ' + str(-code) + ')', flush=True)
    else:
        print('\n== finished with code ' + str(code), flush=True)
    update_job_file(job_file, {'exit_code': code, 'finished': finished})
    # The wrapper ends like its job: 128 + signal for a job killed by a signal, as a shell does.
    if code < 0:
        return 128 - code
    if code > 255:
        return 1
    return code


if __name__ == '__main__':
    sys.exit(main())
