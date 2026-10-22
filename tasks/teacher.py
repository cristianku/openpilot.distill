"""The teachers of the distillation: where to download them and how to check them.

    python -m tasks.teacher [md-v3|084747c7]     (downloads it when missing, checks it)

ensure_teacher() downloads into <file>.part (a stopped download is finished by the next call) and
gives the file its name only after the size and sha256 checks: a partial or wrong file never has
the teacher's name, so no training can start with it.
"""
from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from supervised.teachers import DEFAULT_TEACHER, OLD_TEACHER
from tasks.common import REPO, TaskError, run_main, say

# Teacher md-v3: revision of commaai/openpilot_driving_models on Hugging Face (pinned: the repo gets
# new models, a fixed revision always gives the same file).
HF_TEACHER_REVISION = '8dbc57a6480111bd58487ecc248e30d50fb2f8dd'
# Teacher 084747c7: openpilot commit of its GitHub LFS file.
OPENPILOT_SHA = '084747c75d2cbd23af65ab7a9e770bbd7b98bac9'
CHUNK_BYTES = 4 * 1024 * 1024
TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class Teacher:
    name: str
    # Relative to the repo root, as supervised/teachers.py and the checkpoints have it.
    path: str
    url: str
    sha256: str
    source: str


TEACHERS = {
    'md-v3': Teacher(
        'md-v3', DEFAULT_TEACHER,
        'https://huggingface.co/commaai/openpilot_driving_models/resolve/' + HF_TEACHER_REVISION
        + '/1365126a-e71e-4778-be00-8ab92dc82c09/12864/big_driving_supercombo.onnx',
        '1cfd2c45353d876b5b257b24d77537de58ab90a44ae595f15b1f64b6ea3fd770', 'Hugging Face'),
    '084747c7': Teacher(
        '084747c7', OLD_TEACHER,
        'https://media.githubusercontent.com/media/commaai/openpilot/' + OPENPILOT_SHA
        + '/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx',
        'a501760a9d1d5fef0eab2b8c5d122d06124fc26dc8e0782e0aa94b82a208f0ff', 'GitHub'),
}


def teacher_of_file(path: str) -> Teacher | None:
    """The teacher whose file this is, by file name (what a checkpoint saved as --teacher)."""
    name = Path(path).name
    for teacher in TEACHERS.values():
        if Path(teacher.path).name == name:
            return teacher
    return None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _remote_size(url: str) -> int | None:
    """The size of the file at url (Hugging Face answers with a redirect), None when it does not answer."""
    import requests

    try:
        response = requests.head(url, allow_redirects=True, timeout=TIMEOUT_SECONDS)
    except requests.RequestException:
        return None
    length = response.headers.get('Content-Length')
    if response.status_code != 200 or length is None or not length.isdigit():
        return None
    return int(length)


def _download(url: str, partial: Path, size: int) -> None:
    import requests

    done = partial.stat().st_size if partial.exists() else 0
    headers = {'Range': 'bytes=' + str(done) + '-'} if done else {}
    with requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=TIMEOUT_SECONDS) as response:
        if done and response.status_code == 200:
            # The server ignored the range: start over.
            done = 0
        elif response.status_code not in (200, 206):
            raise TaskError('teacher download failed (HTTP ' + str(response.status_code) + ')')
        next_report = 0
        with partial.open('ab' if done else 'wb') as handle:
            for chunk in response.iter_content(CHUNK_BYTES):
                handle.write(chunk)
                done += len(chunk)
                percent = done * 100 // size
                if percent >= next_report:
                    # The PROGRESS line is the progress bar of the site, the other one is for people.
                    say('PROGRESS ' + str(done // 1_000_000) + '/' + str(size // 1_000_000))
                    say('  ' + str(done // 1_000_000) + ' of ' + str(size // 1_000_000) + ' MB (' + str(percent) + '%)')
                    next_report = percent + 5


def ensure_teacher(name: str, repo: Path = REPO) -> Path:
    """The teacher file, downloaded and checked when its size is not the one of its source."""
    # Imported here: the checks of this module run in the CI without the project dependencies.
    import requests

    if name not in TEACHERS:
        raise TaskError('unknown teacher: ' + name + ' (available: ' + ', '.join(TEACHERS) + ')')
    teacher = TEACHERS[name]
    target = repo / teacher.path
    local_size = target.stat().st_size if target.is_file() else 0
    remote_size = _remote_size(teacher.url)
    if remote_size is None:
        if not target.is_file():
            raise TaskError('teacher ' + name + ' is missing and ' + teacher.source + ' does not answer to download it')
        say('Teacher ' + name + ': ' + teacher.source + ' does not answer, using the file already here ('
            + str(local_size) + ' bytes).')
        return target
    if local_size == remote_size:
        say('Teacher ' + name + ': ok (' + str(local_size) + ' bytes).')
        return target

    partial = target.with_name(target.name + '.part')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial_size = partial.stat().st_size if partial.exists() else 0
    if partial_size > remote_size:
        partial.unlink()
        partial_size = 0
    if partial_size != remote_size:
        say('Teacher ' + name + ': ' + str(partial_size) + ' of ' + str(remote_size) + ' bytes, downloading from '
            + teacher.source + '...')
        try:
            _download(teacher.url, partial, remote_size)
        except (requests.RequestException, OSError) as exc:
            raise TaskError('teacher download interrupted (' + str(exc) + '): run it again to complete it') from exc
    partial_size = partial.stat().st_size
    if partial_size != remote_size:
        raise TaskError('teacher half downloaded (' + str(partial_size) + ' of ' + str(remote_size)
                        + ' bytes): run it again to complete it')
    say('Teacher ' + name + ': downloaded, checking the sha256...')
    local_sha256 = sha256_of(partial)
    if local_sha256 != teacher.sha256:
        partial.unlink()
        raise TaskError('teacher downloaded but its sha256 differs (' + local_sha256 + ' instead of ' + teacher.sha256
                        + '): deleted, run it again to download it again')
    os.replace(partial, target)
    say('Teacher ' + name + ': ok, sha256 correct (' + str(remote_size) + ' bytes).')
    return target


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else 'md-v3'
    ensure_teacher(name)
    return 0


if __name__ == '__main__':
    run_main(main)
