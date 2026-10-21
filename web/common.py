"""Things every module of the site shares: the repo root, the name patterns, the error type and the
settings read from the environment or from the repo .env."""
from __future__ import annotations

import os
import re
from pathlib import Path

# web/common.py -> the repo root. The site runs from the clone it lives in.
CODE_ROOT = Path(__file__).resolve().parents[1]

# A segment folder in data/: route without dashes (8 + 10 hex) + segment number on two digits.
SEGMENT_RE = re.compile(r'^[0-9a-f]{20}$')
# A training run folder in outputs/supervised/: date_time, minutes (old runs) or seconds (new runs).
RUN_RE = re.compile(r'^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{4,6}$')

DEFAULT_HF_REPO = 'cristianku/openpilot-distill-models'
DEFAULT_COMMA_HOST = 'comma'
HF_REPO_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$')
# An ssh host name or user@host; never starting with "-" (it would become an ssh option).
COMMA_HOST_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.@-]*$')


class WebError(Exception):
    """A refusal shown to the user: HTTP status, short machine reason, Italian message."""

    def __init__(self, status: int, reason: str, message: str):
        super().__init__(message)
        self.status = status
        self.reason = reason
        self.message = message


def inside(base: Path, path: Path) -> bool:
    """True when path, with every symlink resolved, is base itself or below it."""
    try:
        resolved = path.resolve()
        base_resolved = base.resolve()
    except (OSError, RuntimeError):
        return False
    if resolved == base_resolved:
        return True
    return resolved.is_relative_to(base_resolved)


def read_env_file(path: Path, keys: tuple[str, ...]) -> dict:
    """The values of `keys` in a KEY=value file (the repo .env). The file is parsed, never executed,
    and every other key (the site token, for one) is ignored. Same rules as helpers/common.sh:
    an "export " prefix and one pair of quotes around the value are accepted."""
    values = {}
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
        return values
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[len('export '):].strip()
        key, separator, value = line.partition('=')
        if not separator:
            continue
        key = key.strip()
        if key not in keys:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        values[key] = value
    return values


def setting(repo: Path, key: str, default: str) -> str:
    """A setting of the site: the environment wins, then the repo .env, then the default."""
    value = os.environ.get(key, '')
    if value:
        return value
    from_file = read_env_file(repo / '.env', (key,))
    value = from_file.get(key, '')
    if value:
        return value
    return default


def hf_repo(repo: Path) -> str:
    value = setting(repo, 'HF_REPO', DEFAULT_HF_REPO)
    if not HF_REPO_RE.fullmatch(value):
        return DEFAULT_HF_REPO
    return value


def comma_host(repo: Path) -> str:
    value = setting(repo, 'COMMA_HOST', DEFAULT_COMMA_HOST)
    if not COMMA_HOST_RE.fullmatch(value):
        return DEFAULT_COMMA_HOST
    return value


def is_heldout(segment: str) -> bool:
    """The hold-out rule of supervised/dataset.py (Comma1MDataset.__init__): a segment whose
    hash(int(name, 16)) % 10 == 0 is never trained on. Mirrored here instead of imported because
    supervised.dataset pulls in torch/cv2/openpilot and the site is standard library only.
    hash() of an int is deterministic (no PYTHONHASHSEED randomization for ints), so the result is
    the same as in the training process. Keep the two in sync."""
    if not SEGMENT_RE.fullmatch(segment):
        return False
    return hash(int(segment, 16)) % 10 == 0


def route_of(segment: str) -> str:
    """0000011e0c8825628a08 -> 0000011e--0c8825628a (the route as the comma names it)."""
    return segment[:8] + '--' + segment[8:18]


def short_text(value, limit: int = 200) -> str:
    """Text from a file or a request made safe to show or print: printable characters only, bounded."""
    text = str(value)
    if len(text) > limit:
        text = text[:limit] + '…'
    result = []
    for character in text:
        if character.isprintable():
            result.append(character)
        else:
            result.append('?')
    return ''.join(result)
