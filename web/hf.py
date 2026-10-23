"""What is already published on Hugging Face (HF_REPO, default cristianku/openpilot-distill-models),
read from the public API without login:
  models/<slug>/                   the published models (helpers/publish_model.sh)
  training/releases/<slug>/        the PyTorch weights published with them
  training/stable.json             the weights helpers/train_supervised.sh --continue starts from
The answer is kept for 5 minutes; the real publish asks again before starting.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import quote

API_TREE = 'https://huggingface.co/api/models/{repo}/tree/main/{path}'
CACHE_SECONDS = 300
TIMEOUT_SECONDS = 10


def fetch_tree(repo: str, path: str) -> list | None:
    """The entries of one folder of the repo, [] when the folder does not exist."""
    url = API_TREE.format(repo=quote(repo, safe='/'), path=quote(path, safe='/'))
    request = urllib.request.Request(url, headers={'User-Agent': 'distill-web', 'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read(4 * 1024 * 1024)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return []
        raise
    data = json.loads(raw.decode('utf-8'))
    if not isinstance(data, list):
        raise ValueError('unexpected answer from Hugging Face')
    return data


def folder_names(entries: list, prefix: str) -> list[str]:
    names = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get('type') != 'directory':
            continue
        path = entry.get('path')
        if isinstance(path, str) and path.startswith(prefix):
            names.append(path[len(prefix):])
    names.sort()
    return names


class HfCatalog:
    def __init__(self, repo: str, fetch=fetch_tree):
        self.repo = repo
        self._fetch = fetch
        self._lock = threading.Lock()
        self._snapshot = None
        self._time = 0.0

    def set_repo(self, repo: str) -> None:
        """Another repo (saved in Settings): the next snapshot asks Hugging Face about it."""
        with self._lock:
            if repo != self.repo:
                self.repo = repo
                self._snapshot = None

    def snapshot(self, force: bool = False) -> dict:
        with self._lock:
            now = time.monotonic()
            if self._snapshot is not None and not force and now - self._time < CACHE_SECONDS:
                return self._snapshot
            result = {'repo': self.repo, 'url': 'https://huggingface.co/' + self.repo, 'ok': False, 'error': '',
                      'checked': time.time(), 'models': [], 'releases': [], 'stable': None}
            try:
                models = self._fetch(self.repo, 'models')
                training = self._fetch(self.repo, 'training')
                releases = self._fetch(self.repo, 'training/releases')
            except (OSError, ValueError, urllib.error.URLError) as exc:
                result['error'] = 'Hugging Face does not answer: ' + type(exc).__name__
            else:
                result['ok'] = True
                result['models'] = folder_names(models, 'models/')
                result['releases'] = folder_names(releases, 'training/releases/')
                stable = False
                for entry in training:
                    if isinstance(entry, dict) and entry.get('path') == 'training/stable.json':
                        stable = True
                result['stable'] = stable
            self._snapshot = result
            self._time = now
            return result
