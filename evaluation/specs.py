"""Model specs: the strings that name a model on the command line and in the web site.

  stock           comma's own CD210 model, the pinned outputs/cd210/driving_supercombo.onnx
  teacher         models/big_driving_supercombo.onnx (the model the student imitates)
  logged          the plan the car really ran: modelV2 from the segment rlog, no inference
  run:<run dir>   outputs/supervised/<run dir>/supercombo.pt (or supercombo.training.pt)
  hf:<slug>       Hugging Face training/releases/<slug>/supercombo.pt, else models/<slug>/driving_supercombo.onnx

Standard library only: the web site can import it to validate what the user typed.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

# Always fullmatch(): with match() the "$" also accepts a trailing newline ("stock\n").
SPEC_PATTERN = re.compile(r"^(stock|teacher|logged|run:[A-Za-z0-9_.-]+|hf:[a-z0-9_]+)$")
SEGMENT_PATTERN = re.compile(r"^[0-9a-f]{20}$")

DEFAULT_HF_REPO = "cristianku/openpilot-distill-models"

# The stock reference is pinned by hash: models/comma_driving_supercombo.onnx is NOT used because
# helpers/export_onnx.sh copies there whatever the comma runs, which is a student after an install.
STOCK_PATH = "outputs/cd210/driving_supercombo.onnx"
STOCK_SHA256 = "65a08adc31d5c456219687d99b7bf5e44d61dae2d49ea67850e76105c7248cce"
STOCK_HF_FILE = "models/cd210_native/driving_supercombo.onnx"
STOCK_CACHE_DIR = "models/cache"

TEACHER_PATH = "models/big_driving_supercombo.onnx"
RUNS_DIR = "outputs/supervised"
HF_CACHE_DIR = "models/cache/hf"

# How a model file is run (evaluation/runners.py).
FORMAT_ONNX_STREAMING = "onnx_streaming"  # one 20 Hz frame per call, state fed back (modeld style)
FORMAT_ONNX_TEACHER = "onnx_teacher"  # big teacher, split + dense like in training
FORMAT_TORCH = "torch"  # trained student checkpoint
FORMAT_LOGGED = "logged"  # modelV2 from the rlog


@dataclass(frozen=True, slots=True)
class ModelSpec:
    text: str
    kind: str
    value: str

    @property
    def key(self) -> str:
        """File-name-safe name: outputs/eval/<key>.json, compare file names."""
        if self.kind == "run":
            return f"run_{self.value}"
        if self.kind == "hf":
            return f"hf_{self.value}"
        return self.kind

    @property
    def label(self) -> str:
        """Name shown in the video and in the site."""
        if self.kind == "stock":
            return "comma CD210 (stock)"
        if self.kind == "teacher":
            return "teacher (big_driving_supercombo)"
        if self.kind == "logged":
            return "modello sull'auto (dal log)"
        if self.kind == "run":
            return f"studente run {self.value}"
        return f"studente HF {self.value}"


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    spec: ModelSpec
    path: Path | None
    format: str
    sha256: str


def parse_model_spec(text: str) -> ModelSpec:
    """Validate a spec string; ValueError with an Italian message when it is not one."""
    if not isinstance(text, str) or not SPEC_PATTERN.fullmatch(text):
        raise ValueError(f"modello '{text}' non valido: usa stock, teacher, logged, run:<cartella> o hf:<nome>")
    if ":" not in text:
        return ModelSpec(text=text, kind=text, value="")
    kind, value = text.split(":", 1)
    # "." and ".." pass the character class but would leave outputs/supervised.
    if value.startswith("."):
        raise ValueError(f"modello '{text}' non valido: il nome non puo' iniziare con un punto")
    return ModelSpec(text=text, kind=kind, value=value)


def validate_segment_name(name: str) -> str:
    if not isinstance(name, str) or not SEGMENT_PATTERN.fullmatch(name):
        raise ValueError(f"segmento '{name}' non valido: 20 caratteri esadecimali (es. 00000126f84ad127f010)")
    return name


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def hf_repo() -> str:
    # HF_REPO comes from the repo .env through helpers/common.sh.
    value = os.environ.get("HF_REPO", "").strip()
    if value:
        return value
    return DEFAULT_HF_REPO


def _hf_download(filename: str, local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo_id=hf_repo(), filename=filename, local_dir=str(local_dir)))


def _hf_file_exists(filename: str) -> bool:
    from huggingface_hub import HfApi

    return bool(HfApi().file_exists(repo_id=hf_repo(), filename=filename))


def _resolve_stock(root: Path, spec: ModelSpec) -> ResolvedModel:
    path = root / STOCK_PATH
    if not path.exists():
        print(f"{STOCK_PATH} manca: scarico {STOCK_HF_FILE} da Hugging Face ({hf_repo()})", flush=True)
        path = _hf_download(STOCK_HF_FILE, root / STOCK_CACHE_DIR)
    sha = file_sha256(path)
    if sha != STOCK_SHA256:
        raise ValueError(f"{path} non e' il modello CD210 originale (sha256 {sha[:16]}, atteso {STOCK_SHA256[:16]})")
    return ResolvedModel(spec=spec, path=path, format=FORMAT_ONNX_STREAMING, sha256=sha)


def _resolve_run(root: Path, spec: ModelSpec) -> ResolvedModel:
    run_dir = root / RUNS_DIR / spec.value
    final = run_dir / "supercombo.pt"
    # supervised/train.py writes supercombo.training.pt while it runs and supercombo.pt at the end.
    in_progress = run_dir / "supercombo.training.pt"
    if final.is_file():
        path = final
    elif in_progress.is_file():
        path = in_progress
    else:
        raise ValueError(f"run '{spec.value}': nessun supercombo.pt ne' supercombo.training.pt in {run_dir}")
    return ResolvedModel(spec=spec, path=path, format=FORMAT_TORCH, sha256=file_sha256(path))


def _resolve_hf(root: Path, spec: ModelSpec) -> ResolvedModel:
    local_dir = root / HF_CACHE_DIR / spec.value
    weights_file = f"training/releases/{spec.value}/supercombo.pt"
    onnx_file = f"models/{spec.value}/driving_supercombo.onnx"
    if _hf_file_exists(weights_file):
        path = _hf_download(weights_file, local_dir)
        file_format = FORMAT_TORCH
    elif _hf_file_exists(onnx_file):
        path = _hf_download(onnx_file, local_dir)
        file_format = FORMAT_ONNX_STREAMING
    else:
        raise ValueError(f"hf:{spec.value}: ne' {weights_file} ne' {onnx_file} su {hf_repo()}")
    return ResolvedModel(spec=spec, path=path, format=file_format, sha256=file_sha256(path))


def resolve_model(spec: ModelSpec, root: Path) -> ResolvedModel:
    """Find (or download) the model file of a spec and hash it."""
    if spec.kind == "stock":
        return _resolve_stock(root, spec)
    if spec.kind == "teacher":
        path = root / TEACHER_PATH
        if not path.is_file():
            raise ValueError(f"{TEACHER_PATH} manca: scaricalo come in supervised/README.md")
        return ResolvedModel(spec=spec, path=path, format=FORMAT_ONNX_TEACHER, sha256=file_sha256(path))
    if spec.kind == "logged":
        return ResolvedModel(spec=spec, path=None, format=FORMAT_LOGGED, sha256="logged")
    if spec.kind == "run":
        return _resolve_run(root, spec)
    if spec.kind == "hf":
        return _resolve_hf(root, spec)
    raise ValueError(f"tipo di modello sconosciuto: {spec.kind}")
