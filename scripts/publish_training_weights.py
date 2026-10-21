#!/usr/bin/env python3
"""Publish validated PyTorch training weights, optionally moving the stable pointer.

By default this command is a read-only dry run. Never upload private recordings,
full optimizer checkpoints, .env files or authentication material.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory


VALID_RELEASE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
VALID_TASKS = {"supercombo", "detector", "reader"}


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_model(argument: str) -> tuple[str, Path]:
    if "=" not in argument:
        raise ValueError("--model must be TASK=PATH")
    task, path = argument.split("=", 1)
    if task not in VALID_TASKS or not path:
        raise ValueError("model task must be supercombo, detector or reader")
    return task, Path(path)


def inspect_weights(path: Path, task: str) -> dict:
    import torch

    if not path.is_file():
        raise ValueError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")
    if checkpoint.get("task", task) != task:
        raise ValueError(f"{path}: task mismatch")
    weights = checkpoint.get("state_dict", checkpoint)
    if not isinstance(weights, dict) or not weights or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor) for name, value in weights.items()
    ):
        raise ValueError(f"{path}: not a model state_dict")
    if "optimizer" in checkpoint or "scaler" in checkpoint:
        raise ValueError("do not publish full training-state checkpoints; publish inference/fine-tuning weights only")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest_file(path),
        "architecture": checkpoint.get("architecture", "supercombo" if task == "supercombo" else None),
        "classes": checkpoint.get("classes"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("HF_REPO"), required=False,
                        help="Hugging Face model repository (or HF_REPO environment variable)")
    parser.add_argument("--release", required=True, help="immutable release ID, e.g. v1")
    parser.add_argument("--model", action="append", required=True, metavar="TASK=PATH",
                        help="repeat for detector and reader, or supply one supercombo")
    parser.add_argument("--upload", action="store_true", help="actually publish public training weights")
    parser.add_argument("--stable", action="store_true",
                        help="update training/stable.json after separately validating the model")
    args = parser.parse_args()
    if not VALID_RELEASE.fullmatch(args.release) or args.release in (".", ".."):
        parser.error("invalid release name")
    if not args.repo:
        parser.error("--repo or HF_REPO is required")
    if args.stable and not args.upload:
        parser.error("--stable requires --upload (otherwise no pointer is changed)")

    models = {}
    for argument in args.model:
        task, path = parse_model(argument)
        if task in models:
            parser.error(f"duplicate task {task}")
        details = inspect_weights(path, task)
        details["hub_path"] = f"training/releases/{args.release}/{task}.pt"
        models[task] = details

    if bool("detector" in models) != bool("reader" in models):
        parser.error("speedlimits releases must include both detector and reader")
    if "supercombo" in models and len(models) != 1:
        parser.error("do not mix supercombo and speedlimits in one release")

    release_info = {
        "schema_version": 1,
        "release": args.release,
        "models": {
            task: {
                "path": meta["hub_path"],
                "sha256": meta["sha256"],
                "bytes": meta["bytes"],
                "architecture": meta["architecture"],
                "classes": meta["classes"],
            }
            for task, meta in sorted(models.items())
        },
    }
    print(json.dumps({"repo": args.repo, "upload": args.upload, "set_stable": args.stable,
                      "release": release_info}, indent=2))
    if not args.upload:
        print("DRY RUN: no files uploaded, no stable pointer changed.")
        return 0

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    api = HfApi()
    current = api.repo_info(repo_id=args.repo, repo_type="model")
    release_manifest_path = f"training/releases/{args.release}/manifest.json"
    # Release names are immutable. A retry with an existing name is rejected.
    try:
        hf_hub_download(repo_id=args.repo, filename=release_manifest_path, revision="main")
    except EntryNotFoundError:
        pass
    else:
        raise ValueError(f"release already exists: {args.release}")

    operations = []
    with TemporaryDirectory() as temporary:
        release_path = Path(temporary) / "manifest.json"
        release_path.write_text(json.dumps(release_info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for task, meta in models.items():
            operations.append(CommitOperationAdd(path_in_repo=meta["hub_path"], path_or_fileobj=meta["path"]))
        operations.append(CommitOperationAdd(path_in_repo=release_manifest_path,
                                             path_or_fileobj=str(release_path)))

        if args.stable:
            try:
                existing = Path(hf_hub_download(repo_id=args.repo, filename="training/stable.json",
                                                revision="main"))
                stable_doc = json.loads(existing.read_text(encoding="utf-8"))
            except EntryNotFoundError:
                stable_doc = {"schema_version": 1, "models": {}}
            if stable_doc.get("schema_version") != 1 or not isinstance(stable_doc.get("models"), dict):
                raise ValueError("existing stable.json is invalid")
            for task, meta in models.items():
                stable_doc["models"][task] = {"path": meta["hub_path"], "sha256": meta["sha256"]}
            stable_path = Path(temporary) / "stable.json"
            stable_path.write_text(json.dumps(stable_doc, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
            operations.append(CommitOperationAdd(path_in_repo="training/stable.json",
                                                 path_or_fileobj=str(stable_path)))
        commit = api.create_commit(repo_id=args.repo, repo_type="model", operations=operations,
                                   commit_message=f"Publish training weights {args.release}",
                                   parent_commit=current.sha)
    print(f"Uploaded reviewed weights: {commit.commit_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
