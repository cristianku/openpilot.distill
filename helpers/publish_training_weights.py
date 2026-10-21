#!/usr/bin/env python3
"""Publish validated PyTorch training weights, optionally moving the stable pointer.

By default this command is a read-only dry run. Never upload private recordings,
full optimizer checkpoints, .env files or authentication material.

--stage-dir writes the files of the release into a folder at their Hugging Face paths
instead of uploading them: helpers/publish_model.sh uploads that folder together with
the ONNX/compiled model in one commit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory


VALID_RELEASE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
VALID_TASKS = {"supercombo", "detector", "reader"}
STABLE_PATH = "training/stable.json"


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


def release_manifest(release: str, models: dict) -> dict:
    return {
        "schema_version": 1,
        "release": release,
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


def updated_stable(stable_doc: dict | None, models: dict) -> dict:
    """training/stable.json pointing every published task to this release, other tasks kept."""
    stable_doc = stable_doc if stable_doc is not None else {"schema_version": 1, "models": {}}
    if stable_doc.get("schema_version") != 1 or not isinstance(stable_doc.get("models"), dict):
        raise ValueError("existing stable.json is invalid")
    for task, meta in models.items():
        stable_doc["models"][task] = {"path": meta["hub_path"], "sha256": meta["sha256"]}
    return stable_doc


def stage_release(stage_dir: Path, release: str, models: dict, existing: dict | None,
                  stable_doc: dict | None, set_stable: bool) -> list[str]:
    """Write the files of the release into stage_dir at their Hugging Face paths.

    existing: manifest.json of a release with this name already on Hugging Face (None: new).
    The same weights again are not staged a second time; different weights under the same
    name are refused, because releases are immutable.
    stable_doc: the current training/stable.json (None: there is none yet).
    Returns the staged paths. No PyTorch and no network: the caller uploads the folder.
    """
    staged = []
    if existing is not None:
        existing_models = existing.get("models")
        if not isinstance(existing_models, dict) or set(existing_models) != set(models) or any(
            existing_models[task].get("sha256") != meta["sha256"] for task, meta in models.items()
        ):
            raise ValueError(f"release already exists with different weights: {release}")
    else:
        for meta in models.values():
            destination = stage_dir / meta["hub_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(meta["path"], destination)
            staged.append(meta["hub_path"])
        manifest_path = f"training/releases/{release}/manifest.json"
        (stage_dir / manifest_path).write_text(
            json.dumps(release_manifest(release, models), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staged.append(manifest_path)
    if set_stable:
        stable_file = stage_dir / STABLE_PATH
        stable_file.parent.mkdir(parents=True, exist_ok=True)
        stable_file.write_text(
            json.dumps(updated_stable(stable_doc, models), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staged.append(STABLE_PATH)
    return staged


def hub_json(repo: str, filename: str) -> dict | None:
    """A JSON file of the Hugging Face repo, None when it does not exist."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        path = hf_hub_download(repo_id=repo, filename=filename, revision="main")
    except EntryNotFoundError:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("HF_REPO"), required=False,
                        help="Hugging Face model repository (or HF_REPO environment variable)")
    parser.add_argument("--release", required=True, help="immutable release ID, e.g. v1")
    parser.add_argument("--model", action="append", required=True, metavar="TASK=PATH",
                        help="repeat for detector and reader, or supply one supercombo")
    parser.add_argument("--upload", action="store_true", help="actually publish public training weights")
    parser.add_argument("--stage-dir", type=Path,
                        help="write the files into this folder at their Hugging Face paths instead of uploading")
    parser.add_argument("--stable", action="store_true",
                        help="update training/stable.json after separately validating the model")
    args = parser.parse_args()
    if not VALID_RELEASE.fullmatch(args.release) or args.release in (".", ".."):
        parser.error("invalid release name")
    if not args.repo:
        parser.error("--repo or HF_REPO is required")
    if args.upload and args.stage_dir:
        parser.error("--upload and --stage-dir are alternatives")
    if args.stable and not (args.upload or args.stage_dir):
        parser.error("--stable requires --upload or --stage-dir (otherwise no pointer is changed)")

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

    release_info = release_manifest(args.release, models)
    print(json.dumps({"repo": args.repo, "upload": args.upload, "set_stable": args.stable,
                      "release": release_info}, indent=2))
    release_manifest_path = f"training/releases/{args.release}/manifest.json"

    if args.stage_dir:
        existing = hub_json(args.repo, release_manifest_path)
        stable_doc = hub_json(args.repo, STABLE_PATH) if args.stable else None
        staged = stage_release(args.stage_dir, args.release, models, existing, stable_doc, args.stable)
        if existing is not None:
            print(f"release {args.release} already on {args.repo} with the same weights: not staged again")
        print(f"staged in {args.stage_dir}: {', '.join(staged) or 'nothing'}")
        return 0

    if not args.upload:
        print("DRY RUN: no files uploaded, no stable pointer changed.")
        return 0

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    current = api.repo_info(repo_id=args.repo, repo_type="model")
    # Release names are immutable. A retry with an existing name is rejected.
    if hub_json(args.repo, release_manifest_path) is not None:
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
            stable_path = Path(temporary) / "stable.json"
            stable_path.write_text(
                json.dumps(updated_stable(hub_json(args.repo, STABLE_PATH), models), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            operations.append(CommitOperationAdd(path_in_repo=STABLE_PATH, path_or_fileobj=str(stable_path)))
        commit = api.create_commit(repo_id=args.repo, repo_type="model", operations=operations,
                                   commit_message=f"Publish training weights {args.release}",
                                   parent_commit=current.sha)
    print(f"Uploaded reviewed weights: {commit.commit_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
