"""Publish a trained model on Hugging Face (HF_REPO), so that sunnypilot can offer it in the Models menu:
export to ONNX, compile it on the comma for sunnypilot's modeld_v2, upload ONNX + compiled model + the
catalog driving_models.json in one commit, together with the PyTorch weights that the training "from
the stable weights" (tasks.train --mode continue) starts from.

    python -m tasks.publish [--host <ip|name>] [--name <name>] [--stable] [--upload]
                            [--run <run> | <supercombo.pt|file.onnx>]
    python -m tasks.publish [--host <ip|name>] [--name <name>] [--upload] --stock

A dry run by default: everything is prepared (export, compile on the comma, catalog) in
outputs/publish/<name>/ and nothing is uploaded. Only --upload writes to Hugging Face.
Without --run and without a file: the newest outputs/supervised/*/supercombo.pt.

Steps:
  1. export: supercombo.pt -> driving_supercombo.onnx next to it, with helpers.export_onnx and the
     comma's original driving_supercombo.onnx as reference (tasks.export: the sha256-checked copy in
     models/comma_driving_supercombo.onnx). Skipped when an .onnx is given.
  2. PyTorch weights: the supercombo.pt (the one given, or the one next to the given .onnx) as
     training/releases/<name>/supercombo.pt + manifest.json, with helpers.publish_training_weights.
     Only the weights: a supercombo.training.pt (with the optimizer) is refused. A release with this
     name already on Hugging Face is left as it is when the weights are the same, refused when they
     differ. Skipped for --stock and for an .onnx without supercombo.pt next to it.
  3. compile on the comma (offroad) with sunnypilot's compile_modeld_v2.py: modeld_v2 loads only that
     format, not the pickle of compile_onnx.py. It uses the comma's tinygrad: after a sunnypilot update
     that changes tinygrad, run again with the .onnx and a new --name.
  4. catalog: a new driving_models.json with only this model (helpers.model_catalog) and the sha256 of
     the compiled file. The models published before (--stock included) leave the menu, their files stay
     on Hugging Face.
  5. (--upload only) one Hugging Face commit with models/<name>/driving_supercombo.onnx,
     models/<name>/<file.pkl>, driving_models.json and the training/ files of step 2.
Nothing is installed on the comma: the build folder there is removed at the end.

Hugging Face: --upload needs a login on this machine (a saved token, or HF_TOKEN) that can write the
repo; it is checked (HfApi.whoami) before anything else is done.

--name: name of the model in the menu and in the files (default: the run folder). Letters, digits,
        _ . -  A name whose folder models/<name>/ is already on Hugging Face is refused before anything
        else is done (public API): replacing a published model changes its sha256 and puts a comma that
        uses it back on Default. In a dry run the check is only a warning.
--stock: publishes the comma's own original driving model (the Default of the Models menu, e.g. CD210),
         compiled the same way, as "<default>-native" (e.g. cd210-native): a baseline to tell problems
         of this compile path from problems of a trained model. Kept in outputs/stock/<name>/.
--stable: also points training/stable.json to these PyTorch weights, so the next training "from the
          stable weights" starts from this model. Without it the pointer stays where it is.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

from tasks.comma import BACKUP, REMOTE_OPENPILOT, Comma, comma_host, original_model_source
from tasks.common import DEFAULT_HF_REPO, REPO, TaskError, load_env, run_main, say
from tasks.export import (ONNX_NAME, add_model_arguments, check_model_arguments, export_onnx, resolve_model,
                          run_module, update_reference)
from tasks.teacher import sha256_of
from web.common import HF_REPO_RE

BUILD_DIR = '/data/openpilot_distill/build'
PUBLISH_DIR = REPO / 'outputs' / 'publish'
STOCK_DIR = REPO / 'outputs' / 'stock'
NAME_RE = re.compile(r'^[A-Za-z0-9_.-]+$')
HF_TREE_URL = 'https://huggingface.co/api/models/{repo}/tree/main/models/{slug}'
HF_TIMEOUT_SECONDS = 30

# onroad state and where this openpilot keeps the files used here (paths asked to git, the layout
# changes between versions). For --stock also the name of the default model. To find the original model
# (--stock and the export reference): the sha256 of the original of this openpilot version (from git,
# the onnx is in LFS), of the current one and of the backup. $1 = the backup path.
PUBLISH_INFO_SCRIPT = r'''
backup="$1"
cd /data/openpilot || exit 1
echo "onroad=$(cat /data/params/d/IsOnroad 2>/dev/null)"
echo "compiler=$(git ls-files '*sunnypilot/modeld_v2/compile_modeld_v2.py' | head -1)"
reference="$(git ls-files '*selfdrive/modeld/models/driving_supercombo.onnx' | head -1)"
echo "reference=$reference"
helpers="$(git ls-files '*sunnypilot/models/helpers.py' | head -1)"
if [ -n "$helpers" ]; then
  echo "selector_version=$(sed -n 's/^REQUIRED_JSON_VERSION *= *\([0-9]*\).*/\1/p' "$helpers")"
fi
model_name="$(git ls-files '*sunnypilot/models/model_name.py' | head -1)"
if [ -n "$model_name" ]; then
  echo "default_model=$(sed -n 's/^DEFAULT_MODEL *= *"\([^"]*\)".*/\1/p' "$model_name" | head -1)"
fi
if [ -n "$reference" ]; then
  if [ "$(git cat-file -s "HEAD:$reference")" -lt 1024 ]; then
    echo "original=$(git cat-file -p "HEAD:$reference" | sed -n 's/^oid sha256://p')"
  else
    echo "original=$(git cat-file -p "HEAD:$reference" | sha256sum | cut -d' ' -f1)"
  fi
  echo "current=$(sha256sum "$reference" | cut -d' ' -f1)"
fi
if [ -f "$backup" ]; then
  echo "backup=$(sha256sum "$backup" | cut -d' ' -f1)"
fi
echo "tinygrad_ref=$(git -C tinygrad_repo rev-parse HEAD 2>/dev/null)"
if [ -x /usr/local/venv/bin/python3 ]; then
  echo "python=/usr/local/venv/bin/python3"
fi
'''

INFO_KEYS = ('onroad', 'compiler', 'reference', 'selector_version', 'default_model', 'original', 'current',
             'backup', 'tinygrad_ref', 'python')

# Compile for modeld_v2, flags as sunnypilot CI (TG_FLAGS_QCOM in
# .github/workflows/sunnypilot-build-model.yaml). No taskset: this comma has only CPUs 0-3 online.
# compile_modeld_v2 writes a temporary file in the current folder.
# $1 = build folder, $2 = openpilot checkout, $3 = python, $4 = compiler (path in the checkout),
# $5 = compiled file name. stdin is the script itself (bash -s), so the compiler gets /dev/null.
COMPILE_SCRIPT = r'''
exec 2>&1
build="$1"
openpilot="$2"
python="$3"
compiler="$4"
pkl="$5"
cd "$build" || exit 1
env DEV=QCOM IMAGE=1 FLOAT16=1 NOLOCALS=1 OPENPILOT_HACKS=1 \
  PYTHONPATH="$openpilot:$openpilot/tinygrad_repo" TINYGRAD_PATH="$openpilot/tinygrad_repo" \
  "$python" "$openpilot/$compiler" driving_supercombo.onnx "$pkl" --device-input '*' --benchmark-runs 1 </dev/null
'''


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser, 'the supercombo.pt to publish, or an .onnx already exported (compile only)')
    parser.add_argument('--name', default=None, help='name of the model (default: the run folder)')
    parser.add_argument('--stable', action='store_true', help='the PyTorch weights become the base of the trainings '
                        '"from the stable weights"')
    parser.add_argument('--stock', action='store_true', help='publishes the comma\'s original model')
    parser.add_argument('--upload', action='store_true', help='really uploads to Hugging Face (without it: dry run)')
    args = parser.parse_args(arguments)
    check_model_arguments(parser, args)
    if args.name is not None and not NAME_RE.fullmatch(args.name):
        parser.error('invalid name: ' + args.name + ' (only letters, digits, _ . -)')
    if args.stock:
        if args.run is not None or args.model is not None:
            parser.error('--stock takes no model: it takes the original one from the comma')
        if args.stable:
            parser.error('--stable with --stock makes no sense: the comma\'s model has no PyTorch weights')
    return args


def slug_of(name: str) -> str:
    """File and folder name: lowercase, anything else becomes "_" (2026-10-20_1910 -> 2026_10_20_1910)."""
    return re.sub(r'[^a-z0-9]', '_', name.lower())


def stock_name(default_model: str) -> str:
    """The name of the comma's own model: CD210 -> cd210-native."""
    return re.sub(r'[^a-z0-9]', '-', default_model.lower()) + '-native'


def read_publish_info(comma: Comma) -> dict:
    try:
        info = comma.key_values(PUBLISH_INFO_SCRIPT, BACKUP)
    except TaskError as exc:
        raise TaskError('cannot read /data/openpilot on the comma (' + str(exc) + ')') from exc
    for key in INFO_KEYS:
        info.setdefault(key, '')
    return info


def check_comma(info: dict) -> None:
    if info['onroad'] == '1':
        raise TaskError('the comma is onroad: turn the car off and try again')
    if not info['compiler']:
        raise TaskError('compile_modeld_v2.py is not on the comma: sunnypilot is needed')
    if not info['reference']:
        raise TaskError('driving_supercombo.onnx not found on the comma')
    if not info['selector_version']:
        raise TaskError('REQUIRED_JSON_VERSION not found on the comma (sunnypilot/models/helpers.py)')
    if not info['python']:
        raise TaskError('/usr/local/venv/bin/python3 is not on the comma')


def check_login(hf_repo: str) -> None:
    """Refuses an upload without a Hugging Face login, or with a read-only token."""
    from huggingface_hub import HfApi

    try:
        user = HfApi().whoami()
    except Exception as exc:  # no saved token, revoked token, Hugging Face not answering
        detail = (str(exc).strip().splitlines() or [type(exc).__name__])[0][:200]
        raise TaskError('Hugging Face: no valid login on this server (' + detail + '): it needs a token '
                        'that can write ' + hf_repo) from exc
    token = (user.get('auth') or {}).get('accessToken') or {}
    if token.get('role') == 'read':
        raise TaskError('Hugging Face: the token of this server is read-only, it cannot write ' + hf_repo)
    say('Hugging Face: logged in as ' + str(user.get('name', '?')))


def check_name_free(hf_repo: str, slug: str, name: str, upload: bool) -> None:
    """A model already published under this name: refused before minutes of export and compile.
    Public API, no token: 200 = models/<slug>/ exists, 404 = free. In a dry run only a warning."""
    import requests

    url = HF_TREE_URL.format(repo=hf_repo, slug=slug)
    try:
        code = requests.get(url, timeout=HF_TIMEOUT_SECONDS, allow_redirects=False).status_code
    except requests.RequestException:
        code = None
    if code == 200:
        if upload:
            raise TaskError('the name ' + name + ' is already published on Hugging Face (models/' + slug + '/ in '
                            + hf_repo + '): choose another name')
        say('WARNING: the name ' + name + ' is already published on Hugging Face (models/' + slug + '/): the dry '
            'run prepares it anyway, but uploading it needs another name')
    elif code == 404:
        say('Name free on Hugging Face: models/' + slug + '/')
    else:
        answer = 'HTTP ' + str(code) if code is not None else 'no answer'
        if upload:
            raise TaskError('cannot check on Hugging Face whether the name ' + name + ' is free (' + url
                            + ', ' + answer + ')')
        say('WARNING: cannot check on Hugging Face whether the name ' + name + ' is free (' + answer
            + '), going on with the dry run')


def upload_stage(hf_repo: str, stage: Path, name: str) -> None:
    """One commit with everything in the stage folder, at the same paths in the repo."""
    from huggingface_hub import HfApi

    try:
        HfApi().upload_folder(repo_id=hf_repo, repo_type='model', folder_path=str(stage),
                              commit_message='Publish model ' + name)
    except Exception as exc:  # network, permissions, a commit refused by the Hub
        detail = (str(exc).strip().splitlines() or [type(exc).__name__])[0][:300]
        raise TaskError('upload to Hugging Face failed: ' + detail) from exc


def compile_on_comma(comma: Comma, onnx: Path, info: dict, pkl_name: str, destination: Path) -> None:
    """Step 3: compiles the ONNX for modeld_v2 in BUILD_DIR and copies the compiled file here."""
    comma.run('rm -rf ' + BUILD_DIR + ' && mkdir -p ' + BUILD_DIR)
    comma.push(onnx, BUILD_DIR + '/' + ONNX_NAME)
    result = comma.script(COMPILE_SCRIPT, BUILD_DIR, REMOTE_OPENPILOT, info['python'], info['compiler'], pkl_name,
                          check=False)
    for line in ((result.stdout or '') + (result.stderr or '')).splitlines():
        say(line)
    if result.returncode != 0:
        raise TaskError('compile failed on the comma (files in ' + BUILD_DIR + ')')
    comma.pull(BUILD_DIR + '/' + pkl_name, destination)
    comma.run('rm -rf ' + BUILD_DIR)


def publish(args: argparse.Namespace, comma: Comma) -> int:
    hf_repo = os.environ.get('HF_REPO') or DEFAULT_HF_REPO
    if not HF_REPO_RE.fullmatch(hf_repo):
        raise TaskError('invalid HF_REPO: ' + hf_repo)
    steps = 5 if args.upload else 4
    model = None if args.stock else resolve_model(args.run, args.model)

    comma.require()
    info = read_publish_info(comma)
    check_comma(info)

    name = args.name
    if args.stock:
        if not info['default_model']:
            raise TaskError('DEFAULT_MODEL not found on the comma (sunnypilot/models/model_name.py)')
        if not name:
            name = stock_name(info['default_model'])
        if not NAME_RE.fullmatch(name):
            raise TaskError('invalid name: ' + name + ' (only letters, digits, _ . -)')
        # The original of this openpilot version, sha256-checked (the export reference is the same file).
        original = update_reference(comma, info, strict=True)
        model = STOCK_DIR / name / ONNX_NAME
        model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, model)
        say('Comma original: ' + info['default_model'] + ' (' + (original_model_source(info) or str(original))
            + ')')

    run_dir = model.parent
    is_onnx = model.suffix == '.onnx'
    # PyTorch weights for the training from the stable weights: the .pt given, or the one of the run of
    # the given .onnx. The comma's own model (--stock) has none.
    weights = None
    if not args.stock and not is_onnx:
        weights = model
    elif not args.stock and (run_dir / 'supercombo.pt').is_file():
        weights = run_dir / 'supercombo.pt'
    if args.stable and weights is None:
        raise TaskError('--stable needs the PyTorch weights: give the supercombo.pt (or an .onnx with a supercombo.pt '
                        'next to it)')

    if not name:
        name = run_dir.name
    if not NAME_RE.fullmatch(name):
        raise TaskError('invalid name: ' + name + ' (only letters, digits, _ . -)')
    slug = slug_of(name)
    pkl_name = 'driving_distill_' + slug + '_tinygrad.pkl'
    stage = PUBLISH_DIR / slug

    if args.upload:
        check_login(hf_repo)
    check_name_free(hf_repo, slug, name, args.upload)

    say('Model: ' + name + ' (' + pkl_name + ')')
    say('Comma: tinygrad ' + info['tinygrad_ref'] + ', catalog version ' + info['selector_version'])

    if stage.exists():
        shutil.rmtree(stage)
    (stage / 'models' / slug).mkdir(parents=True)

    # 1. Export, unless an .onnx was given.
    if is_onnx:
        onnx = model
        say('== 1. export skipped, using ' + str(onnx))
    else:
        onnx = run_dir / ONNX_NAME
        say('== 1. export ' + str(model) + ' -> ' + str(onnx))
        reference = update_reference(comma, info, strict=True)
        export_onnx(model, reference, onnx)
    shutil.copyfile(onnx, stage / 'models' / slug / ONNX_NAME)
    say('PROGRESS 1/' + str(steps))

    # 2. PyTorch weights into the stage training/, uploaded in the same commit. Before the compile: a .pt
    # that cannot be published (training checkpoint, release name taken by other weights) stops here
    # instead of after minutes on the comma.
    if weights is not None:
        say('== 2. PyTorch weights ' + str(weights) + ' -> training/releases/' + slug + '/')
        stable_args = ['--stable'] if args.stable else []
        run_module('helpers.publish_training_weights', '--repo=' + hf_repo, '--release=' + slug,
                   '--model=supercombo=' + str(weights), '--stage-dir=' + str(stage), *stable_args,
                   failure='PyTorch weights cannot be published (error above)')
    else:
        say('== 2. PyTorch weights skipped: no supercombo.pt, a training "from the stable weights" cannot start '
            'from this model')
    say('PROGRESS 2/' + str(steps))

    # 3. Compile on the comma.
    say('== 3. compiling on the comma (a few minutes)')
    compiled = stage / 'models' / slug / pkl_name
    compile_on_comma(comma, onnx, info, pkl_name, compiled)
    sha256 = sha256_of(compiled)
    say('Compiled: ' + pkl_name + ' sha256 ' + sha256)
    say('PROGRESS 3/' + str(steps))

    # 4. Catalog: only this model, the menu shows the last one published.
    say('== 4. catalog')
    base_url = 'https://huggingface.co/' + hf_repo + '/resolve/main/'
    run_module('helpers.model_catalog', str(stage / 'driving_models.json'), '--name=' + name,
               '--file-name=' + pkl_name, '--url=' + base_url + 'models/' + slug + '/' + pkl_name,
               '--sha256=' + sha256, '--selector-version=' + info['selector_version'],
               '--tinygrad-ref=' + info['tinygrad_ref'], failure='catalog not written')
    say('PROGRESS 4/' + str(steps))

    if not args.upload:
        say('Dry run: nothing uploaded. Files ready in ' + str(stage))
        return 0

    # 5. One commit with the model files, the new catalog and the PyTorch weights.
    say('== 5. uploading to https://huggingface.co/' + hf_repo)
    upload_stage(hf_repo, stage, name)
    say('PROGRESS 5/' + str(steps))
    say('Published. Catalog: ' + base_url + 'driving_models.json')
    if weights is not None and args.stable:
        say('PyTorch weights: training/releases/' + slug + '/supercombo.pt, now stable: the next training "from the '
            'stable weights" (Training) starts from here.')
    elif weights is not None:
        say('PyTorch weights: training/releases/' + slug + '/supercombo.pt (stable not moved: publishing it as '
            'stable makes it the base of the trainings "from the stable weights").')
    return 0


def main() -> int:
    args = parse_args()
    load_env()
    return publish(args, Comma(comma_host(args.host)))


if __name__ == '__main__':
    run_main(main)
