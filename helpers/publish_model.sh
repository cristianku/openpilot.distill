#!/usr/bin/env bash
# Publish a trained model on Hugging Face (cristianku/openpilot-distill-models), so that sunnypilot
# can offer it in the Models menu: export to ONNX, compile it on the comma for sunnypilot's
# modeld_v2, upload ONNX + compiled model + the catalog driving_models.json in one commit,
# together with the PyTorch weights that helpers/train_supervised.sh --continue starts from.
#
# Usage: helpers/publish_model.sh [--host=<ip|name>] [--name=<name>] [--stable] [--no-upload]
#                                 [<supercombo.pt|file.onnx>]
#        helpers/publish_model.sh [--host=<ip|name>] [--name=<name>] [--no-upload] --stock
#   helpers/publish_model.sh                       (the latest outputs/supervised/*/supercombo.pt)
#   helpers/publish_model.sh outputs/supervised/2026-10-20_1855/supercombo.pt --name=trial
#   helpers/publish_model.sh outputs/supervised/2026-10-20_1855/supercombo.pt --stable  (base of --continue)
#   helpers/publish_model.sh outputs/supervised/2026-10-20_1855/driving_supercombo.onnx   (only recompiles)
#   helpers/publish_model.sh --stock               (the comma's original model, e.g. cd210-native)
#   helpers/publish_model.sh --no-upload           (prepares everything, uploads nothing)
#
# Steps:
#   1. export: supercombo.pt -> driving_supercombo.onnx next to it, with helpers/export_onnx.py and
#      the comma's original driving_supercombo.onnx as reference for inputs and output layout
#      (from /data/driving_supercombo.onnx.orig when a trained model is installed in its place).
#      Skipped when an .onnx is given.
#   2. PyTorch weights: the supercombo.pt (the one given, or the one next to the given .onnx) as
#      training/releases/<name>/supercombo.pt + manifest.json, with helpers/publish_training_weights.py.
#      Only the weights: a supercombo.training.pt (with the optimizer) is refused. A release with
#      this name already on Hugging Face is left as it is when the weights are the same, refused
#      when they differ. Skipped for --stock and for an .onnx without supercombo.pt next to it.
#   3. compile on the comma (offroad) with sunnypilot's compile_modeld_v2.py: modeld_v2 loads only
#      that format, not the pickle of compile_onnx.py. It uses the comma's tinygrad: after a
#      sunnypilot update that changes tinygrad, run again with the .onnx and a new --name.
#   4. catalog: a new driving_models.json with only this model (helpers/model_catalog.py) and the
#      sha256 of the compiled file. The models published before (--stock included) leave the menu,
#      their files stay on Hugging Face.
#   5. one Hugging Face commit with models/<name>/driving_supercombo.onnx, models/<name>/<file.pkl>,
#      driving_models.json and the training/ files of step 2. Everything is also left in
#      outputs/publish/<name>/.
# Nothing is installed on the comma: the build folder there is removed at the end. The model shows
# up in the Models menu only with the sunnypilot patch that reads this catalog.
#
# Hugging Face: "uv run hf auth login" done once on this machine, with a token that can write
# the repo.
# The comma only accepts the keys registered on GitHub: this machine needs one of them, or connect
# to it with agent forwarding (ssh -A) from the Mac. ssh never asks anything (BatchMode).
#
# --name: name of the model in the menu and in the files (default: the run folder, e.g.
#         2026-10-20_1910). Letters, digits, _ . -  A name whose folder models/<name>/ is already
#         on Hugging Face is refused before anything else is done (checked with the public API):
#         replacing a published model changes its sha256 and puts a comma that uses it back on
#         Default. With --no-upload the check is only a warning.
# --stock: publishes the comma's own original driving model (the Default of the Models menu, e.g.
#          CD210), compiled the same way, as "<default>-native" (e.g. cd210-native): a baseline to
#          tell problems of this compile path from problems of a trained model. Taken from the comma
#          (or from /data/driving_supercombo.onnx.orig when a trained model was installed in its
#          place); kept in outputs/stock/<name>/.
# --stable: also points training/stable.json to these PyTorch weights, so the next
#           helpers/train_supervised.sh --continue starts from this model. Without it the pointer
#           stays where it is: a model published only to try it in the car does not become the base.
# --no-upload: stops before Hugging Face.
# --host: IP or ssh name of the comma; "comma@" is added when no user is given.
# -h, --help: this help.
# Env: COMMA_HOST (default "comma", the Host in ~/.ssh/config), overridden by --host
#      HF_REPO (default cristianku/openpilot-distill-models)
#      Both can also be set in the repo .env (see helpers/common.sh).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
. "$ROOT/helpers/common.sh"
distill_load_env

COMMA_HOST="${COMMA_HOST:-comma}"
HF_REPO="${HF_REPO:-cristianku/openpilot-distill-models}"
REMOTE_OP="/data/openpilot"
BUILD_DIR="/data/openpilot_distill/build"
# Original model saved by helpers/install_model_on_comma.sh.
BACKUP="/data/driving_supercombo.onnx.orig"

die() { echo "ERROR: $*" >&2; exit 1; }

# Prints the comment block at the top of this file.
usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

# --host without an explicit user: the comma login user is "comma".
# The default "comma" is left alone, user and IP come from ~/.ssh/config.
add_comma_user() {
  if [[ "$COMMA_HOST" != *@* ]]; then
    COMMA_HOST="comma@$COMMA_HOST"
  fi
}

# Prints "key=value" lines read from the comma: onroad state and where this openpilot keeps the
# files used here (paths asked to git, the layout changes between versions). For --stock also the
# name of the default model. To find the original model (--stock and the export reference): the
# sha256 of the original of this openpilot version (from git, the onnx is in LFS), of the current
# one and of the backup.
read_comma_info() {
  ssh "${DISTILL_SSH_OPTIONS[@]}" "$COMMA_HOST" bash -s -- "$BACKUP" <<'EOF'
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
EOF
}

name=""
upload=1
stock=0
stable=0
input=""

while [ $# -gt 0 ]; do
  if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
  elif [ "$1" = "--no-upload" ]; then
    upload=0
  elif [ "$1" = "--stock" ]; then
    stock=1
  elif [ "$1" = "--stable" ]; then
    stable=1
  elif [ "$1" = "--name" ]; then
    shift
    [ -n "${1:-}" ] || die "--name needs a name"
    name="$1"
  elif [[ "$1" == --name=* ]]; then
    name="${1#--name=}"
    [ -n "$name" ] || die "--name needs a name"
  elif [ "$1" = "--host" ]; then
    shift
    [ -n "${1:-}" ] || die "--host needs an IP or a name"
    COMMA_HOST="$1"
    add_comma_user
  elif [[ "$1" == --host=* ]]; then
    COMMA_HOST="${1#--host=}"
    [ -n "$COMMA_HOST" ] || die "--host needs an IP or a name"
    add_comma_user
  elif [ -z "$input" ]; then
    input="$1"
  else
    die "extra argument: $1"
  fi
  shift
done

if [ "$stock" -eq 1 ]; then
  [ -z "$input" ] || die "--stock takes no file: it takes the original model from the comma"
  [ "$stable" -eq 0 ] || die "--stable with --stock makes no sense: the comma's model has no PyTorch weights"
elif [ -z "$input" ]; then
  input="$(ls -t "$ROOT"/outputs/supervised/*/supercombo.pt 2>/dev/null | head -1 || true)"
  [ -n "$input" ] || die "no outputs/supervised/*/supercombo.pt: pass the .pt or the .onnx"
fi

ssh "${DISTILL_SSH_OPTIONS[@]}" "$COMMA_HOST" true \
  || die "comma '$COMMA_HOST' not reachable over ssh (on and online? ssh key registered? from the Mac use ssh -A)"

onroad=""
compiler=""
reference=""
selector_version=""
default_model=""
original=""
current=""
backup=""
tinygrad_ref=""
python=""
info="$(read_comma_info)" || die "cannot read /data/openpilot on the comma"
while IFS='=' read -r key value; do
  if [ "$key" = "onroad" ]; then
    onroad="$value"
  elif [ "$key" = "compiler" ]; then
    compiler="$value"
  elif [ "$key" = "reference" ]; then
    reference="$value"
  elif [ "$key" = "selector_version" ]; then
    selector_version="$value"
  elif [ "$key" = "default_model" ]; then
    default_model="$value"
  elif [ "$key" = "original" ]; then
    original="$value"
  elif [ "$key" = "current" ]; then
    current="$value"
  elif [ "$key" = "backup" ]; then
    backup="$value"
  elif [ "$key" = "tinygrad_ref" ]; then
    tinygrad_ref="$value"
  elif [ "$key" = "python" ]; then
    python="$value"
  fi
done <<< "$info"

[ "$onroad" != "1" ] || die "the comma is onroad: turn the car off and try again"
[ -n "$compiler" ] || die "compile_modeld_v2.py is not on the comma: sunnypilot is needed"
[ -n "$reference" ] || die "driving_supercombo.onnx not found on the comma"
[ -n "$selector_version" ] || die "REQUIRED_JSON_VERSION not found on the comma (sunnypilot/models/helpers.py)"
[ -n "$python" ] || die "/usr/local/venv/bin/python3 is not on the comma"

# The original model of this openpilot version, from its place or from the backup that
# helpers/install_model_on_comma.sh keeps while a trained model is installed in its place.
# Used by --stock and as the export reference.
original_source=""
if [ -n "$original" ] && [ "$current" = "$original" ]; then
  original_source="$REMOTE_OP/$reference"
elif [ -n "$original" ] && [ "$backup" = "$original" ]; then
  original_source="$BACKUP"
fi

# --stock: the original model of this openpilot version.
if [ "$stock" -eq 1 ]; then
  [ -n "$default_model" ] || die "DEFAULT_MODEL not found on the comma (sunnypilot/models/model_name.py)"
  [ -n "$original_source" ] || die "the original of this version is not on the comma (neither in its place nor in $BACKUP)"
  if [ -z "$name" ]; then
    name="$(echo "$default_model" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9\n' '-')-native"
  fi
  input="$ROOT/outputs/stock/$name/driving_supercombo.onnx"
  mkdir -p "$(dirname "$input")"
  rsync -a -e "$DISTILL_RSYNC_SSH" "$COMMA_HOST:$original_source" "$input"
  echo "The comma's original: $default_model ($original_source)"
fi

[ -f "$input" ] || die "$input does not exist"
input="$(cd "$(dirname "$input")" && pwd)/$(basename "$input")"
run_dir="$(dirname "$input")"

# PyTorch weights for helpers/train_supervised.sh --continue: the .pt given, or the one of the
# run of the given .onnx. The comma's own model (--stock) has none.
weights=""
if [ "$stock" -eq 0 ] && [[ "$input" != *.onnx ]]; then
  weights="$input"
elif [ "$stock" -eq 0 ] && [ -f "$run_dir/supercombo.pt" ]; then
  weights="$run_dir/supercombo.pt"
fi
[ -n "$weights" ] || [ "$stable" -eq 0 ] \
  || die "--stable needs the PyTorch weights: pass the supercombo.pt (or an .onnx with supercombo.pt next to it)"

if [ -z "$name" ]; then
  name="$(basename "$run_dir")"
fi
[[ "$name" =~ ^[A-Za-z0-9_.-]+$ ]] || die "invalid name: $name (only letters, digits, _ . -)"
# File and folder names: lowercase, anything else becomes "_" (2026-10-20_1910 -> 2026_10_20_1910).
slug="$(echo "$name" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9\n' '_')"
pkl_name="driving_distill_${slug}_tinygrad.pkl"
stage="$ROOT/outputs/publish/$slug"

# A model already published under this name: refused before minutes of export and compile.
# Public API, no token: 200 = models/<slug>/ exists, 404 = free.
hf_tree_url="https://huggingface.co/api/models/$HF_REPO/tree/main/models/$slug"
hf_code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 30 "$hf_tree_url" || true)"
if [ "$hf_code" = "200" ]; then
  if [ "$upload" -eq 1 ]; then
    die "the name $name is already published on Hugging Face (models/$slug/ in $HF_REPO): choose another --name"
  fi
  echo "WARNING: the name $name is already published on Hugging Face (models/$slug/): with --no-upload it is prepared anyway, but uploading it needs another --name"
elif [ "$hf_code" = "404" ]; then
  echo "Name free on Hugging Face: models/$slug/"
elif [ "$upload" -eq 1 ]; then
  die "cannot check on Hugging Face whether the name $name is free ($hf_tree_url, HTTP $hf_code)"
else
  echo "WARNING: cannot check on Hugging Face whether the name $name is free (HTTP $hf_code), going on with --no-upload"
fi

echo "Model: $name ($pkl_name)"
echo "Comma: tinygrad $tinygrad_ref, catalog version $selector_version"

rm -rf "$stage"
mkdir -p "$stage/models/$slug"

# 1. Export, unless an .onnx was given.
if [[ "$input" == *.onnx ]]; then
  onnx="$input"
  echo "== 1. export skipped, using $onnx"
else
  onnx="$run_dir/driving_supercombo.onnx"
  [ -n "$original_source" ] \
    || die "the original of this version is not on the comma (neither in its place nor in $BACKUP): it is needed as the export reference"
  echo "== 1. export $input -> $onnx (reference $original_source)"
  rsync -a -e "$DISTILL_RSYNC_SSH" "$COMMA_HOST:$original_source" "$stage/reference.onnx"
  (cd "$ROOT" && distill_uv_run python -m helpers.export_onnx "$input" --reference "$stage/reference.onnx" --output "$onnx")
  rm -f "$stage/reference.onnx"
fi
cp "$onnx" "$stage/models/$slug/driving_supercombo.onnx"

# 2. PyTorch weights into $stage/training/, uploaded in the same commit. Before the compile:
# a .pt that cannot be published (training checkpoint, release name taken by other weights)
# stops here instead of after minutes on the comma.
if [ -n "$weights" ]; then
  echo "== 2. PyTorch weights $weights -> training/releases/$slug/"
  stable_args=()
  if [ "$stable" -eq 1 ]; then
    stable_args=(--stable)
  fi
  (cd "$ROOT" && distill_uv_run python -m helpers.publish_training_weights --repo "$HF_REPO" --release "$slug" \
    --model "supercombo=$weights" --stage-dir "$stage" "${stable_args[@]}") \
    || die "PyTorch weights cannot be published (error above)"
else
  echo "== 2. PyTorch weights skipped: no supercombo.pt, --continue will not be able to start from this model"
fi

# 3. Compile on the comma, flags as sunnypilot CI (TG_FLAGS_QCOM in .github/workflows/sunnypilot-build-model.yaml).
# No taskset: this comma has only CPUs 0-3 online. compile_modeld_v2 writes a temporary file in the cwd.
echo "== 3. compiling on the comma (a few minutes)"
ssh "${DISTILL_SSH_OPTIONS[@]}" "$COMMA_HOST" "rm -rf $BUILD_DIR && mkdir -p $BUILD_DIR"
rsync -a -e "$DISTILL_RSYNC_SSH" "$onnx" "$COMMA_HOST:$BUILD_DIR/driving_supercombo.onnx"
ssh "${DISTILL_SSH_OPTIONS[@]}" "$COMMA_HOST" "cd $BUILD_DIR && env DEV=QCOM IMAGE=1 FLOAT16=1 NOLOCALS=1 OPENPILOT_HACKS=1 \
  PYTHONPATH=$REMOTE_OP:$REMOTE_OP/tinygrad_repo TINYGRAD_PATH=$REMOTE_OP/tinygrad_repo \
  $python $REMOTE_OP/$compiler driving_supercombo.onnx $pkl_name --device-input '*' --benchmark-runs 1" \
  || die "compilation failed on the comma (files in $BUILD_DIR)"
rsync -a -e "$DISTILL_RSYNC_SSH" "$COMMA_HOST:$BUILD_DIR/$pkl_name" "$stage/models/$slug/"
ssh "${DISTILL_SSH_OPTIONS[@]}" "$COMMA_HOST" "rm -rf $BUILD_DIR"
sha256="$(sha256sum "$stage/models/$slug/$pkl_name" | cut -d' ' -f1)"
echo "Compiled: $pkl_name sha256 $sha256"

# 4. Catalog: only this model, the menu shows the last one published.
echo "== 4. catalog"
catalog_url="https://huggingface.co/$HF_REPO/resolve/main/driving_models.json"
(cd "$ROOT" && distill_uv_run python -m helpers.model_catalog "$stage/driving_models.json" \
  --name "$name" --file-name "$pkl_name" \
  --url "https://huggingface.co/$HF_REPO/resolve/main/models/$slug/$pkl_name" \
  --sha256 "$sha256" --selector-version "$selector_version" --tinygrad-ref "$tinygrad_ref")

if [ "$upload" -eq 0 ]; then
  echo "--no-upload: nothing uploaded. Files ready in $stage"
  exit 0
fi

# 5. One commit with the model files, the new catalog and the PyTorch weights.
echo "== 5. uploading to https://huggingface.co/$HF_REPO"
(cd "$ROOT" && distill_uv_run hf upload "$HF_REPO" "$stage" . --commit-message "Publish model $name")

echo "Published. Catalog: $catalog_url"
if [ -n "$weights" ] && [ "$stable" -eq 1 ]; then
  echo "PyTorch weights: training/releases/$slug/supercombo.pt, now stable: helpers/train_supervised.sh --continue starts from here."
elif [ -n "$weights" ]; then
  echo "PyTorch weights: training/releases/$slug/supercombo.pt (stable not moved: with --stable it becomes the base of --continue)."
fi
