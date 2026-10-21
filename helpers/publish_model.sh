#!/usr/bin/env bash
# Publish a trained model on Hugging Face (cristianku/openpilot-distill-models), so that sunnypilot
# can offer it in the Models menu: export to ONNX, compile it on the comma for sunnypilot's
# modeld_v2, upload ONNX + compiled model + the catalog driving_models.json in one commit,
# together with the PyTorch weights that helpers/train_supervised.sh --continue starts from.
#
# Usage: helpers/publish_model.sh [--host=<ip|nome>] [--name=<nome>] [--stable] [--no-upload]
#                                 [<supercombo.pt|file.onnx>]
#        helpers/publish_model.sh [--host=<ip|nome>] [--name=<nome>] [--no-upload] --stock
#   helpers/publish_model.sh                       (l'ultimo outputs/supervised/*/supercombo.pt)
#   helpers/publish_model.sh outputs/supervised/2026-10-20_1855/supercombo.pt --name=prova
#   helpers/publish_model.sh outputs/supervised/2026-10-20_1855/supercombo.pt --stable  (base di --continue)
#   helpers/publish_model.sh outputs/supervised/2026-10-20_1855/driving_supercombo.onnx   (solo ricompila)
#   helpers/publish_model.sh --stock               (il modello originale del comma, es. cd210-native)
#   helpers/publish_model.sh --no-upload           (prepara tutto, non carica niente)
#
# Steps:
#   1. export: supercombo.pt -> driving_supercombo.onnx next to it, with helpers/export_onnx.py and
#      the comma's own driving_supercombo.onnx as reference for inputs and output layout.
#      Skipped when an .onnx is given.
#   2. PyTorch weights: the supercombo.pt (the one given, or the one next to the given .onnx) as
#      training/releases/<nome>/supercombo.pt + manifest.json, with scripts/publish_training_weights.py.
#      Only the weights: a supercombo.training.pt (with the optimizer) is refused. A release with
#      this name already on Hugging Face is left as it is when the weights are the same, refused
#      when they differ. Skipped for --stock and for an .onnx without supercombo.pt next to it.
#   3. compile on the comma (offroad) with sunnypilot's compile_modeld_v2.py: modeld_v2 loads only
#      that format, not the pickle of compile_onnx.py. It uses the comma's tinygrad: after a
#      sunnypilot update that changes tinygrad, run again with the .onnx.
#   4. catalog: downloads driving_models.json from Hugging Face and adds/replaces this model
#      (helpers/model_catalog.py) with the sha256 of the compiled file.
#   5. one Hugging Face commit with models/<nome>/driving_supercombo.onnx, models/<nome>/<file.pkl>,
#      driving_models.json and the training/ files of step 2. Everything is also left in
#      outputs/publish/<nome>/.
# Nothing is installed on the comma: the build folder there is removed at the end. The model shows
# up in the Models menu only with the sunnypilot patch that reads this catalog.
#
# Hugging Face: "uv run hf auth login" done once on this machine, with a token that can write
# the repo.
# The comma only accepts the keys registered on GitHub: connect to this machine with agent
# forwarding (ssh -A) so the Mac's key is used.
#
# --name: name of the model in the menu and in the files (default: the run folder, e.g.
#         2026-10-20_1910). Letters, digits, _ . -  The same name again replaces the model.
# --stock: publishes the comma's own original driving model (the Default of the Models menu, e.g.
#          CD210), compiled the same way, as "<default>-native" (e.g. cd210-native): a baseline to
#          tell problems of this compile path from problems of a trained model. Taken from the comma
#          (or from /data/driving_supercombo.onnx.orig when a trained model was installed in its
#          place); kept in outputs/stock/<nome>/.
# --stable: also points training/stable.json to these PyTorch weights, so the next
#           helpers/train_supervised.sh --continue starts from this model. Without it the pointer
#           stays where it is: a model published only to try it in the car does not become the base.
# --no-upload: stops before Hugging Face.
# --host: IP or ssh name of the comma; "comma@" is added when no user is given.
# -h, --help: this help.
# Env: COMMA_HOST (default "comma", the Host in ~/.ssh/config), overridden by --host
#      HF_REPO (default cristianku/openpilot-distill-models)

set -euo pipefail

COMMA_HOST="${COMMA_HOST:-comma}"
HF_REPO="${HF_REPO:-cristianku/openpilot-distill-models}"
REMOTE_OP="/data/openpilot"
BUILD_DIR="/data/openpilot_distill/build"
# Original model saved by helpers/install_model_on_comma.sh.
BACKUP="/data/driving_supercombo.onnx.orig"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

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
# name of the default model and the sha256 of the original model of this openpilot version (from
# git, the onnx is in LFS), of the current one and of the backup.
read_comma_info() {
  ssh "$COMMA_HOST" bash -s -- "$BACKUP" <<'EOF'
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
    name="${1:-}"
  elif [[ "$1" == --name=* ]]; then
    name="${1#--name=}"
  elif [ "$1" = "--host" ]; then
    shift
    [ -n "${1:-}" ] || die "--host vuole un IP o un nome"
    COMMA_HOST="$1"
    add_comma_user
  elif [[ "$1" == --host=* ]]; then
    COMMA_HOST="${1#--host=}"
    [ -n "$COMMA_HOST" ] || die "--host vuole un IP o un nome"
    add_comma_user
  elif [ -z "$input" ]; then
    input="$1"
  else
    die "argomento in piu': $1"
  fi
  shift
done

if [ "$stock" -eq 1 ]; then
  [ -z "$input" ] || die "--stock non vuole un file: prende il modello originale dal comma"
  [ "$stable" -eq 0 ] || die "--stable con --stock non ha senso: il modello del comma non ha pesi PyTorch"
elif [ -z "$input" ]; then
  input="$(ls -t "$ROOT"/outputs/supervised/*/supercombo.pt 2>/dev/null | head -1 || true)"
  [ -n "$input" ] || die "nessun outputs/supervised/*/supercombo.pt: passa il .pt o l'.onnx"
fi

ssh -o ConnectTimeout=8 "$COMMA_HOST" true \
  || die "comma '$COMMA_HOST' non raggiungibile via ssh (hai fatto ssh -A?)"

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
info="$(read_comma_info)" || die "non riesco a leggere /data/openpilot sul comma"
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

[ "$onroad" != "1" ] || die "il comma e' onroad: spegni l'auto e riprova"
[ -n "$compiler" ] || die "sul comma non c'e' compile_modeld_v2.py: serve sunnypilot"
[ -n "$reference" ] || die "sul comma non trovo driving_supercombo.onnx"
[ -n "$selector_version" ] || die "sul comma non trovo REQUIRED_JSON_VERSION (sunnypilot/models/helpers.py)"
[ -n "$python" ] || die "sul comma non c'e' /usr/local/venv/bin/python3"

# --stock: the original model of this openpilot version, from its place or from the backup.
if [ "$stock" -eq 1 ]; then
  [ -n "$default_model" ] || die "sul comma non trovo DEFAULT_MODEL (sunnypilot/models/model_name.py)"
  if [ -z "$name" ]; then
    name="$(echo "$default_model" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9\n' '-')-native"
  fi
  if [ -n "$original" ] && [ "$current" = "$original" ]; then
    stock_source="$REMOTE_OP/$reference"
  elif [ -n "$original" ] && [ "$backup" = "$original" ]; then
    stock_source="$BACKUP"
  else
    die "sul comma non trovo l'originale di questa versione (ne' al suo posto ne' in $BACKUP)"
  fi
  input="$ROOT/outputs/stock/$name/driving_supercombo.onnx"
  mkdir -p "$(dirname "$input")"
  rsync -a "$COMMA_HOST:$stock_source" "$input"
  echo "Originale del comma: $default_model ($stock_source)"
fi

[ -f "$input" ] || die "$input non esiste"
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
  || die "--stable vuole i pesi PyTorch: passa il supercombo.pt (o un .onnx con supercombo.pt accanto)"

if [ -z "$name" ]; then
  name="$(basename "$run_dir")"
fi
[[ "$name" =~ ^[A-Za-z0-9_.-]+$ ]] || die "nome non valido: $name (solo lettere, numeri, _ . -)"
# File and folder names: lowercase, anything else becomes "_" (2026-10-20_1910 -> 2026_10_20_1910).
slug="$(echo "$name" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9\n' '_')"
pkl_name="driving_distill_${slug}_tinygrad.pkl"
stage="$ROOT/outputs/publish/$slug"

echo "Modello: $name ($pkl_name)"
echo "Comma: tinygrad $tinygrad_ref, catalogo versione $selector_version"

rm -rf "$stage"
mkdir -p "$stage/models/$slug"

# 1. Export, unless an .onnx was given.
if [[ "$input" == *.onnx ]]; then
  onnx="$input"
  echo "== 1. export saltato, uso $onnx"
else
  onnx="$run_dir/driving_supercombo.onnx"
  echo "== 1. export $input -> $onnx"
  rsync -a "$COMMA_HOST:$REMOTE_OP/$reference" "$stage/reference.onnx"
  (cd "$ROOT" && uv run python -m helpers.export_onnx "$input" --reference "$stage/reference.onnx" --output "$onnx")
  rm -f "$stage/reference.onnx"
fi
cp "$onnx" "$stage/models/$slug/driving_supercombo.onnx"

# 2. PyTorch weights into $stage/training/, uploaded in the same commit. Before the compile:
# a .pt that cannot be published (training checkpoint, release name taken by other weights)
# stops here instead of after minutes on the comma.
if [ -n "$weights" ]; then
  echo "== 2. pesi PyTorch $weights -> training/releases/$slug/"
  stable_args=()
  if [ "$stable" -eq 1 ]; then
    stable_args=(--stable)
  fi
  (cd "$ROOT" && uv run python scripts/publish_training_weights.py --repo "$HF_REPO" --release "$slug" \
    --model "supercombo=$weights" --stage-dir "$stage" "${stable_args[@]}") \
    || die "pesi PyTorch non pubblicabili (errore sopra)"
else
  echo "== 2. pesi PyTorch saltati: nessun supercombo.pt, --continue non potra' partire da questo modello"
fi

# 3. Compile on the comma, flags as sunnypilot CI (TG_FLAGS_QCOM in .github/workflows/sunnypilot-build-model.yaml).
# No taskset: this comma has only CPUs 0-3 online. compile_modeld_v2 writes a temporary file in the cwd.
echo "== 3. compilo sul comma (qualche minuto)"
ssh "$COMMA_HOST" "rm -rf $BUILD_DIR && mkdir -p $BUILD_DIR"
rsync -a "$onnx" "$COMMA_HOST:$BUILD_DIR/driving_supercombo.onnx"
ssh "$COMMA_HOST" "cd $BUILD_DIR && env DEV=QCOM IMAGE=1 FLOAT16=1 NOLOCALS=1 OPENPILOT_HACKS=1 \
  PYTHONPATH=$REMOTE_OP:$REMOTE_OP/tinygrad_repo TINYGRAD_PATH=$REMOTE_OP/tinygrad_repo \
  $python $REMOTE_OP/$compiler driving_supercombo.onnx $pkl_name --device-input '*' --benchmark-runs 1" \
  || die "compilazione fallita sul comma (file in $BUILD_DIR)"
rsync -a "$COMMA_HOST:$BUILD_DIR/$pkl_name" "$stage/models/$slug/"
ssh "$COMMA_HOST" "rm -rf $BUILD_DIR"
sha256="$(sha256sum "$stage/models/$slug/$pkl_name" | cut -d' ' -f1)"
echo "Compilato: $pkl_name sha256 $sha256"

# 4. Catalog: the one on Hugging Face plus this model.
echo "== 4. catalogo"
catalog_url="https://huggingface.co/$HF_REPO/resolve/main/driving_models.json"
http_code="$(curl -sS -L -w '%{http_code}' -o "$stage/driving_models.json" "$catalog_url" || true)"
if [ "$http_code" = "404" ]; then
  rm -f "$stage/driving_models.json"
  echo "Nessun catalogo su $HF_REPO: ne creo uno nuovo"
elif [ "$http_code" != "200" ]; then
  die "non riesco a scaricare $catalog_url (HTTP $http_code)"
fi
(cd "$ROOT" && uv run python -m helpers.model_catalog "$stage/driving_models.json" \
  --name "$name" --file-name "$pkl_name" \
  --url "https://huggingface.co/$HF_REPO/resolve/main/models/$slug/$pkl_name" \
  --sha256 "$sha256" --selector-version "$selector_version" --tinygrad-ref "$tinygrad_ref")

if [ "$upload" -eq 0 ]; then
  echo "--no-upload: niente caricato. File pronti in $stage"
  exit 0
fi

# 5. One commit with the model files, the new catalog and the PyTorch weights.
echo "== 5. carico su https://huggingface.co/$HF_REPO"
(cd "$ROOT" && uv run hf upload "$HF_REPO" "$stage" . --commit-message "Publish model $name")

echo "Pubblicato. Catalogo: $catalog_url"
if [ -n "$weights" ] && [ "$stable" -eq 1 ]; then
  echo "Pesi PyTorch: training/releases/$slug/supercombo.pt, ora stable: helpers/train_supervised.sh --continue parte da qui."
elif [ -n "$weights" ]; then
  echo "Pesi PyTorch: training/releases/$slug/supercombo.pt (stable non spostato: con --stable diventa la base di --continue)."
fi
