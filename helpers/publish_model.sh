#!/usr/bin/env bash
# Publish a trained model on Hugging Face (cristianku/openpilot-distill-models), so that sunnypilot
# can offer it in the Models menu: export to ONNX, compile it on the comma for sunnypilot's
# modeld_v2, upload ONNX + compiled model + the catalog driving_models.json in one commit.
#
# Usage: helpers/publish_model.sh [--host=<ip|nome>] [--name=<nome>] [--no-upload] [<supercombo.pt|file.onnx>]
#   helpers/publish_model.sh                       (l'ultimo outputs/supervised/*/supercombo.pt)
#   helpers/publish_model.sh outputs/supervised/2026-10-20_1855/supercombo.pt --name=prova
#   helpers/publish_model.sh outputs/supervised/2026-10-20_1855/driving_supercombo.onnx   (solo ricompila)
#   helpers/publish_model.sh --no-upload           (prepara tutto, non carica niente)
#
# Steps:
#   1. export: supercombo.pt -> driving_supercombo.onnx next to it, with helpers/export_onnx.py and
#      the comma's own driving_supercombo.onnx as reference for inputs and output layout.
#      Skipped when an .onnx is given.
#   2. compile on the comma (offroad) with sunnypilot's compile_modeld_v2.py: modeld_v2 loads only
#      that format, not the pickle of compile_onnx.py. It uses the comma's tinygrad: after a
#      sunnypilot update that changes tinygrad, run again with the .onnx.
#   3. catalog: downloads driving_models.json from Hugging Face and adds/replaces this model
#      (helpers/model_catalog.py) with the sha256 of the compiled file.
#   4. one Hugging Face commit with models/<nome>/driving_supercombo.onnx, models/<nome>/<file.pkl>
#      and driving_models.json. Everything is also left in outputs/publish/<nome>/.
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
# files used here (paths asked to git, the layout changes between versions).
read_comma_info() {
  ssh "$COMMA_HOST" bash -s <<'EOF'
cd /data/openpilot || exit 1
echo "onroad=$(cat /data/params/d/IsOnroad 2>/dev/null)"
echo "compiler=$(git ls-files '*sunnypilot/modeld_v2/compile_modeld_v2.py' | head -1)"
echo "reference=$(git ls-files '*selfdrive/modeld/models/driving_supercombo.onnx' | head -1)"
helpers="$(git ls-files '*sunnypilot/models/helpers.py' | head -1)"
if [ -n "$helpers" ]; then
  echo "selector_version=$(sed -n 's/^REQUIRED_JSON_VERSION *= *\([0-9]*\).*/\1/p' "$helpers")"
fi
echo "tinygrad_ref=$(git -C tinygrad_repo rev-parse HEAD 2>/dev/null)"
if [ -x /usr/local/venv/bin/python3 ]; then
  echo "python=/usr/local/venv/bin/python3"
fi
EOF
}

name=""
upload=1
input=""

while [ $# -gt 0 ]; do
  if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
  elif [ "$1" = "--no-upload" ]; then
    upload=0
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

if [ -z "$input" ]; then
  input="$(ls -t "$ROOT"/outputs/supervised/*/supercombo.pt 2>/dev/null | head -1 || true)"
  [ -n "$input" ] || die "nessun outputs/supervised/*/supercombo.pt: passa il .pt o l'.onnx"
fi
[ -f "$input" ] || die "$input non esiste"
input="$(cd "$(dirname "$input")" && pwd)/$(basename "$input")"
run_dir="$(dirname "$input")"

if [ -z "$name" ]; then
  name="$(basename "$run_dir")"
fi
[[ "$name" =~ ^[A-Za-z0-9_.-]+$ ]] || die "nome non valido: $name (solo lettere, numeri, _ . -)"
# File and folder names: lowercase, anything else becomes "_" (2026-10-20_1910 -> 2026_10_20_1910).
slug="$(echo "$name" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9\n' '_')"
pkl_name="driving_distill_${slug}_tinygrad.pkl"
stage="$ROOT/outputs/publish/$slug"

ssh -o ConnectTimeout=8 "$COMMA_HOST" true \
  || die "comma '$COMMA_HOST' non raggiungibile via ssh (hai fatto ssh -A?)"

onroad=""
compiler=""
reference=""
selector_version=""
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

# 2. Compile on the comma, flags as sunnypilot CI (TG_FLAGS_QCOM in .github/workflows/sunnypilot-build-model.yaml).
# No taskset: this comma has only CPUs 0-3 online. compile_modeld_v2 writes a temporary file in the cwd.
echo "== 2. compilo sul comma (qualche minuto)"
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

# 3. Catalog: the one on Hugging Face plus this model.
echo "== 3. catalogo"
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

# 4. One commit with the model files and the new catalog.
echo "== 4. carico su https://huggingface.co/$HF_REPO"
(cd "$ROOT" && uv run hf upload "$HF_REPO" "$stage" . --commit-message "Publish model $name")

echo "Pubblicato. Catalogo: $catalog_url"
