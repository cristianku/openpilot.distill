#!/usr/bin/env bash
# Try a trained model on the comma: export supercombo.pt to ONNX, check that the comma can compile it,
# put it in place of openpilot's driving model (driving_supercombo.onnx) and reboot the comma.
# --restore puts the original back.
#
# Usage: helpers/install_model_on_comma.sh [--host=<ip|nome>] [--no-reboot] [<supercombo.pt>]
#        helpers/install_model_on_comma.sh [--host=<ip|nome>] [--no-reboot] --restore
#   helpers/install_model_on_comma.sh                 (l'ultimo outputs/supervised/*/supercombo.pt)
#   helpers/install_model_on_comma.sh outputs/supervised/2026-10-20_1910/supercombo.pt
#   helpers/install_model_on_comma.sh --restore       (rimette il modello originale)
#
# What it does:
#   1. checks that the comma is reachable, offroad, and (sunnypilot) on the Default model of the
#      Models menu: a model chosen there runs in place of openpilot's one
#   2. keeps a copy of the original model on the comma (/data/driving_supercombo.onnx.orig), taken
#      while the comma still has the original of its openpilot version (same sha256 as in git)
#   3. exports the .pt with helpers/export_onnx.py, with the original as reference; the ONNX is
#      written next to the .pt (driving_supercombo.onnx)
#   4. compiles it on the comma the way the openpilot build does (tinygrad compile_onnx.py, flags
#      from selfdrive/modeld/SConscript), in a temporary folder: if it fails nothing is changed.
#      The last lines show the milliseconds per frame: they must stay well under 50.
#   5. puts the ONNX in place of driving_supercombo.onnx and reboots: at boot the build compiles it
#      again into driving_tinygrad.pkl, so the first start takes a few minutes more.
#
# The software update from the comma screen also brings the original back.
# With dashcamOnly the model does not drive: it only logs its predictions.
#
# --restore: puts back the original saved at step 2 and reboots.
# --no-reboot: does not reboot (the model is used from the next start).
# --host: IP or ssh name of the comma; "comma@" is added when no user is given.
# -h, --help: this help.
# Env: COMMA_HOST (default "comma", the Host in ~/.ssh/config), overridden by --host

set -euo pipefail

COMMA_HOST="${COMMA_HOST:-comma}"
REMOTE_OP="/data/openpilot"
BACKUP="/data/driving_supercombo.onnx.orig"
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

# Prints "key=value" lines read from the comma: onroad state, selected Models entry, where the model
# is (path asked to git, the layout changes between versions), the sha256 of the original model of
# this openpilot version (from git, the onnx is in LFS), of the current one and of the backup.
read_comma_info() {
  ssh "$COMMA_HOST" bash -s -- "$BACKUP" <<'EOF'
backup="$1"
cd /data/openpilot || exit 1
echo "onroad=$(cat /data/params/d/IsOnroad 2>/dev/null)"
echo "active_bundle=$(cat /data/params/d/ModelManager_ActiveBundle 2>/dev/null | head -c 200)"
onnx="$(git ls-files '*selfdrive/modeld/models/driving_supercombo.onnx' | head -1)"
echo "onnx=$onnx"
if [ -z "$onnx" ]; then
  exit 0
fi
sconscript="$(dirname "$(dirname "$onnx")")/SConscript"
echo "tg_flags=$(sed -n "s/^ *tg_flags = '\([^']*\)'.*/\1/p" "$sconscript" | head -1)"
size="$(git cat-file -s "HEAD:$onnx")"
if [ "$size" -lt 1024 ]; then
  echo "original=$(git cat-file -p "HEAD:$onnx" | sed -n 's/^oid sha256://p')"
else
  echo "original=$(git cat-file -p "HEAD:$onnx" | sha256sum | cut -d' ' -f1)"
fi
echo "current=$(sha256sum "$onnx" | cut -d' ' -f1)"
if [ -f "$backup" ]; then
  echo "backup=$(sha256sum "$backup" | cut -d' ' -f1)"
fi
if [ -x /usr/local/venv/bin/python3 ]; then
  echo "python=/usr/local/venv/bin/python3"
fi
EOF
}

# Copies a file on the comma over the model, through a temporary name so the swap is one rename.
put_model() {
  local source="$1"
  ssh "$COMMA_HOST" "cp $source $REMOTE_OP/$onnx.new && mv $REMOTE_OP/$onnx.new $REMOTE_OP/$onnx"
}

reboot_comma() {
  if [ "$do_reboot" -eq 0 ]; then
    echo "--no-reboot: il comma usa il modello dal prossimo avvio"
    return
  fi
  echo "Riavvio il comma (il primo avvio ricompila il modello, qualche minuto in piu')"
  ssh "$COMMA_HOST" "sudo reboot" || true
}

restore=0
do_reboot=1
input=""

while [ $# -gt 0 ]; do
  if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
  elif [ "$1" = "--restore" ]; then
    restore=1
  elif [ "$1" = "--no-reboot" ]; then
    do_reboot=0
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

ssh -o ConnectTimeout=8 "$COMMA_HOST" true \
  || die "comma '$COMMA_HOST' non raggiungibile via ssh (acceso? hai fatto ssh -A?)"

onroad=""
active_bundle=""
onnx=""
tg_flags=""
original=""
current=""
backup=""
python=""
info="$(read_comma_info)" || die "non riesco a leggere /data/openpilot sul comma"
while IFS='=' read -r key value; do
  if [ "$key" = "onroad" ]; then
    onroad="$value"
  elif [ "$key" = "active_bundle" ]; then
    active_bundle="$value"
  elif [ "$key" = "onnx" ]; then
    onnx="$value"
  elif [ "$key" = "tg_flags" ]; then
    tg_flags="$value"
  elif [ "$key" = "original" ]; then
    original="$value"
  elif [ "$key" = "current" ]; then
    current="$value"
  elif [ "$key" = "backup" ]; then
    backup="$value"
  elif [ "$key" = "python" ]; then
    python="$value"
  fi
done <<< "$info"

[ "$onroad" != "1" ] || die "il comma e' onroad: spegni l'auto e riprova"
[ -n "$onnx" ] || die "sul comma non trovo driving_supercombo.onnx"
[ -n "$original" ] || die "non riesco a leggere da git lo sha256 del modello originale"

if [ "$restore" -eq 1 ]; then
  [ -z "$input" ] || die "--restore non vuole un file"
  if [ "$current" = "$original" ]; then
    echo "Il comma ha gia' il modello originale."
    exit 0
  fi
  [ "$backup" = "$original" ] \
    || die "il backup $BACKUP manca o non e' l'originale di questa versione: rimetti l'originale con un aggiornamento dallo schermo"
  put_model "$BACKUP"
  echo "Rimesso il modello originale."
  reboot_comma
  exit 0
fi

if [ -n "$active_bundle" ] && [ "$active_bundle" != "{}" ]; then
  die "nel menu Models di sunnypilot e' scelto un modello: metti Default, se no il tuo non viene usato"
fi
[ -n "$tg_flags" ] || die "non trovo i flag di tinygrad (tg_flags) nello SConscript del comma"
[ -n "$python" ] || die "sul comma non c'e' /usr/local/venv/bin/python3"

if [ -z "$input" ]; then
  input="$(ls -t "$ROOT"/outputs/supervised/*/supercombo.pt 2>/dev/null | head -1 || true)"
  [ -n "$input" ] || die "nessun outputs/supervised/*/supercombo.pt: passa il .pt"
fi
[ -f "$input" ] || die "$input non esiste"
input="$(cd "$(dirname "$input")" && pwd)/$(basename "$input")"
run_dir="$(dirname "$input")"
student="$run_dir/driving_supercombo.onnx"
echo "Modello: $input"

# 2. Backup of the original, refreshed while the comma has it (it changes with openpilot updates).
if [ "$current" = "$original" ]; then
  ssh "$COMMA_HOST" "cp $REMOTE_OP/$onnx $BACKUP"
  echo "== 2. salvato l'originale in $BACKUP"
elif [ "$backup" = "$original" ]; then
  echo "== 2. sul comma c'e' gia' un modello tuo, l'originale e' in $BACKUP"
else
  die "sul comma c'e' un modello diverso dall'originale e non c'e' un backup valido: rimetti l'originale con un aggiornamento dallo schermo"
fi

# 3. Export, with the original as reference for inputs and output layout.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
rsync -a "$COMMA_HOST:$BACKUP" "$tmp/reference.onnx"
echo "== 3. export -> $student"
(cd "$ROOT" && uv run python -m helpers.export_onnx "$input" --reference "$tmp/reference.onnx" --output "$student")

# 4. Compile check on the comma, same command as the openpilot build. compile_onnx writes a
# temporary file in the current folder.
echo "== 4. compilo sul comma per controllo (qualche minuto)"
ssh "$COMMA_HOST" "rm -rf $BUILD_DIR && mkdir -p $BUILD_DIR"
rsync -a "$student" "$COMMA_HOST:$BUILD_DIR/driving_supercombo.onnx"
log="$run_dir/comma_compile.log"
if ! ssh "$COMMA_HOST" "cd $BUILD_DIR && env $tg_flags PYTHONPATH=$REMOTE_OP/tinygrad_repo:$REMOTE_OP \
  $python $REMOTE_OP/tinygrad_repo/examples/openpilot/compile_onnx.py \
  driving_supercombo.onnx driving_tinygrad.pkl --out-of-band --benchmark-runs 1" > "$log" 2>&1; then
  tail -20 "$log"
  ssh "$COMMA_HOST" "rm -rf $BUILD_DIR"
  die "il comma non riesce a compilare il modello: non ho cambiato niente (log completo: $log)"
fi
grep -E " ms$" "$log" | tail -3 || true

# 5. In place of the original, then reboot.
put_model "$BUILD_DIR/driving_supercombo.onnx"
ssh "$COMMA_HOST" "rm -rf $BUILD_DIR"
echo "== 5. installato: per tornare all'originale helpers/install_model_on_comma.sh --restore"
reboot_comma
