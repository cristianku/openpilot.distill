#!/usr/bin/env bash
# Export a trained model (supercombo.pt) to the driving_supercombo.onnx that openpilot runs on the comma.
#
# Usage: helpers/export_onnx.sh [--host=<ip|name>] [<supercombo.pt>]
#   helpers/export_onnx.sh                 (the latest outputs/supervised/*/supercombo.pt)
#   helpers/export_onnx.sh outputs/supervised/2026-10-20_1910/supercombo.pt
#
# The ONNX is written next to the .pt: outputs/supervised/<date_time>/driving_supercombo.onnx
# It needs the comma's ORIGINAL driving_supercombo.onnx as reference (inputs and output layout of
# the openpilot version on the comma): it is copied from the comma into
# models/comma_driving_supercombo.onnx and reused from there when the comma is off.
# The original is the file whose sha256 is the one in the comma's openpilot git: in its place, or
# /data/driving_supercombo.onnx.orig while a trained model is installed in its place
# (helpers/install_model_on_comma.sh), like helpers/publish_model.sh does. A trained model on the
# comma never replaces models/comma_driving_supercombo.onnx: the copy is checked before it is kept.
# The comma only accepts the keys registered on GitHub: this machine needs one of them, or connect
# to it with agent forwarding (ssh -A) from the Mac. ssh never asks anything (BatchMode).
#
# --host: IP or ssh name of the comma; "comma@" is added when no user is given.
# -h, --help: this help.
# Env: COMMA_HOST (default "comma", the Host in ~/.ssh/config, also from the repo .env), overridden by --host

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
. "$ROOT/helpers/common.sh"
distill_load_env

COMMA_HOST="${COMMA_HOST:-comma}"
REMOTE_OP="/data/openpilot"
# Original model saved by helpers/install_model_on_comma.sh.
BACKUP="/data/driving_supercombo.onnx.orig"
REFERENCE="$ROOT/models/comma_driving_supercombo.onnx"

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

# Prints "key=value" lines read from the comma: where this openpilot keeps driving_supercombo.onnx
# (path asked to git, the layout changes between versions), the sha256 of the original model of
# this openpilot version (from git, the onnx is in LFS), of the current one and of the backup.
# Same reading as helpers/publish_model.sh.
read_comma_info() {
  ssh "${DISTILL_SSH_OPTIONS[@]}" "$COMMA_HOST" bash -s -- "$BACKUP" <<'EOF'
backup="$1"
cd /data/openpilot || exit 1
reference="$(git ls-files '*selfdrive/modeld/models/driving_supercombo.onnx' | head -1)"
echo "reference=$reference"
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
EOF
}

input=""

while [ $# -gt 0 ]; do
  if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
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

if [ -z "$input" ]; then
  input="$(ls -t "$ROOT"/outputs/supervised/*/supercombo.pt 2>/dev/null | head -1 || true)"
  [ -n "$input" ] || die "no outputs/supervised/*/supercombo.pt: pass the .pt"
fi
[ -f "$input" ] || die "$input does not exist"
input="$(cd "$(dirname "$input")" && pwd)/$(basename "$input")"
output="$(dirname "$input")/driving_supercombo.onnx"

mkdir -p "$(dirname "$REFERENCE")"
if ssh "${DISTILL_SSH_OPTIONS[@]}" "$COMMA_HOST" true 2>/dev/null; then
  reference=""
  original=""
  current=""
  backup=""
  info="$(read_comma_info)" || die "cannot read /data/openpilot on the comma"
  while IFS='=' read -r key value; do
    if [ "$key" = "reference" ]; then
      reference="$value"
    elif [ "$key" = "original" ]; then
      original="$value"
    elif [ "$key" = "current" ]; then
      current="$value"
    elif [ "$key" = "backup" ]; then
      backup="$value"
    fi
  done <<< "$info"

  # The original of this openpilot version: in its place, or the backup while a trained model
  # is installed in its place.
  original_source=""
  if [ -n "$original" ] && [ "$current" = "$original" ]; then
    original_source="$REMOTE_OP/$reference"
  elif [ -n "$original" ] && [ "$backup" = "$original" ]; then
    original_source="$BACKUP"
  fi

  cached=""
  if [ -f "$REFERENCE" ]; then
    cached="$(sha256sum "$REFERENCE" | cut -d' ' -f1)"
  fi
  if [ -n "$original" ] && [ "$cached" = "$original" ]; then
    echo "Reference: $REFERENCE, already identical to the comma's original"
  elif [ -n "$original_source" ]; then
    # Into a temporary file first, kept only when it is really the original.
    rsync -a -e "$DISTILL_RSYNC_SSH" "$COMMA_HOST:$original_source" "$REFERENCE.new"
    copied="$(sha256sum "$REFERENCE.new" | cut -d' ' -f1)"
    if [ "$copied" != "$original" ]; then
      rm -f "$REFERENCE.new"
      die "the copy of $original_source does not have the sha256 of the original ($copied instead of $original): $REFERENCE left untouched"
    fi
    mv "$REFERENCE.new" "$REFERENCE"
    echo "Reference: the comma's original driving_supercombo.onnx ($original_source)"
  elif [ -f "$REFERENCE" ]; then
    echo "WARNING: the original of this version is not on the comma (neither in its place nor in $BACKUP): using the reference already copied ($REFERENCE)"
  else
    die "the original of this version is not on the comma (neither in its place nor in $BACKUP) and there is no $REFERENCE"
  fi
elif [ -f "$REFERENCE" ]; then
  echo "Comma not reachable: using the reference already copied ($REFERENCE)"
else
  die "comma not reachable and no $REFERENCE: turn the comma on (ssh key registered? from the Mac use ssh -A)"
fi

echo "Export: $input"
(cd "$ROOT" && distill_uv_run python -m helpers.export_onnx "$input" --reference "$REFERENCE" --output "$output")
echo "Done: $output"
