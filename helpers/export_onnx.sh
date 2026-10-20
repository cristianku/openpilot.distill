#!/usr/bin/env bash
# Export a trained model (supercombo.pt) to the driving_supercombo.onnx that openpilot runs on the comma.
#
# Usage: helpers/export_onnx.sh [--host=<ip|nome>] [<supercombo.pt>]
#   helpers/export_onnx.sh                 (l'ultimo outputs/supervised/*/supercombo.pt)
#   helpers/export_onnx.sh outputs/supervised/2026-10-20_1910/supercombo.pt
#
# The ONNX is written next to the .pt: outputs/supervised/<data_ora>/driving_supercombo.onnx
# It needs the comma's own driving_supercombo.onnx as reference (inputs and output layout of the
# openpilot version on the comma): it is copied from the comma into models/comma_driving_supercombo.onnx
# and reused from there when the comma is off.
# The comma only accepts the keys registered on GitHub: connect to this machine with agent
# forwarding (ssh -A) so the Mac's key is used.
#
# --host: IP or ssh name of the comma; "comma@" is added when no user is given.
# -h, --help: this help.
# Env: COMMA_HOST (default "comma", the Host in ~/.ssh/config), overridden by --host

set -euo pipefail

COMMA_HOST="${COMMA_HOST:-comma}"
REMOTE_ONNX="/data/openpilot/openpilot/selfdrive/modeld/models/driving_supercombo.onnx"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
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

input=""

while [ $# -gt 0 ]; do
  if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
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
  [ -n "$input" ] || die "nessun outputs/supervised/*/supercombo.pt: passa il .pt"
fi
[ -f "$input" ] || die "$input non esiste"
input="$(cd "$(dirname "$input")" && pwd)/$(basename "$input")"
output="$(dirname "$input")/driving_supercombo.onnx"

mkdir -p "$(dirname "$REFERENCE")"
if ssh -o ConnectTimeout=8 "$COMMA_HOST" true 2>/dev/null; then
  rsync -a "$COMMA_HOST:$REMOTE_ONNX" "$REFERENCE"
  echo "Riferimento: driving_supercombo.onnx del comma"
elif [ -f "$REFERENCE" ]; then
  echo "Comma non raggiungibile: uso il riferimento gia' copiato ($REFERENCE)"
else
  die "comma non raggiungibile e nessun $REFERENCE: accendi il comma (hai fatto ssh -A?)"
fi

echo "Export: $input"
(cd "$ROOT" && uv run python -m helpers.export_onnx "$input" --reference "$REFERENCE" --output "$output")
echo "Fatto: $output"
