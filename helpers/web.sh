#!/usr/bin/env bash
# The control site of openpilot.distill (distill-web): server and comma state, data, training with
# live charts, evaluation, comparison videos, publish and install, and the jobs started from the
# browser. It starts in the background and prints the link to open from the PC or the phone.
#
# Usage: helpers/web.sh [--local] [--foreground] [--stop]
#   helpers/web.sh                (avvia, o se gia' acceso ristampa il link)
#   helpers/web.sh --stop         (spegne il sito; i job avviati dal sito continuano)
#   helpers/web.sh --local        (solo da questo server: dal PC serve il tunnel ssh)
#   helpers/web.sh --foreground   (resta attaccato al terminale, Ctrl+C per spegnere)
#
# By default it listens on the LAN (0.0.0.0, port 8766) and every request needs the access token:
# the link printed here already has it. The token is created once in the .env of this clone
# (DISTILL_WEB_TOKEN=..., permissions 600, ignored by git), the same one the service uses, so the
# link stays the same after a restart and can be bookmarked. When the systemd service distill-web
# (deploy/install_web_service.sh) is active it does not start a second server: it prints the link
# of the service. Site log: outputs/web/web.log; jobs and their logs: outputs/web/jobs/.
# -h, --help: this help.
# Env: DISTILL_WEB_PORT (default 8766)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
. "$ROOT/helpers/common.sh"
distill_load_env

PY="$ROOT/.venv/bin/python"
PORT="${DISTILL_WEB_PORT:-8766}"
REPO_ENV="$ROOT/.env"
WEB_DIR="$ROOT/outputs/web"
LOG="$WEB_DIR/web.log"
PID_FILE="$WEB_DIR/web.pid"
SERVICE="distill-web"
SERVICE_PORT=8766
SERVICE_ENV="/etc/distill-web.env"

die() { echo "ERROR: $*" >&2; exit 1; }

# Prints the comment block at the top of this file.
usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

server_ip() {
  local address=""
  address="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
  if [ -z "$address" ]; then
    address="$(hostname)"
  fi
  echo "$address"
}

# Pid of the site started by this script, when it is still running.
running_pid() {
  local pid=""
  [ -f "$PID_FILE" ] || return 1
  pid="$(cat "$PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  ps -o args= -p "$pid" 2>/dev/null | grep -q "web.server" || return 1
  echo "$pid"
}

port_busy() {
  (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null
}

service_active() {
  command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet "$SERVICE" 2>/dev/null
}

# The access token: the DISTILL_WEB_TOKEN line of the .env of the clone, created once (permissions
# 600) and appended to the other lines of that file when it already exists.
token() {
  local value=""
  if [ -f "$REPO_ENV" ]; then
    value="$(sed -n 's/^DISTILL_WEB_TOKEN=//p' "$REPO_ENV" | tail -1)"
  fi
  if [ -z "$value" ]; then
    value="$("$PY" -c 'import secrets; print(secrets.token_urlsafe(24))')"
    (
      umask 077
      if [ ! -f "$REPO_ENV" ]; then
        echo "# Segreti e impostazioni locali di questo clone: il file e' in .gitignore, non va mai in git." > "$REPO_ENV"
      fi
      echo "# Token di accesso al sito distill-web (servizio distill-web e helpers/web.sh)." >> "$REPO_ENV"
      echo "DISTILL_WEB_TOKEN=$value" >> "$REPO_ENV"
    )
    chmod 600 "$REPO_ENV"
    echo "Token creato e salvato in $REPO_ENV" >&2
  fi
  echo "$value"
}

print_lan_link() {
  echo "Apri dal PC o dal telefono:  http://$(server_ip):$PORT/?token=$1"
  echo "(il link resta lo stesso anche dopo un riavvio; il token e' in $REPO_ENV)"
}

print_local_link() {
  echo "Dal PC apri prima il tunnel (lascia aperto quel terminale):"
  echo "  ssh -L $PORT:127.0.0.1:$PORT root@$(server_ip)"
  echo "poi nel browser:  http://localhost:$PORT/"
}

local_only=0
foreground=0
stop=0

while [ $# -gt 0 ]; do
  if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
  elif [ "$1" = "--local" ]; then
    local_only=1
  elif [ "$1" = "--foreground" ]; then
    foreground=1
  elif [ "$1" = "--stop" ]; then
    stop=1
  else
    die "opzione non valida: $1 (vedi --help)"
  fi
  shift
done

[ -x "$PY" ] || die "manca $PY: crea l'ambiente con uv sync (docs/INSTALL.md)"
mkdir -p "$WEB_DIR"
chmod 700 "$WEB_DIR"

if [ "$stop" -eq 1 ]; then
  if pid="$(running_pid)"; then
    kill "$pid"
    rm -f "$PID_FILE"
    echo "Sito spento (pid $pid). I job avviati dal sito continuano."
  else
    rm -f "$PID_FILE"
    echo "Nessun sito avviato da questo script."
    if service_active; then
      echo "C'e' il servizio $SERVICE: si spegne con  systemctl stop $SERVICE"
    fi
  fi
  exit 0
fi

# Already on: the systemd service, or a site started earlier by this script.
if service_active; then
  echo "Il sito gira gia' come servizio $SERVICE (porta $SERVICE_PORT)."
  PORT="$SERVICE_PORT"
  if [ -r "$SERVICE_ENV" ]; then
    print_lan_link "$(sed -n 's/^DISTILL_WEB_TOKEN=//p' "$SERVICE_ENV" | tail -1)"
  else
    echo "Il token e' in $SERVICE_ENV (leggibile solo da root)."
  fi
  exit 0
fi
if pid="$(running_pid)"; then
  echo "Il sito e' gia' acceso (pid $pid, log $LOG)."
  if grep -q "in ascolto su 127.0.0.1" "$LOG" 2>/dev/null; then
    print_local_link
  else
    print_lan_link "$(token)"
  fi
  exit 0
fi
if port_busy; then
  die "la porta $PORT e' gia' occupata da un altro programma (DISTILL_WEB_PORT=<altra> per cambiarla)"
fi

if [ "$local_only" -eq 1 ]; then
  host="127.0.0.1"
  unset DISTILL_WEB_TOKEN
else
  host="0.0.0.0"
  DISTILL_WEB_TOKEN="$(token)"
  export DISTILL_WEB_TOKEN
fi
# python -m web.server needs the repo root as working directory (the web package is there).
cd "$ROOT"
command=("$PY" -m web.server --host "$host" --port "$PORT")

if [ "$foreground" -eq 1 ]; then
  if [ "$local_only" -eq 1 ]; then print_local_link; else print_lan_link "$DISTILL_WEB_TOKEN"; fi
  echo "Ctrl+C per spegnere (i job avviati dal sito continuano)."
  exec "${command[@]}"
fi

nohup "${command[@]}" < /dev/null > "$LOG" 2>&1 &
echo $! > "$PID_FILE"
# The site stops at once when it cannot start (port taken, bad token): wait for its first line.
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if grep -q "in ascolto" "$LOG" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    break
  fi
  sleep 0.5
done
if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  rm -f "$PID_FILE"
  tail -5 "$LOG" >&2
  die "il sito non e' partito (log completo: $LOG)"
fi

echo "Sito acceso in background (pid $(cat "$PID_FILE"), log $LOG)."
if [ "$local_only" -eq 1 ]; then print_local_link; else print_lan_link "$DISTILL_WEB_TOKEN"; fi
echo "Spegni con:  helpers/web.sh --stop"
