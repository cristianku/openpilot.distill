#!/usr/bin/env bash
# The control site of openpilot.distill (distill-web): server and comma state, data, training with
# live charts, evaluation, comparison videos, publish and install, and the jobs started from the
# browser. It starts in the background and prints the link to open from the PC or the phone.
#
# Usage: helpers/web.sh [--local] [--foreground] [--stop]
#   helpers/web.sh                (start it, or print the link again when it is already on)
#   helpers/web.sh --stop         (stop the site; the jobs started from the site keep running)
#   helpers/web.sh --local        (this server only: from the PC you need the ssh tunnel)
#   helpers/web.sh --foreground   (stay attached to the terminal, Ctrl+C to stop)
#
# By default it listens on the LAN (0.0.0.0, port 8766) and every page needs the administrator
# signed in (name and password). The first visit creates the administrator with the setup code,
# printed here until that is done: it is created once in the .env of this clone
# (DISTILL_WEB_TOKEN=..., permissions 600, ignored by git), the same one the service uses. When the
# systemd service distill-web (deploy/install_web_service.sh) is active it does not start a second
# server: it prints the link of the service. Site log: outputs/web/web.log; jobs and their logs: outputs/web/jobs/.
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

# The setup code: the DISTILL_WEB_TOKEN line of the .env of the clone, created once (permissions
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
        echo "# Secrets and local settings of this clone: the file is in .gitignore, it never goes into git." > "$REPO_ENV"
      fi
      echo "# Setup code of the distill-web site (distill-web service and helpers/web.sh)." >> "$REPO_ENV"
      echo "DISTILL_WEB_TOKEN=$value" >> "$REPO_ENV"
    )
    chmod 600 "$REPO_ENV"
    echo "Setup code created and saved in $REPO_ENV" >&2
  fi
  echo "$value"
}

# The setup code is shown only while the administrator does not exist yet.
print_setup_code() {
  if [ -f "$WEB_DIR/auth/admin.json" ]; then
    echo "Sign in as the administrator of the site."
  elif [ -n "$1" ]; then
    echo "First visit: create the administrator (name and password) with the setup code  $1"
    echo "(it is needed only once; it stays in $REPO_ENV)"
  else
    echo "First visit: create the administrator with the setup code printed in $LOG."
  fi
}

print_lan_link() {
  echo "Open from the PC or the phone:  http://$(server_ip):$PORT/"
  print_setup_code "$1"
}

print_local_link() {
  echo "From the PC open the tunnel first (leave that terminal open):"
  echo "  ssh -L $PORT:127.0.0.1:$PORT root@$(server_ip)"
  echo "then in the browser:  http://localhost:$PORT/"
  print_setup_code ""
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
    die "invalid option: $1 (see --help)"
  fi
  shift
done

[ -x "$PY" ] || die "$PY is missing: create the environment with uv sync (docs/INSTALL.md)"
mkdir -p "$WEB_DIR"
chmod 700 "$WEB_DIR"

if [ "$stop" -eq 1 ]; then
  if pid="$(running_pid)"; then
    kill "$pid"
    rm -f "$PID_FILE"
    echo "Site stopped (pid $pid). The jobs started from the site keep running."
  else
    rm -f "$PID_FILE"
    echo "No site started by this script."
    if service_active; then
      echo "The $SERVICE service is running: stop it with  systemctl stop $SERVICE"
    fi
  fi
  exit 0
fi

# Already on: the systemd service, or a site started earlier by this script.
if service_active; then
  echo "The site already runs as the $SERVICE service (port $SERVICE_PORT)."
  PORT="$SERVICE_PORT"
  if [ -r "$SERVICE_ENV" ]; then
    print_lan_link "$(sed -n 's/^DISTILL_WEB_TOKEN=//p' "$SERVICE_ENV" | tail -1)"
  else
    echo "The setup code is in $SERVICE_ENV (readable by root only)."
  fi
  exit 0
fi
if pid="$(running_pid)"; then
  echo "The site is already on (pid $pid, log $LOG)."
  if grep -q "listening on 127.0.0.1" "$LOG" 2>/dev/null; then
    print_local_link
  else
    print_lan_link "$(token)"
  fi
  exit 0
fi
if port_busy; then
  die "port $PORT is already taken by another program (DISTILL_WEB_PORT=<other> to change it)"
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
  echo "Ctrl+C to stop (the jobs started from the site keep running)."
  exec "${command[@]}"
fi

nohup "${command[@]}" < /dev/null > "$LOG" 2>&1 &
echo $! > "$PID_FILE"
# The site stops at once when it cannot start (port taken, bad token): wait for its first line.
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if grep -q "listening on" "$LOG" 2>/dev/null; then
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
  die "the site did not start (full log: $LOG)"
fi

echo "Site started in the background (pid $(cat "$PID_FILE"), log $LOG)."
if [ "$local_only" -eq 1 ]; then print_local_link; else print_lan_link "$DISTILL_WEB_TOKEN"; fi
echo "Stop it with:  helpers/web.sh --stop"
