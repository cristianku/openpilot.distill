#!/usr/bin/env bash
# Install (or remove) the control site of openpilot.distill (distill-web, python -m web.server) as
# the systemd service "distill-web" on the LXC. Safe to run again: the token file is created only
# when it is missing, the unit is rewritten only when it changed.
#
#   sudo deploy/install_web_service.sh             install / update and start
#   deploy/install_web_service.sh --dry-run        show what would be done, change nothing
#   sudo deploy/install_web_service.sh --uninstall stop and remove the service (token file kept)
#
# The token is the setup code of the site: the first visit asks for it to create the administrator
# (name and password), after that it opens nothing. It is printed at the end of an install while
# the administrator does not exist yet. It is also kept in the .env of this clone (DISTILL_WEB_TOKEN=..., permissions 600, ignored by
# git), the one place to look it up, shared with helpers/web.sh: the first install takes the token
# already there, otherwise it creates one and writes it in both files.
# DISTILL_WEB_ENV_FILE, DISTILL_WEB_UNIT_FILE and DISTILL_WEB_REPO_ENV override the three paths
# (for tests).
set -euo pipefail

SERVICE_NAME="distill-web"
PORT=8766
ENV_FILE="${DISTILL_WEB_ENV_FILE:-/etc/distill-web.env}"
UNIT_FILE="${DISTILL_WEB_UNIT_FILE:-/etc/systemd/system/${SERVICE_NAME}.service}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
REPO_ENV="${DISTILL_WEB_REPO_ENV:-$REPO/.env}"
TEMPLATE="$REPO/deploy/distill-web.service"
PYTHON="$REPO/.venv/bin/python"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--dry-run] [--uninstall] [--help]

Installs the control site of openpilot.distill (distill-web) as the systemd
service "$SERVICE_NAME", listening on 0.0.0.0:$PORT, started at every boot.

  (no option)        install or update and start the service (needs root)
  --dry-run          show what it would do, without changing anything
  --uninstall        stop and remove the service; the token file is kept
  --help             this help

Files used:
  token     $ENV_FILE (created only when missing, permissions 600)
            and a copy in $REPO_ENV (DISTILL_WEB_TOKEN=..., permissions 600, outside git)
  service   $UNIT_FILE
  clone     $REPO
USAGE
}

say() { printf '%s\n' "$*"; }
fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }

MODE="install"
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --uninstall) MODE="uninstall" ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n\n' "$arg" >&2; usage >&2; exit 2 ;;
  esac
done

# The paths go into the unit through sed and into ExecStart unquoted: plain characters only.
check_plain_path() {
  case "$1" in
    *[!A-Za-z0-9._/+-]*) fail "path with unsupported characters (only letters, digits, . _ / + -): $1" ;;
  esac
}

need_writable_dir() {
  if [ ! -w "$(dirname "$1")" ]; then
    fail "cannot write in $(dirname "$1"): run it again with sudo."
  fi
}

need_systemctl() {
  if ! command -v systemctl >/dev/null 2>&1; then
    fail "systemctl not found: this script is meant for a machine with systemd (the LXC)."
  fi
}

render_unit() {
  sed -e "s|__DISTILL_REPO__|$REPO|g" -e "s|__DISTILL_ENV_FILE__|$ENV_FILE|g" "$TEMPLATE"
}

new_token() {
  "$PYTHON" - <<'PY'
import secrets
import string

alphabet = string.ascii_letters + string.digits
token = ''
for _ in range(32):
    token = token + secrets.choice(alphabet)
print(token)
PY
}

read_token() {
  local file="${1:-$ENV_FILE}"
  local line
  [ -f "$file" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      DISTILL_WEB_TOKEN=*) printf '%s\n' "${line#DISTILL_WEB_TOKEN=}"; return 0 ;;
    esac
  done < "$file"
  return 0
}

# Writes the token in the .env of the clone, replacing an older DISTILL_WEB_TOKEN line and keeping
# every other line (DISTILL_UV_EXTRA, HF_REPO, ...).
save_repo_token() {
  local token="$1"
  local temporary="$REPO_ENV.tmp"
  (
    umask 077
    if [ -f "$REPO_ENV" ]; then
      grep -v '^DISTILL_WEB_TOKEN=' "$REPO_ENV" > "$temporary" || true
    else
      say '# Secrets and local settings of this clone: the file is in .gitignore, it never goes into git.' > "$temporary"
      say '# Access token of the distill-web site (distill-web service and helpers/web.sh).' >> "$temporary"
    fi
    say "DISTILL_WEB_TOKEN=$token" >> "$temporary"
    mv "$temporary" "$REPO_ENV"
  )
  chmod 600 "$REPO_ENV"
}

valid_token() {
  case "$1" in
    ''|*[!A-Za-z0-9_-]*) return 1 ;;
  esac
  [ "${#1}" -ge 16 ]
}

lan_ips() {
  local found=""
  found="$(hostname -I 2>/dev/null || true)"
  if [ -z "${found// /}" ] && command -v ip >/dev/null 2>&1; then
    found="$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1 || true)"
  fi
  if [ -z "${found// /}" ]; then
    found="$(hostname)"
  fi
  local address
  for address in $found; do
    case "$address" in
      *:*) ;;          # IPv6: the server listens on IPv4 0.0.0.0
      *) say "$address" ;;
    esac
  done
}

write_env_file() {
  local token="$1"
  (
    umask 077
    {
      say '# Access token of the distill-web site (distill-web service). Keep it secret.'
      say '# To change it: write a new value (at least 16 characters among letters, digits, - and _),'
      say '# the same in the DISTILL_WEB_TOKEN= line of the .env of the clone, and restart: systemctl restart distill-web'
      say "DISTILL_WEB_TOKEN=$token"
    } > "$ENV_FILE"
  )
  chmod 600 "$ENV_FILE"
}

uninstall_service() {
  if [ "$DRY_RUN" -eq 1 ]; then
    say "[dry run] would stop and disable $SERVICE_NAME, delete $UNIT_FILE and reload systemd."
    say "[dry run] the jobs started from the site would keep running (KillMode=process)."
    say "[dry run] the token file $ENV_FILE would be kept."
    return 0
  fi
  need_systemctl
  if [ -e "$UNIT_FILE" ]; then
    need_writable_dir "$UNIT_FILE"
  fi
  systemctl disable --now "$SERVICE_NAME.service" >/dev/null 2>&1 || true
  if [ -e "$UNIT_FILE" ]; then
    rm -f "$UNIT_FILE"
    say "Service removed: $UNIT_FILE"
  else
    say "Service not installed: nothing to remove."
  fi
  systemctl daemon-reload
  if [ -e "$ENV_FILE" ]; then
    say "The token file is kept: $ENV_FILE (delete it by hand if it is no longer needed)."
  fi
}

install_service() {
  [ -f "$TEMPLATE" ] || fail "the service template is missing: $TEMPLATE"
  [ -x "$PYTHON" ] || fail "$PYTHON is missing: create the environment first: sudo ./install.sh"
  [ -f "$REPO/web/server.py" ] || fail "$REPO/web/server.py is missing: this clone does not have the site."
  check_plain_path "$REPO"
  check_plain_path "$ENV_FILE"

  if [ "$DRY_RUN" -eq 1 ]; then
    if [ -e "$ENV_FILE" ]; then
      say "[dry run] the token file $ENV_FILE already exists: would leave it as it is (permissions 600)."
    elif valid_token "$(read_token "$REPO_ENV")"; then
      say "[dry run] would create $ENV_FILE with the token already in $REPO_ENV (permissions 600)."
    else
      say "[dry run] would create $ENV_FILE with a random 32-character token (permissions 600)."
    fi
    say "[dry run] would keep the same token in $REPO_ENV too (DISTILL_WEB_TOKEN= line, permissions 600, outside git)."
    say "[dry run] would write $UNIT_FILE with this content:"
    render_unit
    say "[dry run] then: systemctl daemon-reload; systemctl enable --now $SERVICE_NAME.service"
    say "[dry run] at the end would print the addresses to open (http://<IP>:$PORT/) and, before the first visit, the setup code."
    return 0
  fi

  need_systemctl
  need_writable_dir "$ENV_FILE"
  need_writable_dir "$UNIT_FILE"

  if [ -e "$ENV_FILE" ]; then
    chmod 600 "$ENV_FILE"
    say "Token file already present: $ENV_FILE (left as it is)."
  else
    # The first install keeps the token already in the .env of the clone (helpers/web.sh).
    local first
    first="$(read_token "$REPO_ENV")"
    if ! valid_token "$first"; then
      first="$(new_token)"
    fi
    write_env_file "$first"
    say "Token file created: $ENV_FILE (permissions 600)."
  fi
  local token
  token="$(read_token)"
  if ! valid_token "$token"; then
    fail "$ENV_FILE has no valid DISTILL_WEB_TOKEN (at least 16 characters among letters, digits, - and _): fix it, or delete the file and run again."
  fi
  if [ "$(read_token "$REPO_ENV")" != "$token" ]; then
    save_repo_token "$token"
    say "Token also saved in $REPO_ENV (permissions 600, outside git)."
  fi

  local was_active=0
  if systemctl is-active --quiet "$SERVICE_NAME.service"; then
    was_active=1
  fi
  local rendered
  rendered="$(mktemp)"
  render_unit > "$rendered"
  local changed=0
  if [ -f "$UNIT_FILE" ] && cmp -s "$rendered" "$UNIT_FILE"; then
    say "Service already up to date: $UNIT_FILE"
  else
    install -m 644 "$rendered" "$UNIT_FILE"
    changed=1
    say "Service written: $UNIT_FILE"
  fi
  rm -f "$rendered"

  systemctl daemon-reload
  systemctl enable --now "$SERVICE_NAME.service"
  if [ "$changed" -eq 1 ] && [ "$was_active" -eq 1 ]; then
    systemctl restart "$SERVICE_NAME.service"
    say "Service restarted with the new configuration (the running jobs keep going)."
  fi
  if systemctl is-active --quiet "$SERVICE_NAME.service"; then
    say "Service $SERVICE_NAME active, it starts by itself at every boot."
  else
    say "Warning: the service does not show as active. Look at: journalctl -u $SERVICE_NAME -n 50"
  fi

  say ""
  say "Open in the browser:"
  local address
  while IFS= read -r address; do
    say "  http://$address:$PORT/"
  done < <(lan_ips)
  if [ -f "$REPO/outputs/web/auth/admin.json" ]; then
    say "Sign in as the administrator of the site."
  else
    say "First visit: create the administrator (name and password) with this setup code: $token"
    say "(it is needed only once; it stays in $REPO_ENV)"
  fi
}

if [ "$MODE" = "uninstall" ]; then
  uninstall_service
else
  install_service
fi
