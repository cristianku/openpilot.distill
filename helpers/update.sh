#!/usr/bin/env bash
# Update this clone from GitHub and put the new code to work, always the same way:
#   0. refuses while a job uses this clone: a training, an evaluation or a comparison video
#      (GPU lock /run/lock/v100-gpu.lock taken, or a supervised.train process), or another helper
#      that runs Python from .venv (fetch, localize, export, publish, install). uv sync would change
#      the packages under it. The GPU lock stays held until the update ends: no GPU job starts
#      in the middle of it.
#   1. git pull --ff-only of the current branch. Refuses when files of the repo are modified
#      locally (it shows which) or the branch has diverged: nothing is changed then.
#   2. uv sync, with --extra $DISTILL_UV_EXTRA when the repo .env sets it (v100 on the Tesla V100
#      server, see helpers/common.sh). --locked: uv.lock is used as it is and never rewritten
#      (a rewritten uv.lock would stop the next git pull); when it does not match pyproject.toml
#      it stops with an error.
#   3. with the systemd service distill-web installed: restarts it, so the site runs the new code.
#      The jobs started from the site keep running (KillMode=process in the unit).
#      When deploy/ changed, it says to run deploy/install_web_service.sh for the new unit file.
#
# Usage: helpers/update.sh
# -h, --help: this help.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
. "$ROOT/helpers/common.sh"
distill_load_env

SERVICE="distill-web"
# Helpers and Python modules that run from .venv: updating the packages under them would break them.
BUSY_PATTERN='helpers/(fetch_comma_segments|localize_segments|export_onnx|publish_model|install_model_on_comma|compare_video|eval_holdout|train_supervised)[.]sh|localizer[.]run|evaluation[.](compare|holdout)|helpers[.]export_onnx'

die() { echo "ERROR: $*" >&2; exit 1; }

# Prints the comment block at the top of this file.
usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

while [ $# -gt 0 ]; do
  if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
  else
    die "invalid option: $1 (see --help)"
  fi
done

command -v uv > /dev/null 2>&1 || die "uv not found in PATH (on the server it is /root/.local/bin/uv)"

# 0. Jobs that use this clone.
if pgrep -f "supervised[.]train" > /dev/null; then
  die "a training is running (supervised.train): update when it has finished"
fi
busy="$(pgrep -a -f -- "$BUSY_PATTERN" || true)"
if [ -n "$busy" ]; then
  echo "Running jobs:" >&2
  echo "$busy" >&2
  die "a job that uses .venv is running: update when it has finished"
fi
# The same lock as every GPU job (helpers/common.sh), held until this script exits.
if command -v flock > /dev/null 2>&1 && [ -d "$(dirname "$DISTILL_GPU_LOCK_FILE")" ]; then
  exec 9>>"$DISTILL_GPU_LOCK_FILE"
  if ! flock -n 9; then
    holder="$(distill_gpu_lock_holder)"
    if [ -n "$holder" ]; then
      die "The GPU is busy with another job ($holder): update when it has finished"
    fi
    die "The GPU is busy with another job (lock $DISTILL_GPU_LOCK_FILE taken): update when it has finished"
  fi
fi

cd "$ROOT"
git rev-parse --is-inside-work-tree > /dev/null 2>&1 || die "$ROOT is not a git clone: nothing to update"

# 1. git pull.
modified="$(git status --porcelain --untracked-files=no)"
if [ -n "$modified" ]; then
  echo "Repo files modified locally:" >&2
  echo "$modified" >&2
  die "the repo files have local changes (above): sort them out first, they are not overwritten"
fi
before="$(git rev-parse HEAD)"
echo "== 1. git pull ($(git branch --show-current))"
# GIT_TERMINAL_PROMPT=0: a remote that wants a password fails instead of waiting for it.
GIT_TERMINAL_PROMPT=0 git pull --ff-only || die "git pull failed (diverged branch or network?): nothing changed"
after="$(git rev-parse HEAD)"
if [ "$before" = "$after" ]; then
  echo "Already up to date: $(git log --oneline -1)"
else
  git log --oneline "$before..$after"
fi

# 2. Packages of .venv.
if [ -n "${DISTILL_UV_EXTRA:-}" ]; then
  echo "== 2. uv sync --locked --extra $DISTILL_UV_EXTRA"
  uv sync --locked --extra "$DISTILL_UV_EXTRA" \
    || die "uv sync failed: the code is updated but .venv is not (uv.lock out of sync with pyproject.toml?)"
else
  echo "== 2. uv sync --locked"
  uv sync --locked \
    || die "uv sync failed: the code is updated but .venv is not (uv.lock out of sync with pyproject.toml?)"
fi

# 3. The web site.
installed=0
if command -v systemctl > /dev/null 2>&1; then
  if systemctl list-unit-files "$SERVICE.service" 2>/dev/null | grep -q "^$SERVICE.service"; then
    installed=1
  fi
fi
if [ "$installed" -eq 0 ]; then
  echo "== 3. service $SERVICE not installed: nothing to restart"
  echo "Done."
  exit 0
fi

if [ "$before" != "$after" ] && git diff --name-only "$before" "$after" | grep -q '^deploy/'; then
  echo "WARNING: deploy/ changed: run deploy/install_web_service.sh for the new service file"
fi
echo "== 3. restarting the service $SERVICE"
systemctl restart "$SERVICE"
sleep 2
systemctl is-active --quiet "$SERVICE" || die "the service $SERVICE does not start again: journalctl -u $SERVICE -n 50"
echo "Service restarted with the new code."
echo "Done."
