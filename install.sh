#!/usr/bin/env bash
# All-in-one install of openpilot.distill on a Linux server with an NVIDIA GPU (Debian/Ubuntu,
# systemd), from a fresh clone to the control site running. Safe to run again: every step checks
# what is already there and only does what is missing.
#
#   1. system packages: git, curl, ca-certificates, openssh-client, rsync, util-linux (flock),
#      iproute2, procps, unzip (apt-get, only the missing ones)
#   2. uv (the Python package manager, https://docs.astral.sh/uv/) in ~/.local/bin, when missing
#   3. the GPU: a Tesla V100 (Volta, compute capability 7.0) needs the CUDA 12 builds, so
#      DISTILL_UV_EXTRA=v100 is written in the .env of this clone (permissions 600, outside git);
#      a newer GPU uses the default stack; no GPU is only a warning (training needs one)
#   4. the Python environment: the submodules when the clone was made without them
#      (perception_repo), then uv sync --locked (with the extra of step 3); uv also downloads
#      the right Python. Refused while a job of this clone runs (it would change packages under it)
#   5. the Android app (helpers/build_android.sh): its build tools in /opt/android, the signing key of
#      this server in ~/.config/distill-android (made once, never in git), then the APK the site
#      offers in Settings › Distill for Android. A failure (no network, not x86_64) is only a
#      warning: the site does not need the app
#   6. the control site as the systemd service distill-web (deploy/install_web_service.sh):
#      started now and at every boot on port 8766; it prints the address and, until the
#      administrator exists, the setup code to create it on the first visit
#
# Usage: sudo ./install.sh [--dry-run] [--no-android] [--no-service] [--help]
#   --dry-run      show what would be done, change nothing (no root needed)
#   --no-android   skip step 5 (no Android app on this server)
#   --no-service   stop after step 5 (start the site by hand with helpers/web.sh)
# Then open the printed address from the PC or the phone: everything else (Hugging Face, W&B,
# the comma address) is set from the Settings page of the site.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ENV="$ROOT/.env"
PACKAGES=(git curl ca-certificates openssh-client rsync util-linux iproute2 procps unzip)
# Jobs that run Python from .venv: uv sync must not change the packages under them.
BUSY_PATTERN='supervised[.]train|helpers/(fetch_comma_segments|localize_segments|export_onnx|publish_model|install_model_on_comma|compare_video|eval_holdout|train_supervised)[.]sh|localizer[.]run|evaluation[.](compare|holdout)|helpers[.]export_onnx'

DRY_RUN=0
SERVICE=1
ANDROID=1

say() { printf '%s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --no-service) SERVICE=0 ;;
    --no-android) ANDROID=0 ;;
    -h|--help) usage ;;
    *) die "invalid option: $1 (see --help)" ;;
  esac
  shift
done

# Runs a command, or only prints it with --dry-run.
run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    say "[dry run] $*"
  else
    "$@"
  fi
}

# Sets KEY=value in the .env of the clone, keeping every other line; permissions 600.
set_env_key() {
  local key="$1" value="$2" temporary="$REPO_ENV.tmp"
  if [ "$DRY_RUN" -eq 1 ]; then
    say "[dry run] would write $key=$value in $REPO_ENV (permissions 600)"
    return 0
  fi
  (
    umask 077
    if [ -f "$REPO_ENV" ]; then
      grep -v "^$key=" "$REPO_ENV" > "$temporary" || true
    else
      say '# Secrets and local settings of this clone: the file is in .gitignore, it never goes into git.' > "$temporary"
    fi
    say "$key=$value" >> "$temporary"
    mv "$temporary" "$REPO_ENV"
  )
  chmod 600 "$REPO_ENV"
}

env_value() {
  [ -f "$REPO_ENV" ] || return 0
  sed -n "s/^$1=//p" "$REPO_ENV" | tail -1
}

# -- 0. checks --------------------------------------------------------------------------------------
[ "$(uname -s)" = "Linux" ] || die "this installer is for a Linux server (on a Mac follow README.md by hand)"
if [ "$DRY_RUN" -eq 0 ] && [ "$(id -u)" -ne 0 ]; then
  die "run it as root: sudo $0 (or --dry-run to only see what it would do)"
fi
[ -f "$ROOT/pyproject.toml" ] && [ -f "$ROOT/web/server.py" ] || die "$ROOT is not a clone of openpilot.distill"
say "openpilot.distill: all-in-one install in $ROOT"

# -- 1. system packages -----------------------------------------------------------------------------
step "1. system packages"
missing=()
for package in "${PACKAGES[@]}"; do
  if command -v dpkg-query >/dev/null 2>&1; then
    if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q 'install ok installed'; then
      missing+=("$package")
    fi
  fi
done
if ! command -v apt-get >/dev/null 2>&1; then
  for command_name in git curl ssh rsync flock; do
    command -v "$command_name" >/dev/null 2>&1 || die "no apt-get on this machine and $command_name is missing: install it first"
  done
  say "no apt-get: the needed commands are there"
elif [ "${#missing[@]}" -eq 0 ]; then
  say "all there"
else
  say "installing: ${missing[*]}"
  run apt-get update -q
  run env DEBIAN_FRONTEND=noninteractive apt-get install -y -q "${missing[@]}"
fi

# -- 2. uv ------------------------------------------------------------------------------------------
step "2. uv"
export PATH="$HOME/.local/bin:$PATH"
if command -v uv >/dev/null 2>&1; then
  say "already there: $(uv --version) ($(command -v uv))"
else
  say "installing uv in $HOME/.local/bin (official installer from astral.sh)"
  if [ "$DRY_RUN" -eq 1 ]; then
    say "[dry run] curl -LsSf https://astral.sh/uv/install.sh | sh"
  else
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
    command -v uv >/dev/null 2>&1 || die "uv did not install in $HOME/.local/bin"
    say "installed: $(uv --version)"
  fi
fi

# -- 3. GPU -----------------------------------------------------------------------------------------
step "3. GPU"
extra="$(env_value DISTILL_UV_EXTRA)"
if command -v nvidia-smi >/dev/null 2>&1 && gpu="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1)" \
    && [ -n "$gpu" ]; then
  say "found: $gpu"
  capability="$(printf '%s' "$gpu" | awk -F', *' '{print $NF}')"
  if [ "$capability" = "7.0" ]; then
    if [ "$extra" != "v100" ]; then
      say "Volta GPU: the CUDA 12 builds are needed (DISTILL_UV_EXTRA=v100)"
      set_env_key DISTILL_UV_EXTRA v100
    fi
    extra="v100"
  elif [ -n "$extra" ]; then
    say "keeping DISTILL_UV_EXTRA=$extra from $REPO_ENV"
  fi
else
  say "WARNING: no NVIDIA GPU found (nvidia-smi). The site and the downloads work; training, localization"
  say "and evaluation need an NVIDIA GPU with NVDEC and its driver."
fi
if [ -n "$extra" ]; then
  say "Python packages: default stack + extra $extra"
else
  say "Python packages: default stack"
fi

# -- 4. Python environment --------------------------------------------------------------------------
step "4. Python environment (uv sync)"
busy="$(pgrep -a -f -- "$BUSY_PATTERN" || true)"
if [ -n "$busy" ]; then
  say "$busy"
  die "a job of this clone is running (above): run the installer again when it has finished"
fi
if [ ! -e "$ROOT/perception_repo/.git" ]; then
  say "downloading the submodules (perception_repo)"
  # As the owner of the clone, so a later update from the site can change them.
  owner="$(stat -c %U "$ROOT")"
  as_owner=()
  if [ "$(id -u)" -eq 0 ] && [ "$owner" != root ]; then
    as_owner=(runuser -u "$owner" --)
  fi
  run "${as_owner[@]}" git -C "$ROOT" submodule update --init --recursive || die "git submodule update failed (above)"
fi
sync_command=(uv sync --locked)
if [ -n "$extra" ]; then
  sync_command+=(--extra "$extra")
fi
if [ "$DRY_RUN" -eq 1 ]; then
  say "[dry run] cd $ROOT && ${sync_command[*]}"
else
  (cd "$ROOT" && "${sync_command[@]}") || die "uv sync failed (above)"
  (cd "$ROOT" && "$ROOT/.venv/bin/python" -c 'import web.server') || die "the environment does not load the site"
  say "environment ready: $ROOT/.venv"
fi

# -- 5. Android app ---------------------------------------------------------------------------------
step "5. Android app (tools, signing key, APK)"
if [ "$ANDROID" -eq 0 ]; then
  say "skipped (--no-android)"
elif [ "$DRY_RUN" -eq 1 ]; then
  say "[dry run] $ROOT/helpers/build_android.sh"
elif "$ROOT/helpers/build_android.sh"; then
  say "Android app ready: Settings › Distill for Android on the site"
else
  say "WARNING: the Android app was not built (above). The site works without it; run the installer again,"
  say "or helpers/build_android.sh, to retry."
fi

# -- 6. site ----------------------------------------------------------------------------------------
step "6. control site"
if [ "$SERVICE" -eq 0 ]; then
  say "skipped (--no-service): start it with  helpers/web.sh"
  exit 0
fi
if ! command -v systemctl >/dev/null 2>&1 || [ ! -d /run/systemd/system ]; then
  say "no systemd on this machine: start the site with  helpers/web.sh  (it prints the address)"
  exit 0
fi
if [ "$DRY_RUN" -eq 1 ]; then
  "$ROOT/deploy/install_web_service.sh" --dry-run
else
  was_active=0
  if systemctl is-active --quiet distill-web.service; then
    was_active=1
  fi
  "$ROOT/deploy/install_web_service.sh"
  # Already running before: restart it so it runs the code and packages just installed (the jobs
  # started from the site keep running, KillMode=process).
  if [ "$was_active" -eq 1 ]; then
    systemctl restart distill-web.service
    say "Site restarted with the installed code."
  fi
fi

say ""
say "Done. Open the address above from the PC or the phone; Hugging Face, Weights & Biases and the comma"
say "address are set in the Settings page of the site."
