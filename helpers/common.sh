#!/usr/bin/env bash
# Shared functions of the helper scripts. Not a command: every helpers/*.sh loads it with
#   . "$(dirname "$0")/common.sh"
#   distill_load_env
#
# distill_load_env
#   Reads <repo>/.env when it exists (gitignored, chmod 600) and exports only these keys:
#     DISTILL_UV_EXTRA  uv extra to run with: "v100" on the Tesla V100 server, empty elsewhere
#     HF_REPO           Hugging Face repo of the published models
#     COMMA_HOST        ssh host of the comma
#     COMMA_VIDEOS_DIR  shared folder of the files downloaded from the comma
#   Lines are KEY=value (an "export " prefix and quotes around the value are accepted); blank
#   lines and lines starting with # are skipped, other keys are ignored. The file is parsed,
#   never executed. A key already set in the environment wins over the .env value.
#   Call it before the script reads these variables or sets their defaults.
#
# distill_uv_run <args...>
#   "uv run --extra $DISTILL_UV_EXTRA <args...>" when DISTILL_UV_EXTRA is set, else "uv run <args...>".
#   A plain "uv run" on the V100 server would re-sync the default CUDA 13 torch, which has no
#   Volta kernels (see [project.optional-dependencies] in pyproject.toml).
#
# distill_gpu_lock_or_die
#   Takes the GPU lock shared by every GPU job on the server (also sunnypilot.speedlimits and
#   "flock /run/lock/v100-gpu.lock <command>"), or stops the script when the GPU is busy:
#     - opens fd 9 on /run/lock/v100-gpu.lock and takes it with "flock -n 9";
#     - refuses when nvidia-smi lists a compute process that is not this script or one of its
#       children (a job that does not use the lock, e.g. a speedlimits training).
#   The lock stays held until the script exits. Child processes inherit fd 9, so a training
#   started in background keeps the lock until it ends. A helper started by a helper that holds
#   the lock does not take it again. Without nvidia-smi (e.g. the Mac) there is no GPU to share
#   and the function does nothing.

# The repo root, from the position of this file (helpers/common.sh).
DISTILL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DISTILL_ENV_FILE="$DISTILL_ROOT/.env"
DISTILL_GPU_LOCK_FILE="/run/lock/v100-gpu.lock"

# Run directly (helpers/common.sh --help): print the comment block above.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

distill_die() { echo "ERROR: $*" >&2; exit 1; }

distill_load_env() {
  local line=""
  local line_number=0
  local key=""
  local value=""
  local current_value=""
  # Patterns in variables: the portable way to use [[ =~ ]] (bash 3.2 on the Mac too).
  local skip_pattern='^[[:space:]]*(#|$)'
  local assignment_pattern='^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$'
  local extra_name_pattern='^[A-Za-z0-9_-]+$'

  if [ ! -f "$DISTILL_ENV_FILE" ]; then
    return 0
  fi

  # "|| [ -n "$line" ]" also reads a last line without a final newline.
  while IFS= read -r line || [ -n "$line" ]; do
    line_number=$((line_number + 1))
    # Files edited on Windows end their lines with \r.
    line="${line%$'\r'}"

    if [[ "$line" =~ $skip_pattern ]]; then
      continue
    fi
    if [[ ! "$line" =~ $assignment_pattern ]]; then
      # The line itself is not printed: it could hold a secret.
      echo "ATTENZIONE: $DISTILL_ENV_FILE riga $line_number ignorata (formato atteso CHIAVE=valore)" >&2
      continue
    fi
    key="${BASH_REMATCH[2]}"
    value="${BASH_REMATCH[3]}"

    case "$key" in
      DISTILL_UV_EXTRA|HF_REPO|COMMA_HOST|COMMA_VIDEOS_DIR)
        ;;
      *)
        # Any other key (e.g. a token) is not for the helpers: ignored, never printed.
        continue
        ;;
    esac

    # Trailing spaces, then one pair of quotes around the whole value.
    value="$(printf '%s' "$value" | sed 's/[[:space:]]*$//')"
    if [ "${#value}" -ge 2 ]; then
      if [ "${value:0:1}" = '"' ] && [ "${value: -1}" = '"' ]; then
        value="${value:1:${#value}-2}"
      elif [ "${value:0:1}" = "'" ] && [ "${value: -1}" = "'" ]; then
        value="${value:1:${#value}-2}"
      fi
    fi

    if [ "$key" = "DISTILL_UV_EXTRA" ] && [ -n "$value" ]; then
      if [[ ! "$value" =~ $extra_name_pattern ]]; then
        distill_die "$DISTILL_ENV_FILE riga $line_number: DISTILL_UV_EXTRA='$value' non e' il nome di un extra di uv (es. v100)"
      fi
    fi

    # The environment wins: "COMMA_HOST=... helpers/x.sh" overrides the .env value.
    # ${!key} is the value of the variable whose name is in $key.
    current_value="${!key:-}"
    if [ -n "$current_value" ]; then
      continue
    fi
    export "$key=$value"
  done < "$DISTILL_ENV_FILE"
}

distill_uv_run() {
  if [ -n "${DISTILL_UV_EXTRA:-}" ]; then
    uv run --extra "$DISTILL_UV_EXTRA" "$@"
  else
    uv run "$@"
  fi
}

# True when process $1 is process $2 or one of its descendants (walks up the parent pids).
distill_pid_descends_from() {
  local pid="$1"
  local ancestor="$2"
  local parent=""

  while true; do
    case "$pid" in
      ''|*[!0-9]*)
        return 1
        ;;
    esac
    if [ "$pid" -le 1 ]; then
      return 1
    fi
    if [ "$pid" = "$ancestor" ]; then
      return 0
    fi
    # A process that is gone (or not visible from this container) has no parent: not ours.
    if ! parent="$(ps -o ppid= -p "$pid" 2>/dev/null)"; then
      return 1
    fi
    pid="$(echo "$parent" | tr -d '[:space:]')"
  done
}

# Prints the processes (at most 3) that have the GPU lock file open, apart from this script and
# its children: the job holding the lock and the children that inherited its fd. Nothing when it
# cannot tell. /proc/locks is no use here: inside the LXC it does not list these locks.
distill_gpu_lock_holder() {
  local open_fds=""
  local fd_path=""
  local pid=""
  local seen_pids=" "
  local command_line=""
  local found=0
  local holders=""

  # One line per open fd: /proc/<pid>/fd/<n>. find exits non-zero when a process ends while it
  # looks: what it printed is still good. "9>&-": find must not inherit our fd 9 and list itself.
  open_fds="$(find /proc/[0-9]*/fd -maxdepth 1 -lname "$DISTILL_GPU_LOCK_FILE" 2>/dev/null 9>&-)" || true

  while IFS= read -r fd_path; do
    if [ -z "$fd_path" ]; then
      continue
    fi
    pid="${fd_path#/proc/}"
    pid="${pid%%/*}"
    # One entry per process, even when it has the file open more than once.
    case "$seen_pids" in
      *" $pid "*)
        continue
        ;;
    esac
    seen_pids="$seen_pids$pid "
    # This script opened the file too (fd 9), and so did its children.
    if distill_pid_descends_from "$pid" "$$"; then
      continue
    fi
    # A process that already ended holds nothing.
    if ! command_line="$(ps -o args= -p "$pid" 2>/dev/null 9>&-)"; then
      continue
    fi
    if [ -n "$holders" ]; then
      holders="$holders; "
    fi
    holders="${holders}pid $pid: $command_line"
    found=$((found + 1))
    if [ "$found" -ge 3 ]; then
      break
    fi
  done <<< "$open_fds"
  echo "$holders"
}

distill_gpu_lock_or_die() {
  local lock_dir=""
  local holder=""
  local gpu_processes=""
  local line=""
  local gpu_pid=""
  local gpu_name=""
  local foreign_processes=""

  # No NVIDIA driver here (e.g. the Mac): no GPU shared with other jobs.
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi non trovato: nessuna GPU NVIDIA da condividere, lock GPU non necessario" >&2
    return 0
  fi

  # Already held by this job: a second call in the same script, or a helper started by a
  # helper that took the lock (it inherited fd 9). Taking it again would refuse our own job.
  if [ -n "${DISTILL_GPU_LOCK_PID:-}" ]; then
    if distill_pid_descends_from "$$" "$DISTILL_GPU_LOCK_PID"; then
      return 0
    fi
  fi

  if ! command -v flock >/dev/null 2>&1; then
    distill_die "flock non trovato (pacchetto util-linux): serve per il lock della GPU"
  fi
  lock_dir="$(dirname "$DISTILL_GPU_LOCK_FILE")"
  if [ ! -d "$lock_dir" ] || [ ! -w "$lock_dir" ]; then
    distill_die "cartella del lock GPU $lock_dir mancante o non scrivibile"
  fi

  # fd 9 stays open until the script exits: the lock lasts as long as the script (and its children).
  if ! exec 9>>"$DISTILL_GPU_LOCK_FILE"; then
    distill_die "non riesco ad aprire il lock GPU $DISTILL_GPU_LOCK_FILE"
  fi
  if ! flock -n 9; then
    holder="$(distill_gpu_lock_holder)"
    if [ -n "$holder" ]; then
      distill_die "La GPU e' occupata da un altro job ($holder). Riprova quando ha finito."
    else
      distill_die "La GPU e' occupata da un altro job (lock $DISTILL_GPU_LOCK_FILE preso). Riprova quando ha finito."
    fi
  fi

  # Jobs that do not use the lock (e.g. a speedlimits training) still show up in nvidia-smi.
  if ! gpu_processes="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>&1)"; then
    distill_die "nvidia-smi non risponde, non so se la GPU e' libera: $gpu_processes"
  fi
  while IFS= read -r line; do
    if [ -z "$line" ]; then
      continue
    fi
    # "<pid>, <process name>"
    gpu_pid="$(echo "${line%%,*}" | tr -d '[:space:]')"
    gpu_name="$(echo "${line#*,}" | sed 's/^[[:space:]]*//')"
    if distill_pid_descends_from "$gpu_pid" "$$"; then
      continue
    fi
    if [ -n "$foreign_processes" ]; then
      foreign_processes="$foreign_processes; "
    fi
    foreign_processes="${foreign_processes}pid $gpu_pid: $gpu_name"
  done <<< "$gpu_processes"
  if [ -n "$foreign_processes" ]; then
    distill_die "La GPU e' occupata da un altro job (processi sulla GPU: $foreign_processes). Riprova quando ha finito."
  fi

  export DISTILL_GPU_LOCK_PID="$$"
}
