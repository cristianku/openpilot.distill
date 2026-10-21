#!/usr/bin/env bash
# Train (distill) the driving model on the segments in data/ with supervised/train.py:
# it learns from your driving (localizer targets) and from the big openpilot model (teacher).
#
# Usage: helpers/train_supervised.sh [--steps=<n>] [--batch-size=<n>] [--workers=<n>] [--test]
#                                    [--background | --foreground-log] [--wandb=online|offline|disabled]
#                                    [--continue | --resume=<run>]
#   helpers/train_supervised.sh                  (5000 step, in primo piano)
#   helpers/train_supervised.sh --test           (10 step, solo per vedere che tutto parte)
#   helpers/train_supervised.sh --background     (continua anche se chiudi la ssh)
#   helpers/train_supervised.sh --steps=20000 --background
#   helpers/train_supervised.sh --continue --background                    (dai pesi "stable" su Hugging Face)
#   helpers/train_supervised.sh --resume=outputs/supervised/2026-10-21_1544  (riprende un run interrotto)
#   helpers/train_supervised.sh --resume=2026-10-21_1544                     (uguale, solo il nome del run)
#   helpers/train_supervised.sh --foreground-log --wandb=offline             (come lo lancia il sito)
#
# Start of the model (supervised/train.py --mode):
#   (default)   fresh: a new model, from the pretrained vision backbone.
#   --continue  a new run that starts from the weights pointed to by training/stable.json on
#               Hugging Face (HF_REPO): the ones published with helpers/publish_model.sh --stable.
#               Only the weights, not the optimizer: it is a new run with its own folder.
#   --resume    the same run, interrupted (ssh dropped, server restarted): model and optimizer
#               from <run>/supercombo.training.pt (saved every 500 steps), same folder, train.log
#               and metrics.jsonl, same --steps (read from the checkpoint when not given).
#               <run> is the run folder or only its name in outputs/supervised/, old names
#               (2026-10-21_1544) and new ones (2026-10-21_154412) alike.
#
# One step = --batch-size moments (default 8) taken at random from ONE segment.
# 5000 steps (the README value) x 8 = 40000 samples: with ~45 segments every segment is
# seen about a hundred times. train.py does no validation: judge from the loss
# (gt = your driving, distill = the teacher) going down and flattening (metrics.jsonl or wandb).
#
# --workers (default 6): processes that load and decode the videos in parallel. With 1
# the GPU sits waiting for data most of the time.
#
# Where the output goes:
#   (default)         foreground; the training output is printed and appended to train.log.
#   --background      the training goes on after the ssh is closed; its output only in train.log.
#   --foreground-log  foreground; everything from the creation of the run folder on (checks,
#                     errors, training) is printed AND appended to train.log. The web site uses
#                     it: it keeps what is printed in its own job log.
#   In every case the run folder is printed early, on a line of its own:  RUN_DIR=<absolute path>
#
# --wandb=online|offline|disabled: WANDB_MODE of this run. Without it: online when "uv run wandb
#   login" was done (key in ~/.netrc) or WANDB_API_KEY is set, offline otherwise; a WANDB_MODE
#   already set is kept. offline keeps everything in the run folder (wandb sync later), disabled
#   writes nothing. The numbers of every step are in metrics.jsonl anyway.
#
# GPU: one GPU job at a time. Before anything else it takes /run/lock/v100-gpu.lock (the lock of
# every GPU job on the server, see helpers/common.sh) and stops when another job holds it or uses
# the GPU. The training keeps the lock until it ends, with --background too.
#
# Before training it checks:
#   - the teacher models/big_driving_supercombo.onnx: downloaded (or completed) when missing or partial
#   - every segment in data/ is localized (localizer.safetensors + frame_info.safetensors):
#     one without would stop the training, run helpers/localize_segments.sh first
#   - fcamera.hevc / ecamera.hevc in data/ are not links to a missing file of the shared
#     folder (COMMA_VIDEOS_DIR): one would stop the training too
#
# Every run gets its own folder outputs/supervised/<AAAA-MM-GG_hhmmss>/ (time of the server) with
# supercombo.pt (the trained model, written only at the end), supercombo.training.pt (model +
# optimizer, for --resume), train.log, metrics.jsonl (one JSON line per step: loss, gt, distill,
# grad_norm, step_time and the single losses), run.json (settings, start, end) and the wandb files.
# -h, --help: this help.
# Env: HF_REPO (default cristianku/openpilot-distill-models, as helpers/publish_model.sh)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
. "$ROOT/helpers/common.sh"
distill_load_env

DATA_DIR="$ROOT/data"
RUNS_DIR="$ROOT/outputs/supervised"
OPENPILOT_SHA="084747c75d2cbd23af65ab7a9e770bbd7b98bac9"
TEACHER="$ROOT/models/big_driving_supercombo.onnx"
TEACHER_URL="https://media.githubusercontent.com/media/commaai/openpilot/${OPENPILOT_SHA}/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx"
HF_REPO="${HF_REPO:-cristianku/openpilot-distill-models}"
# Run folder names: with seconds, and the old ones to the minute (runs before 2026-10-22).
RUN_NAME_PATTERN='^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{4}([0-9]{2})?$'

# Python keeps its prints in a buffer when the output is a file or a pipe: train.log and the
# site's job log would only move every few kilobytes.
export PYTHONUNBUFFERED=1

die() { echo "ERROR: $*" >&2; exit 1; }

# Prints the comment block at the top of this file.
usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

steps=5000
batch_size=8
# Data-loader workers: with 1 the GPU waits ~3.5 s per step for the videos to be decoded
# (one core busy, GPU mostly idle). 6 of the 8 cores, ~2.5 GB RAM each.
workers=6
background=0
foreground_log=0
wandb_mode=""
mode="fresh"
resume_dir=""
# --resume takes the steps from the checkpoint unless they are given.
steps_given=0

while [ $# -gt 0 ]; do
  if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
  elif [ "$1" = "--test" ]; then
    steps=10
    steps_given=1
  elif [ "$1" = "--background" ]; then
    background=1
  elif [ "$1" = "--foreground-log" ]; then
    foreground_log=1
  elif [[ "$1" == --wandb=* ]]; then
    wandb_mode="${1#--wandb=}"
  elif [ "$1" = "--continue" ]; then
    [ "$mode" != "resume" ] || die "--continue e --resume insieme non hanno senso"
    mode="continue"
  elif [[ "$1" == --resume=* ]]; then
    [ "$mode" != "continue" ] || die "--continue e --resume insieme non hanno senso"
    mode="resume"
    resume_dir="${1#--resume=}"
    [ -n "$resume_dir" ] || die "--resume vuole il run (es. outputs/supervised/2026-10-21_1544 o 2026-10-21_1544)"
  elif [[ "$1" == --steps=* ]]; then
    steps="${1#--steps=}"
    steps_given=1
  elif [[ "$1" == --batch-size=* ]]; then
    batch_size="${1#--batch-size=}"
  elif [[ "$1" == --workers=* ]]; then
    workers="${1#--workers=}"
  else
    die "opzione non valida: $1 (vedi --help)"
  fi
  shift
done

[[ "$steps" =~ ^[0-9]+$ ]] || die "--steps vuole un numero"
[[ "$batch_size" =~ ^[0-9]+$ ]] || die "--batch-size vuole un numero"
[[ "$workers" =~ ^[0-9]+$ ]] || die "--workers vuole un numero"
if [ "$background" -eq 1 ] && [ "$foreground_log" -eq 1 ]; then
  die "--background e --foreground-log insieme non hanno senso"
fi
if [ -n "$wandb_mode" ] && [ "$wandb_mode" != "online" ] && [ "$wandb_mode" != "offline" ] \
  && [ "$wandb_mode" != "disabled" ]; then
  die "--wandb vuole online, offline o disabled (non '$wandb_mode')"
fi

# GPU lock first: two trainings started together must not both prepare a run.
distill_gpu_lock_or_die

# --continue: without training/stable.json on Hugging Face train.py would stop after the teacher
# download and the dataset checks, so check it first.
if [ "$mode" = "continue" ]; then
  stable_url="https://huggingface.co/$HF_REPO/resolve/main/training/stable.json"
  http_code="$(curl -sS -L -o /dev/null -w '%{http_code}' "$stable_url" || true)"
  if [ "$http_code" = "404" ]; then
    die "su $HF_REPO non c'e' ancora training/stable.json: pubblica prima un modello con helpers/publish_model.sh --stable"
  fi
  [ "$http_code" = "200" ] || die "non riesco a leggere $stable_url (HTTP $http_code)"
fi

# --resume: the folder of an interrupted run, with its checkpoint and without the final model.
# Given as a path, or only as the run name (what the web site passes).
if [ "$mode" = "resume" ]; then
  if [ -d "$resume_dir" ]; then
    run_dir="$(cd "$resume_dir" && pwd)"
  elif [[ "$resume_dir" =~ $RUN_NAME_PATTERN ]] && [ -d "$RUNS_DIR/$resume_dir" ]; then
    run_dir="$RUNS_DIR/$resume_dir"
  else
    die "non trovo il run $resume_dir (la cartella, o il nome di una cartella in outputs/supervised/)"
  fi
  [ -f "$run_dir/supercombo.training.pt" ] \
    || die "in $run_dir non c'e' supercombo.training.pt: il run e' partito senza checkpoint, non si puo' riprendere"
  [ ! -f "$run_dir/supercombo.pt" ] \
    || die "$run_dir e' gia' finito (c'e' supercombo.pt): per allenarlo ancora pubblicalo con --stable e usa --continue"
  if [ "$steps_given" -eq 0 ]; then
    steps="$(cd "$ROOT" && distill_uv_run python -c 'import sys, torch
print(torch.load(sys.argv[1], map_location="cpu", weights_only=True)["total_steps"])' \
      "$run_dir/supercombo.training.pt")"
  fi
fi

# wandb: --wandb wins. Without it as before: "uv run wandb login" saves the key in ~/.netrc,
# without a key it goes offline (wandb.init would ask for a login otherwise).
wandb_logged_in=0
if [ -n "${WANDB_API_KEY:-}" ] || grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
  wandb_logged_in=1
fi
if [ -n "$wandb_mode" ]; then
  if [ "$wandb_mode" = "online" ] && [ "$wandb_logged_in" -eq 0 ]; then
    die "--wandb=online ma wandb non e' loggato (uv run wandb login): usa --wandb=offline"
  fi
  export WANDB_MODE="$wandb_mode"
elif [ -z "${WANDB_MODE:-}" ] && [ "$wandb_logged_in" -eq 0 ]; then
  export WANDB_MODE=offline
  echo "wandb: non loggato, vado offline (i numeri restano in train.log e metrics.jsonl)."
fi

# A new run: its own folder, named after the time with seconds (two starts in the same minute
# used to get the same folder, and train.py refused the second).
if [ "$mode" != "resume" ]; then
  mkdir -p "$RUNS_DIR"
  run_dir="$RUNS_DIR/$(date '+%Y-%m-%d_%H%M%S')"
  [ ! -e "$run_dir" ] || die "$run_dir esiste gia' (due partenze nello stesso secondo?): rilancia"
  mkdir "$run_dir"
fi
log="$run_dir/train.log"
# On a line of its own: the web site reads the run folder of its job from here.
echo "RUN_DIR=$run_dir"

# Teacher and segment checks, then the training. A function, so that with --foreground-log its
# whole output can go through one tee into train.log.
prepare_and_train() {
  # A download of the teacher still running elsewhere: a second curl on the same file would break it.
  if pgrep -f "curl.*big_driving_supercombo.onnx" > /dev/null; then
    die "il teacher si sta ancora scaricando (curl in corso): aspetta che finisca e rilancia"
  fi

  # Teacher: download it, or finish a partial download, when its size is not the one on GitHub.
  remote_size="$(curl -sIL "$TEACHER_URL" | tr -d '\r' | sed -n 's/^[Cc]ontent-[Ll]ength: //p' | tail -1)"
  local_size=0
  if [ -f "$TEACHER" ]; then
    local_size="$(stat -c %s "$TEACHER")"
  fi
  if [ -z "$remote_size" ]; then
    [ -f "$TEACHER" ] || die "manca $TEACHER e GitHub non risponde per scaricarlo"
    echo "Teacher: GitHub non risponde, uso $TEACHER cosi' com'e' ($local_size byte)."
  elif [ "$local_size" != "$remote_size" ]; then
    echo "Teacher: $local_size di $remote_size byte, scarico..."
    mkdir -p "$(dirname "$TEACHER")"
    curl --fail --location --continue-at - "$TEACHER_URL" --output "$TEACHER"
    local_size="$(stat -c %s "$TEACHER")"
    [ "$local_size" = "$remote_size" ] || die "teacher scaricato a meta' ($local_size di $remote_size byte), rilancia"
  else
    echo "Teacher: ok ($local_size byte)."
  fi

  # Segments: the dataset takes every folder of data/ with fcamera.hevc, and one without
  # the localizer outputs would stop the training halfway. The videos are links into the
  # shared folder (COMMA_VIDEOS_DIR): a link to a file that is not there (shared disk not
  # mounted, segment removed from it) is still listed by the dataset, which then stops when
  # it tries to read it.
  total=0
  missing=""
  broken=""
  for dir in "$DATA_DIR"/*/; do
    if [ -L "$dir/fcamera.hevc" ] && [ ! -e "$dir/fcamera.hevc" ]; then
      broken="$broken $(basename "$dir")"
      continue
    fi
    if [ ! -f "$dir/fcamera.hevc" ]; then
      continue
    fi
    if [ -L "$dir/ecamera.hevc" ] && [ ! -e "$dir/ecamera.hevc" ]; then
      broken="$broken $(basename "$dir")"
      continue
    fi
    total=$((total + 1))
    if [ ! -f "$dir/localizer.safetensors" ] || [ ! -f "$dir/frame_info.safetensors" ]; then
      missing="$missing $(basename "$dir")"
    fi
  done
  [ -z "$broken" ] || die "link rotti verso la cartella comune (COMMA_VIDEOS_DIR), il training si fermerebbe:$broken
Controlla che la cartella comune ci sia (disco montato) o riscarica con helpers/fetch_comma_segments.sh."
  [ "$total" -gt 0 ] || die "nessun segmento in data/ (scaricali con helpers/fetch_comma_segments.sh)"
  [ -z "$missing" ] || die "segmenti non localizzati, fermerebbero il training:$missing
Lancia prima helpers/localize_segments.sh (quelli inutilizzabili li sposta in data_scartati/)."
  echo "Segmenti: $total, tutti localizzati."

  mode_args=(--mode "$mode")
  if [ "$mode" = "fresh" ]; then
    start="da zero"
  elif [ "$mode" = "continue" ]; then
    start="dai pesi stable di $HF_REPO"
    mode_args+=(--hf-repo "$HF_REPO")
  else
    start="ripresa dall'ultimo checkpoint"
  fi
  echo "Training ($start): $steps step, batch $batch_size, workers $workers, wandb ${WANDB_MODE:-online} -> $run_dir"

  # The log is appended to: --resume keeps writing the train.log of its run.
  # The GPU lock (fd 9) is inherited by the training: with --background it stays held after this
  # script ends, until the training ends.
  cd "$ROOT"
  if [ "$background" -eq 1 ]; then
    # nohup cannot run a shell function: the uv command as an array (helpers/common.sh).
    distill_uv_command
    nohup "${DISTILL_UV_COMMAND[@]}" torchrun --standalone --nproc-per-node=1 -m supervised.train "${mode_args[@]}" \
      --steps "$steps" --batch-size "$batch_size" --workers "$workers" \
      --output "$run_dir/supercombo.pt" >> "$log" 2>&1 &
    echo "Partito in background (pid $!). Segui con:  tail -f $log"
    echo "Alla fine il modello e' in $run_dir/supercombo.pt"
  elif [ "$foreground_log" -eq 1 ]; then
    # The output of this function already goes through tee into train.log.
    distill_uv_run torchrun --standalone --nproc-per-node=1 -m supervised.train "${mode_args[@]}" \
      --steps "$steps" --batch-size "$batch_size" --workers "$workers" \
      --output "$run_dir/supercombo.pt" 2>&1
    echo "Fatto. Modello: $run_dir/supercombo.pt"
  else
    distill_uv_run torchrun --standalone --nproc-per-node=1 -m supervised.train "${mode_args[@]}" \
      --steps "$steps" --batch-size "$batch_size" --workers "$workers" \
      --output "$run_dir/supercombo.pt" 2>&1 | tee -a "$log"
    echo "Fatto. Modello: $run_dir/supercombo.pt"
  fi
}

if [ "$foreground_log" -eq 1 ]; then
  # Everything printed from here on: on the screen (the site's job log) and in train.log.
  # With pipefail the exit code is the one of prepare_and_train, not the one of tee.
  prepare_and_train 2>&1 | tee -a "$log"
else
  prepare_and_train
fi
