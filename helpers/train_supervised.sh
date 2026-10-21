#!/usr/bin/env bash
# Train (distill) the driving model on the segments in data/ with supervised/train.py:
# it learns from your driving (localizer targets) and from the big openpilot model (teacher).
#
# Usage: helpers/train_supervised.sh [--steps=<n>] [--batch-size=<n>] [--workers=<n>] [--test] [--background]
#                                    [--continue | --resume=<cartella del run>]
#   helpers/train_supervised.sh                  (5000 step, in primo piano)
#   helpers/train_supervised.sh --test           (10 step, solo per vedere che tutto parte)
#   helpers/train_supervised.sh --background     (continua anche se chiudi la ssh)
#   helpers/train_supervised.sh --steps=20000 --background
#   helpers/train_supervised.sh --continue --background                    (dai pesi "stable" su Hugging Face)
#   helpers/train_supervised.sh --resume=outputs/supervised/2026-10-21_1544  (riprende un run interrotto)
#
# Start of the model (supervised/train.py --mode):
#   (default)   fresh: a new model, from the pretrained vision backbone.
#   --continue  a new run that starts from the weights pointed to by training/stable.json on
#               Hugging Face (HF_REPO): the ones published with helpers/publish_model.sh --stable.
#               Only the weights, not the optimizer: it is a new run with its own folder.
#   --resume    the same run, interrupted (ssh dropped, server restarted): model and optimizer
#               from <cartella del run>/supercombo.training.pt (saved every 500 steps), same
#               folder and train.log, same --steps (read from the checkpoint when not given).
#
# One step = --batch-size moments (default 8) taken at random from ONE segment.
# 5000 steps (the README value) x 8 = 40000 samples: with ~45 segments every segment is
# seen about a hundred times. train.py does no validation: judge from the loss
# (gt = your driving, distill = the teacher) going down and flattening on wandb.
#
# --workers (default 6): processes that load and decode the videos in parallel. With 1
# the GPU sits waiting for data most of the time.
#
# Before training it checks:
#   - the teacher models/big_driving_supercombo.onnx: downloaded (or completed) when missing or partial
#   - every segment in data/ is localized (localizer.safetensors + frame_info.safetensors):
#     one without would stop the training, run helpers/localize_segments.sh first
#   - fcamera.hevc / ecamera.hevc in data/ are not links to a missing file of the shared
#     folder (COMMA_VIDEOS_DIR): one would stop the training too
#   - wandb: without "uv run wandb login" done, it runs offline (WANDB_MODE=offline)
#
# Every run gets its own folder outputs/supervised/<data_ora>/ with supercombo.pt (the
# trained model, written only at the end), supercombo.training.pt (model + optimizer, for
# --resume), train.log and the wandb files.
# -h, --help: this help.
# Env: HF_REPO (default cristianku/openpilot-distill-models, as helpers/publish_model.sh)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT/data"
OPENPILOT_SHA="084747c75d2cbd23af65ab7a9e770bbd7b98bac9"
TEACHER="$ROOT/models/big_driving_supercombo.onnx"
TEACHER_URL="https://media.githubusercontent.com/media/commaai/openpilot/${OPENPILOT_SHA}/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx"
HF_REPO="${HF_REPO:-cristianku/openpilot-distill-models}"

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
  elif [ "$1" = "--continue" ]; then
    [ "$mode" != "resume" ] || die "--continue e --resume insieme non hanno senso"
    mode="continue"
  elif [[ "$1" == --resume=* ]]; then
    [ "$mode" != "continue" ] || die "--continue e --resume insieme non hanno senso"
    mode="resume"
    resume_dir="${1#--resume=}"
    [ -n "$resume_dir" ] || die "--resume vuole la cartella del run (es. outputs/supervised/2026-10-21_1544)"
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
if [ "$mode" = "resume" ]; then
  [ -d "$resume_dir" ] || die "non trovo $resume_dir"
  run_dir="$(cd "$resume_dir" && pwd)"
  [ -f "$run_dir/supercombo.training.pt" ] \
    || die "in $run_dir non c'e' supercombo.training.pt: il run e' partito senza checkpoint, non si puo' riprendere"
  [ ! -f "$run_dir/supercombo.pt" ] \
    || die "$run_dir e' gia' finito (c'e' supercombo.pt): per allenarlo ancora pubblicalo con --stable e usa --continue"
  if [ "$steps_given" -eq 0 ]; then
    steps="$(cd "$ROOT" && uv run python -c 'import sys, torch
print(torch.load(sys.argv[1], map_location="cpu", weights_only=True)["total_steps"])' \
      "$run_dir/supercombo.training.pt")"
  fi
fi

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

# wandb: "uv run wandb login" saves the key in ~/.netrc; without it, offline.
if [ -z "${WANDB_MODE:-}" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  if ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    export WANDB_MODE=offline
    echo "wandb: non loggato, vado offline (i numeri restano nel terminale e in train.log)."
  fi
fi

if [ "$mode" != "resume" ]; then
  run_dir="$ROOT/outputs/supervised/$(date '+%Y-%m-%d_%H%M')"
  mkdir -p "$run_dir"
fi
log="$run_dir/train.log"
mode_args=(--mode "$mode")
if [ "$mode" = "fresh" ]; then
  start="da zero"
elif [ "$mode" = "continue" ]; then
  start="dai pesi stable di $HF_REPO"
  mode_args+=(--hf-repo "$HF_REPO")
else
  start="ripresa dall'ultimo checkpoint"
fi
echo "Training ($start): $steps step, batch $batch_size, workers $workers -> $run_dir"

# The log is appended to: --resume keeps writing the train.log of its run.
cd "$ROOT"
if [ "$background" -eq 1 ]; then
  nohup uv run torchrun --standalone --nproc-per-node=1 -m supervised.train "${mode_args[@]}" \
    --steps "$steps" --batch-size "$batch_size" --workers "$workers" \
    --output "$run_dir/supercombo.pt" >> "$log" 2>&1 &
  echo "Partito in background (pid $!). Segui con:  tail -f $log"
  echo "Alla fine il modello e' in $run_dir/supercombo.pt"
else
  uv run torchrun --standalone --nproc-per-node=1 -m supervised.train "${mode_args[@]}" \
    --steps "$steps" --batch-size "$batch_size" --workers "$workers" \
    --output "$run_dir/supercombo.pt" 2>&1 | tee -a "$log"
  echo "Fatto. Modello: $run_dir/supercombo.pt"
fi
