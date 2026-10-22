#!/usr/bin/env bash
# Train (distill) the driving model on the segments in data/ with supervised/train.py:
# it learns from your driving (localizer targets) and from the big openpilot model (teacher).
#
# Usage: helpers/train_supervised.sh [--steps=<n>] [--batch-size=<n>] [--workers=<n>] [--test]
#                                    [--background | --foreground-log] [--wandb=online|offline|disabled]
#                                    [--continue | --resume=<run>] [--teacher=md-v3|084747c7]
#   helpers/train_supervised.sh                  (5000 steps, in the foreground)
#   helpers/train_supervised.sh --test           (10 steps, only to see that everything starts)
#   helpers/train_supervised.sh --background     (keeps going after the ssh is closed)
#   helpers/train_supervised.sh --steps=20000 --background
#   helpers/train_supervised.sh --continue --background                    (from the "stable" weights on Hugging Face)
#   helpers/train_supervised.sh --resume=outputs/supervised/2026-10-21_1544  (resumes an interrupted run)
#   helpers/train_supervised.sh --resume=2026-10-21_1544                     (the same, only the run name)
#   helpers/train_supervised.sh --foreground-log --wandb=offline             (as the web site starts it)
#   helpers/train_supervised.sh --teacher=084747c7 --background              (with the old teacher)
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
# Teacher (--teacher, the big model the student imitates; supervised/teachers.py):
#   md-v3     (default) the big model of openpilot master since "MD v3" (commit 1e219d778):
#             commaai/openpilot_driving_models on Hugging Face, 1365126a-e71e-4778-be00-8ab92dc82c09/12864,
#             at a pinned revision -> models/big_driving_supercombo_1365126a_12864.onnx (0.8 GB).
#             Its history is inside the ONNX: supervised/onnx_helpers.py cuts it for the training
#             (helpers/check_teacher.py checks the cut). distill/hidden_state compares 32 tokens x 512.
#   084747c7  the big model of openpilot 084747c75 from GitHub, the one of upstream openpilot.distill
#             -> models/big_driving_supercombo.onnx (1.8 GB). distill/hidden_state compares one
#             averaged 512 vector: its values are not comparable with the md-v3 ones.
#   --resume always takes the teacher of the interrupted run (saved in its checkpoint), so it does
#   not take --teacher. --continue is a new run and can change teacher (distill/hidden_state then
#   starts high).
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
#   - the teacher file: downloaded (or completed) into <file>.part when missing, renamed only after
#     its size and sha256 are checked
#   - every segment in data/ is localized (localizer.safetensors + frame_info.safetensors):
#     one without would stop the training, run helpers/localize_segments.sh first
#   - fcamera.hevc / ecamera.hevc in data/ are not links to a missing file of the shared
#     folder (COMMA_VIDEOS_DIR): one would stop the training too
#
# Every run gets its own folder outputs/supervised/<YYYY-MM-DD_hhmmss>/ (time of the server) with
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
# Teacher 084747c7: openpilot commit of its GitHub LFS file.
OPENPILOT_SHA="084747c75d2cbd23af65ab7a9e770bbd7b98bac9"
# Teacher md-v3: revision of commaai/openpilot_driving_models on Hugging Face (pinned: the repo
# gets new models, a fixed revision always gives the same file).
HF_TEACHER_REVISION="8dbc57a6480111bd58487ecc248e30d50fb2f8dd"
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
teacher_name="md-v3"
# --resume takes the teacher from the checkpoint and refuses --teacher.
teacher_given=0

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
    [ "$mode" != "resume" ] || die "--continue and --resume together make no sense"
    mode="continue"
  elif [[ "$1" == --resume=* ]]; then
    [ "$mode" != "continue" ] || die "--continue and --resume together make no sense"
    mode="resume"
    resume_dir="${1#--resume=}"
    [ -n "$resume_dir" ] || die "--resume needs the run (e.g. outputs/supervised/2026-10-21_1544 or 2026-10-21_1544)"
  elif [[ "$1" == --steps=* ]]; then
    steps="${1#--steps=}"
    steps_given=1
  elif [[ "$1" == --batch-size=* ]]; then
    batch_size="${1#--batch-size=}"
  elif [[ "$1" == --workers=* ]]; then
    workers="${1#--workers=}"
  elif [[ "$1" == --teacher=* ]]; then
    teacher_name="${1#--teacher=}"
    teacher_given=1
  else
    die "invalid option: $1 (see --help)"
  fi
  shift
done

[[ "$steps" =~ ^[0-9]+$ ]] || die "--steps needs a number"
[[ "$batch_size" =~ ^[0-9]+$ ]] || die "--batch-size needs a number"
[[ "$workers" =~ ^[0-9]+$ ]] || die "--workers needs a number"
if [ "$background" -eq 1 ] && [ "$foreground_log" -eq 1 ]; then
  die "--background and --foreground-log together make no sense"
fi
if [ -n "$wandb_mode" ] && [ "$wandb_mode" != "online" ] && [ "$wandb_mode" != "offline" ] \
  && [ "$wandb_mode" != "disabled" ]; then
  die "--wandb needs online, offline or disabled (not '$wandb_mode')"
fi
# --teacher: one of the two this script can download (file, URL and sha256 below). --resume takes
# the teacher of the interrupted run from its checkpoint, so it refuses --teacher.
if [ "$teacher_name" != "md-v3" ] && [ "$teacher_name" != "084747c7" ]; then
  die "--teacher needs md-v3 or 084747c7 (not '$teacher_name')"
fi
if [ "$mode" = "resume" ] && [ "$teacher_given" -eq 1 ]; then
  die "with --resume the teacher is the one of the interrupted run: drop --teacher"
fi

# GPU lock first: two trainings started together must not both prepare a run.
distill_gpu_lock_or_die

# --continue: without training/stable.json on Hugging Face train.py would stop after the teacher
# download and the dataset checks, so check it first.
if [ "$mode" = "continue" ]; then
  stable_url="https://huggingface.co/$HF_REPO/resolve/main/training/stable.json"
  http_code="$(curl -sS -L -o /dev/null -w '%{http_code}' "$stable_url" || true)"
  if [ "$http_code" = "404" ]; then
    die "$HF_REPO has no training/stable.json yet: first publish a model with helpers/publish_model.sh --stable"
  fi
  [ "$http_code" = "200" ] || die "cannot read $stable_url (HTTP $http_code)"
fi

# --resume: the folder of an interrupted run, with its checkpoint and without the final model.
# Given as a path, or only as the run name (what the web site passes).
if [ "$mode" = "resume" ]; then
  if [ -d "$resume_dir" ]; then
    run_dir="$(cd "$resume_dir" && pwd)"
  elif [[ "$resume_dir" =~ $RUN_NAME_PATTERN ]] && [ -d "$RUNS_DIR/$resume_dir" ]; then
    run_dir="$RUNS_DIR/$resume_dir"
  else
    die "run $resume_dir not found (give the folder, or the name of a folder in outputs/supervised/)"
  fi
  [ -f "$run_dir/supercombo.training.pt" ] \
    || die "$run_dir has no supercombo.training.pt: the run started without a checkpoint and cannot be resumed"
  [ ! -f "$run_dir/supercombo.pt" ] \
    || die "$run_dir has already finished (supercombo.pt is there): to train it further, publish it with --stable and use --continue"
  # Steps and teacher of the run in one read (the checkpoint also holds the optimizer: slow to load).
  resume_info="$(cd "$ROOT" && distill_uv_run python -c 'import sys, torch
saved = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
print(saved["total_steps"])
print(saved.get("teacher", ""))' \
    "$run_dir/supercombo.training.pt")"
  saved_steps="$(printf '%s\n' "$resume_info" | sed -n 1p)"
  saved_teacher="$(printf '%s\n' "$resume_info" | sed -n 2p)"
  if [ "$steps_given" -eq 0 ]; then
    steps="$saved_steps"
  fi
  # The file name says which teacher it is (the runs before --teacher all used 084747c7).
  saved_teacher_file="$(basename "$saved_teacher")"
  if [ "$saved_teacher_file" = "big_driving_supercombo_1365126a_12864.onnx" ]; then
    teacher_name="md-v3"
  elif [ "$saved_teacher_file" = "big_driving_supercombo.onnx" ]; then
    teacher_name="084747c7"
  else
    die "the run uses the teacher '$saved_teacher', which this script cannot download: resume it by hand with supervised/train.py --mode resume --teacher ..."
  fi
fi

# Teacher: file, where to download it and its sha256 (supervised/teachers.py, supervised/README.md).
if [ "$teacher_name" = "md-v3" ]; then
  TEACHER="$ROOT/models/big_driving_supercombo_1365126a_12864.onnx"
  TEACHER_URL="https://huggingface.co/commaai/openpilot_driving_models/resolve/${HF_TEACHER_REVISION}/1365126a-e71e-4778-be00-8ab92dc82c09/12864/big_driving_supercombo.onnx"
  TEACHER_SHA256="1cfd2c45353d876b5b257b24d77537de58ab90a44ae595f15b1f64b6ea3fd770"
  TEACHER_SOURCE="Hugging Face"
elif [ "$teacher_name" = "084747c7" ]; then
  TEACHER="$ROOT/models/big_driving_supercombo.onnx"
  TEACHER_URL="https://media.githubusercontent.com/media/commaai/openpilot/${OPENPILOT_SHA}/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx"
  TEACHER_SHA256="a501760a9d1d5fef0eab2b8c5d122d06124fc26dc8e0782e0aa94b82a208f0ff"
  TEACHER_SOURCE="GitHub"
else
  die "--teacher needs md-v3 or 084747c7 (not '$teacher_name')"
fi
# TEACHER is the file that is checked here, TEACHER_ARG the --teacher given to train.py.
# --resume: TEACHER_ARG is exactly the path saved in the checkpoint, train.py compares it as a
# string (the repo path of the run may be written differently, e.g. through a symlink). A relative
# one (train.py started by hand with --teacher models/...) is relative to the repo root, where
# train.py runs: the checks below look for the file there too, wherever this script is started from.
TEACHER_ARG="$TEACHER"
if [ "$mode" = "resume" ]; then
  TEACHER_ARG="$saved_teacher"
  if [[ "$saved_teacher" == /* ]]; then
    TEACHER="$saved_teacher"
  else
    TEACHER="$ROOT/$saved_teacher"
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
    die "--wandb=online but wandb is not logged in (uv run wandb login): use --wandb=offline"
  fi
  export WANDB_MODE="$wandb_mode"
elif [ -z "${WANDB_MODE:-}" ] && [ "$wandb_logged_in" -eq 0 ]; then
  export WANDB_MODE=offline
  echo "wandb: not logged in, going offline (the numbers stay in train.log and metrics.jsonl)."
fi

# A new run: its own folder, named after the time with seconds (two starts in the same minute
# used to get the same folder, and train.py refused the second).
if [ "$mode" != "resume" ]; then
  mkdir -p "$RUNS_DIR"
  run_dir="$RUNS_DIR/$(date '+%Y-%m-%d_%H%M%S')"
  [ ! -e "$run_dir" ] || die "$run_dir already exists (two starts in the same second?): start it again"
  mkdir "$run_dir"
fi
log="$run_dir/train.log"
# On a line of its own: the web site reads the run folder of its job from here.
echo "RUN_DIR=$run_dir"

# Teacher and segment checks, then the training. A function, so that with --foreground-log its
# whole output can go through one tee into train.log.
prepare_and_train() {
  # A download of this teacher file still running elsewhere (this script or the curl of
  # supervised/README.md): a second curl on the same file would break it. Matched on the file that
  # curl writes (--output), not on the URL: every model of commaai/openpilot_driving_models is
  # called big_driving_supercombo.onnx, and downloading another one must not stop the training.
  teacher_file_regex="$(basename "$TEACHER" | sed 's/[.]/[.]/g')"
  if pgrep -f "curl .*--output ([^ ]*/)?${teacher_file_regex}([.]part)?( |\$)" > /dev/null; then
    die "the teacher is still downloading (curl running): wait for it to finish and start again"
  fi

  # Teacher: download it when its size is not the one of its source. Hugging Face answers with a
  # redirect: the last Content-Length is the one of the file.
  remote_size="$(curl -sIL "$TEACHER_URL" | tr -d '\r' | sed -n 's/^[Cc]ontent-[Ll]ength: //p' | tail -1)"
  local_size=0
  if [ -f "$TEACHER" ]; then
    local_size="$(stat -c %s "$TEACHER")"
  fi
  if [ -z "$remote_size" ]; then
    [ -f "$TEACHER" ] || die "$TEACHER is missing and $TEACHER_SOURCE does not answer to download it"
    echo "Teacher $teacher_name: $TEACHER_SOURCE does not answer, using $TEACHER as it is ($local_size bytes)."
  elif [ "$local_size" != "$remote_size" ]; then
    # The download goes to $TEACHER.part (a stopped one is finished by the next start) and gets
    # the teacher's name only after the size and sha256 checks: a partial or wrong file never has
    # it, so no later start can find it with the right size and train with it.
    partial="$TEACHER.part"
    partial_size=0
    if [ -f "$partial" ]; then
      partial_size="$(stat -c %s "$partial")"
    fi
    mkdir -p "$(dirname "$TEACHER")"
    # Already complete when the previous start stopped during the sha256: nothing left to download
    # (curl would get "range not satisfiable").
    if [ "$partial_size" != "$remote_size" ]; then
      echo "Teacher $teacher_name: $partial_size of $remote_size bytes, downloading from $TEACHER_SOURCE..."
      curl --fail --location --continue-at - "$TEACHER_URL" --output "$partial"
      partial_size="$(stat -c %s "$partial")"
    fi
    [ "$partial_size" = "$remote_size" ] || die "teacher only partly downloaded ($partial_size of $remote_size bytes), start again"
    # Only after a download, not at every start (0.8-1.8 GB to read).
    echo "Teacher $teacher_name: downloaded, checking the sha256..."
    local_sha256="$(sha256sum "$partial" | cut -d ' ' -f 1)"
    if [ "$local_sha256" != "$TEACHER_SHA256" ]; then
      rm -f "$partial"
      die "teacher downloaded but its sha256 differs ($local_sha256 instead of $TEACHER_SHA256): deleted, start again to download it again"
    fi
    mv "$partial" "$TEACHER"
    echo "Teacher $teacher_name: ok, sha256 matches ($remote_size bytes)."
  else
    echo "Teacher $teacher_name: ok ($local_size bytes)."
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
  [ -z "$broken" ] || die "broken links to the shared folder (COMMA_VIDEOS_DIR), the training would stop:$broken
Check that the shared folder is there (disk mounted) or download again with helpers/fetch_comma_segments.sh."
  [ "$total" -gt 0 ] || die "no segments in data/ (download them with helpers/fetch_comma_segments.sh)"
  [ -z "$missing" ] || die "segments not localized, they would stop the training:$missing
Run helpers/localize_segments.sh first (it moves the unusable ones to data_scartati/)."
  echo "Segments: $total, all localized."

  mode_args=(--mode "$mode")
  if [ "$mode" = "fresh" ]; then
    start="from scratch"
  elif [ "$mode" = "continue" ]; then
    start="from the stable weights of $HF_REPO"
    mode_args+=(--hf-repo "$HF_REPO")
  else
    start="resumed from the last checkpoint"
  fi
  echo "Training ($start): $steps steps, batch $batch_size, workers $workers, teacher $teacher_name, wandb ${WANDB_MODE:-online} -> $run_dir"

  # The log is appended to: --resume keeps writing the train.log of its run.
  # The GPU lock (fd 9) is inherited by the training: with --background it stays held after this
  # script ends, until the training ends.
  cd "$ROOT"
  if [ "$background" -eq 1 ]; then
    # nohup cannot run a shell function: the uv command as an array (helpers/common.sh).
    distill_uv_command
    nohup "${DISTILL_UV_COMMAND[@]}" torchrun --standalone --nproc-per-node=1 -m supervised.train "${mode_args[@]}" \
      --steps "$steps" --batch-size "$batch_size" --workers "$workers" --teacher "$TEACHER_ARG" \
      --output "$run_dir/supercombo.pt" >> "$log" 2>&1 &
    echo "Started in the background (pid $!). Follow it with:  tail -f $log"
    echo "At the end the model is in $run_dir/supercombo.pt"
  elif [ "$foreground_log" -eq 1 ]; then
    # The output of this function already goes through tee into train.log.
    distill_uv_run torchrun --standalone --nproc-per-node=1 -m supervised.train "${mode_args[@]}" \
      --steps "$steps" --batch-size "$batch_size" --workers "$workers" --teacher "$TEACHER_ARG" \
      --output "$run_dir/supercombo.pt" 2>&1
    echo "Done. Model: $run_dir/supercombo.pt"
  else
    distill_uv_run torchrun --standalone --nproc-per-node=1 -m supervised.train "${mode_args[@]}" \
      --steps "$steps" --batch-size "$batch_size" --workers "$workers" --teacher "$TEACHER_ARG" \
      --output "$run_dir/supercombo.pt" 2>&1 | tee -a "$log"
    echo "Done. Model: $run_dir/supercombo.pt"
  fi
}

if [ "$foreground_log" -eq 1 ]; then
  # Everything printed from here on: on the screen (the site's job log) and in train.log.
  # With pipefail the exit code is the one of prepare_and_train, not the one of tee.
  prepare_and_train 2>&1 | tee -a "$log"
else
  prepare_and_train
fi
