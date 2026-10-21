#!/usr/bin/env bash
# Short trial training (10 steps), to see that GPU, videos, teacher and wandb all work
# before a long run. Same checks and output folder as helpers/train_supervised.sh --test.
#
# Usage: helpers/train_supervised_test.sh [--batch-size=<n>] [--workers=<n>]
# While it runs, "nvidia-smi" in another terminal shows the GPU memory it uses.

set -euo pipefail

exec "$(dirname "$0")/train_supervised.sh" --test "$@"
