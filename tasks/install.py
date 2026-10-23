"""Try a trained model on the comma: export supercombo.pt to ONNX, check that the comma can compile it,
put it in place of openpilot's driving model (driving_supercombo.onnx) and reboot the comma.
--restore puts the original back.

    python -m tasks.install [--host <ip|name>] [--no-reboot] [--run <run> | <supercombo.pt>]
    python -m tasks.install [--host <ip|name>] [--no-reboot] --restore

Without --run and without a file: the newest outputs/supervised/*/supercombo.pt.

What it does:
  1. checks that the comma is reachable, offroad, and (sunnypilot) on the Default model of the Models
     menu: a model chosen there runs in place of openpilot's one
  2. keeps a copy of the original model on the comma (tasks.comma.BACKUP), taken while the comma still
     has the original of its openpilot version (same sha256 as in git)
  3. exports the .pt with helpers.export_onnx, with the original as reference (tasks.export: the
     sha256-checked copy in models/comma_driving_supercombo.onnx); the ONNX is written next to the .pt
  4. compiles it on the comma the way the openpilot build does (tinygrad compile_onnx.py, flags from
     selfdrive/modeld/SConscript), in a temporary folder: if it fails nothing is changed. The output goes
     to <run>/comma_compile.log; the last lines show the milliseconds per frame: they must stay well
     under 50.
  5. puts the ONNX in place of driving_supercombo.onnx (a copy to a .new name, then one rename) and
     reboots: at boot the build compiles it again into driving_tinygrad.pkl, so the first start takes a
     few minutes more.

The software update from the comma screen also brings the original back.
With dashcamOnly the model does not drive: it only logs its predictions.

--restore: puts back the original saved at step 2 and reboots.
--no-reboot: does not reboot (the model is used from the next start).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tasks.comma import BACKUP, REMOTE_OPENPILOT, Comma, comma_host
from tasks.common import TaskError, load_env, run_main, say
from tasks.export import (ONNX_NAME, add_model_arguments, check_model_arguments, export_onnx, resolve_model,
                          update_reference)

BUILD_DIR = '/data/openpilot_distill/build'
STEPS = 5

# onroad state, selected Models entry, where the model is (path asked to git, the layout changes between
# versions), the tinygrad flags of the openpilot build, the sha256 of the original model of this
# openpilot version (from git, the onnx is in LFS), of the current one and of the backup, and the
# comma's python. $1 = the backup path. "reference" is the model path, as in tasks.comma.MODEL_INFO_SCRIPT.
INSTALL_INFO_SCRIPT = r'''
backup="$1"
cd /data/openpilot || exit 1
echo "onroad=$(cat /data/params/d/IsOnroad 2>/dev/null)"
echo "active_bundle=$(cat /data/params/d/ModelManager_ActiveBundle 2>/dev/null | head -c 200)"
reference="$(git ls-files '*selfdrive/modeld/models/driving_supercombo.onnx' | head -1)"
echo "reference=$reference"
if [ -z "$reference" ]; then
  exit 0
fi
sconscript="$(dirname "$(dirname "$reference")")/SConscript"
echo "tg_flags=$(sed -n "s/^ *tg_flags = '\([^']*\)'.*/\1/p" "$sconscript" | head -1)"
size="$(git cat-file -s "HEAD:$reference")"
if [ "$size" -lt 1024 ]; then
  echo "original=$(git cat-file -p "HEAD:$reference" | sed -n 's/^oid sha256://p')"
else
  echo "original=$(git cat-file -p "HEAD:$reference" | sha256sum | cut -d' ' -f1)"
fi
echo "current=$(sha256sum "$reference" | cut -d' ' -f1)"
if [ -f "$backup" ]; then
  echo "backup=$(sha256sum "$backup" | cut -d' ' -f1)"
fi
if [ -x /usr/local/venv/bin/python3 ]; then
  echo "python=/usr/local/venv/bin/python3"
fi
'''

INFO_KEYS = ('onroad', 'active_bundle', 'reference', 'tg_flags', 'original', 'current', 'backup', 'python')

# $1 = source, $2 = destination.
COPY_SCRIPT = r'''
cp "$1" "$2"
'''

# A file over the model through a temporary name, so the swap is one rename. $1 = source, $2 = model.
PUT_MODEL_SCRIPT = r'''
cp "$1" "$2.new" && mv "$2.new" "$2"
'''

# The compile of the openpilot build, in the build folder (compile_onnx writes a temporary file in the
# current folder). $1 = build folder, $2 = openpilot checkout, $3 = tg_flags (VAR=value words, split
# on purpose), $4 = python. stderr goes with stdout, in order, into the log; stdin is the script
# itself (bash -s), so the compiler gets /dev/null.
COMPILE_CHECK_SCRIPT = r'''
exec 2>&1
set -f
build="$1"
openpilot="$2"
flags="$3"
python="$4"
cd "$build" || exit 1
env $flags PYTHONPATH="$openpilot/tinygrad_repo:$openpilot" \
  "$python" "$openpilot/tinygrad_repo/examples/openpilot/compile_onnx.py" \
  driving_supercombo.onnx driving_tinygrad.pkl --out-of-band --benchmark-runs 1 </dev/null
'''


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser, 'the supercombo.pt to install')
    parser.add_argument('--restore', action='store_true', help='puts the original model back')
    parser.add_argument('--no-reboot', action='store_true', help='does not reboot the comma')
    args = parser.parse_args(arguments)
    check_model_arguments(parser, args)
    if args.restore and (args.run is not None or args.model is not None):
        parser.error('--restore takes no model')
    return args


def read_install_info(comma: Comma) -> dict:
    try:
        info = comma.key_values(INSTALL_INFO_SCRIPT, BACKUP)
    except TaskError as exc:
        raise TaskError('cannot read /data/openpilot on the comma (' + str(exc) + ')') from exc
    for key in INFO_KEYS:
        info.setdefault(key, '')
    return info


def put_model(comma: Comma, source: str, info: dict) -> None:
    comma.script(PUT_MODEL_SCRIPT, source, REMOTE_OPENPILOT + '/' + info['reference'])


def reboot(comma: Comma, do_reboot: bool) -> None:
    if not do_reboot:
        say('No reboot: the comma uses the model from its next start')
        return
    say('Rebooting the comma (the first start compiles the model again, a few minutes more)')
    try:
        # The connection drops while the comma goes down: its exit code means nothing.
        comma.run('sudo reboot', check=False, timeout=60)
    except TaskError:
        pass


def restore(comma: Comma, info: dict, do_reboot: bool) -> int:
    if info['current'] == info['original']:
        say('The comma already has the original model.')
        return 0
    if info['backup'] != info['original']:
        raise TaskError('the backup ' + BACKUP + ' is missing or is not the original of this version: put the '
                        'original back with a software update from the comma screen')
    put_model(comma, BACKUP, info)
    say('Original model put back.')
    reboot(comma, do_reboot)
    return 0


def compile_check(comma: Comma, student: Path, info: dict, log: Path) -> None:
    """Step 4: compiles the ONNX on the comma in BUILD_DIR; refuses (build folder removed) when it fails."""
    comma.run('rm -rf ' + BUILD_DIR + ' && mkdir -p ' + BUILD_DIR)
    comma.push(student, BUILD_DIR + '/' + ONNX_NAME)
    result = comma.script(COMPILE_CHECK_SCRIPT, BUILD_DIR, REMOTE_OPENPILOT, info['tg_flags'], info['python'],
                          check=False)
    output = (result.stdout or '') + (result.stderr or '')
    log.write_text(output, encoding='utf-8')
    lines = output.splitlines()
    if result.returncode != 0:
        for line in lines[-20:]:
            say(line)
        comma.run('rm -rf ' + BUILD_DIR, check=False)
        raise TaskError('the comma cannot compile the model: nothing was changed (full log: '
                        + str(log) + ')')
    for line in [line for line in lines if line.endswith(' ms')][-3:]:
        say(line)


def install(args: argparse.Namespace, comma: Comma) -> int:
    comma.require()
    info = read_install_info(comma)
    if info['onroad'] == '1':
        raise TaskError('the comma is onroad: turn the car off and try again')
    if not info['reference']:
        raise TaskError('driving_supercombo.onnx not found on the comma')
    if not info['original']:
        raise TaskError('cannot read the original model\'s sha256 from git')

    if args.restore:
        return restore(comma, info, not args.no_reboot)

    if info['active_bundle'] and info['active_bundle'] != '{}':
        raise TaskError('a model is chosen in the sunnypilot Models menu: set it to Default, otherwise yours is '
                        'not used')
    if not info['tg_flags']:
        raise TaskError('tinygrad flags (tg_flags) not found in the comma\'s SConscript')
    if not info['python']:
        raise TaskError('/usr/local/venv/bin/python3 is not on the comma')

    checkpoint = resolve_model(args.run, args.model)
    run_dir = checkpoint.parent
    student = run_dir / ONNX_NAME
    say('Model: ' + str(checkpoint))
    say('PROGRESS 1/' + str(STEPS))

    # 2. Backup of the original, refreshed while the comma has it (it changes with openpilot updates).
    if info['current'] == info['original']:
        comma.script(COPY_SCRIPT, REMOTE_OPENPILOT + '/' + info['reference'], BACKUP)
        say('== 2. original saved in ' + BACKUP)
    elif info['backup'] == info['original']:
        say('== 2. one of your models is already on the comma, the original is in ' + BACKUP)
    else:
        raise TaskError('the comma has a model other than the original and no valid backup: '
                        'put the original back with a software update from the comma screen')
    say('PROGRESS 2/' + str(STEPS))

    # 3. Export, with the original as reference for inputs and output layout.
    reference = update_reference(comma, info, strict=True)
    say('== 3. export -> ' + str(student))
    export_onnx(checkpoint, reference, student)
    say('PROGRESS 3/' + str(STEPS))

    # 4. Compile check on the comma, same command as the openpilot build.
    say('== 4. test compile on the comma (a few minutes)')
    compile_check(comma, student, info, run_dir / 'comma_compile.log')
    say('PROGRESS 4/' + str(STEPS))

    # 5. In place of the original, then reboot.
    put_model(comma, BUILD_DIR + '/' + ONNX_NAME, info)
    comma.run('rm -rf ' + BUILD_DIR)
    say('== 5. installed: to go back to the original, use "Put the original model back" in Publish')
    say('PROGRESS 5/' + str(STEPS))
    reboot(comma, not args.no_reboot)
    return 0


def main() -> int:
    args = parse_args()
    load_env()
    return install(args, Comma(comma_host(args.host)))


if __name__ == '__main__':
    run_main(main)
