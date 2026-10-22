"""Export a trained model (supercombo.pt) to the driving_supercombo.onnx that openpilot runs on the comma.

    python -m tasks.export [--host <ip|name>] [--run <run> | <supercombo.pt>]

Without --run and without a file: the newest outputs/supervised/*/supercombo.pt. The ONNX is written
next to the .pt: outputs/supervised/<run>/driving_supercombo.onnx.

The export needs the comma's ORIGINAL driving_supercombo.onnx as reference (inputs and output layout
of the openpilot version on the comma). It is copied from the comma into
models/comma_driving_supercombo.onnx and reused from there when the comma is off. The original is the
file whose sha256 is the one in the comma's openpilot git: in its place, or the backup
(tasks.comma.BACKUP) while a trained model is installed in its place (tasks.install). The copy goes to
a .new file first and replaces the kept one only when its sha256 is the original's: a trained model on
the comma never becomes the reference.

The reference and the export (update_reference, export_onnx) are shared with tasks.install and
tasks.publish, and so are the model choice (resolve_model) and the run of a Python module (run_module).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from tasks.comma import BACKUP, MODEL_INFO_SCRIPT, Comma, comma_host, original_model_source
from tasks.common import REPO, TaskError, load_env, run_main, say
from tasks.teacher import sha256_of
from tasks.train import RUN_NAME_RE, RUNS_DIR

REFERENCE = REPO / 'models' / 'comma_driving_supercombo.onnx'
ONNX_NAME = 'driving_supercombo.onnx'
MISSING_ORIGINAL = ('the original model of this version is not on the comma (neither in its place nor in '
                    + BACKUP + ')')


def add_model_arguments(parser: argparse.ArgumentParser, file_help: str) -> None:
    """--host, --run and the optional model file: the arguments of the tasks that take a trained model."""
    parser.add_argument('--host', default=None,
                        help='IP or ssh name of the comma ("comma@" added when it has no user); default COMMA_HOST')
    parser.add_argument('--run', default=None, help='the run of outputs/supervised/ (its supercombo.pt)')
    parser.add_argument('model', nargs='?', default=None,
                        help=file_help + ' (without --run and without a file: the newest supercombo.pt)')


def check_model_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.host == '':
        parser.error('--host needs an IP or a name')
    if args.run is not None and args.model is not None:
        parser.error('give --run or a file, not both')
    if args.run is not None and not RUN_NAME_RE.fullmatch(args.run):
        parser.error('invalid --run: ' + args.run)


def latest_model() -> Path:
    """The newest outputs/supervised/*/supercombo.pt (by modification time)."""
    models = [path for path in RUNS_DIR.glob('*/supercombo.pt') if path.is_file()]
    if not models:
        raise TaskError('no finished run in outputs/supervised/ (no supercombo.pt): finish a training first '
                        '(Training)')
    return max(models, key=lambda path: path.stat().st_mtime)


def resolve_model(run: str | None, path: str | None) -> Path:
    """The absolute path of the model to use: the run's supercombo.pt, the file given, or the newest one."""
    if run is not None:
        model = RUNS_DIR / run / 'supercombo.pt'
        if not RUN_NAME_RE.fullmatch(run) or not model.is_file():
            raise TaskError('run ' + run + ' has no supercombo.pt (it does not exist or has not finished)')
    elif path is not None:
        model = Path(path)
        if not model.is_file():
            raise TaskError(path + ' does not exist')
    else:
        model = latest_model()
    return Path(os.path.abspath(model))


def run_module(module: str, *arguments: str, failure: str) -> None:
    """python -m <module> <arguments> from the repo root, its output in the job log; TaskError(failure) when
    it fails."""
    environment = dict(os.environ)
    environment['PYTHONUNBUFFERED'] = '1'
    code = subprocess.run([sys.executable, '-m', module, *arguments], cwd=REPO, env=environment,
                          stdin=subprocess.DEVNULL, check=False).returncode
    if code != 0:
        raise TaskError(failure + ' (code ' + str(code) + ')')


def export_onnx(checkpoint: Path, reference: Path, output: Path) -> None:
    run_module('helpers.export_onnx', str(checkpoint), '--reference', str(reference), '--output', str(output),
               failure='ONNX export failed (error above)')


def read_model_info(comma: Comma) -> dict:
    """reference, original, current and backup of tasks.comma.MODEL_INFO_SCRIPT, '' when missing. Unlike
    tasks.comma.model_info it does not refuse a checkout without the model: the kept reference is still
    usable then."""
    try:
        info = comma.key_values(MODEL_INFO_SCRIPT, BACKUP)
    except TaskError as exc:
        raise TaskError('cannot read /data/openpilot on the comma (' + str(exc) + ')') from exc
    for key in ('reference', 'original', 'current', 'backup'):
        info.setdefault(key, '')
    return info


def update_reference(comma: Comma, info: dict, *, strict: bool, reference: Path | None = None) -> Path:
    """The original model of the comma's openpilot version in models/comma_driving_supercombo.onnx.

    info: the reference/original/current/backup values read from the comma (read_model_info, or a task's
    own info script with the same keys). The kept copy is used as it is when its sha256 is the original's;
    otherwise the original is copied from the comma into <reference>.new and kept only when its sha256 is
    the original's. When the comma has the original in neither place: strict refuses, otherwise the kept
    copy is used with a warning (it may belong to an older openpilot version).
    """
    if reference is None:
        reference = REFERENCE
    reference.parent.mkdir(parents=True, exist_ok=True)
    original = info.get('original', '')
    source = original_model_source(info)
    cached = sha256_of(reference) if reference.is_file() else ''
    if original and cached == original:
        say('Reference: ' + str(reference) + ', already the same as the comma\'s original')
        return reference
    if source:
        # Into a temporary file first, kept only when it is really the original.
        partial = reference.with_name(reference.name + '.new')
        comma.pull(source, partial)
        copied = sha256_of(partial)
        if copied != original:
            partial.unlink(missing_ok=True)
            raise TaskError('the copy of ' + source + ' does not have the original\'s sha256 (' + copied
                            + ' instead of ' + original + '): ' + str(reference) + ' left untouched')
        os.replace(partial, reference)
        say('Reference: the comma\'s original driving_supercombo.onnx (' + source + ')')
        return reference
    if strict:
        raise TaskError(MISSING_ORIGINAL)
    if reference.is_file():
        say('WARNING: ' + MISSING_ORIGINAL + ': using the reference copied earlier (' + str(reference) + ')')
        return reference
    raise TaskError(MISSING_ORIGINAL + ' and there is no ' + str(reference))


def reference_for_export(comma: Comma) -> Path:
    """The reference of a plain export: from the comma when it answers, else the kept copy."""
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    if comma.reachable():
        return update_reference(comma, read_model_info(comma), strict=False)
    if REFERENCE.is_file():
        say('Comma not reachable: using the reference copied earlier (' + str(REFERENCE) + ')')
        return REFERENCE
    raise TaskError('comma not reachable and no ' + str(REFERENCE) + ': turn the comma on (it needs an ssh key '
                    'registered on its GitHub account)')


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser, 'the supercombo.pt to export')
    args = parser.parse_args(arguments)
    check_model_arguments(parser, args)
    return args


def export(args: argparse.Namespace) -> int:
    load_env()
    comma = Comma(comma_host(args.host))
    checkpoint = resolve_model(args.run, args.model)
    output = checkpoint.parent / ONNX_NAME
    reference = reference_for_export(comma)
    say('Export: ' + str(checkpoint))
    export_onnx(checkpoint, reference, output)
    say('Done: ' + str(output))
    return 0


def main() -> int:
    return export(parse_args())


if __name__ == '__main__':
    run_main(main)
