#!/usr/bin/env python3
"""Read comma qlogs/rlogs for safe route filtering; identical in both projects.

  python helpers/comma_fetch_log.py dashcam <first-qlog.zst>
  python helpers/comma_fetch_log.py duration <first-qlog.zst> <last-qlog.zst>
  python helpers/comma_fetch_log.py reverse <segment-rlog.zst>

Prefer openpilot's LogReader (openpilot.distill); fall back to pycapnp + zstandard
and the schemas selected by NNSLR_OPENPILOT_ROOT / NNSLR_OPENDBC_ROOT
(sunnypilot.speedlimits). Missing or unreadable data is always 'unknown'.
"""
from __future__ import annotations

import bz2
import os
from pathlib import Path
import sys


def _schema_dirs() -> tuple[Path, Path]:
    value = os.environ.get("NNSLR_OPENPILOT_ROOT")
    if not value:
        raise RuntimeError("set NNSLR_OPENPILOT_ROOT for the schema reader")
    root = Path(value).expanduser()
    cereal = next((root / p for p in ("openpilot/cereal", "cereal")
                   if (root / p / "log.capnp").is_file()), None)
    if cereal is None:
        raise RuntimeError("missing log.capnp in NNSLR_OPENPILOT_ROOT")
    dbc_name = os.environ.get("NNSLR_OPENDBC_ROOT")
    dbc = Path(dbc_name).expanduser() if dbc_name else root
    car = next((dbc / p for p in ("opendbc/opendbc/car", "opendbc/car",
                                  "openpilot/opendbc/opendbc/car", "openpilot/opendbc/car",
                                  "car", ".") if (dbc / p / "car.capnp").is_file()), None)
    if car is None:
        raise RuntimeError("missing car.capnp; set NNSLR_OPENDBC_ROOT")
    return cereal, car


def _messages(path: Path):
    try:
        from openpilot.tools.lib.logreader import LogReader
    except ImportError:
        import capnp
        import zstandard
        cereal, car = _schema_dirs()
        capnp.remove_import_hook()
        schema = capnp.load(str(cereal / "log.capnp"), imports=[str(cereal), str(car)])
        if path.suffix == ".zst":
            # stream_reader handles concatenated zstd frames.
            with path.open("rb") as compressed:
                with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                    data = reader.read()
        elif path.suffix == ".bz2":
            data = bz2.decompress(path.read_bytes())
        else:
            data = path.read_bytes()
        if not data:
            raise ValueError("empty comma log")
        yield from schema.Event.read_multiple_bytes(data)
    else:
        yield from LogReader(str(path))


def _times(path: Path) -> tuple[int, int]:
    first = last = None
    for event in _messages(path):
        stamp = int(event.logMonoTime)
        first = stamp if first is None else min(first, stamp)
        last = stamp if last is None else max(last, stamp)
    if first is None or last is None:
        raise ValueError("no log timestamps")
    return first, last


def inspect(mode: str, paths: list[Path]) -> str:
    if mode == "duration" and len(paths) == 2:
        beginning, _ = _times(paths[0])
        _, ending = _times(paths[1])
        if ending < beginning:
            raise ValueError("route timestamps out of order")
        return str((ending - beginning) / 1e9)

    if mode == "dashcam" and len(paths) == 1:
        for event in _messages(paths[0]):
            if event.which() == "carParams":
                return "dashcam" if event.carParams.dashcamOnly else "normal"
        return "unknown"

    if mode == "reverse" and len(paths) == 1:
        has_state = False
        for event in _messages(paths[0]):
            if event.which() != "carState":
                continue
            has_state = True
            if str(event.carState.gearShifter).lower() == "reverse":
                return "reverse"
        return "forward" if has_state else "unknown"

    raise ValueError("usage: comma_fetch_log.py dashcam|reverse <log> | duration <first> <last>")


def main() -> int:
    if len(sys.argv) < 3:
        print("unknown")
        return 2
    try:
        print(inspect(sys.argv[1], [Path(p) for p in sys.argv[2:]]))
        return 0
    except Exception as exc:
        print(f"comma_fetch_log: {exc}", file=sys.stderr)
        print("unknown")
        return 2


if __name__ == "__main__":
    sys.exit(main())
