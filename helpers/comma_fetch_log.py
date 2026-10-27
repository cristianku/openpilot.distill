#!/usr/bin/env python3
"""Read comma qlogs/rlogs for safe route filtering; identical in both projects.

  python helpers/comma_fetch_log.py dashcam <first-qlog.zst>
  python helpers/comma_fetch_log.py duration <first-qlog.zst> <last-qlog.zst>
  python helpers/comma_fetch_log.py reverse <segment-rlog.zst>

Prefer openpilot's LogReader (openpilot.distill); fall back to pycapnp + zstandard
and the schemas selected by NNSLR_OPENPILOT_ROOT / NNSLR_OPENDBC_ROOT, or without
them the pinned schema of the author's comma, downloaded once into
~/.cache/comma-fetch-log and checked by SHA-256 (the same pins as openpilot.perception).
Missing or unreadable data is always 'unknown'.
"""
from __future__ import annotations

import bz2
import hashlib
import os
from pathlib import Path
import sys
import urllib.request

SCHEMA_COMMIT = "1b6a472ae61d2d61866902cb48a5b3ab8bd11d07"
CAR_COMMIT = "6810f033b9e60738365cfecb8504c980dfaad137"
_RAW = "https://raw.githubusercontent.com"
# (file in the cache, URL, SHA-256): log.capnp imports "/car.capnp" and "/include/c++.capnp" from the cache folder.
SCHEMA_FILES = (
    ("log.capnp", f"{_RAW}/cristianku/openpilot/{SCHEMA_COMMIT}/openpilot/cereal/log.capnp",
     "be3b81471c18ad7ffbac3890042cd93823c69832fd88130ac9400bd02f8b049b"),
    ("custom.capnp", f"{_RAW}/cristianku/openpilot/{SCHEMA_COMMIT}/openpilot/cereal/custom.capnp",
     "439f46f148c6e72571bdc232e29b42ab6bb361c304cff9eaba0afeab66885f19"),
    ("deprecated.capnp", f"{_RAW}/cristianku/openpilot/{SCHEMA_COMMIT}/openpilot/cereal/deprecated.capnp",
     "b3aa5c13de40001ed7376cef858500fb003f7698f96131cd0db18d878fa1af0d"),
    ("include/c++.capnp", f"{_RAW}/cristianku/openpilot/{SCHEMA_COMMIT}/openpilot/cereal/include/c++.capnp",
     "fb306076cd38c27af1aed20fac6395e9a46fbe5b5df6a248b5f1b6845a079c44"),
    ("car.capnp", f"{_RAW}/cristianku/opendbc/{CAR_COMMIT}/opendbc/car/car.capnp",
     "b4a6502704b0315b8277fb7fe125c9780ec3aae47d4ba3dfa9b3d665160b545a"),
)


def _pinned_schema(fetch=None) -> Path:
    """The cache folder with the pinned schema files, downloaded and checked when missing."""
    folder = Path.home() / ".cache" / "comma-fetch-log" / f"{SCHEMA_COMMIT[:12]}-{CAR_COMMIT[:12]}"
    fetch = fetch or (lambda url: urllib.request.urlopen(url, timeout=60).read())
    for name, url, sha in SCHEMA_FILES:
        path = folder / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == sha:
            continue
        data = fetch(url)
        if hashlib.sha256(data).hexdigest() != sha:
            raise RuntimeError(f"{name}: the downloaded schema does not match its pinned SHA-256")
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        part.write_bytes(data)
        part.replace(path)
    return folder


def _schema_dirs() -> tuple[Path, Path]:
    value = os.environ.get("NNSLR_OPENPILOT_ROOT")
    if not value:
        folder = _pinned_schema()
        return folder, folder
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
