"""What a person needs to recognise a drive: when it was, how fast, how curvy, the shape of the road and
a picture of it.

For every localized segment of data/:
  start        unix time of the first frame, from the GPS clock in the rlog (gpsLocationExternal)
  duration_s   span of the video frames
  avg_kmh, max_kmh, distance_km   from the localizer velocities and positions
  turn_deg     total heading change of the path (how curvy the segment is)
  trace        the path seen from above, at most TRACE_POINTS [x, y] in 0..1 (north up)
The numbers are computed once and kept in outputs/web/drives/<segment>.json, computed again when
localizer.safetensors changes. A background thread fills them in (about half a second per segment),
so the first answers can be partial: "pending" says how many are still to do.

Thumbnails: outputs/web/thumbs/<segment>.jpg, one road-camera frame scaled to THUMB_WIDTH, made with
ffmpeg on first request.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import threading
from pathlib import Path

from web.common import SEGMENT_RE, inside, is_heldout, route_of

CACHE_VERSION = 1
TRACE_POINTS = 40
THUMB_WIDTH = 384
# Frame of the thumbnail: a little after the start, past any keyframe artefact.
THUMB_FRAME = 20
# Earth radius for the local east/north plane of a segment (a few km: flat is fine).
EARTH_RADIUS_M = 6_371_000.0


def _gps_offset(rlog: Path) -> float | None:
    """unix time minus the log's monotonic time, from the first GPS fix of the rlog."""
    from openpilot.tools.lib.logreader import LogReader

    for message in LogReader(str(rlog)):
        if message.which() != 'gpsLocationExternal':
            continue
        gps = message.gpsLocationExternal
        if gps.hasFix and gps.unixTimestampMillis > 0:
            return gps.unixTimestampMillis / 1000.0 - message.logMonoTime / 1e9
    return None


def compute(folder: Path) -> dict:
    """The drive numbers of one localized segment folder."""
    import numpy as np
    from safetensors.numpy import load_file

    localizer = load_file(str(folder / 'localizer.safetensors'))
    frame_t = localizer['frame_t']
    states = localizer['frame_states']
    position = states[:, 0:3]
    speed = np.linalg.norm(states[:, 7:10], axis=1)

    # Local east/north plane at the first position (geocentric latitude: close enough for a picture).
    x, y, z = position[0]
    lon = math.atan2(y, x)
    lat = math.atan2(z, math.hypot(x, y))
    east_axis = np.array([-math.sin(lon), math.cos(lon), 0.0])
    north_axis = np.array([-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)])
    relative = position - position[0]
    east = relative @ east_axis
    north = relative @ north_axis

    steps = np.hypot(np.diff(east), np.diff(north))
    distance = float(steps.sum())
    # Heading changes between points at least 3 m apart (standing still has no heading).
    turn = 0.0
    previous = None
    last_e, last_n = east[0], north[0]
    for e, n in zip(east[1:], north[1:], strict=True):
        if math.hypot(e - last_e, n - last_n) < 3.0:
            continue
        heading = math.atan2(n - last_n, e - last_e)
        if previous is not None:
            change = (heading - previous + math.pi) % (2 * math.pi) - math.pi
            turn += abs(change)
        previous = heading
        last_e, last_n = e, n

    pick = np.linspace(0, len(east) - 1, min(TRACE_POINTS, len(east))).astype(int)
    span = max(float(np.ptp(east)), float(np.ptp(north)), 1.0)
    trace = [[round(float((east[i] - east.min()) / span), 3), round(float((north[i] - north.min()) / span), 3)]
             for i in pick]

    start = None
    rlog = folder / 'rlog.zst'
    if rlog.exists():
        try:
            offset = _gps_offset(rlog)
        except Exception:  # an unreadable log: the drive is shown without a clock time
            offset = None
        if offset is not None:
            start = round(float(frame_t[0]) + offset, 1)

    return {'version': CACHE_VERSION, 'start': start, 'duration_s': round(float(frame_t[-1] - frame_t[0]), 1),
            'avg_kmh': round(float(speed.mean()) * 3.6, 1), 'max_kmh': round(float(speed.max()) * 3.6, 1),
            'distance_km': round(distance / 1000, 2), 'turn_deg': round(math.degrees(turn)),
            'lat': round(math.degrees(lat), 4), 'lon': round(math.degrees(lon), 4), 'trace': trace}


class DriveIndex:
    def __init__(self, repo: Path):
        self.repo = repo
        self.data_dir = repo / 'data'
        self.cache_dir = repo / 'outputs' / 'web' / 'drives'
        self.thumb_dir = repo / 'outputs' / 'web' / 'thumbs'
        self._lock = threading.Lock()
        self._worker = None
        self._thumb_lock = threading.Lock()

    def _folders(self) -> list[Path]:
        if not self.data_dir.is_dir():
            return []
        folders = []
        for entry in sorted(self.data_dir.iterdir()):
            if SEGMENT_RE.fullmatch(entry.name) and (entry / 'localizer.safetensors').is_file():
                folders.append(entry)
        return folders

    def _cached(self, folder: Path) -> dict | None:
        path = self.cache_dir / (folder.name + '.json')
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            source = (folder / 'localizer.safetensors').stat().st_mtime
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get('version') != CACHE_VERSION or data.get('source_mtime') != source:
            return None
        return data

    def _fill(self, folders: list[Path]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for folder in folders:
            if self._cached(folder) is not None:
                continue
            try:
                data = compute(folder)
                data['source_mtime'] = (folder / 'localizer.safetensors').stat().st_mtime
            except Exception as exc:  # a broken segment must not stop the others
                print('drive info of ' + folder.name + ' failed: ' + str(exc), flush=True)
                continue
            temporary = self.cache_dir / (folder.name + '.json.tmp')
            temporary.write_text(json.dumps(data), encoding='utf-8')
            os.replace(temporary, self.cache_dir / (folder.name + '.json'))

    def drives(self) -> dict:
        """Every localized segment with its numbers (when computed already)."""
        folders = self._folders()
        result = []
        missing = []
        for folder in folders:
            info = self._cached(folder)
            row = {'segment': folder.name, 'route': route_of(folder.name), 'number': int(folder.name[18:20]),
                   'heldout': is_heldout(folder.name)}
            if info is None:
                missing.append(folder)
            else:
                for key in ('start', 'duration_s', 'avg_kmh', 'max_kmh', 'distance_km', 'turn_deg', 'trace'):
                    row[key] = info.get(key)
            result.append(row)
        if missing:
            with self._lock:
                if self._worker is None or not self._worker.is_alive():
                    self._worker = threading.Thread(target=self._fill, args=(missing,), daemon=True)
                    self._worker.start()
        return {'drives': result, 'pending': len(missing)}

    def thumbnail(self, segment: str) -> Path | None:
        """The JPEG of a segment, made on first request; None when it cannot be made."""
        if not SEGMENT_RE.fullmatch(segment):
            return None
        folder = self.data_dir / segment
        video = folder / 'fcamera.hevc'
        if not video.exists() or not inside(self.data_dir, folder):
            return None
        target = self.thumb_dir / (segment + '.jpg')
        if target.is_file():
            return target
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg is None:
            return None
        with self._thumb_lock:
            if target.is_file():
                return target
            self.thumb_dir.mkdir(parents=True, exist_ok=True)
            temporary = self.thumb_dir / (segment + '.tmp.jpg')
            command = [ffmpeg, '-nostdin', '-v', 'error', '-y', '-f', 'hevc', '-i', str(video),
                       '-vf', 'select=eq(n\\,' + str(THUMB_FRAME) + '),scale=' + str(THUMB_WIDTH) + ':-2',
                       '-frames:v', '1', '-q:v', '4', str(temporary)]
            try:
                subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, timeout=60, check=True)
            except (OSError, subprocess.SubprocessError):
                temporary.unlink(missing_ok=True)
                return None
            os.replace(temporary, target)
        return target
