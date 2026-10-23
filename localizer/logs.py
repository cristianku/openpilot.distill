from collections import defaultdict
import json
from pathlib import Path

import laika.raw_gnss as raw
import numpy as np
from openpilot.tools.lib.framereader import get_video_index

from .errors import UnusableSegment
from .localizer import LocalizationInput

CAMERA_STREAMS = {
  "fcamera": "narrowRoadEncodeIdx",
  "ecamera": "wideRoadEncodeIdx",
}
POSE_STREAM = "deviceMotion"
CALIBRATION_STREAM = "extrinsicsCalibration"
# Largest share of calibration messages flagged not valid in a usable segment (stable_calibration). A driving
# model that drops 7 % of its frames flags about 10 %; the start of a drive about 20 % of a short segment.
MAX_FLAGGED_CALIBRATION = 0.25
FRAME_INFO_METADATA = {
  "encoded_numpy_dtypes": json.dumps({
    f"{camera}/{field}": {"dtype": dtype, "shape": [1]}
    for camera in CAMERA_STREAMS
    for field, dtype in (("codec_name", "<U4"), ("global_prefix", "|S88"))
  }, separators=(",", ":")),
  "schema_version": "1",
}


def _xyz(measurement) -> list[float]:
  return [measurement.x, measurement.y, measurement.z]


def _xyz_std(measurement) -> list[float]:
  return [measurement.xStd, measurement.yStd, measurement.zStd]


def _frame_info(streams, device_type: int, video_dir: Path) -> dict[str, np.ndarray]:
  frame_info = {"device_type": np.asarray(device_type, dtype=np.int64)}
  for camera, stream in CAMERA_STREAMS.items():
    events = sorted(streams[stream], key=lambda event: getattr(event, stream).segmentId)
    indexes = [getattr(event, stream) for event in events]
    segment_ids = np.asarray([index.segmentId for index in indexes])
    if (
      not indexes
      or any(not event.valid or str(index.type) != "fullHEVC" or not index.timestampEof or not index.len
             for event, index in zip(events, indexes, strict=True))
      or not np.array_equal(segment_ids, np.arange(len(indexes)))
    ):
      raise UnusableSegment(f"{stream} must contain a complete fullHEVC segment")

    video = get_video_index(str(video_dir / f"{camera}.hevc"))
    video_index = video["index"]
    video_stream = video["probe"]["streams"][0]
    prefix = video["global_prefix"]
    if len(video_index) != len(indexes) + 1:
      raise UnusableSegment(f"{camera}.hevc does not match {stream}")
    codec = np.asarray([video_stream["codec_name"]], dtype="<U4").view(np.uint8).reshape(1, -1)
    frame_info.update({
      f"{camera}/codec_name": codec.copy(),
      f"{camera}/frame_count": np.asarray([len(indexes)], dtype=np.int64),
      f"{camera}/global_prefix": np.frombuffer(prefix, dtype=np.uint8)[None].copy(),
      f"{camera}/height": np.asarray([video_stream["height"]], dtype=np.int64),
      f"{camera}/index": video_index,
      f"{camera}/t": np.asarray([index.timestampEof * 1e-9 - 0.008 for index in indexes]),
      f"{camera}/width": np.asarray([video_stream["width"]], dtype=np.int64),
    })
  return frame_info


def stable_calibration(events) -> tuple[np.ndarray, np.ndarray]:
  """rpyCalib and wideFromDeviceEuler of a segment whose calibration stays in force all along: from the first
  calibrated message on, every message is calibrated and validBlocks never goes down.

  calibrationd also marks its message not valid (event.valid) whenever one of its inputs is, for example
  cameraOdometry after a frame the driving model dropped, or in the first seconds after the start: the
  calibration it carries is still the one in force. So a flagged message only counts against the segment when
  its content breaks the rule above; its values are never used; and a segment where more than
  MAX_FLAGGED_CALIBRATION of the messages are flagged is refused."""
  calibration_values = wide_values = None
  last_valid_blocks = None
  total = flagged = 0
  for event in events:
    calibration = getattr(event, CALIBRATION_STREAM)
    rpy, wide = list(calibration.rpyCalib), list(calibration.wideFromDeviceEuler)
    calibrated = (
      str(calibration.calStatus) == "calibrated"
      and len(rpy) == len(wide) == 3
      and np.all(np.isfinite(rpy + wide))
    )
    total += 1
    if not event.valid:
      flagged += 1
    if last_valid_blocks is not None and (not calibrated or calibration.validBlocks < last_valid_blocks):
      raise UnusableSegment(f"{CALIBRATION_STREAM} changed during the segment (no longer calibrated, or recalibrating)")
    if calibrated and event.valid:
      calibration_values = np.asarray(rpy, dtype=np.float64)
      wide_values = np.asarray(wide, dtype=np.float64)
      last_valid_blocks = calibration.validBlocks
  if calibration_values is None:
    raise UnusableSegment(f"no stable calibrated {CALIBRATION_STREAM} for the segment")
  if flagged > MAX_FLAGGED_CALIBRATION * total:
    raise UnusableSegment(f"{CALIBRATION_STREAM} flagged not valid in {round(100 * flagged / total)} % of the segment")
  return calibration_values, wide_values


def read_log(messages, video_dir: Path) -> tuple[LocalizationInput, dict[str, np.ndarray]]:
  streams = defaultdict(list)
  for event in messages:
    which = event.which()
    wrong_device = (
      which == "initData" and event.valid and event.initData.version
      and str(event.initData.deviceType) != "mici"
    )
    wrong_gps = which == "gpsLocation" and event.valid and event.gpsLocation.hasFix
    if wrong_device:
      raise UnusableSegment(
        f"only comma four (mici) segments are supported, this one is {event.initData.deviceType}"
      )
    if wrong_gps:
      raise UnusableSegment(
        "GPS comes from gpsLocation, not gpsLocationExternal: only comma four (mici) segments are supported"
      )
    streams[which].append(event)

  device_types = {
    str(event.initData.deviceType)
    for event in streams["initData"]
    if event.valid and event.initData.version
  }
  gps_events = [
    event for event in streams["gpsLocationExternal"]
    if event.valid and event.gpsLocationExternal.hasFix
  ]
  if not device_types:
    raise UnusableSegment("no valid initData in the log")
  if not streams["gpsLocationExternal"]:
    raise UnusableSegment("no gpsLocationExternal messages in the log")
  if not gps_events:
    raise UnusableSegment(
      f"no GPS fix in the segment ({len(streams['gpsLocationExternal'])} gpsLocationExternal messages, "
      "none valid with hasFix)"
    )

  missing = [
    name for name in (POSE_STREAM, CALIBRATION_STREAM, *CAMERA_STREAMS.values()) if not streams[name]
  ]
  if missing:
    raise UnusableSegment(f"missing required log streams: {', '.join(missing)}")

  device_motion = []
  for event in streams[POSE_STREAM]:
    if not event.valid:
      continue
    pose = getattr(event, POSE_STREAM)
    if pose.orientationNED.valid and pose.velocityDevice.valid:
      event_t = event.logMonoTime * 1e-9
      device_motion.append((
        pose.timestamp * 1e-9 if pose.timestamp else event_t,
        _xyz(pose.orientationNED),
        _xyz(pose.velocityDevice),
        _xyz_std(pose.velocityDevice),
        _xyz(pose.accelerationDevice),
        _xyz(pose.angularVelocityDevice),
      ))
  if not device_motion:
    raise UnusableSegment(f"no valid {POSE_STREAM} stream found in the log")

  gps_positions = []
  for event in gps_events:
    gps = event.gpsLocationExternal
    position = [gps.latitude, gps.longitude, gps.altitude]
    if not np.all(np.isfinite(position)):
      raise UnusableSegment("gpsLocationExternal requires a finite position")
    gps_positions.append(position)

  raw_gnss_t = []
  raw_gnss_measurements = []
  for event in streams["ubloxGnss"]:
    if not event.valid or event.ubloxGnss.which() != "measurementReport":
      continue
    for measurement in raw.read_raw_ublox(event.ubloxGnss.measurementReport):
      raw_gnss_t.append(event.logMonoTime * 1e-9)
      raw_gnss_measurements.append(raw.array_from_normal_meas(measurement))
  if not raw_gnss_measurements:
    raise UnusableSegment("no raw UBlox measurement reports found in the log")

  calibration_values, wide_values = stable_calibration(streams[CALIBRATION_STREAM])

  device_type = next(
    event.initData.deviceType.raw
    for event in streams["initData"] if event.valid and event.initData.version
  )
  frame_info = _frame_info(streams, device_type, video_dir)

  car_state = np.asarray([
    (event.logMonoTime * 1e-9, event.carState.vEgo)
    for event in streams["carState"] if event.valid
  ]).reshape(-1, 2)

  return LocalizationInput(
    pose_t=np.asarray([row[0] for row in device_motion]),
    orientation=np.asarray([row[1] for row in device_motion]),
    velocity=np.asarray([row[2] for row in device_motion]),
    velocity_std=np.asarray([row[3] for row in device_motion]),
    acceleration_device=np.asarray([row[4] for row in device_motion]),
    angular_velocity_device=np.asarray([row[5] for row in device_motion]),
    gps_seed_geodetic=np.asarray(gps_positions[0]),
    raw_gnss_t=np.asarray(raw_gnss_t),
    raw_gnss_measurements=np.asarray(raw_gnss_measurements),
    calibration=calibration_values,
    wide_from_device_euler=wide_values,
    frame_t=frame_info["fcamera/t"],
    car_state=car_state,
  ), frame_info
