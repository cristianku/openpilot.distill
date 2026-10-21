"""Offline localizer error classification tests (no device or GNSS downloads)."""
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from openpilot.cereal import log

from localizer import gnss, logs
from localizer.errors import UnusableSegment


def event(service, *, valid=True, **fields):
    message = log.Event.new_message()
    message.valid = valid
    payload = message.init(service)
    for name, value in fields.items():
        setattr(payload, name, value)
    return message


def init_data(device="mici", **kwargs):
    return event("initData", deviceType=device, version="test", **kwargs)


def rows(count, prn=1):
    values = np.zeros((count, 10))
    values[:, gnss.GNSSMeasurement.PRN] = np.arange(prn, prn + count)
    return values


class ReadLogErrorTests(unittest.TestCase):
    def assert_error(self, messages, expected):
        with self.assertRaises(UnusableSegment) as raised:
            logs.read_log(iter(messages), Path("unused"))
        self.assertEqual(str(raised.exception), expected)

    def test_wrong_device_exits_without_reading_rest(self):
        def messages():
            yield init_data("tici")
            self.fail("read_log must reject unsupported devices immediately")
        self.assert_error(messages(), "only comma four (mici) segments are supported, this one is tici")

    def test_internal_gps_exits_without_reading_rest(self):
        def messages():
            yield event("gpsLocation", hasFix=True)
            self.fail("read_log must reject internal GPS immediately")
        self.assert_error(messages(), "GPS comes from gpsLocation, not gpsLocationExternal: "
                          "only comma four (mici) segments are supported")

    def test_no_valid_init_data(self):
        for messages in ([], [init_data(valid=False)], [event("initData", deviceType="mici")]):
            with self.subTest(messages=messages):
                self.assert_error(messages, "no valid initData in the log")

    def test_no_external_gps_messages(self):
        self.assert_error([init_data()], "no gpsLocationExternal messages in the log")

    def test_external_gps_without_valid_fix_counts_all_messages(self):
        self.assert_error([init_data(), event("gpsLocationExternal", hasFix=False),
                           event("gpsLocationExternal", valid=False, hasFix=True)],
                          "no GPS fix in the segment (2 gpsLocationExternal messages, none valid with hasFix)")

    def test_valid_device_and_fix_reach_existing_stream_validation(self):
        self.assert_error([init_data(), event("gpsLocationExternal", hasFix=True)],
                          "missing required log streams: deviceMotion, extrinsicsCalibration, "
                          "narrowRoadEncodeIdx, wideRoadEncodeIdx")


class GnssErrorTests(unittest.TestCase):
    def test_insufficient_raw_gps_is_unusable_before_downloads(self):
        # Plenty of GLONASS rows must not hide a shortage of raw GPS satellites.
        epochs = [np.concatenate([rows(3), rows(5, 65)]), rows(2)]
        with patch.object(gnss, "_gnss_epochs", return_value=(np.array([0., 1.]), epochs)), \
             patch.object(gnss, "_AstroDog", side_effect=AssertionError("must not download")):
            with self.assertRaises(UnusableSegment) as raised:
                gnss.localize_raw_gnss(np.empty(0), np.empty((0, 10)), np.zeros(3))
        self.assertEqual(str(raised.exception), "fewer than four GPS satellites in every epoch of the segment")

    def test_satellites_lost_during_correction_remain_retryable(self):
        with patch.object(gnss, "_normal_measurements", return_value=[]), \
             patch.object(gnss.raw, "process_measurements", return_value=[]), \
             patch.object(gnss.raw, "correct_measurements", return_value=[]):
            with self.assertRaises(RuntimeError) as raised:
                gnss._correct_epochs(object(), np.zeros(3), [rows(4)])
        self.assertNotIsInstance(raised.exception, UnusableSegment)
        self.assertIn("correction discarded satellites", str(raised.exception))

    def test_four_corrected_satellites_keep_original_values(self):
        values = np.arange(56, dtype=float).reshape(4, 14)
        from types import SimpleNamespace
        measurements = [SimpleNamespace(as_array=lambda row=row: row) for row in values]
        with patch.object(gnss, "_normal_measurements", return_value=[]), \
             patch.object(gnss.raw, "process_measurements", return_value=[]), \
             patch.object(gnss.raw, "correct_measurements", return_value=measurements):
            corrected = gnss._correct_epochs(object(), np.zeros(3), [rows(4)])
        np.testing.assert_array_equal(corrected[0], values)


if __name__ == "__main__":
    unittest.main()
