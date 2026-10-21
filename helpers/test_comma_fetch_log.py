"""Offline tests for the shared comma fetch metadata policy (no device needed)."""
from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from helpers import comma_fetch_log as reader


class Message:
    def __init__(self, service, timestamp, *, gear=None, dashcam=None):
        self._service = service
        self.logMonoTime = timestamp
        if gear is not None:
            self.carState = SimpleNamespace(gearShifter=gear)
        if dashcam is not None:
            self.carParams = SimpleNamespace(dashcamOnly=dashcam)

    def which(self):
        return self._service


class FetchPolicyTests(unittest.TestCase):
    def test_duration_counts_entire_trip_including_removed_edges(self):
        def messages(path):
            if path.name == "first":
                return iter([Message("carParams", 100_000_000_000)])
            return iter([Message("carParams", 399_999_999_999)])
        with patch.object(reader, "_messages", side_effect=messages):
            self.assertLess(float(reader.inspect("duration", [Path("first"), Path("last")])), 300)
        with patch.object(reader, "_messages", side_effect=lambda p: iter(
            [Message("carParams", 100_000_000_000 if p.name == "first" else 400_000_000_000)]
        )):
            self.assertEqual(reader.inspect("duration", [Path("first"), Path("last")]), "300.0")

    def test_reverse_anywhere_rejects_segment(self):
        events = [Message("carState", 1, gear="drive"),
                  Message("carState", 2, gear="reverse"),
                  Message("carState", 3, gear="drive")]
        with patch.object(reader, "_messages", return_value=iter(events)):
            self.assertEqual(reader.inspect("reverse", [Path("rlog.zst")]), "reverse")

    def test_no_car_state_is_unknown(self):
        with patch.object(reader, "_messages", return_value=iter([Message("carParams", 1)])):
            self.assertEqual(reader.inspect("reverse", [Path("rlog.zst")]), "unknown")

    def test_dashcam_only_auto_routes(self):
        with patch.object(reader, "_messages", return_value=iter([Message("carParams", 1, dashcam=True)])):
            self.assertEqual(reader.inspect("dashcam", [Path("qlog.zst")]), "dashcam")
        with patch.object(reader, "_messages", return_value=iter([Message("carParams", 1, dashcam=False)])):
            self.assertEqual(reader.inspect("dashcam", [Path("qlog.zst")]), "normal")


if __name__ == "__main__":
    unittest.main()
