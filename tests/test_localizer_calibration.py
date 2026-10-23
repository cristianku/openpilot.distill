"""localizer.logs.stable_calibration: which calibration a segment gets, and when it is refused."""
import unittest
from types import SimpleNamespace

import numpy as np

from localizer.errors import UnusableSegment
from localizer.logs import stable_calibration


def message(valid=True, status='calibrated', blocks=50, rpy=(0.0, 0.1, 0.05), wide=(0.0, 0.0, 0.0)):
  calibration = SimpleNamespace(calStatus=status, validBlocks=blocks, rpyCalib=list(rpy),
                                wideFromDeviceEuler=list(wide))
  return SimpleNamespace(valid=valid, extrinsicsCalibration=calibration)


class StableCalibration(unittest.TestCase):
  def test_flagged_messages_with_the_same_calibration_are_skipped(self):
    # The driving model dropped frames: one message in ten is flagged, the calibration does not change.
    events = [message(valid=index % 10 != 3, rpy=(0.0, 0.1159, 0.0531)) for index in range(240)]
    rpy, wide = stable_calibration(events)
    np.testing.assert_allclose(rpy, [0.0, 0.1159, 0.0531])
    np.testing.assert_allclose(wide, [0.0, 0.0, 0.0])

  def test_a_flagged_message_never_gives_its_values(self):
    events = [message(rpy=(0.0, 0.1, 0.0)), message(valid=False, rpy=(0.0, 0.9, 0.0))]
    events += [message(rpy=(0.0, 0.1, 0.0))] * 8
    np.testing.assert_allclose(stable_calibration(events)[0], [0.0, 0.1, 0.0])

  def test_recalibration_is_refused_even_when_flagged(self):
    for broken in (message(valid=False, status='recalibrating'), message(valid=False, blocks=40),
                   message(status='uncalibrated')):
      with self.assertRaises(UnusableSegment):
        stable_calibration([message()] * 5 + [broken] + [message()] * 5)

  def test_too_many_flagged_or_never_calibrated(self):
    with self.assertRaisesRegex(UnusableSegment, 'flagged not valid in 30 %'):
      stable_calibration([message(valid=index % 10 >= 3) for index in range(100)])
    with self.assertRaisesRegex(UnusableSegment, 'no stable calibrated'):
      stable_calibration([message(status='uncalibrated')] * 10)
    with self.assertRaisesRegex(UnusableSegment, 'no stable calibrated'):
      stable_calibration([message(valid=False)] * 10)

  def test_calibration_reached_during_the_segment(self):
    events = [message(status='uncalibrated', blocks=10)] * 3 + [message(blocks=50)] * 7
    np.testing.assert_allclose(stable_calibration(events)[0], [0.0, 0.1, 0.05])


if __name__ == '__main__':
  unittest.main()
