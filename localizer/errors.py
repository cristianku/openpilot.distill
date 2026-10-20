"""Errors the localizer raises because of the data of the segment itself."""

# localizer.run exits with this code when the segment cannot be localized because of its
# data (car never above 18 km/h, no GPS, no calibration yet, incomplete segment, ...):
# running it again gives the same result, so helpers/localize_segments.sh moves the
# segment out of data/ (into data_scartati/) and the training does not see it.
UNUSABLE_SEGMENT_EXIT_CODE = 3


class UnusableSegment(RuntimeError):
  """The segment cannot be localized because of its data: running again does not help.
  Network or GNSS download problems are NOT this error: those can work on a later run."""
