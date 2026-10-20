"""Tell whether comma logs were recorded in dashcam mode (carParams.dashcamOnly).

Usage: uv run python -m helpers.dashcam_mode <qlog.zst> [<qlog.zst> ...]
Prints one line per file: "<file> dashcam", "<file> normal" or "<file> unknown"
(unknown = no carParams in the file, or the file could not be read).
carParams is logged every 50 s and goes to the qlog too, so any segment has one.
"""
import sys

from openpilot.tools.lib.logreader import LogReader


def dashcam_state(path: str) -> str:
  try:
    car_params = LogReader(path).first("carParams")
  except Exception as error:
    print(f"{path}: {error}", file=sys.stderr)
    return "unknown"

  if car_params is None:
    return "unknown"
  if car_params.dashcamOnly:
    return "dashcam"
  return "normal"


def main() -> None:
  for path in sys.argv[1:]:
    print(f"{path} {dashcam_state(path)}")


if __name__ == "__main__":
  main()
