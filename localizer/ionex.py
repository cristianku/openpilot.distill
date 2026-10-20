"""Ionosphere maps (IONEX) for the localizer, recording which map was used for each day.

CODE publishes the map of a day more than once, each version better than the one before:
  P1D  prediction, out the day before (the only one there is for a drive of today)
  RAP  rapid, computed from the real data of the day, out about one day later
  FIN  final, out about two weeks later
(P2D, the old two-day prediction, is only a last resort.)

Laika alone takes FIN or RAP, so a drive of today cannot be localized. Here P1D is
accepted too, and localizer.run writes the map used for each day in localizer.ionex
next to the outputs ("2026/293 P1D"). helpers/localize_segments.sh runs a segment
again when a better map has come out, one step at a time: P1D -> RAP -> FIN.

Usage: uv run python -m localizer.ionex <segment folder> [...]
Prints "<folder> <map used> <better map out now, or ->" for every folder with a
localizer.ionex (the worst day counts when a segment spans midnight).
"""
import contextlib
import sys
from datetime import datetime, timedelta
from pathlib import Path

from laika.downloader import CDDIS_BASE_URL, DownloadFailed, download_and_cache_file, mirror_url
from laika.gps_time import GPSTime

IONEX_FILE = "localizer.ionex"
# From GPS week 2238 (27 Nov 2022) CODE uses the long file names (same switch as laika).
LONG_NAMES_FROM = GPSTime(2238, 0.0)


def ionex_filename(time: GPSTime, product: str) -> str | None:
  """File name of one map of the day of `time`, None when that map has no file."""
  t = time.as_datetime()
  if time >= LONG_NAMES_FROM:
    if product == "FIN":
      return t.strftime("COD0OPSFIN_%Y%j0000_01D_01H_GIM.INX.gz")
    elif product == "RAP":
      return t.strftime("COD0OPSRAP_%Y%j0000_01D_01H_GIM.INX.gz")
    elif product == "P1D":
      return t.strftime("COD0OPSP1D_%Y%j0000_01D_01H_GIM.INX.gz")
    elif product == "P2D":
      return t.strftime("c2pg%j0.%yi.Z")
  else:
    if product == "FIN":
      return t.strftime("codg%j0.%yi.Z")
    elif product == "P1D":
      return t.strftime("c1pg%j0.%yi.Z")
    elif product == "P2D":
      return t.strftime("c2pg%j0.%yi.Z")
  return None


def better_products(product: str) -> tuple[str, ...]:
  """The maps better than `product`, best first."""
  if product == "FIN":
    return ()
  elif product == "RAP":
    return ("FIN",)
  elif product == "P1D":
    return ("FIN", "RAP")
  elif product == "P2D":
    return ("FIN", "RAP", "P1D")
  else:
    raise ValueError(f"unknown ionosphere map: {product}")


def day_of(time: GPSTime) -> str:
  return time.as_datetime().strftime("%Y/%j")


def time_of_day(day: str) -> GPSTime:
  """Noon of a "2026/293" day, as GPS time."""
  year, day_of_year = day.split("/")
  noon = datetime(int(year), 1, 1, 12) + timedelta(days=int(day_of_year) - 1)
  return GPSTime.from_datetime(noon)


def _download(time: GPSTime, product: str, cache_dir: str) -> str | None:
  """Path of one map in laika's cache (downloaded if needed), None when not out yet.
  Same cache and the same one hour pause after a failed try as laika."""
  filename = ionex_filename(time, product)
  if filename is None:
    return None
  url_bases = (mirror_url(CDDIS_BASE_URL, "/gnss/products/ionex/"),)
  folder_path = time.as_datetime().strftime("%Y/%j/")
  try:
    return download_and_cache_file(url_bases, folder_path, cache_dir + "ionex/", filename)
  except DownloadFailed:
    return None


def download_ionex(time: GPSTime, cache_dir: str) -> tuple[str, str]:
  """Best map out for the day of `time`: (file path, product)."""
  for product in ("FIN", "RAP", "P1D", "P2D"):
    path = _download(time, product, cache_dir)
    if path is not None:
      return path, product
  raise DownloadFailed(f"no ionosphere map out yet for {day_of(time)}")


def better_ionex_out(day: str, product: str, cache_dir: str) -> str | None:
  """The best map better than `product` out now for `day` (left in the cache), or None."""
  time = time_of_day(day)
  for better in better_products(product):
    if _download(time, better, cache_dir) is not None:
      return better
  return None


def write_ionex_file(segment_dir: Path, ionex_used: dict[str, str]) -> None:
  lines = []
  for day in sorted(ionex_used):
    lines.append(f"{day} {ionex_used[day]}\n")
  (segment_dir / IONEX_FILE).write_text("".join(lines))


def read_ionex_file(segment_dir: Path) -> dict[str, str]:
  ionex_used = {}
  for line in (segment_dir / IONEX_FILE).read_text().splitlines():
    if line.strip():
      day, product = line.split()
      ionex_used[day] = product
  return ionex_used


def main() -> None:
  from laika.astro_dog import AstroDog

  cache_dir = AstroDog().cache_dir
  results = []
  for folder in sys.argv[1:]:
    segment_dir = Path(folder)
    if not (segment_dir / IONEX_FILE).is_file():
      continue
    ionex_used = read_ionex_file(segment_dir)

    # The worst map used, and whether a better one is out for any day.
    worst = "FIN"
    better = "-"
    for day in sorted(ionex_used):
      product = ionex_used[day]
      if len(better_products(product)) > len(better_products(worst)):
        worst = product
      if better == "-":
        # laika prints "HTTPS error 404" for every map not out yet: keep stdout for the results.
        with contextlib.redirect_stdout(sys.stderr):
          found = better_ionex_out(day, product, cache_dir)
        if found is not None:
          better = found
    results.append(f"{folder} {worst} {better}")

  for line in results:
    print(line)


if __name__ == "__main__":
  main()
