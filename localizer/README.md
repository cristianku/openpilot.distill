# Localizer

The most minimal localization pipeline. It globalizes logged `deviceMotion` with raw GNSS (GPS+GLONASS), stable calibration, and wheel speed. Runs a small KF/RTS smoother. Adjusts GPS fix delays.

## Use

```bash
uv sync 
uv run python -m localizer.run
```

Run these commands **from the repository root**. The default `--segment` is the example segment defined in `localizer/run.py`; for your own comma-four recording, set the segment ID explicitly:

```bash
# Place the comma-four recording under data/<segment-id>/
# Required for this localizer: fcamera.hevc, ecamera.hevc and rlog.zst
uv run python -m localizer.run --segment <segment-id>
```

`localizer.run` writes `data/<segment-id>/localizer.safetensors` (localization state) and `data/<segment-id>/frame_info.safetensors` (camera timestamps, metadata and the HEVC frame/byte-offset indices). These are consumed by [`supervised/dataset.py`](../supervised/dataset.py) and [`helpers/video_helpers.py`](../helpers/video_helpers.py). See the [video helpers guide](../helpers/README.md) for a working decoding example.

**Input constraint:** `localizer/logs.py` currently accepts only complete comma-four (`mici`) full-HEVC camera/log streams with external GNSS (`gpsLocationExternal`). A generic `qlog` or an incomplete/reordered HEVC index is not a drop-in replacement. Investigate localizer errors rather than making up missing frame metadata.

The first run may download precise orbit and ionosphere data into Laika's cache.

Measured against the comma1M reference `localizer.safetensors` for the included segment:

| Metric | error |
| --- | ---: |
| Median 3D position  | `0.449 m` |
| Median absolute longitudinal | `0.396 m` |
| Median absolute lateral | `0.071 m` |
| Median velocity  | `0.117 m/s` |
