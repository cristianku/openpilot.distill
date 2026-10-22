"""Write the catalog that the sunnypilot Models menu reads, with only this model.

Usage: uv run python -m helpers.model_catalog <driving_models.json> --name <nome> --file-name <file.pkl>
           --url <url> --sha256 <hex> --selector-version <n> [--tinygrad-ref <sha>]

Same layout as sunnypilot's driving_models_vNN.json ({"bundles": [...]}), parsed by the same ModelParser
(openpilot/sunnypilot/models/fetcher.py). The file is overwritten: the models published before leave the
menu, their files stay on Hugging Face.
When the model selected on a comma leaves the catalog, or is republished with the same name (new sha256),
sunnypilot resets that comma to Default at the next start: select the new one.

The values follow the default model CD210: generation 12, 20 Hz, runner tinygrad, and the smoothing of
stock openpilot (lat .0, long .3; without "long" modeld_v2 uses 0).
"""

import argparse
import datetime
import json
import re
from pathlib import Path


FOLDER = "Distill cristianku"
# The official catalog uses 0..76: higher indexes put this folder at the top of the menu.
FIRST_INDEX = 1000
MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def make_bundle(args: argparse.Namespace, index: int) -> dict:
    today = datetime.date.today()
    date = f"{MONTHS[today.month - 1]} {today.day}, {today.year}"
    return {
        "short_name": args.name,
        "display_name": f"{args.name} ({date})",
        "index": index,
        "generation": "12",
        "environment": "distill",
        "runner": "tinygrad",
        "is_20hz": True,
        "minimum_selector_version": str(args.selector_version),
        "ref": f"distill-{args.name}",
        # Not read by sunnypilot: the tinygrad of the comma that compiled the file.
        "tinygrad_ref": args.tinygrad_ref,
        "overrides": {"folder": FOLDER, "lat": ".0", "long": ".3"},
        "models": [
            {
                "type": "supercombo",
                "artifact": {
                    "file_name": args.file_name,
                    "download_uri": {"url": args.url, "sha256": args.sha256},
                },
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("catalog", type=Path, help="driving_models.json to write (overwritten)")
    parser.add_argument("--name", required=True, help="model name: letters, digits, _ . -")
    parser.add_argument("--file-name", required=True, help="compiled file name in /data/media/0/models")
    parser.add_argument("--url", required=True, help="download url of the compiled file")
    parser.add_argument("--sha256", required=True, help="sha256 of the compiled file")
    parser.add_argument("--selector-version", type=int, required=True, help="REQUIRED_JSON_VERSION of sunnypilot")
    parser.add_argument("--tinygrad-ref", default="", help="tinygrad commit used to compile")
    args = parser.parse_args()

    # The ref ends up in the ';'-separated favorites list and in file names.
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
        raise SystemExit(f"invalid name: {args.name} (only letters, digits, _ . -)")
    if not re.fullmatch(r"[0-9a-f]{64}", args.sha256):
        raise SystemExit(f"invalid sha256: {args.sha256}")

    catalog = {"bundles": [make_bundle(args, FIRST_INDEX)]}
    args.catalog.write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"catalog: only distill-{args.name}")


if __name__ == "__main__":
    main()
