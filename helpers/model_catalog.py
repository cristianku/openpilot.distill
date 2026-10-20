"""Add or replace one model in the catalog that the sunnypilot Models menu reads.

Usage: uv run python -m helpers.model_catalog <driving_models.json> --name <nome> --file-name <file.pkl>
           --url <url> --sha256 <hex> --selector-version <n> [--tinygrad-ref <sha>]

Same layout as sunnypilot's driving_models_vNN.json ({"bundles": [...]}), parsed by the same ModelParser
(openpilot/sunnypilot/models/fetcher.py). A missing file starts a new catalog; the bundle with the same
ref (distill-<name>) is replaced in place, keeping its index.
Republishing a name that is selected on a comma changes its sha256: sunnypilot resets that comma to
Default at the next start, select it again.

The values follow the default model CD210: generation 12, 20 Hz, runner tinygrad, and the smoothing of
stock openpilot (lat .0, long .3; without "long" modeld_v2 uses 0).
"""

import argparse
import datetime
import json
import re
from pathlib import Path


FOLDER = "Distill Models"
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
        "display_name": f"Distill {args.name} ({date})",
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
    parser.add_argument("catalog", type=Path, help="driving_models.json to update (created if missing)")
    parser.add_argument("--name", required=True, help="model name: letters, digits, _ . -")
    parser.add_argument("--file-name", required=True, help="compiled file name in /data/media/0/models")
    parser.add_argument("--url", required=True, help="download url of the compiled file")
    parser.add_argument("--sha256", required=True, help="sha256 of the compiled file")
    parser.add_argument("--selector-version", type=int, required=True, help="REQUIRED_JSON_VERSION of sunnypilot")
    parser.add_argument("--tinygrad-ref", default="", help="tinygrad commit used to compile")
    args = parser.parse_args()

    # The ref ends up in the ';'-separated favorites list and in file names.
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
        raise SystemExit(f"nome non valido: {args.name} (solo lettere, numeri, _ . -)")
    if not re.fullmatch(r"[0-9a-f]{64}", args.sha256):
        raise SystemExit(f"sha256 non valido: {args.sha256}")

    if args.catalog.exists():
        catalog = json.loads(args.catalog.read_text())
    else:
        catalog = {"bundles": []}

    ref = f"distill-{args.name}"
    position = None
    highest = FIRST_INDEX - 1
    for number, bundle in enumerate(catalog["bundles"]):
        if bundle["ref"] == ref:
            position = number
        if int(bundle["index"]) > highest:
            highest = int(bundle["index"])

    if position is None:
        catalog["bundles"].append(make_bundle(args, highest + 1))
        print(f"catalogo: aggiunto {ref}")
    else:
        index = int(catalog["bundles"][position]["index"])
        catalog["bundles"][position] = make_bundle(args, index)
        print(f"catalogo: sostituito {ref}")

    args.catalog.write_text(json.dumps(catalog, indent=2) + "\n")


if __name__ == "__main__":
    main()
