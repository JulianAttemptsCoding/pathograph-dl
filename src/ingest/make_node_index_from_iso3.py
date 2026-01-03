import csv
import sys
from pathlib import Path

import pycountry

ROOT = Path.cwd()
COUNTRY_LIST = ROOT / "config" / "countries_194_iso3.txt"
OUT_DIR = ROOT / "data" / "processed" / "meta"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "node_index.csv"


def read_iso3_list(path: Path):
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            lines.append(ln.upper())
    return lines


def iso3_to_country(iso3: str):
    c = pycountry.countries.get(alpha_3=iso3)
    if c is not None:
        name = getattr(c, "name", iso3)
        iso2 = getattr(c, "alpha_2", "")
        return iso2, name
    if iso3 == "TWN":
        return "TW", "Taiwan"
    return "", "UNKNOWN"


def main():
    if not COUNTRY_LIST.exists():
        print(f"COUNTRIES LIST MISSING: Paste the 194 ISO3 codes (one per line) into `{COUNTRY_LIST}` and re-run.")
        sys.exit(2)

    iso3_list = read_iso3_list(COUNTRY_LIST)
    if len(iso3_list) != 194:
        print(f"COUNTRIES LIST MISSING: Paste the 194 ISO3 codes (one per line) into `{COUNTRY_LIST}` and re-run. Found {len(iso3_list)}.")
        sys.exit(2)

    iso3_sorted = sorted(set(iso3_list))
    if len(iso3_sorted) != 194:
        print("DUPLICATES/INVALID: ISO3 list must contain 194 unique ISO3 codes.")
        sys.exit(2)

    rows = []
    for node_id, iso3 in enumerate(iso3_sorted):
        iso2, name = iso3_to_country(iso3)
        rows.append({"node_id": node_id, "iso3": iso3, "iso2": iso2, "name": name})

    with OUT_FILE.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["node_id", "iso3", "iso2", "name"])
        w.writeheader()
        w.writerows(rows)

    print(f"WROTE {OUT_FILE} with {len(rows)} rows (node_id 0..{len(rows)-1}).")


if __name__ == "__main__":
    main()

