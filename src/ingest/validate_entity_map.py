import csv
import sys
from pathlib import Path

ROOT = Path.cwd()
META = ROOT / "data" / "processed" / "meta"
NODE_INDEX = META / "node_index.csv"
ROSETTA = META / "rosetta_codes.csv"


def read_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    if not NODE_INDEX.exists():
        print(f"Missing {NODE_INDEX}. Run: python -m src.ingest.make_node_index_from_iso3")
        sys.exit(2)
    if not ROSETTA.exists():
        print(f"Missing {ROSETTA}. Run: python -m src.ingest.build_entity_map")
        sys.exit(2)

    nodes = read_rows(NODE_INDEX)
    ros = read_rows(ROSETTA)

    assert len(nodes) == 194, f"Expected 194 nodes, found {len(nodes)}"
    assert len(ros) == 194, f"Expected 194 rosetta rows, found {len(ros)}"

    node_ids = sorted(int(r["node_id"]) for r in nodes if r.get("node_id") is not None)
    assert node_ids == list(range(194)), "node_id must be exactly 0..193"

    iso3_nodes = {r["iso3"].strip().upper() for r in nodes}
    iso3_ros = {r["iso3"].strip().upper() for r in ros}
    assert "TWN" in iso3_nodes, "TWN missing from node_index.csv"
    assert "TWN" in iso3_ros, "TWN missing from rosetta_codes.csv"

    unmapped = [r for r in ros if not (r.get("imf_ref_area") or "").strip()]

    print("=== EntityMap Validation ===")
    print(f"node_index rows: {len(nodes)}")
    print(f"rosetta rows: {len(ros)}")
    print(f"unmapped IMF REF_AREA rows: {len(unmapped)}")
    if unmapped:
        for r in unmapped[:20]:
            print("UNMAPPED:", r.get("iso3"), "|", r.get("name"), "| notes=", r.get("notes"))
    print("PASS: Step 3 invariants satisfied.")


if __name__ == "__main__":
    main()

