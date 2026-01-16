from __future__ import annotations

import argparse
import hashlib
import json
import os
from glob import glob
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Missing PyYAML. Install:\n"
            "  conda run -n pathograph-pre python -m pip install pyyaml\n"
            f"Error: {e}"
        )
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    paths = cfg["paths"]
    proc = cfg["processing"]

    try:
        import zarr  # type: ignore
    except Exception as e:
        raise SystemExit(f"Missing zarr. Error: {e}")

    try:
        import pyarrow  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "Missing pyarrow (required to read Parquet). Install:\n"
            "  conda run -n pathograph-pre python -m pip install pyarrow\n"
            f"Error: {e}"
        )

    ti_master = np.load(paths["time_index_master"]).astype(np.int32)
    T = int(len(ti_master))
    if T != 908:
        raise SystemExit(f"Expected T=908; found T={T}")

    # Map month_index value -> position in time axis
    pos = {int(v): i for i, v in enumerate(ti_master.tolist())}

    node_index = pd.read_csv(paths["node_index"])
    if "node_id" not in node_index.columns:
        raise SystemExit("node_index missing node_id")
    N = int(node_index["node_id"].nunique())
    if N != 194:
        raise SystemExit(f"Expected N=194; found N={N}")

    feature_names = list(proc["feature_order_locked"])
    F = len(feature_names)
    if F != 10:
        raise SystemExit(f"Expected F=10; found F={F}")

    # Prepare arrays
    climate = np.full((T, N, F), np.nan, dtype=np.float32)
    mask = np.zeros((T, N, F), dtype=np.uint8)

    # Read all parquet files (can be partial; missing months remain NaN/mask=0)
    parquet_root = Path(paths["processed_country_month_dir"])
    files = sorted(glob(str(parquet_root / "year=*/country_month_*.parquet")))
    if not files:
        raise SystemExit(f"No Parquet inputs found under {parquet_root}/year=*/country_month_*.parquet")

    required_cols = set(["node_id", "month_index"]) | set(feature_names)
    filled_cells = 0

    for fp in files:
        df = pd.read_parquet(fp)
        missing = required_cols - set(df.columns)
        if missing:
            raise SystemExit(f"Parquet {fp} missing required columns: {sorted(missing)}")

        for _, row in df.iterrows():
            mi = int(row["month_index"])
            if mi not in pos:
                raise SystemExit(f"month_index={mi} not in master time index; STOP.")
            t = pos[mi]
            n = int(row["node_id"])
            if not (0 <= n < N):
                raise SystemExit(f"node_id out of range: {n}")

            for j, feat in enumerate(feature_names):
                v = row[feat]
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                climate[t, n, j] = np.float32(v)
                mask[t, n, j] = 1
                filled_cells += 1

    out_zarr = Path(paths["processed_tensor_zarr"])
    out_zarr.parent.mkdir(parents=True, exist_ok=True)

    # Write Zarr v3 group
    g = zarr.open_group(str(out_zarr), mode="w")
    chunks = (int(proc["chunk_time"]), N, F)

    g.create_array("climate", data=climate, chunks=chunks)
    g.create_array("mask", data=mask, chunks=chunks)
    g.create_array("time_index", data=ti_master, chunks=(T,))
    g.create_array("feature_names", data=np.array(feature_names, dtype="U32"), chunks=(F,))

    # Assert time_index equality by reading back
    ti_back = np.asarray(g["time_index"][:]).astype(np.int32)
    if not np.array_equal(ti_back, ti_master):
        raise SystemExit("time_index mismatch after write; STOP.")

    # Manifest
    man_dir = Path(paths["manifests_dir"])
    man_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "output_zarr": str(out_zarr).replace("\\", "/"),
        "arrays": {
            "climate": {"shape": [T, N, F], "dtype": "float32", "chunks": list(chunks)},
            "mask": {"shape": [T, N, F], "dtype": "uint8", "chunks": list(chunks)},
            "time_index": {"shape": [T], "dtype": "int32"},
            "feature_names": {"shape": [F], "dtype": "U32"}
        },
        "feature_order": feature_names,
        "inputs": {
            "config": str(Path(args.config)).replace("\\", "/"),
            "config_sha256": _sha256(Path(args.config)),
            "time_index_master": paths["time_index_master"],
            "time_index_master_sha256": _sha256(Path(paths["time_index_master"])),
            "node_index": paths["node_index"],
            "node_index_sha256": _sha256(Path(paths["node_index"]))
        },
        "parquet_files_count": len(files),
        "filled_cells": int(filled_cells),
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z"
    }

    man_path = man_dir / "climate_step3_tensor_manifest.json"
    tmp = man_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
    os.replace(tmp, man_path)

    print(f"[OK] Wrote {out_zarr} and manifest {man_path}")

if __name__ == "__main__":
    main()
