"""Climate Step 4: Compute anomalies and z-scores relative to baseline climatology.

Baseline: 1991-2020, computed per node, per feature, per month-of-year.
Anomaly = value - climatology_mean[month-of-year]
Z-score = anomaly / climatology_std (where std > 0 and count >= 2)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


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


def _sha256_str(s: str) -> str:
    """Compute SHA256 of a string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def month_index_to_year_month(month_index: int) -> tuple[int, int]:
    """Convert month_index (months since 1950-01) to (year, month)."""
    year = 1950 + (month_index // 12)
    month = (month_index % 12) + 1
    return year, month


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/climate_step4.yaml")
    ap.add_argument("--baseline", help="Override baseline as start:end, e.g. 1991:2020")
    ap.add_argument("--compute-zscore", action="store_true", default=None)
    ap.add_argument("--no-compute-zscore", action="store_false", dest="compute_zscore")
    ap.add_argument("--out-dir", help="Override output directory")
    ap.add_argument("--in-zarr", help="Override input climate tensor zarr")
    ap.add_argument("--time-index-master", help="Override time_index_master.npy path")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    
    # Parse baseline
    if args.baseline:
        parts = args.baseline.split(":")
        if len(parts) != 2:
            raise SystemExit("--baseline must be start:end, e.g. 1991:2020")
        baseline_start = int(parts[0])
        baseline_end = int(parts[1])
    else:
        baseline_start = cfg["baseline"]["year_start"]
        baseline_end = cfg["baseline"]["year_end"]
    
    compute_zscore = args.compute_zscore if args.compute_zscore is not None else cfg["compute_zscore"]
    
    paths = cfg["paths"]
    time_index_master_path = args.time_index_master or paths["time_index_master"]
    climate_zarr_path = args.in_zarr or paths["climate_tensor_zarr"]
    output_dir = Path(args.out_dir or paths["output_dir"])
    output_zarr_name = paths["output_zarr"]
    
    arrays = cfg["arrays"]
    chunk_cfg = cfg["chunking"]
    
    # Import zarr
    try:
        import zarr  # type: ignore
    except Exception as e:
        raise SystemExit(f"Missing zarr. Error: {e}")
    
    print(f"[Step 4] Loading climate tensor from {climate_zarr_path}")
    g_in = zarr.open_group(str(climate_zarr_path), mode="r")
    
    if "climate" not in g_in:
        raise SystemExit(f"Missing 'climate' array in {climate_zarr_path}")
    if "mask" not in g_in:
        raise SystemExit(f"Missing 'mask' array in {climate_zarr_path}")
    if "time_index" not in g_in:
        raise SystemExit(f"Missing 'time_index' array in {climate_zarr_path}")
    
    climate_arr = g_in["climate"]
    mask_arr = g_in["mask"]
    time_index_in = np.asarray(g_in["time_index"][:]).astype(np.int32)
    
    T, N, F = climate_arr.shape
    print(f"[Step 4] Input shape: T={T}, N={N}, F={F}")
    
    # Verify time_index vs master if provided
    time_index_master = np.load(time_index_master_path).astype(np.int32)
    if not np.array_equal(time_index_in, time_index_master):
        raise SystemExit(f"time_index from {climate_zarr_path} does not match {time_index_master_path}")
    
    print(f"[Step 4] Baseline period: {baseline_start}-{baseline_end}")
    print(f"[Step 4] Compute z-score: {compute_zscore}")
    
    # Prepare output arrays
    output_dir.mkdir(parents=True, exist_ok=True)
    out_zarr_path = output_dir / output_zarr_name
    
    # Prepare climatology arrays: [12, N, F] for each month-of-year
    climo_mean = np.full((12, N, F), np.nan, dtype=np.float32)
    climo_std = np.full((12, N, F), np.nan, dtype=np.float32)
    climo_count = np.zeros((12, N, F), dtype=np.int32)
    
    # First pass: compute climatology for each month-of-year
    print("[Step 4] Computing baseline climatology (per month-of-year)...")
    
    # Collect values for each (month_of_year, n, f)
    # Use lists to avoid large intermediate arrays
    # For efficiency, we'll use a chunked approach
    baseline_samples = [[[] for _ in range(F)] for _ in range(12)]  # [12][F] -> list of (n, value) tuples per month
    
    # Iterate over time in chunks
    chunk_size = chunk_cfg["time"]
    for t_start in range(0, T, chunk_size):
        t_end = min(t_start + chunk_size, T)
        climate_chunk = np.asarray(climate_arr[t_start:t_end, :, :])
        mask_chunk = np.asarray(mask_arr[t_start:t_end, :, :])
        
        for t_offset in range(t_end - t_start):
            t = t_start + t_offset
            mi = int(time_index_in[t])
            year, month = month_index_to_year_month(mi)
            
            if not (baseline_start <= year <= baseline_end):
                continue
            
            month_of_year = month - 1  # 0-indexed
            
            for n in range(N):
                for f in range(F):
                    if mask_chunk[t_offset, n, f] == 0:
                        continue
                    val = climate_chunk[t_offset, n, f]
                    if not np.isnan(val):
                        # Store in climatology accumulator
                        # For memory, we'll compute statistics on the fly
                        # Use a simpler approach: accumulate per (month, n, f)
                        # Build a dict structure instead
                        pass  # Will refactor below
    
    # Refactor: use dict accumulator for baseline samples
    # Key: (month_of_year, n, f) -> list of values
    baseline_dict: dict[tuple[int, int, int], list[float]] = {}
    
    for t_start in range(0, T, chunk_size):
        t_end = min(t_start + chunk_size, T)
        climate_chunk = np.asarray(climate_arr[t_start:t_end, :, :])
        mask_chunk = np.asarray(mask_arr[t_start:t_end, :, :])
        
        for t_offset in range(t_end - t_start):
            t = t_start + t_offset
            mi = int(time_index_in[t])
            year, month = month_index_to_year_month(mi)
            
            if not (baseline_start <= year <= baseline_end):
                continue
            
            month_of_year = month - 1  # 0-indexed
            
            for n in range(N):
                for f in range(F):
                    if mask_chunk[t_offset, n, f] == 0:
                        continue
                    val = float(climate_chunk[t_offset, n, f])
                    if np.isnan(val):
                        continue
                    
                    key = (month_of_year, n, f)
                    if key not in baseline_dict:
                        baseline_dict[key] = []
                    baseline_dict[key].append(val)
    
    # Compute climatology statistics
    print(f"[Step 4] Computing climatology stats from {len(baseline_dict)} unique (month, node, feature) keys...")
    for (moy, n, f), vals in baseline_dict.items():
        vals_arr = np.array(vals, dtype=np.float32)
        count = len(vals_arr)
        climo_count[moy, n, f] = count
        
        if count >= 1:
            climo_mean[moy, n, f] = np.mean(vals_arr)
        
        if count >= 2:
            climo_std[moy, n, f] = np.std(vals_arr, ddof=0)
    
    # Now compute anomalies and z-scores
    print("[Step 4] Computing anomalies and z-scores...")
    anomaly = np.full((T, N, F), np.nan, dtype=np.float32)
    zscore = np.full((T, N, F), np.nan, dtype=np.float32)
    anomaly_mask = np.zeros((T, N, F), dtype=np.uint8)
    zscore_mask = np.zeros((T, N, F), dtype=np.uint8)
    
    for t_start in range(0, T, chunk_size):
        t_end = min(t_start + chunk_size, T)
        climate_chunk = np.asarray(climate_arr[t_start:t_end, :, :])
        mask_chunk = np.asarray(mask_arr[t_start:t_end, :, :])
        
        for t_offset in range(t_end - t_start):
            t = t_start + t_offset
            mi = int(time_index_in[t])
            year, month = month_index_to_year_month(mi)
            month_of_year = month - 1
            
            for n in range(N):
                for f in range(F):
                    if mask_chunk[t_offset, n, f] == 0:
                        continue
                    
                    val = climate_chunk[t_offset, n, f]
                    if np.isnan(val):
                        continue
                    
                    # Anomaly
                    if climo_count[month_of_year, n, f] >= 1:
                        mean_val = climo_mean[month_of_year, n, f]
                        if not np.isnan(mean_val):
                            anom = val - mean_val
                            anomaly[t, n, f] = anom
                            anomaly_mask[t, n, f] = 1
                            
                            # Z-score
                            if compute_zscore and climo_count[month_of_year, n, f] >= 2:
                                std_val = climo_std[month_of_year, n, f]
                                if not np.isnan(std_val) and std_val > 0:
                                    z = anom / std_val
                                    zscore[t, n, f] = z
                                    zscore_mask[t, n, f] = 1
    
    # Write output Zarr
    print(f"[Step 4] Writing output to {out_zarr_path}")
    g_out = zarr.open_group(str(out_zarr_path), mode="w")
    
    chunk_t = chunk_cfg["time"]
    chunk_n = chunk_cfg["nodes"]
    chunks = (chunk_t, chunk_n, F)
    
    g_out.create_array(arrays["anomaly"], data=anomaly, chunks=chunks)
    g_out.create_array(arrays["zscore"], data=zscore, chunks=chunks)
    g_out.create_array(arrays["mask"], data=anomaly_mask, chunks=chunks)  # Use anomaly_mask as primary
    g_out.create_array("zscore_mask", data=zscore_mask, chunks=chunks)
    g_out.create_array(arrays["time_index"], data=time_index_in, chunks=(T,))
    g_out.create_array(arrays["climatology_mean"], data=climo_mean, chunks=(12, N, F))
    g_out.create_array(arrays["climatology_std"], data=climo_std, chunks=(12, N, F))
    g_out.create_array(arrays["climatology_count"], data=climo_count, chunks=(12, N, F))
    
    # Verify time_index equality
    ti_back = np.asarray(g_out[arrays["time_index"]][:]).astype(np.int32)
    if not np.array_equal(ti_back, time_index_master):
        raise SystemExit("time_index mismatch after write; STOP.")
    
    # QC report
    valid_anom = np.sum(anomaly_mask)
    valid_z = np.sum(zscore_mask)
    total_cells = T * N * F
    
    anom_vals = anomaly[anomaly_mask == 1]
    z_vals = zscore[zscore_mask == 1] if compute_zscore else np.array([])
    
    qc = {
        "total_cells": int(total_cells),
        "valid_anomalies": int(valid_anom),
        "valid_zscores": int(valid_z),
        "fraction_anomaly_valid": float(valid_anom / total_cells) if total_cells > 0 else 0.0,
        "fraction_zscore_valid": float(valid_z / total_cells) if total_cells > 0 and compute_zscore else 0.0,
        "anomaly_stats": {
            "min": float(np.min(anom_vals)) if len(anom_vals) > 0 else None,
            "max": float(np.max(anom_vals)) if len(anom_vals) > 0 else None,
            "mean": float(np.mean(anom_vals)) if len(anom_vals) > 0 else None,
            "std": float(np.std(anom_vals)) if len(anom_vals) > 0 else None,
        },
        "zscore_stats": {
            "min": float(np.min(z_vals)) if len(z_vals) > 0 else None,
            "max": float(np.max(z_vals)) if len(z_vals) > 0 else None,
            "mean": float(np.mean(z_vals)) if len(z_vals) > 0 else None,
            "std": float(np.std(z_vals)) if len(z_vals) > 0 else None,
        } if compute_zscore else {},
        "baseline_coverage": {
            "year_start": baseline_start,
            "year_end": baseline_end,
            "unique_climatology_keys": len(baseline_dict),
            "mean_samples_per_key": float(np.mean([len(v) for v in baseline_dict.values()])) if baseline_dict else 0.0,
        },
    }
    
    qc_path = output_dir / "qc_anomalies.json"
    with qc_path.open("w", encoding="utf-8") as f:
        json.dump(qc, f, indent=2)
    
    print(f"[Step 4] QC report written to {qc_path}")
    
    # Manifest
    manifest = {
        "output_zarr": str(out_zarr_path).replace("\\", "/"),
        "arrays": {
            arrays["anomaly"]: {"shape": [T, N, F], "dtype": "float32", "chunks": list(chunks)},
            arrays["zscore"]: {"shape": [T, N, F], "dtype": "float32", "chunks": list(chunks)},
            arrays["mask"]: {"shape": [T, N, F], "dtype": "uint8", "chunks": list(chunks)},
            "zscore_mask": {"shape": [T, N, F], "dtype": "uint8", "chunks": list(chunks)},
            arrays["time_index"]: {"shape": [T], "dtype": "int32"},
            arrays["climatology_mean"]: {"shape": [12, N, F], "dtype": "float32"},
            arrays["climatology_std"]: {"shape": [12, N, F], "dtype": "float32"},
            arrays["climatology_count"]: {"shape": [12, N, F], "dtype": "int32"},
        },
        "inputs": {
            "config": str(Path(args.config)).replace("\\", "/"),
            "config_sha256": _sha256_str(Path(args.config).read_text(encoding="utf-8")),
            "climate_tensor_zarr": str(climate_zarr_path).replace("\\", "/"),
            "time_index_master": str(time_index_master_path).replace("\\", "/"),
        },
        "baseline": {
            "year_start": baseline_start,
            "year_end": baseline_end,
        },
        "compute_zscore": compute_zscore,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    
    manifest_path = output_dir / "manifest_anomalies.json"
    tmp = manifest_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, manifest_path)
    
    print(f"[Step 4] Manifest written to {manifest_path}")
    print(f"[OK] Step 4 complete: {out_zarr_path}")


if __name__ == "__main__":
    main()
