"""Preprocessing Acceptance Verifier.

Assert existence + alignment across:
- TRADE (base + risk)
- CLIMATE (tensor + anomalies)
- PATHOGEN (status tensor)
- META (spatial + time_index_master)

Exit code 0 on pass, nonzero on failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, UTC
from pathlib import Path

import numpy as np


def fail(msg: str) -> None:
    """Print failure message and exit."""
    print(f"[FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


def check_file_exists(path: Path, name: str) -> None:
    """Check that a file exists."""
    if not path.exists():
        fail(f"{name} not found: {path}")
    print(f"[OK] {name}: {path}")


def check_zarr_group(path: Path, name: str, required_arrays: list[str]) -> object:
    """Check that a Zarr group exists and contains required arrays."""
    try:
        import zarr  # type: ignore
    except Exception as e:
        fail(f"Missing zarr library: {e}")
    
    if not path.exists():
        fail(f"{name} not found: {path}")
    
    try:
        g = zarr.open_group(str(path), mode="r")
    except Exception as e:
        fail(f"{name} is not a valid Zarr group: {path} - {e}")
    
    for arr in required_arrays:
        if arr not in g:
            fail(f"{name} missing required array '{arr}': {path}")
    
    print(f"[OK] {name}: {path} (arrays: {', '.join(required_arrays)})")
    return g


def resolve_array_name(group: object, candidates: list[str], label: str, zarr_path: Path) -> str:
    """Try candidate names in order; fail if none exist."""
    for name in candidates:
        if name in group:
            return name
    available = list(group.keys())
    fail(f"{label}: none of {candidates} found in {zarr_path} (available: {available})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--time-index-master", default="data/processed/meta/time_index_master.npy")
    ap.add_argument("--trade-base-zarr", default="data/processed/trade/imf_imts_step1/trade_fob_tensor.zarr")
    ap.add_argument("--trade-risk-zarr", default="data/processed/trade/faostat_step2/trade_risk_tensor.zarr")
    ap.add_argument("--climate-zarr", default="data/processed/climate/climate_tensor.zarr")
    ap.add_argument("--climate-anoms-zarr", default="data/processed/climate/climate_step4/climate_anomalies.zarr")
    ap.add_argument("--pathogen-zarr", default="data/processed/pathogen/status_tensor.zarr")
    ap.add_argument("--meta-dir", default="data/processed/meta")
    ap.add_argument("--mode", choices=["fast", "full"], default="fast")
    ap.add_argument("--require-meta", action="store_true", help="Fail if distance/adjacency matrices are missing (default: skip)")
    args = ap.parse_args()
    
    print("[Preprocessing Acceptance Verifier]")
    print(f"Mode: {args.mode}")
    print()
    
    # Load master time index
    ti_master_path = Path(args.time_index_master)
    check_file_exists(ti_master_path, "time_index_master")
    ti_master = np.load(ti_master_path).astype(np.int32)
    T = len(ti_master)
    print(f"  Time axis length T = {T}")
    print()
    
    # Expected N
    N = 194
    
    results = {
        "mode": args.mode,
        "time_index_master": {
            "path": str(ti_master_path).replace("\\", "/"),
            "T": int(T),
        },
        "checks": {},
        "summary": {},
    }
    
    # Check TRADE base
    print("[TRADE BASE]")
    trade_base_path = Path(args.trade_base_zarr)
    g_trade_base = check_zarr_group(
        trade_base_path,
        "Trade base tensor",
        ["trade", "mask", "time_index", "is_estimated"]
    )
    
    trade_base_arr = g_trade_base["trade"]
    trade_base_shape = trade_base_arr.shape
    trade_base_ti = np.asarray(g_trade_base["time_index"][:]).astype(np.int32)
    
    if len(trade_base_shape) != 4:
        fail(f"Trade base shape {trade_base_shape} is not 4D")
    
    T_trade, N_trade, N2_trade, F_trade = trade_base_shape
    
    if N_trade != N or N2_trade != N:
        fail(f"Trade base N dims ({N_trade}, {N2_trade}) != {N}")
    
    if F_trade != 2:
        fail(f"Trade base feature dim {F_trade} != 2 (expected import/export)")
    
    if not np.array_equal(trade_base_ti, ti_master):
        fail(f"Trade base time_index does not match master")
    
    print(f"  Shape: {trade_base_shape}")
    print(f"  time_index matches master: OK")
    
    results["checks"]["trade_base"] = {
        "path": str(trade_base_path).replace("\\", "/"),
        "shape": list(trade_base_shape),
        "time_match": True,
    }
    print()
    
    # Check TRADE risk
    print("[TRADE RISK]")
    trade_risk_path = Path(args.trade_risk_zarr)
    
    # Import zarr for manual group opening
    try:
        import zarr  # type: ignore
    except Exception as e:
        fail(f"Missing zarr library: {e}")
    
    if not trade_risk_path.exists():
        fail(f"Trade risk tensor not found: {trade_risk_path}")
    
    try:
        g_trade_risk = zarr.open_group(str(trade_risk_path), mode="r")
    except Exception as e:
        fail(f"Trade risk tensor is not a valid Zarr group: {trade_risk_path} - {e}")
    
    # Resolve array names with aliases (prefer canonical FAOSTAT schema)
    risk_key = resolve_array_name(g_trade_risk, ["trade_risk", "risk_trade", "risk"], "Trade risk data", trade_risk_path)
    mask_key = resolve_array_name(g_trade_risk, ["observed_mask", "mask"], "Trade risk mask", trade_risk_path)
    
    # Require hard keys
    if "is_estimated" not in g_trade_risk:
        fail(f"Trade risk missing 'is_estimated' array: {trade_risk_path}")
    if "time_index" not in g_trade_risk:
        fail(f"Trade risk missing 'time_index' array: {trade_risk_path}")
    
    print(f"[OK] Trade risk tensor: {trade_risk_path}")
    print(f"  Selected arrays: risk={risk_key}, mask={mask_key}")
    
    # Get arrays
    trade_risk_arr = g_trade_risk[risk_key]
    trade_risk_mask = g_trade_risk[mask_key]
    trade_risk_is_est = g_trade_risk["is_estimated"]
    trade_risk_ti = np.asarray(g_trade_risk["time_index"][:]).astype(np.int32)
    
    # Validate dtypes
    if str(trade_risk_arr.dtype) not in ("float32", "<f4"):
        fail(f"Trade risk array dtype {trade_risk_arr.dtype} is not float32")
    if str(trade_risk_mask.dtype) not in ("uint8", "ubyte", "|u1", "bool"):
        fail(f"Trade risk mask dtype {trade_risk_mask.dtype} is not uint8/bool")
    if str(trade_risk_is_est.dtype) not in ("uint8", "ubyte", "|u1", "bool"):
        fail(f"Trade risk is_estimated dtype {trade_risk_is_est.dtype} is not uint8/bool")
    
    trade_risk_shape = trade_risk_arr.shape
    
    # Validate shapes match
    if trade_risk_mask.shape != trade_risk_shape:
        fail(f"Trade risk mask shape {trade_risk_mask.shape} != risk shape {trade_risk_shape}")
    if trade_risk_is_est.shape != trade_risk_shape:
        fail(f"Trade risk is_estimated shape {trade_risk_is_est.shape} != risk shape {trade_risk_shape}")
    
    # Validate 5D shape
    if len(trade_risk_shape) != 5:
        fail(f"Trade risk shape {trade_risk_shape} is not 5D")
    
    T_risk, N_risk, N2_risk, K_risk, F_risk = trade_risk_shape
    
    if N_risk != N or N2_risk != N:
        fail(f"Trade risk N dims ({N_risk}, {N2_risk}) != {N}")
    
    if F_risk != 2:
        fail(f"Trade risk feature dim {F_risk} != 2")
    
    if not np.array_equal(trade_risk_ti, ti_master):
        fail(f"Trade risk time_index does not match master")
    
    print(f"  Shape: {trade_risk_shape}")
    print(f"  K (risk products) = {K_risk}")
    print(f"  dtype: risk={trade_risk_arr.dtype}, mask={trade_risk_mask.dtype}")
    print(f"  time_index matches master: OK")
    
    results["checks"]["trade_risk"] = {
        "path": str(trade_risk_path).replace("\\", "/"),
        "shape": list(trade_risk_shape),
        "K": int(K_risk),
        "time_match": True,
        "arrays": {
            "risk": risk_key,
            "mask": mask_key,
            "is_estimated": "is_estimated",
            "time_index": "time_index"
        }
    }
    print()
    
    # Check CLIMATE tensor
    print("[CLIMATE TENSOR]")
    climate_path = Path(args.climate_zarr)
    g_climate = check_zarr_group(
        climate_path,
        "Climate tensor",
        ["climate", "mask", "time_index", "feature_names"]
    )
    
    climate_arr = g_climate["climate"]
    climate_shape = climate_arr.shape
    climate_ti = np.asarray(g_climate["time_index"][:]).astype(np.int32)
    
    if len(climate_shape) != 3:
        fail(f"Climate shape {climate_shape} is not 3D")
    
    T_climate, N_climate, F_climate = climate_shape
    
    if N_climate != N:
        fail(f"Climate N dim {N_climate} != {N}")
    
    if not np.array_equal(climate_ti, ti_master):
        fail(f"Climate time_index does not match master")
    
    print(f"  Shape: {climate_shape}")
    print(f"  F (features) = {F_climate}")
    print(f"  time_index matches master: OK")
    
    results["checks"]["climate_tensor"] = {
        "path": str(climate_path).replace("\\", "/"),
        "shape": list(climate_shape),
        "F": int(F_climate),
        "time_match": True,
    }
    print()
    
    # Check CLIMATE anomalies
    print("[CLIMATE ANOMALIES]")
    climate_anoms_path = Path(args.climate_anoms_zarr)
    g_climate_anoms = check_zarr_group(
        climate_anoms_path,
        "Climate anomalies",
        ["anomaly", "zscore", "mask", "time_index"]
    )
    
    anom_arr = g_climate_anoms["anomaly"]
    anom_shape = anom_arr.shape
    anom_ti = np.asarray(g_climate_anoms["time_index"][:]).astype(np.int32)
    
    if len(anom_shape) != 3:
        fail(f"Climate anomaly shape {anom_shape} is not 3D")
    
    T_anom, N_anom, F_anom = anom_shape
    
    if N_anom != N:
        fail(f"Climate anomaly N dim {N_anom} != {N}")
    
    if not np.array_equal(anom_ti, ti_master):
        fail(f"Climate anomaly time_index does not match master")
    
    print(f"  Shape: {anom_shape}")
    print(f"  time_index matches master: OK")
    
    results["checks"]["climate_anomalies"] = {
        "path": str(climate_anoms_path).replace("\\", "/"),
        "shape": list(anom_shape),
        "time_match": True,
    }
    print()
    
    # Check PATHOGEN
    print("[PATHOGEN STATUS]")
    pathogen_path = Path(args.pathogen_zarr)
    
    # Import zarr
    try:
        import zarr  # type: ignore
    except Exception as e:
        fail(f"Missing zarr library: {e}")
    
    if not pathogen_path.exists():
        fail(f"Pathogen status tensor not found: {pathogen_path}")
    
    try:
        g_pathogen = zarr.open_group(str(pathogen_path), mode="r")
    except Exception as e:
        fail(f"Pathogen status tensor is not a valid Zarr group: {pathogen_path} - {e}")
    
    # Resolve mask name (pathogen uses 'status_mask' canonically, 'mask' as fallback)
    pathogen_mask_key = resolve_array_name(g_pathogen, ["status_mask", "mask"], "Pathogen mask", pathogen_path)
    
    # Require hard keys
    if "status" not in g_pathogen:
        fail(f"Pathogen missing 'status' array: {pathogen_path}")
    if "time_index" not in g_pathogen:
        fail(f"Pathogen missing 'time_index' array: {pathogen_path}")
    
    print(f"[OK] Pathogen status tensor: {pathogen_path}")
    print(f"  Selected mask: {pathogen_mask_key}")
    
    pathogen_arr = g_pathogen["status"]
    pathogen_shape = pathogen_arr.shape
    pathogen_ti = np.asarray(g_pathogen["time_index"][:]).astype(np.int32)
    
    if len(pathogen_shape) != 3:
        fail(f"Pathogen shape {pathogen_shape} is not 3D")
    
    T_pathogen, N_pathogen, P_pathogen = pathogen_shape
    
    if N_pathogen != N:
        fail(f"Pathogen N dim {N_pathogen} != {N}")
    
    if not np.array_equal(pathogen_ti, ti_master):
        fail(f"Pathogen time_index does not match master")
    
    print(f"  Shape: {pathogen_shape}")
    print(f"  P (pathogens) = {P_pathogen}")
    print(f"  time_index matches master: OK")
    
    results["checks"]["pathogen_status"] = {
        "path": str(pathogen_path).replace("\\", "/"),
        "shape": list(pathogen_shape),
        "P": int(P_pathogen),
        "time_match": True,
    }
    print()
    
    # Check META spatial (optional for now)
    print("[META SPATIAL]")
    meta_dir = Path(args.meta_dir)
    distance_path = meta_dir / "distance_km.npy"
    adjacency_path = meta_dir / "adjacency_border.npy"
    
    spatial_ok = True
    distance_present = False
    adjacency_present = False
    
    if distance_path.exists():
        dist = np.load(distance_path)
        if dist.shape != (N, N):
            fail(f"Distance matrix shape {dist.shape} != ({N}, {N})")
        if not np.all(np.isfinite(dist)):
            fail(f"Distance matrix contains non-finite values")
        diag = np.diag(dist)
        if not np.allclose(diag, 0.0, atol=1e-3):
            fail(f"Distance matrix diagonal not near-zero (max: {np.max(diag)})")
        print(f"[OK] Distance matrix: {distance_path}")
        results["checks"]["distance"] = {
            "path": str(distance_path).replace("\\", "/"),
            "shape": list(dist.shape),
        }
        distance_present = True
    else:
        if args.require_meta:
            fail(f"Meta matrices required (--require-meta) but distance_km.npy not found: {distance_path}")
        print(f"[SKIP] Distance matrix not found (optional): {distance_path}")
        spatial_ok = False
    
    if adjacency_path.exists():
        adj = np.load(adjacency_path)
        if adj.shape != (N, N):
            fail(f"Adjacency matrix shape {adj.shape} != ({N}, {N})")
        if not np.all(np.isfinite(adj)):
            fail(f"Adjacency matrix contains non-finite values")
        print(f"[OK] Adjacency matrix: {adjacency_path}")
        results["checks"]["adjacency"] = {
            "path": str(adjacency_path).replace("\\", "/"),
            "shape": list(adj.shape),
        }
        adjacency_present = True
    else:
        if args.require_meta:
            fail(f"Meta matrices required (--require-meta) but adjacency_border.npy not found: {adjacency_path}")
        print(f"[SKIP] Adjacency matrix not found (optional): {adjacency_path}")
        spatial_ok = False
    
    # Record meta spatial status
    if "meta_spatial" not in results["checks"]:
        results["checks"]["meta_spatial"] = {}
    results["checks"]["meta_spatial"]["require_meta"] = args.require_meta
    results["checks"]["meta_spatial"]["distance"] = {
        "present": distance_present,
        "path": str(distance_path).replace("\\", "/")
    }
    results["checks"]["meta_spatial"]["adjacency"] = {
        "present": adjacency_present,
        "path": str(adjacency_path).replace("\\", "/")
    }
    
    print()
    
    # Summary
    results["summary"] = {
        "all_time_indices_aligned": True,
        "N_nodes": N,
        "T_timesteps": T,
        "K_risk_products": int(K_risk),
        "F_climate_features": int(F_climate),
        "P_pathogens": int(P_pathogen),
        "spatial_matrices_present": spatial_ok,
    }
    
    # Write reports
    report_json = Path("data/processed/preprocessing_acceptance_report.json")
    report_txt = Path("data/processed/preprocessing_acceptance_report.txt")
    
    results["created_at"] = datetime.now(UTC).isoformat(timespec="seconds") + "Z"
    
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"[OK] JSON report: {report_json}")
    
    # Text summary
    with report_txt.open("w", encoding="utf-8") as f:
        f.write("Preprocessing Acceptance Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {results['created_at']}\n")
        f.write(f"Mode: {args.mode}\n\n")
        f.write("Summary:\n")
        f.write(f"  T (timesteps): {T}\n")
        f.write(f"  N (nodes): {N}\n")
        f.write(f"  K (risk products): {K_risk}\n")
        f.write(f"  F (climate features): {F_climate}\n")
        f.write(f"  P (pathogens): {P_pathogen}\n")
        f.write(f"  All time indices aligned: {results['summary']['all_time_indices_aligned']}\n")
        f.write(f"  Spatial matrices present: {spatial_ok}\n\n")
        f.write("All checks passed.\n")
    
    print(f"[OK] Text summary: {report_txt}")
    print()
    print("[PASS] All preprocessing acceptance checks passed.")


if __name__ == "__main__":
    main()
