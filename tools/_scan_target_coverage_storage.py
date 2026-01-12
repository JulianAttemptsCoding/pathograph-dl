"""Scan Zarr storage to measure target mask coverage per time index.

This produces a JSON report with earliest observed t_y for base and risk masks.
Observed defined as: mask_codes != 0 (typically code=1).
"""
import json
import sys
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parent.parent
STEP1_ZARR = ROOT / "data/processed/trade/imf_imts_step1/trade_fob_tensor.zarr"
STEP2_ZARR = ROOT / "data/processed/trade/faostat_step2/trade_risk_tensor.zarr"
REPORT_PATH = ROOT / "docs/reports/target_coverage_storage.json"

# Splits from config/trade_step3.yaml
SPLITS = {
    "train": (0, 815),
    "val": (816, 851),
    "test": (852, 907),
}


def main():
    print(">>> Scanning storage-level target coverage...")
    
    # Open Zarr groups
    base_grp = zarr.open(str(STEP1_ZARR), mode="r")
    risk_grp = zarr.open(str(STEP2_ZARR), mode="r")
    
    base_mask = base_grp["mask"]  # (T, N, N) or (T, N, N, 2)
    risk_mask = risk_grp["observed_mask"]  # (T, N, N, K) or (T, N, N, K, 2)
    
    T_base = base_mask.shape[0]
    T_risk = risk_mask.shape[0]
    
    print(f"Base mask shape: {base_mask.shape}")
    print(f"Risk mask shape: {risk_mask.shape}")
    
    # Compute observed counts per time index
    # For base: count_nonzero(mask[t] != 0)
    # For risk: count_nonzero(mask[t] != 0)
    
    base_counts = []
    for t in range(T_base):
        m = base_mask[t]
        c = int(np.count_nonzero(m != 0))
        base_counts.append(c)
    base_counts = np.array(base_counts)
    
    risk_counts = []
    for t in range(T_risk):
        m = risk_mask[t]
        c = int(np.count_nonzero(m != 0))
        risk_counts.append(c)
    risk_counts = np.array(risk_counts)
    
    # Find earliest t with observed cells
    base_observed = np.where(base_counts > 0)[0]
    risk_observed = np.where(risk_counts > 0)[0]
    
    earliest_base_t = int(base_observed[0]) if len(base_observed) > 0 else None
    earliest_risk_t = int(risk_observed[0]) if len(risk_observed) > 0 else None
    
    # Per-split stats
    def split_stats(counts, t_min, t_max):
        subset = counts[t_min:t_max+1]
        if len(subset) == 0:
            return {"n": 0, "min": None, "max": None, "median": None, "nonzero_count": 0}
        nonzero = subset[subset > 0]
        return {
            "n": int(len(subset)),
            "min": int(subset.min()),
            "max": int(subset.max()),
            "median": int(np.median(subset)),
            "nonzero_count": int(len(nonzero)),
            "first_nonzero_t": int(t_min + np.argmax(subset > 0)) if len(nonzero) > 0 else None,
        }
    
    report = {
        "base_mask": {
            "shape": list(base_mask.shape),
            "T": T_base,
            "earliest_observed_t": earliest_base_t,
            "total_observed_times": int(len(base_observed)),
            "splits": {name: split_stats(base_counts, t_min, t_max) for name, (t_min, t_max) in SPLITS.items()},
        },
        "risk_mask": {
            "shape": list(risk_mask.shape),
            "T": T_risk,
            "earliest_observed_t": earliest_risk_t,
            "total_observed_times": int(len(risk_observed)),
            "splits": {name: split_stats(risk_counts, t_min, t_max) for name, (t_min, t_max) in SPLITS.items()},
        },
        "observed_definition": "mask_codes != 0",
    }
    
    # Write report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    print(f">>> Report written: {REPORT_PATH}")
    print(f"    Earliest base observed t: {earliest_base_t}")
    print(f"    Earliest risk observed t: {earliest_risk_t}")
    print(f"    Base total observed times: {len(base_observed)}/{T_base}")
    print(f"    Risk total observed times: {len(risk_observed)}/{T_risk}")


if __name__ == "__main__":
    main()
