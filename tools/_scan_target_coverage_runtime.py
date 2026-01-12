"""Scan runtime dataset target coverage.

Samples indices from TradeDatasetZarr and records target mask coverage.
Writes JSON report showing mask counts for sampled indices.
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pathograph.data.trade_dataset import TradeDatasetConfig, TradeDatasetZarr, TradeSplit

CONFIG_PATH = ROOT / "config" / "trade_step3.yaml"

# Output path - caller can set via environment or we use default
REPORT_PATH_BEFORE = ROOT / "docs/reports/target_coverage_runtime_before.json"
REPORT_PATH_AFTER = ROOT / "docs/reports/target_coverage_runtime_after.json"


def sample_indices(length: int):
    """Return indices to sample for diagnostics."""
    if length == 0:
        return []
    indices = [0, 1, 2, length // 2, length - 3, length - 2, length - 1]
    # Filter valid and dedupe
    indices = sorted(set(i for i in indices if 0 <= i < length))
    return indices


def main():
    import os
    
    # Check if we should use "after" path (set by caller after fixes)
    report_path = REPORT_PATH_AFTER if os.environ.get("SCAN_AFTER") else REPORT_PATH_BEFORE
    
    print(f">>> Scanning runtime dataset target coverage...")
    print(f"    Report will be written to: {report_path}")
    
    # Load config
    with open(CONFIG_PATH, "r") as f:
        raw_cfg = yaml.safe_load(f)["trade_step3"]
    
    # Check if we should enable filtering (for "after" scan)
    enable_filtering = bool(os.environ.get("SCAN_AFTER"))
    
    # Build dataset config with targets enabled
    # Use standardize=False to speed up scan (we only care about masks)
    cfg = TradeDatasetConfig(
        base_zarr_path=str(ROOT / raw_cfg["inputs"]["base_zarr"]),
        risk_zarr_path=str(ROOT / raw_cfg["inputs"]["risk_zarr"]),
        lookback=raw_cfg["windowing"]["lookback_months"],
        horizon=raw_cfg["windowing"]["horizon_months"],
        split_train=TradeSplit(**raw_cfg["splits"]["train"]),
        split_val=TradeSplit(**raw_cfg["splits"]["val"]),
        split_test=TradeSplit(**raw_cfg["splits"]["test"]),
        split="train",
        apply_log1p=False,
        standardize=False,
        return_targets=True,
        target_kind="both",
        include_target_masks=True,
        require_target_observed=enable_filtering,
        min_target_observed=1,
        require_target_observed_kind="both",
    )
    
    print(f"    Dataset config: lookback={cfg.lookback}, horizon={cfg.horizon}")
    
    ds = TradeDatasetZarr(cfg)
    length = len(ds)
    print(f"    Dataset length: {length}")
    print(f"    t_start={ds.t_start}, t_end={ds.t_end}")
    
    # Sample indices
    indices = sample_indices(length)
    print(f"    Sampling indices: {indices}")
    
    samples = []
    for idx in indices:
        item = ds[idx]
        t = int(item["t"])
        t_y = int(item["t_y"])
        
        y_base_mask = item.get("y_base_mask")
        y_risk_mask = item.get("y_risk_mask")
        
        base_count = int(np.count_nonzero(y_base_mask)) if y_base_mask is not None else None
        risk_count = int(np.count_nonzero(y_risk_mask)) if y_risk_mask is not None else None
        
        samples.append({
            "idx": idx,
            "t": t,
            "t_y": t_y,
            "y_base_mask_count": base_count,
            "y_risk_mask_count": risk_count,
        })
        print(f"      idx={idx}, t={t}, t_y={t_y}, base_mask_count={base_count}, risk_mask_count={risk_count}")
    
    # Check if any samples have empty masks
    empty_base = sum(1 for s in samples if s["y_base_mask_count"] == 0)
    empty_risk = sum(1 for s in samples if s["y_risk_mask_count"] == 0)
    
    report = {
        "config": {
            "lookback": cfg.lookback,
            "horizon": cfg.horizon,
            "split": cfg.split,
            "target_kind": cfg.target_kind,
            "t_start": ds.t_start,
            "t_end": ds.t_end,
        },
        "dataset_length": length,
        "samples": samples,
        "summary": {
            "samples_with_empty_base_mask": empty_base,
            "samples_with_empty_risk_mask": empty_risk,
        },
    }
    
    # Write report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    print(f">>> Report written: {report_path}")
    if empty_base > 0 or empty_risk > 0:
        print(f"    WARNING: Found samples with empty masks (base={empty_base}, risk={empty_risk})")


if __name__ == "__main__":
    main()
