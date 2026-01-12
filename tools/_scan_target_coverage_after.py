"""Generate the 'after' runtime scan with filtering enabled."""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pathograph.data.trade_dataset import TradeDatasetConfig, TradeDatasetZarr, TradeSplit

CONFIG_PATH = ROOT / "config" / "trade_step3.yaml"
REPORT_PATH = ROOT / "docs/reports/target_coverage_runtime_after.json"


def sample_indices(length: int):
    if length == 0:
        return []
    indices = [0, 1, 2, length // 2, length - 3, length - 2, length - 1]
    return sorted(set(i for i in indices if 0 <= i < length))


def main():
    print(">>> Generating AFTER scan with filtering enabled...")
    
    with open(CONFIG_PATH, "r") as f:
        raw_cfg = yaml.safe_load(f)["trade_step3"]
    
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
        require_target_observed=True,  # ENABLED
        min_target_observed=1,
        require_target_observed_kind="both",
    )
    
    ds = TradeDatasetZarr(cfg)
    length = len(ds)
    print(f"    Filtered dataset length: {length}")
    print(f"    First valid t: {ds.first_valid_t}, Last valid t: {ds.last_valid_t}")
    
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
        print(f"      idx={idx}, t={t}, t_y={t_y}, base={base_count}, risk={risk_count}")
    
    empty_base = sum(1 for s in samples if s["y_base_mask_count"] == 0)
    empty_risk = sum(1 for s in samples if s["y_risk_mask_count"] == 0)
    
    report = {
        "config": {
            "lookback": cfg.lookback,
            "horizon": cfg.horizon,
            "split": cfg.split,
            "target_kind": cfg.target_kind,
            "require_target_observed": True,
            "require_target_observed_kind": "both",
            "first_valid_t": ds.first_valid_t,
            "last_valid_t": ds.last_valid_t,
        },
        "dataset_length_unfiltered": 792,
        "dataset_length_filtered": length,
        "samples": samples,
        "summary": {
            "samples_with_empty_base_mask": empty_base,
            "samples_with_empty_risk_mask": empty_risk,
        },
    }
    
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    print(f">>> Report written: {REPORT_PATH}")
    if empty_base == 0 and empty_risk == 0:
        print("    SUCCESS: All sampled indices have non-empty target masks!")
    else:
        print(f"    WARNING: Still have empty masks (base={empty_base}, risk={empty_risk})")


if __name__ == "__main__":
    main()
