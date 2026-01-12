import sys
import os
import json
import zarr
import numpy as np
from pathlib import Path

# Ensure repo root is on path
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO_ROOT)

def audit_mask_semantics():
    inventory_path = Path("docs/reports/trade_spec_v1_2_zarr_inventory.json")
    if not inventory_path.exists():
        print("Error: Inventory file not found.")
        sys.exit(1)
        
    with open(inventory_path, 'r', encoding='utf-8') as f:
        inventory = json.load(f)
        
    def analyze_step(name, store_path, trade_key, mask_key):
        print(f"\nAnalyzing {name} semantics...")
        if not os.path.exists(store_path):
            return {"error": "Path not found"}
            
        root = zarr.open(store_path, mode='r')
        trade_arr = root[trade_key]
        mask_arr = root[mask_key]
        
        T = trade_arr.shape[0]
        indices = [0, 100, 300, 600, 815, 900]
        indices = [t for t in indices if t < T]
        
        informative_samples = []
        for t in indices:
            trade_raw = trade_arr[t]
            mask_raw = mask_arr[t].astype(np.int32)
            
            # unique codes
            unique_codes = np.unique(mask_raw)
            if len(unique_codes) == 1 and 0 in unique_codes and np.abs(trade_raw).sum() == 0:
                print(f"  t={t}: Uninformative (empty)")
                continue
                
            stats = {"t": t, "codes": {}}
            trade_abs = np.abs(trade_raw)
            
            for code in unique_codes:
                mask_c = (mask_raw == code)
                sum_trade = float(trade_abs[mask_c].sum())
                count = int(mask_c.sum())
                stats["codes"][int(code)] = {"sum_trade": sum_trade, "count": count}
                
            informative_samples.append(stats)
            
        if not informative_samples:
            return {"error": "No informative samples found"}
            
        # Aggregate mass per code
        total_mass_per_code = {}
        for sample in informative_samples:
            for code, s in sample["codes"].items():
                total_mass_per_code[code] = total_mass_per_code.get(code, 0.0) + s["sum_trade"]
                
        if not total_mass_per_code:
            return {"error": "No trade mass found"}
            
        observed_code = max(total_mass_per_code, key=total_mass_per_code.get)
        
        return {
            "informative_samples": informative_samples,
            "total_mass_per_code": total_mass_per_code,
            "observed_code": int(observed_code)
        }

    report = {
        "step1": analyze_step("Step 1", "data/processed/trade/imf_imts_step1/trade_fob_tensor.zarr", "trade", "mask"),
        "step2": analyze_step("Step 2", "data/processed/trade/faostat_step2/trade_risk_tensor.zarr", "trade_risk", "observed_mask")
    }
    
    with open("docs/reports/trade_spec_v1_2_mask_semantics.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
        
    print("\nMask Semantics Results:")
    for step, res in report.items():
        if "error" in res:
            print(f"  {step}: ERROR: {res['error']}")
        else:
            print(f"  {step}: observed_code={res['observed_code']}")

if __name__ == "__main__":
    audit_mask_semantics()
