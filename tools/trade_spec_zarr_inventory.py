import sys
import os
import json
import zarr
import numpy as np
from pathlib import Path

# Ensure repo root is on path
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO_ROOT)

def get_zarr_inventory(store_path):
    if not os.path.exists(store_path):
        return {"error": f"Path {store_path} not found"}
    
    inventory = {}
    try:
        # Open group
        root = zarr.open(store_path, mode='r')
        
        # In Zarr v3, use members()
        def walk(group, path=""):
            for name, member in group.members():
                full_name = f"{path}/{name}" if path else name
                if isinstance(member, zarr.Array):
                    inventory[full_name] = {
                        "shape": member.shape,
                        "data_type": str(member.dtype),
                        "chunk_shape": member.chunks,
                        "fill_value": str(member.fill_value) if member.fill_value is not None else None,
                    }
                elif isinstance(member, zarr.Group):
                    walk(member, full_name)
                    
        walk(root)
        
    except Exception as e:
        inventory["error"] = f"{type(e).__name__}: {str(e)}"
    
    return inventory

def main():
    step1_path = "data/processed/trade/imf_imts_step1/trade_fob_tensor.zarr"
    step2_path = "data/processed/trade/faostat_step2/trade_risk_tensor.zarr"
    
    inventory = {
        "step1": get_zarr_inventory(step1_path),
        "step2": get_zarr_inventory(step2_path)
    }
    
    os.makedirs("docs/reports", exist_ok=True)
    with open("docs/reports/trade_spec_v1_2_zarr_inventory.json", 'w', encoding='utf-8') as f:
        json.dump(inventory, f, indent=2)
        
    # Print summary
    print("Zarr Inventory Summary:")
    for step, items in inventory.items():
        print(f"\n{step.upper()}:")
        if "error" in items:
            print(f"  Error: {items['error']}")
            continue
        for name, info in items.items():
            print(f"  - {name}: shape={info['shape']}, dtype={info['data_type']}")

if __name__ == "__main__":
    main()
