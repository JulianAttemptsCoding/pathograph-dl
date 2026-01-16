"""Debug helper to discover and verify Climate Step 3 outputs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import zarr  # type: ignore


def main() -> None:
    zarr_path = Path("data/processed/climate/climate_tensor.zarr")
    
    if not zarr_path.exists():
        print(f"[GATE FAIL] Climate tensor not found at {zarr_path}", file=sys.stderr)
        sys.exit(1)
    
    g = zarr.open_group(str(zarr_path), mode="r")
    
    required = ["climate", "mask", "time_index", "feature_names"]
    for k in required:
        if k not in g:
            print(f"[GATE FAIL] Missing array '{k}' in {zarr_path}", file=sys.stderr)
            sys.exit(1)
    
    climate = g["climate"]
    mask = g["mask"]
    time_index = np.asarray(g["time_index"][:]).astype(np.int32)
    feature_names = list(np.asarray(g["feature_names"][:]).astype(str))
    
    print(f"climate_tensor.zarr discovered at: {zarr_path}")
    print(f"  climate shape: {climate.shape}, dtype: {climate.dtype}")
    print(f"  mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"  time_index shape: {time_index.shape}, dtype: {time_index.dtype}")
    print(f"    time_index min: {time_index.min()}, max: {time_index.max()}")
    print(f"  feature_names: {feature_names}")
    
    # Gate invariants from Step 1 requirements
    T, N, F = climate.shape
    
    if time_index.shape != (T,):
        print(f"[GATE FAIL] time_index shape {time_index.shape} != ({T},)", file=sys.stderr)
        sys.exit(1)
    
    if climate.shape != (T, N, F):
        print(f"[GATE FAIL] climate shape {climate.shape} not (T, N, F)", file=sys.stderr)
        sys.exit(1)
    
    if N != 194:
        print(f"[GATE FAIL] Expected N=194, got N={N}", file=sys.stderr)
        sys.exit(1)
    
    if F != 10:
        print(f"[GATE FAIL] Expected F=10, got F={F}", file=sys.stderr)
        sys.exit(1)
    
    if str(climate.dtype) not in ("float32", "<f4"):
        print(f"[GATE FAIL] climate dtype {climate.dtype} not float32", file=sys.stderr)
        sys.exit(1)
    
    if str(mask.dtype) not in ("uint8", "ubyte", "|u1"):
        print(f"[GATE FAIL] mask dtype {mask.dtype} not uint8", file=sys.stderr)
        sys.exit(1)
    
    if mask.shape != climate.shape:
        print(f"[GATE FAIL] mask shape {mask.shape} != climate shape {climate.shape}", file=sys.stderr)
        sys.exit(1)
    
    print(f"\n[OK] All gate invariants passed: T={T}, N={N}, F={F}")
    print(f"[OK] Output Zarr: {zarr_path}")
    print(f"[OK] Array names: climate, mask, time_index")


if __name__ == "__main__":
    main()
