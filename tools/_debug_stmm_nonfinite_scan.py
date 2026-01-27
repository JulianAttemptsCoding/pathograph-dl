"""
Debug tool to scan STMM input artifacts for NaN/Inf values.

Scans:
- Trade tensors (base, risk)
- Climate tensor
- Climate anomalies (anomaly, zscore if present)
- Pathogen status tensor
- Meta matrices (adjacency, distance)

Exits with code 1 if any NaN/Inf found in critical arrays.
"""
import sys
from pathlib import Path
import numpy as np
import zarr

def check_array(name: str, arr: np.ndarray, sample_only: bool = False) -> bool:
    """Check array for NaN/Inf. Returns True if issues found."""
    if sample_only and arr.size > 10_000_000:
        # Sample for very large arrays
        sample_size = min(1000000, arr.size // 100)
        indices = np.random.choice(arr.size, sample_size, replace=False)
        flat = arr.flat
        sample = np.array([flat[i] for i in indices])
        arr_check = sample
        note = f" (sampled {sample_size}/{arr.size})"
    else:
        arr_check = arr
        note = ""
    
    nan_count = np.isnan(arr_check).sum()
    inf_count = np.isinf(arr_check).sum()
    total = arr_check.size
    
    if nan_count > 0 or inf_count > 0:
        print(f"❌ {name}: NaN={nan_count}/{total}, Inf={inf_count}/{total}{note}")
        return True
    else:
        print(f"✅ {name}: all finite{note}")
        return False

def main():
    repo_root = Path(__file__).parent.parent
    has_issues = False
    
    print("=== Scanning STMM Input Artifacts for NaN/Inf ===\n")
    
    # Climate tensor
    print("--- Climate Tensor ---")
    try:
        climate_path = repo_root / "data/processed/climate/climate_tensor.zarr"
        g = zarr.open_group(climate_path, mode='r')
        climate = g['climate'][:]
        has_issues |= check_array("climate_tensor", climate)
    except Exception as e:
        print(f"⚠️  Could not load climate tensor: {e}")
    
    # Climate anomalies
    print("\n--- Climate Anomalies ---")
    try:
        anoms_path = repo_root / "data/processed/climate/climate_step4/climate_anomalies.zarr"
        g_anoms = zarr.open_group(anoms_path, mode='r')
        
        if 'anomaly' in g_anoms:
            anomaly = g_anoms['anomaly'][:]
            has_issues |= check_array("climate_anomaly", anomaly)
        
        if 'zscore' in g_anoms:
            zscore = g_anoms['zscore'][:]
            has_issues |= check_array("climate_zscore", zscore)
    except Exception as e:
        print(f"⚠️  Could not load climate anomalies: {e}")
    
    # Pathogen status
    print("\n--- Pathogen Status ---")
    try:
        pathogen_path = repo_root / "data/processed/pathogen/status_tensor.zarr"
        g_path = zarr.open_group(pathogen_path, mode='r')
        status = g_path['status'][:]
        has_issues |= check_array("pathogen_status", status)
        
        status_mask = g_path['status_mask'][:]
        has_issues |= check_array("pathogen_status_mask", status_mask)
    except Exception as e:
        print(f"⚠️  Could not load pathogen status: {e}")
    
    # Meta adjacency
    print("\n--- Meta Matrices ---")
    try:
        adj_path = repo_root / "data/processed/meta/adjacency_border.npy"
        adjacency = np.load(adj_path)
        has_issues |= check_array("adjacency_border", adjacency)
        
        # Check for zero-degree rows
        row_sums = adjacency.sum(axis=1)
        zero_degree = (row_sums == 0).sum()
        if zero_degree > 0:
            print(f"⚠️  adjacency_border has {zero_degree} nodes with zero degree (isolated nodes)")
            print(f"   This may cause NaN in graph normalization!")
            has_issues = True
    except Exception as e:
        print(f"⚠️  Could not load adjacency: {e}")
    
    try:
        dist_path = repo_root / "data/processed/meta/distance_km.npy"
        distance = np.load(dist_path)
        has_issues |= check_array("distance_km", distance)
    except Exception as e:
        print(f"⚠️  Could not load distance: {e}")
    
    # Trade tensors (sample only - they're large)
    print("\n--- Trade Tensors (sampled) ---")
    try:
        base_path = repo_root / "data/processed/trade/imf_imts_step1/trade_fob_tensor.zarr"
        g_base = zarr.open_group(base_path, mode='r')
        base_trade = g_base['trade'][:100]  # First 100 timesteps
        has_issues |= check_array("base_trade (t=0:100)", base_trade)
    except Exception as e:
        print(f"⚠️  Could not load base trade: {e}")
    
    try:
        risk_path = repo_root / "data/processed/trade/faostat_step2/trade_risk_tensor.zarr"
        g_risk = zarr.open_group(risk_path, mode='r')
        risk_trade = g_risk['trade'][:100]  # First 100 timesteps
        has_issues |= check_array("risk_trade (t=0:100)", risk_trade)
    except Exception as e:
        print(f"⚠️  Could not load risk trade: {e}")
    
    print("\n" + "="*50)
    if has_issues:
        print("❌ FOUND NaN/Inf or zero-degree issues!")
        sys.exit(1)
    else:
        print("✅ All scanned arrays are finite")
        sys.exit(0)

if __name__ == "__main__":
    main()
