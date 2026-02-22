import pandas as pd
import numpy as np
import zarr
from numcodecs import Blosc
import json
import argparse
from pathlib import Path
import datetime

def load_node_index(path):
    df = pd.read_csv(path)
    return {row['iso3']: int(row['node_id']) for _, row in df.iterrows()}, len(df)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-path", default="data/processed/trade/dots_long.parquet")
    parser.add_argument("--node-index-path", default="data/processed/meta/node_index.csv")
    parser.add_argument("--out-zarr", default="data/processed/trade/trade_tensor_pilot.zarr")
    parser.add_argument("--out-qc", default="data/processed/trade/trade_qc_report_pilot.json")
    parser.add_argument("--start-month", default="2024-01")
    parser.add_argument("--end-month", default="2024-03")
    args = parser.parse_args()
    
    # 1. Setup dimensions
    iso_map, N = load_node_index(args.node_index_path)
    channels = ['exports', 'imports']
    months = pd.date_range(start=args.start_month, end=args.end_month, freq='MS').strftime('%Y-%m').tolist()
    T = len(months)
    C = len(channels)
    
    month_to_idx = {m: i for i, m in enumerate(months)}
    channel_to_idx = {c: i for i, c in enumerate(channels)}
    
    print(f"Tensor Shape: (T={T}, N={N}, N={N}, C={C})")
    
    # 2. Init Tensor
    # If using zarr, we can use a disk/memory store
    # For pilot size (3 * 194 * 194 * 2 * 4 bytes) ~ 900KB, memory is fine
    tensor = np.zeros((T, N, N, C), dtype=np.float32)
    mask = np.ones((T, N, N, C), dtype=np.bool_) # True = missing initially? Or False=observed?
    # Usually mask=1 means Observed, mask=0 means Missing. Or vice versa.
    # Let's define: Mask=1 if data present/imputed, 0 if genuinely missing/unknown. 
    # Actually, usually 1=valid.
    # But here, we default to 0 value. Let's assume missingness mask is separate.
    # If no record exists, is it 0 trade or missing report?
    # For DOTS, usually missing report. 
    # For this pilot, let's track "presence of report".
    
    # Initialize mask to False (Missing)
    # We will flip to True (Present) if we see a record.
    report_mask = np.zeros((T, N, N, C), dtype=np.bool_) 
    
    # 3. Load Data
    df = pd.read_parquet(args.long_path)
    
    # Filter to time range
    df = df[df['month'].isin(months)]
    
    print(f"Processing {len(df)} rows...")
    
    missing_nodes = 0
    mapped_rows = 0
    
    for _, row in df.iterrows():
        t_str = row['month']
        rep = row['reporter_iso3']
        par = row['partner_iso3']
        ch = row['channel']
        val = row['value']
        
        if rep not in iso_map or par not in iso_map:
            missing_nodes += 1
            continue
            
        t = month_to_idx.get(t_str)
        if t is None: continue
        
        i = iso_map[rep]
        j = iso_map[par]
        k = channel_to_idx.get(ch)
        
        if k is None: continue
        
        # Diagonal check (should be 0, but if data has it, we might want to drop or keep)
        if i == j: continue 
        
        # Accumulate (handle duplicates by summing)
        tensor[t, i, j, k] += val
        report_mask[t, i, j, k] = True
        mapped_rows += 1

    print(f"Mapped {mapped_rows} flows. Ignored {missing_nodes} flows involving non-canonical nodes.")
    
    # 4. Save Zarr
    # Use simple open() which handles directory stores automatically
    root = zarr.open_group(store=args.out_zarr, mode='w')
    
    root.create_dataset('trade_data', data=tensor, shape=tensor.shape, chunks=(1, N, N, C))
    root.create_dataset('mask', data=report_mask, shape=report_mask.shape, chunks=(1, N, N, C))
    
    # Save Metadata attributes
    root.attrs['months'] = months
    root.attrs['channels'] = channels
    root.attrs['node_count'] = N
    root.attrs['created_at'] = datetime.datetime.now(datetime.UTC).isoformat()
    
    # 5. QC Report
    non_zeros = int(np.count_nonzero(tensor))
    total_val = float(np.sum(tensor))
    
    # Top 5 corridors
    # Flatten to list of (t, i, j, k, val) is expensive for full tensor, but okay for pilot
    # Efficient top-k indices:
    flat_indices = np.argsort(tensor.flatten())[-20:][::-1]
    top_flows = []
    
    # Recover subscripts
    # idx = t*(N*N*C) + i*(N*C) + j*(C) + k
    for idx in flat_indices:
        val = float(tensor.flat[idx])
        if val == 0: continue
        
        linear_idx = idx
        k = linear_idx % C
        linear_idx //= C
        j = linear_idx % N
        linear_idx //= N
        i = linear_idx % N
        t = linear_idx // N # wait, t is remaining
        
        # Verify: t, i, j, k
        # t * (N*N*C) + i * (N*C) + j * C + k
        # Re-calc to be sure
        # shape is (T, N, N, C)
        # unravelling...
        t, i, j, k = np.unravel_index(idx, (T, N, N, C))
        
        # Get labels
        rep_iso = [k for k,v in iso_map.items() if v == i][0]
        par_iso = [k for k,v in iso_map.items() if v == j][0]
        
        top_flows.append({
            "month": months[t],
            "reporter": rep_iso,
            "partner": par_iso,
            "channel": channels[k],
            "value": val
        })

    qc = {
        "tensor_shape": list(tensor.shape),
        "non_zero_elements": non_zeros,
        "sparsity": 1.0 - (non_zeros / tensor.size),
        "total_value": total_val,
        "negative_values_count": int(np.sum(tensor < 0)),
        "missing_mask_coverage": float(np.mean(report_mask)),
        "top_20_flows": top_flows,
        "channels": channels,
        "time_range": [months[0], months[-1]]
    }
    
    with open(args.out_qc, "w") as f:
        json.dump(qc, f, indent=2)
        
    print(f"QC report written to {args.out_qc}")

if __name__ == "__main__":
    main()
