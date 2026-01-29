import argparse
import json
import yaml
import torch
import zarr
import numpy as np
from pathlib import Path
from tqdm import tqdm

from pathograph.data.trade_datamodule import TradeDataModule, TradeDataModuleConfig

def analyze_split(name, dataloader, num_pathogens=8):
    print(f"Analyzing split: {name}")
    stats = {
        "split": name,
        "total_batches": 0,
        "total_samples": 0,
        "pathogen_stats": {},
        "macro_valid_pathogens": 0
    }
    
    # Initialize counts
    pos_counts = torch.zeros(num_pathogens, dtype=torch.long)
    neg_counts = torch.zeros(num_pathogens, dtype=torch.long)
    obs_counts = torch.zeros(num_pathogens, dtype=torch.long)
    
    for batch in tqdm(dataloader, desc=name):
        stats["total_batches"] += 1
        y_next = batch['y_next'] # (B, N, P)
        y_mask = batch['y_mask'] # (B, N, P)
        
        B, N, P = y_next.shape
        stats["total_samples"] += B * N
        
        # Flatten
        y_flat = y_next.view(-1, P)
        m_flat = y_mask.view(-1, P)
        
        # Count
        observed = m_flat > 0.5
        pos = (y_flat > 0.5) & observed
        neg = (y_flat < 0.5) & observed
        
        pos_counts += pos.sum(dim=0).cpu()
        neg_counts += neg.sum(dim=0).cpu()
        obs_counts += observed.sum(dim=0).cpu()
        
    # Process stats per pathogen
    valid_p_count = 0
    for p in range(num_pathogens):
        pc = int(pos_counts[p])
        nc = int(neg_counts[p])
        oc = int(obs_counts[p])
        
        is_degenerate = (pc == 0) or (nc == 0)
        
        p_stat = {
            "positives": pc,
            "negatives": nc,
            "observed": oc,
            "prevalence": pc / oc if oc > 0 else 0.0,
            "is_degenerate": is_degenerate
        }
        stats["pathogen_stats"][f"p{p}"] = p_stat
        
        if not is_degenerate:
            valid_p_count += 1
            
    stats["macro_valid_pathogens"] = valid_p_count
    
    # Global totals
    stats["total_positives"] = int(pos_counts.sum())
    stats["total_negatives"] = int(neg_counts.sum())
    
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()
    
    print(f"Loading config: {args.config}")
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    print("Initializing DataModule...")
    # Force single worker for stability in script
    cfg['datamodule']['num_workers'] = 0
    dm_config = TradeDataModuleConfig(**cfg['datamodule'])
    dm = TradeDataModule(dm_config)
    dm.setup()
    
    results = {
        "config_split_indices": {
            "train": cfg['datamodule']['split_train'],
            "val": cfg['datamodule']['split_val'],
            "test": cfg['datamodule']['split_test']
        },
        "splits": {}
    }
    
    # Train
    train_stats = analyze_split("train", dm.train_dataloader())
    results["splits"]["train"] = train_stats
    
    # Val
    val_stats = analyze_split("val", dm.val_dataloader())
    results["splits"]["val"] = val_stats
    
    # Test
    test_stats = analyze_split("test", dm.test_dataloader())
    results["splits"]["test"] = test_stats
    
    # Check PASS/FAIL
    test_valid_p = test_stats["macro_valid_pathogens"]
    test_pos_total = test_stats["total_positives"]
    
    # Criteria:
    # 1. Macro average min valid pathogens >= 2
    # 2. Min total positives test >= 10
    
    pass_gate = True
    fail_reasons = []
    
    if test_valid_p < 2:
        pass_gate = False
        fail_reasons.append(f"Test split has only {test_valid_p} valid pathogens (min 2 required for macro).")
        
    if test_pos_total < 10:
        pass_gate = False
        fail_reasons.append(f"Test split has only {test_pos_total} total positives (min 10 required).")
        
    results["gate_status"] = "PASS" if pass_gate else "FAIL"
    results["fail_reasons"] = fail_reasons
    
    # Write output
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
        
    print(f"Gate Status: {results['gate_status']}")
    if not pass_gate:
        print("Failures:", fail_reasons)
        exit(1) # Fail the script if gate fails

if __name__ == "__main__":
    main()
