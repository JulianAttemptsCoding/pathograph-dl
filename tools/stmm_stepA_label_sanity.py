"""
STMM Label Sanity Checker.

Verifies that validation and test splits contain non-degenerate label distributions
under mask for each pathogen. Computes per-pathogen valid/pos/neg counts and flags
invariant violations.

Usage:
    python tools/stmm_stepA_label_sanity.py --config config/stmm_stepA.yaml --split all --strict
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from tqdm import tqdm

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathograph.data.trade_datamodule import TradeDataModule, TradeDataModuleConfig


def analyze_split(name: str, dataloader, num_pathogens: int = 8, max_batches: int = None) -> Dict:
    """
    Analyze a data split for per-pathogen label distribution under mask.
    
    Args:
        name: split name (e.g., 'val', 'test')
        dataloader: DataLoader to iterate
        num_pathogens: number of pathogens (default 8)
        max_batches: optional limit on batches to scan
    
    Returns:
        dict with per-pathogen stats and summary
    """
    print(f"[{name}] Analyzing split...")
    
    # Accumul ators
    pos_counts = torch.zeros(num_pathogens, dtype=torch.long)
    neg_counts = torch.zeros(num_pathogens, dtype=torch.long)
    obs_counts = torch.zeros(num_pathogens, dtype=torch.long)
    
    total_batches = 0
    total_samples = 0
    
    # Optional time index tracking
    time_indices_seen = []
    has_time_index = None
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"{name} batches")):
        if max_batches and batch_idx >= max_batches:
            break
        
        total_batches += 1
        
        y_next = batch['y_next']  # (B, N, P)
        y_mask = batch['y_mask']  # (B, N, P)
        
        B, N, P = y_next.shape
        total_samples += B * N
        
        # Check for optional time index
        if has_time_index is None:
            has_time_index = 't_next' in batch or 'time_index' in batch
        
        if has_time_index:
            t_key = 't_next' if 't_next' in batch else 'time_index'
            t_vals = batch[t_key]
            if t_vals.numel() > 0:
                time_indices_seen.extend(t_vals.flatten().tolist())
        
        # Flatten for per-pathogen counting
        y_flat = y_next.view(-1, P)  # (B*N, P)
        m_flat = y_mask.view(-1, P)  # (B*N, P)
        
        # Count observed, positives, negatives per pathogen
        observed = m_flat > 0.5
        pos = (y_flat > 0.5) & observed
        neg = (y_flat < 0.5) & observed
        
        pos_counts += pos.sum(dim=0).cpu()
        neg_counts += neg.sum(dim=0).cpu()
        obs_counts += observed.sum(dim=0).cpu()
    
    # Build per-pathogen stats
    per_pathogen = {}
    n_degenerate = 0
    n_with_any_valid = 0
    
    for p in range(num_pathogens):
        pc = int(pos_counts[p])
        nc = int(neg_counts[p])
        oc = int(obs_counts[p])
        
        is_degenerate = (pc == 0) or (nc == 0)
        if is_degenerate:
            n_degenerate += 1
        
        if oc > 0:
            n_with_any_valid += 1
        
        per_pathogen[f"p{p}"] = {
            "valid": oc,
            "pos": pc,
            "neg": nc,
            "prevalence": pc / oc if oc > 0 else 0.0,
            "is_degenerate": is_degenerate,
        }
    
    # Summary stats
    summary = {
        "total_batches": total_batches,
        "total_samples": total_samples,
        "total_valid_entries": int(obs_counts.sum()),
        "total_pos_entries": int(pos_counts.sum()),
        "total_neg_entries": int(neg_counts.sum()),
        "n_pathogens_with_any_valid": n_with_any_valid,
        "n_degenerate_pathogens": n_degenerate,
    }
    
    result = {
        "per_pathogen": per_pathogen,
        "summary": summary,
    }
    
    # Add time index info if available
    if has_time_index and time_indices_seen:
        time_unique = sorted(set(time_indices_seen))
        result["time_index_info"] = {
            "has_time_index": True,
            "min_t": min(time_unique),
            "max_t": max(time_unique),
            "unique_t_count": len(time_unique),
            "sample_t_values": time_unique[:10],  # First 10 for reference
        }
    else:
        result["time_index_info"] = {
            "has_time_index": False,
        }
    
    return result


def main():
    parser = argparse.ArgumentParser(description='STMM Label Sanity Check')
    parser.add_argument('--config', type=str, default='config/stmm_stepA.yaml',
                        help='Path to config YAML')
    parser.add_argument('--split', type=str, default='all', choices=['val', 'test', 'all'],
                        help='Which split(s) to check')
    parser.add_argument('--max_batches', type=int, default=50,
                        help='Maximum batches to scan per split')
    parser.add_argument('--strict', action='store_true',
                        help='Exit with error on invariant violations')
    parser.add_argument('--out_json', type=str, default=None,
                        help='Optional output JSON path')
    args = parser.parse_args()
    
    print(f"Loading config: {args.config}")
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Force single worker for stability
    cfg['datamodule']['num_workers'] = 0
    
    print("Instantiating DataModule...")
    dm_config = TradeDataModuleConfig(**cfg['datamodule'])
    dm = TradeDataModule(dm_config)
    dm.setup()
    
    # Determine splits to check
    splits_to_check = []
    if args.split == 'all':
        splits_to_check = ['val', 'test']
    else:
        splits_to_check = [args.split]
    
    results = {
        "meta": {
            "config_path": args.config,
            "max_batches": args.max_batches,
            "timestamp": datetime.now().isoformat(),
        },
        "splits": {},
    }
    
    # Try to get git commit
    try:
        import subprocess
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent.parent).decode('utf-8').strip()
        results["meta"]["git_commit"] = git_hash
    except:
        results["meta"]["git_commit"] = "unknown"
    
    # Analyze each split
    fail_reasons = []
    
    for split_name in splits_to_check:
        if split_name == 'val':
            loader = dm.val_dataloader()
        elif split_name == 'test':
            loader = dm.test_dataloader()
        else:
            continue
        
        split_result = analyze_split(split_name, loader, num_pathogens=8, max_batches=args.max_batches)
        results["splits"][split_name] = split_result
        
        # Check invariants in strict mode
        if args.strict:
            summary = split_result["summary"]
            per_pathogen = split_result["per_pathogen"]
            
            # Invariant: at least some pathogens should have valid entries
            if summary["n_pathogens_with_any_valid"] == 0:
                fail_reasons.append(
                    f"[{split_name}] FAIL: NO pathogens have any valid entries under mask. "
                    f"Mask may be wrong or batch construction is broken."
                )
            
            # Per-pathogen check: flag if ANY pathogen has zero valid
            for p_key, p_stats in per_pathogen.items():
                if p_stats["valid"] == 0:
                    fail_reasons.append(
                        f"[{split_name}] WARN: {p_key} has zero valid entries in scanned batches (max={args.max_batches})"
                    )
    
    # Print summary
    print("\n" + "="*60)
    print("LABEL SANITY CHECK RESULTS")
    print("="*60)
    
    for split_name, split_data in results["splits"].items():
        print(f"\n[{split_name.upper()}]")
        print(f"  Total batches scanned: {split_data['summary']['total_batches']}")
        print(f"  Total valid entries: {split_data['summary']['total_valid_entries']}")
        print(f"  Total pos entries: {split_data['summary']['total_pos_entries']}")
        print(f"  Total neg entries: {split_data['summary']['total_neg_entries']}")
        print(f"  Pathogens with any valid: {split_data['summary']['n_pathogens_with_any_valid']}/8")
        print(f"  Degenerate pathogens (no pos OR no neg): {split_data['summary']['n_degenerate_pathogens']}/8")
        
        if split_data["time_index_info"]["has_time_index"]:
            ti = split_data["time_index_info"]
            print(f"  Time index range: [{ti['min_t']}, {ti['max_t']}], unique count: {ti['unique_t_count']}")
        else:
            print("  Time index: NOT AVAILABLE in batch")
        
        # Show degenerate pathogens
        degenerate_list = [k for k, v in split_data["per_pathogen"].items() if v["is_degenerate"]]
        if degenerate_list:
            print(f"  Degenerate: {', '.join(degenerate_list)}")
    
    print("\n" + "="*60)
    
    # Output JSON
    results["gate_status"] = "PASS" if len(fail_reasons) == 0 else "FAIL"
    results["fail_reasons"] = fail_reasons
    
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to: {out_path}")
    else:
        print("\nJSON output (stdout):")
        print(json.dumps(results, indent=2))
    
    if fail_reasons:
        print("\nFAILURES:")
        for reason in fail_reasons:
            print(f"  - {reason}")
        if args.strict:
            sys.exit(1)
    else:
        print("\n[OK] All invariants passed")


if __name__ == "__main__":
    main()
