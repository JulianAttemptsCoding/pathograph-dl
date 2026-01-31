"""
STMM Step-A Evaluation & Report Tool.

Performs evaluation of STMM model and/or persistence baseline on validation/test splits.
Includes temperature scaling (fit on VAL, apply to TEST) and comparative reporting.

Usage:
    # Baseline only
    python tools/stmm_stepA_eval_report.py --config config/stmm_stepA.yaml --split all --max_batches 10
    
    # With model checkpoint
    python tools/stmm_stepA_eval_report.py --config config/stmm_stepA.yaml --ckpt runs/xxx/last.ckpt --split all
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from pathograph.baselines.persistence import PersistenceBaseline
from pathograph.calibration.temperature_scaling import TemperatureScaling
from pathograph.data.trade_datamodule import TradeDataModule, TradeDataModuleConfig
from pathograph.metrics import macro_nanmean, per_pathogen_metrics


def eval_model_on_split(model, dataloader, device, max_batches=None, desc="Eval"):
    """
    Run model on a dataloader and return accumulated predictions.
    
    Returns:
        (probs_all, logits_all, targets_all, mask_all) as CPU tensors
    """
    probs_batches = []
    logits_batches = []
    targets_batches = []
    mask_batches = []
    
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=desc)):
            if max_batches and batch_idx >= max_batches:
                break
            
            # Move batch to device
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
            
            # Forward
            if hasattr(model, 'forward'):
                logits = model(batch_dev)
            else:
                # Persistence baseline returns probs directly
                probs = model._predict_persistence(batch_dev)
                logits = torch.logit(torch.clamp(probs, 1e-7, 1-1e-7))
            
            probs = torch.sigmoid(logits)
            
            # Store CPU tensors
            probs_batches.append(probs.detach().cpu())
            logits_batches.append(logits.detach().cpu())
            targets_batches.append(batch['y_next'].detach().cpu())
            mask_batches.append(batch['y_mask'].detach().cpu())
    
    if len(probs_batches) == 0:
        return None, None, None, None
    
    probs_all = torch.cat(probs_batches, dim=0)
    logits_all = torch.cat(logits_batches, dim=0)
    targets_all = torch.cat(targets_batches, dim=0)
    mask_all = torch.cat(mask_batches, dim=0)
    
    return probs_all, logits_all, targets_all, mask_all


def main():
    parser = argparse.ArgumentParser(description='STMM Evaluation & Report')
    parser.add_argument('--config', type=str, default='config/stmm_stepA.yaml')
    parser.add_argument('--ckpt', type=str, default=None, 
                        help='Model checkpoint (if omitted, baseline-only mode)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: runs/stmm_eval/<timestamp>)')
    parser.add_argument('--split', type=str, default='all', choices=['val', 'test', 'all'])
    parser.add_argument('--max_batches', type=int, default=None)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    
    # Setup output directory
    if args.out_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = Path('runs') / 'stmm_eval' / timestamp
    else:
        out_dir = Path(args.out_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    
    # Load config
    print(f"Loading config: {args.config}")
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Save config copy
    with open(out_dir / 'config_used.yaml', 'w') as f:
        yaml.dump(cfg, f)
    
    # Get git commit
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], 
            cwd=Path(__file__).parent.parent
        ).decode('utf-8').strip()
    except:
        git_hash = "unknown"
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    
    # Load DataModule
    print("Loading DataModule...")
    cfg['datamodule']['num_workers'] = 0  # Force single worker for stability
    dm_config = TradeDataModuleConfig(**cfg['datamodule'])
    dm = TradeDataModule(dm_config)
    dm.setup()
    
    # Determine splits to evaluate
    eval_splits = ['val', 'test'] if args.split == 'all' else [args.split]
    
    # Load models
    print("Loading baseline...")
    baseline = PersistenceBaseline(num_pathogens=8)
    
    stmm_model = None
    if args.ckpt:
        print(f"Loading STMM model from: {args.ckpt}")
        from pathograph.pl.stmm_pl_module import STMMPLModule
        from pathograph.models.stmm_gwnet import STMMGraphWaveNet
        
        model_arch = STMMGraphWaveNet(**cfg['model'])
        stmm_model = STMMPLModule.load_from_checkpoint(args.ckpt, model=model_arch)
    
    # Results storage
    results = {
        'meta': {
            'config_path': args.config,
            'git_commit': git_hash,
            'timestamp': datetime.now().isoformat(),
            'max_batches': args.max_batches,
            'device': args.device,
            'has_model': stmm_model is not None,
        },
        'splits': {}
    }
    
    # Temperature scaling placeholder
    temperature_results = None
    
    # Evaluate each split
    for split_name in eval_splits:
        print(f"\n{'='*60}")
        print(f"Evaluating {split_name.upper()} split")
        print(f"{'='*60}")
        
        if split_name == 'val':
            loader = dm.val_dataloader()
        else:
            loader = dm.test_dataloader()
        
        split_results = {'split': split_name}
        
        # Evaluate baseline
        print(f"\n[{split_name}] Running persistence baseline...")
        bl_probs, bl_logits, bl_targets, bl_mask = eval_model_on_split(
            baseline, loader, device, 
            max_batches=args.max_batches,
            desc=f"{split_name}_baseline"
        )
        
        if bl_probs is None:
            print(f"WARNING: No data for {split_name} split")
            continue
        
        # Compute baseline metrics
        bl_metrics = per_pathogen_metrics(bl_probs, bl_targets, bl_mask)
        bl_macro_auroc, bl_n_auroc = macro_nanmean(bl_metrics['auroc'])
        bl_macro_auprc, bl_n_auprc = macro_nanmean(bl_metrics['auprc'])
        
        split_results['baseline'] = {
            'per_pathogen': {},
            'macro': {
                'auroc': float(bl_macro_auroc),
                'auprc': float(bl_macro_auprc),
                'n_valid_auroc': int(bl_n_auroc),
                'n_valid_auprc': int(bl_n_auprc),
            }
        }
        
        for p in range(8):
            split_results['baseline']['per_pathogen'][f'p{p}'] = {
                'auroc': float(bl_metrics['auroc'][p]),
                'auprc': float(bl_metrics['auprc'][p]),
                'valid': int(bl_metrics['valid'][p]),
                'pos': int(bl_metrics['pos'][p]),
                'neg': int(bl_metrics['neg'][p]),
            }
        
        # Evaluate model if provided
        if stmm_model:
            print(f"\n[{split_name}] Running STMM model...")
            model_probs, model_logits, model_targets, model_mask = eval_model_on_split(
                stmm_model, loader, device,
                max_batches=args.max_batches,
                desc=f"{split_name}_model"
            )
            
            # Compute raw model metrics
            model_metrics_raw = per_pathogen_metrics(model_probs, model_targets, model_mask)
            model_macro_auroc_raw, model_n_auroc_raw = macro_nanmean(model_metrics_raw['auroc'])
            model_macro_auprc_raw, model_n_auprc_raw = macro_nanmean(model_metrics_raw['auprc'])
            
            split_results['model'] = {
                'per_pathogen': {},
                'macro_raw': {
                    'auroc': float(model_macro_auroc_raw),
                    'auprc': float(model_macro_auprc_raw),
                    'n_valid_auroc': int(model_n_auroc_raw),
                    'n_valid_auprc': int(model_n_auprc_raw),
                }
            }
            
            for p in range(8):
                split_results['model']['per_pathogen'][f'p{p}'] = {
                    'auroc_raw': float(model_metrics_raw['auroc'][p]),
                    'auprc_raw': float(model_metrics_raw['auprc'][p]),
                    'valid': int(model_metrics_raw['valid'][p]),
                    'pos': int(model_metrics_raw['pos'][p]),
                    'neg': int(model_metrics_raw['neg'][p]),
                }
            
            # Temperature scaling: fit on VAL, apply to TEST
            if split_name == 'val':
                print(f"\n[VAL] Fitting temperature scaling...")
                temp_scaler = TemperatureScaling()
                temp_scaler.fit(model_logits, model_targets, model_mask, max_iter=50)
                temperature = float(temp_scaler.temperature.item())
                
                # Compute NLL before/after on VAL
                from torch.nn.functional import binary_cross_entropy_with_logits
                mask_flat = model_mask.flatten()
                observed = mask_flat > 0.5
                
                logits_obs = model_logits.flatten()[observed]
                targets_obs = model_targets.flatten()[observed]
                
                nll_before = binary_cross_entropy_with_logits(
                    logits_obs, targets_obs, reduction='mean'
                ).item()
                
                scaled_logits_obs = logits_obs / temperature
                nll_after = binary_cross_entropy_with_logits(
                    scaled_logits_obs, targets_obs, reduction='mean'
                ).item()
                
                temperature_results = {
                    'temperature': temperature,
                    'val_nll_before': nll_before,
                    'val_nll_after': nll_after,
                    'fit_sample_count': int(targets_obs.numel()),
                }
                
                print(f"  Temperature: {temperature:.4f}")
                print(f"  VAL NLL: {nll_before:.4f} -> {nll_after:.4f}")
            
            elif split_name == 'test' and temperature_results is not None:
                print(f"\n[TEST] Applying temperature scaling...")
                temperature = temperature_results['temperature']
                
                # Apply temperature
                scaled_logits = model_logits / temperature
                calibrated_probs = torch.sigmoid(scaled_logits)
                
                # Compute calibrated metrics
                model_metrics_cal = per_pathogen_metrics(calibrated_probs, model_targets, model_mask)
                model_macro_auroc_cal, model_n_auroc_cal = macro_nanmean(model_metrics_cal['auroc'])
                model_macro_auprc_cal, model_n_auprc_cal = macro_nanmean(model_metrics_cal['auprc'])
                
                split_results['model']['macro_calibrated'] = {
                    'auroc': float(model_macro_auroc_cal),
                    'auprc': float(model_macro_auprc_cal),
                    'n_valid_auroc': int(model_n_auroc_cal),
                    'n_valid_auprc': int(model_n_auprc_cal),
                }
                
                for p in range(8):
                    split_results['model']['per_pathogen'][f'p{p}']['auroc_cal'] = float(model_metrics_cal['auroc'][p])
                    split_results['model']['per_pathogen'][f'p{p}']['auprc_cal'] = float(model_metrics_cal['auprc'][p])
                
                # Compute TEST NLL before/after
                mask_flat = model_mask.flatten()
                observed = mask_flat > 0.5
                logits_obs = model_logits.flatten()[observed]
                targets_obs = model_targets.flatten()[observed]
                
                from torch.nn.functional import binary_cross_entropy_with_logits
                nll_before = binary_cross_entropy_with_logits(
                    logits_obs, targets_obs, reduction='mean'
                ).item()
                
                scaled_logits_obs = logits_obs / temperature
                nll_after = binary_cross_entropy_with_logits(
                    scaled_logits_obs, targets_obs, reduction='mean'
                ).item()
                
                temperature_results['test_nll_before'] = nll_before
                temperature_results['test_nll_after'] = nll_after
                
                print(f"  TEST NLL: {nll_before:.4f} -> {nll_after:.4f}")
        
        results['splits'][split_name] = split_results
    
    # Write outputs
    print(f"\n{'='*60}")
    print("Writing outputs...")
    print(f"{'='*60}")
    
    # Per-split JSON
    for split_name, split_data in results['splits'].items():
        out_path = out_dir / f"{split_name}_metrics.json"
        with open(out_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"Wrote: {out_path}")
    
    # Temperature scaling JSON
    if temperature_results:
        out_path = out_dir / "temperature_scaling.json"
        with open(out_path, 'w') as f:
            json.dump(temperature_results, f, indent=2)
        print(f"Wrote: {out_path}")
    
    # Per-pathogen CSV
    csv_rows = []
    for p in range(8):
        row = {'pathogen': f'p{p}'}
        
        # Get data from test split if available, else val
        source_split = 'test' if 'test' in results['splits'] else 'val'
        split_data = results['splits'][source_split]
        
        # Baseline data
        bl_data = split_data['baseline']['per_pathogen'][f'p{p}']
        row['valid'] = bl_data['valid']
        row['pos'] = bl_data['pos']
        row['neg'] = bl_data['neg']
        row['baseline_auroc'] = bl_data['auroc']
        row['baseline_auprc'] = bl_data['auprc']
        
        # Model data (if available)
        if 'model' in split_data:
            model_data = split_data['model']['per_pathogen'][f'p{p}']
            row['model_auroc_raw'] = model_data['auroc_raw']
            row['model_auprc_raw'] = model_data['auprc_raw']
            
            if 'auroc_cal' in model_data:
                row['model_auroc_cal'] = model_data['auroc_cal']
                row['model_auprc_cal'] = model_data['auprc_cal']
        
        csv_rows.append(row)
    
    csv_path = out_dir / "per_pathogen_metrics.csv"
    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_path, index=False)
    print(f"Wrote: {csv_path}")
    
    # Generate report markdown
    report_lines = [
        "# STMM Layer-A Evaluation Report",
        "",
        "## Configuration",
        f"- Config: `{args.config}`",
        f"- Git commit: `{git_hash}`",
        f"- Timestamp: {results['meta']['timestamp']}",
        f"- Max batches: {args.max_batches if args.max_batches else 'full split'}",
        f"- Device: {args.device}",
        "",
        "## Results Summary",
        ""
    ]
    
    # Add per-split results
    for split_name, split_data in results['splits'].items():
        report_lines.append(f"### {split_name.upper()} Split")
        report_lines.append("")
        
        # Baseline
        bl_macro = split_data['baseline']['macro']
        report_lines.append("**Persistence Baseline:**")
        report_lines.append(f"- Macro AUROC: {bl_macro['auroc']:.4f} (n={bl_macro['n_valid_auroc']}/8)")
        report_lines.append(f"- Macro AUPRC: {bl_macro['auprc']:.4f} (n={bl_macro['n_valid_auprc']}/8)")
        report_lines.append("")
        
        # Model (if present)
        if 'model' in split_data:
            model_macro_raw = split_data['model']['macro_raw']
            report_lines.append("**STMM Model (Raw):**")
            report_lines.append(f"- Macro AUROC: {model_macro_raw['auroc']:.4f} (n={model_macro_raw['n_valid_auroc']}/8)")
            report_lines.append(f"- Macro AUPRC: {model_macro_raw['auprc']:.4f} (n={model_macro_raw['n_valid_auprc']}/8)")
            
            if 'macro_calibrated' in split_data['model']:
                model_macro_cal = split_data['model']['macro_calibrated']
                report_lines.append("")
                report_lines.append("**STMM Model (Calibrated):**")
                report_lines.append(f"- Macro AUROC: {model_macro_cal['auroc']:.4f} (n={model_macro_cal['n_valid_auroc']}/8)")
                report_lines.append(f"- Macro AUPRC: {model_macro_cal['auprc']:.4f} (n={model_macro_cal['n_valid_auprc']}/8)")
            
            report_lines.append("")
        
        report_lines.append("")
    
    # GO/NO-GO decision
    if 'test' in results['splits'] and 'model' in results['splits']['test']:
        test_data = results['splits']['test']
        bl_auprc = test_data['baseline']['macro']['auprc']
        model_auprc = test_data['model']['macro_raw']['auprc']
        
        report_lines.append("## GO/NO-GO Decision")
        report_lines.append("")
        report_lines.append(f"**Rule:** Model macro AUPRC must exceed baseline by at least 0.01")
        report_lines.append("")
        report_lines.append(f"- Baseline AUPRC: {bl_auprc:.4f}")
        report_lines.append(f"- Model AUPRC (raw): {model_auprc:.4f}")
        report_lines.append(f"- Delta: {model_auprc - bl_auprc:+.4f}")
        report_lines.append("")
        
        if np.isnan(model_auprc):
            decision = "INCONCLUSIVE"
            reason = "Model AUPRC is NaN (degenerate metrics)"
        elif model_auprc >= bl_auprc + 0.01:
            decision = "GO"
            reason = "Model beats baseline by required margin"
        elif model_auprc < bl_auprc:
            decision = "NO-GO"
            reason = "Model underperforms baseline"
        else:
            decision = "INCONCLUSIVE"
            reason = "Model improvement is marginal (< 0.01)"
        
        report_lines.append(f"**Decision: {decision}**")
        report_lines.append(f"- Reason: {reason}")
        report_lines.append("")
    
    report_path = out_dir / "report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"Wrote: {report_path}")
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
