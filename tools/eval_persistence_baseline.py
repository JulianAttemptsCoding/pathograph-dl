"""
Evaluate persistence baseline on ST-MM-GNN splits.

Usage:
    python tools/eval_persistence_baseline.py --config config/stmm_stepA.yaml --run_dir runs/persistence_baseline/run1

This script uses identical splits and DataModule as STMM training for fair comparison.
"""

import argparse
import json
import shutil
import subprocess
import pandas as pd
from pathlib import Path
import time

import pytorch_lightning as pl
import torch
import yaml

from pathograph.baselines import PersistenceBaseline
from pathograph.data.trade_datamodule import TradeDataModule, TradeDataModuleConfig



def get_git_info():
    """Get current git commit SHA."""
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.STDOUT,
            text=True
        ).strip()
        return sha
    except Exception:
        return 'unknown'

def load_metrics_from_csv(csv_path):
    """Load metrics from the latest version in CSV logs."""
    try:
        df = pd.read_csv(csv_path)
        # Get the last valid value for each column
        metrics = {}
        for col in df.columns:
            # Drop NaNs and take last
            valid = df[col].dropna()
            if not valid.empty:
                metrics[col] = float(valid.iloc[-1])
        return metrics
    except Exception as e:
        print(f"Error reading metrics CSV: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description='Evaluate persistence baseline')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--run_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--fast_dev_run', action='store_true', help='Fast dev run for testing')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Set seed
    pl.seed_everything(args.seed, workers=True)
    
    # Create run directory
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config snapshot
    config_snapshot = run_dir / 'config.yaml'
    shutil.copy(args.config, config_snapshot)
    
    # Create run manifest
    manifest = {
        'model': 'persistence_baseline',
        'config': args.config,
        'seed': args.seed,
        'git_commit': get_git_info(),
        'timestamp': time.time()
    }
    with open(run_dir / 'run_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Create DataModule (same as STMM)
    # TradeDataModule uses config from YAML to ensure identical splits
    dm_config_dict = config['datamodule']
    # Ensure workers=0 for stability if needed, or keep config
    # dm_config_dict['num_workers'] = 0 
    dm_config = TradeDataModuleConfig(**dm_config_dict)
    datamodule = TradeDataModule(dm_config)
    
    # Create persistence baseline model
    model = PersistenceBaseline(num_pathogens=8)
    
    # Create Trainer
    csv_logger = pl.loggers.CSVLogger(save_dir=run_dir, name='logs')
    
    trainer = pl.Trainer(
        default_root_dir=str(run_dir),
        max_epochs=1,  # Persistence baseline doesn't train
        accelerator='auto',
        devices=1,
        deterministic=True,
        fast_dev_run=args.fast_dev_run,
        enable_checkpointing=False,
        logger=csv_logger,
    )
    
    print(f"\n{'='*80}")
    print("Evaluating Persistence Baseline")
    print(f"{'='*80}")
    print(f"Config: {args.config}")
    print(f"Run dir: {run_dir}")
    print(f"Seed: {args.seed}")
    print(f"Git commit: {manifest['git_commit']}")
    print(f"{'='*80}\n")
    
    # Validate (evaluation on val split)
    print("\nRunning validation...")
    val_results = trainer.validate(model, datamodule=datamodule)
    
    # Test (evaluation on test split)
    print("\nRunning test...")
    test_results = trainer.test(model, datamodule=datamodule)
    
    # Consolidate metrics
    # Trainer returns list of dicts (one per dataloader)
    final_metrics = {}
    if val_results:
        final_metrics.update(val_results[0])
    if test_results:
        final_metrics.update(test_results[0])
        
    # Also try to read from CSV to capture everything logged during epochs
    csv_path = Path(csv_logger.log_dir) / 'metrics.csv'
    if csv_path.exists():
        logged_metrics = load_metrics_from_csv(csv_path)
        # Update final_metrics with logged ones (prefer returned ones? actually logged ones might contain accumulators if not returned)
        # val/test results from trainer usually contain the returns of validation_epoch_end if returned, or just logged metrics.
        # Let's merge, preferring final_metrics for conflicts but taking logged for extras.
        for k, v in logged_metrics.items():
            if k not in final_metrics:
                final_metrics[k] = v
    
    # Write metrics.json
    metrics_json_path = run_dir / 'metrics.json'
    with open(metrics_json_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
        
    # Write metrics.md summary
    metrics_md_path = run_dir / 'metrics.md'
    with open(metrics_md_path, 'w') as f:
        f.write("# Persistence Baseline Results\n\n")
        f.write(f"**Run Dir**: `{run_dir}`\n")
        f.write(f"**Config**: `{args.config}`\n\n")
        f.write("## Key Metrics\n\n")
        
        keys_of_interest = [
            'val_auprc_macro', 'test_auprc_macro', 
            'val_auroc_macro', 'test_auroc_macro',
            'test_pos_total', 'macro_valid_pathogens'
        ]
        
        f.write("| Metric | Value |\n")
        f.write("|---|---|\n")
        for k in keys_of_interest:
            val = final_metrics.get(k, 'N/A')
            if isinstance(val, float):
                f.write(f"| {k} | {val:.4f} |\n")
            else:
                f.write(f"| {k} | {val} |\n")
        
        f.write("\n## All Metrics\n\n")
        f.write("```json\n")
        f.write(json.dumps(final_metrics, indent=2))
        f.write("\n```\n")
    
    print(f"\n{'='*80}")
    print("[OK] Persistence baseline evaluation complete")
    print(f"Metrics saved to: {metrics_json_path}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
