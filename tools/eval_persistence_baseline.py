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
from pathlib import Path

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
    }
    with open(run_dir / 'run_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Create DataModule (same as STMM)
    # TradeDataModule uses config from YAML to ensure identical splits
    dm_config = TradeDataModuleConfig(**config['datamodule'])
    datamodule = TradeDataModule(dm_config)
    
    # Create persistence baseline model
    model = PersistenceBaseline(num_pathogens=8)
    
    # Create Trainer
    trainer = pl.Trainer(
        default_root_dir=str(run_dir),
        max_epochs=1,  # Persistence baseline doesn't train
        accelerator='auto',
        devices=1,
        deterministic=True,
        fast_dev_run=args.fast_dev_run,
        enable_checkpointing=False,
        logger=pl.loggers.CSVLogger(save_dir=run_dir, name='logs'),
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
    trainer.validate(model, datamodule=datamodule)
    
    # Test (evaluation on test split)
    print("\nRunning test...")
    trainer.test(model, datamodule=datamodule)
    
    print(f"\n{'='*80}")
    print("[OK] Persistence baseline evaluation complete")
    print(f"Results saved to: {run_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
