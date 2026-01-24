"""
ST-MM-GNN Step A Training Entrypoint.

Usage:
    python tools/stmm_stepA_train.py --config config/stmm_stepA.yaml [--fast-dev-run]
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathograph.data.trade_datamodule import TradeDataModule, TradeDataModuleConfig
from pathograph.models.stmm_gwnet import STMMGraphWaveNet
from pathograph.pl.stmm_pl_module import STMMPLModule


def main():
    parser = argparse.ArgumentParser(description='Train ST-MM-GNN Layer A model')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--fast-dev-run', action='store_true', help='Run minimal dev loop')
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Set seed
    seed = cfg.get('seed', 1337)
    print(f"Setting seed: {seed}")
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)
    
    # Create run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path('runs') / 'stmm_stepA' / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    
    # Instantiate DataModule
    print("Instantiating DataModule...")
    dm_config = TradeDataModuleConfig(**cfg['datamodule'])
    dm = TradeDataModule(dm_config)
    
    # Instantiate Model
    print("Instantiating Model...")
    model = STMMGraphWaveNet(**cfg['model'])
    
    # Instantiate Lightning Module
    print("Instantiating Lightning Module...")
    pl_module = STMMPLModule(model, **cfg['optim'])
    
    # Configure Trainer
    trainer_cfg = cfg['trainer'].copy()
    trainer_cfg['default_root_dir'] = str(run_dir)
    
    if args.fast_dev_run:
        print("Running fast_dev_run...")
        trainer_cfg['fast_dev_run'] = True
    
    print("Creating Trainer...")
    trainer = pl.Trainer(**trainer_cfg)
    
    # Train
    print("Starting training...")
    trainer.fit(pl_module, datamodule=dm)
    
    print(f"Training complete. Logs/checkpoints saved to: {run_dir}")


if __name__ == '__main__':
    main()
