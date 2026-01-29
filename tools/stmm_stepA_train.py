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
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--run_dir', type=str, default=None, help='Override run directory')
    parser.add_argument('--max_epochs', type=int, default=None, help='Override max epochs')
    parser.add_argument('--early_stop_metric', type=str, default=None, help='Early stopping metric')
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Set seed
    seed = args.seed if args.seed is not None else cfg.get('seed', 1337)
    print(f"Setting seed: {seed}")
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)
    
    # Create run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = Path('runs') / 'stmm_stepA' / timestamp
    
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    
    # Save config copy
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f)
    
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
    
    if args.max_epochs is not None:
        trainer_cfg['max_epochs'] = args.max_epochs
        
    callbacks = []
    # Checkpoint callback
    checkpoint_cfg = cfg.get('checkpoint', {})
    if args.early_stop_metric:
        checkpoint_cfg['monitor'] = args.early_stop_metric
        
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=run_dir,
        filename='{epoch}-{step}-{' + checkpoint_cfg.get('monitor', 'val_loss') + ':.4f}',
        **checkpoint_cfg
    )
    callbacks.append(checkpoint_callback)
    
    if args.early_stop_metric:
        print(f"Adding EarlyStopping on {args.early_stop_metric}")
        early_stop = pl.callbacks.EarlyStopping(
            monitor=args.early_stop_metric,
            patience=5,
            mode='max' if 'au' in args.early_stop_metric else 'min'
        )
        callbacks.append(early_stop)
    
    if args.fast_dev_run:
        print("Running fast_dev_run...")
        trainer_cfg['fast_dev_run'] = True
    
    print("Creating Trainer...")
    # Remove callbacks from cfg if present to avoid dupe or conflict if we pass list
    if 'callbacks' in trainer_cfg:
        del trainer_cfg['callbacks']
        
    trainer = pl.Trainer(callbacks=callbacks, **trainer_cfg)
    
    # Train
    print("Starting training...")
    trainer.fit(pl_module, datamodule=dm)
    
    print(f"Training complete. Logs/checkpoints saved to: {run_dir}")


if __name__ == '__main__':
    main()
