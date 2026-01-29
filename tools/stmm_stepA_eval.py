"""
ST-MM-GNN Step A Evaluation Entrypoint.

Usage:
    python tools/stmm_stepA_eval.py --config config/stmm_stepA.yaml --ckpt runs/stmm_stepA/.../best.ckpt --run_dir runs/stmm_stepA/eval_gate_seed42
"""

import argparse
import yaml
import pytorch_lightning as pl
from pathlib import Path
import torch

from pathograph.data.trade_datamodule import TradeDataModule, TradeDataModuleConfig
from pathograph.pl.stmm_pl_module import STMMPLModule

def main():
    parser = argparse.ArgumentParser(description='Evaluate ST-MM-GNN Layer A model')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--run_dir', type=str, required=True, help='Directory to save results')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    pl.seed_everything(cfg.get('seed', 1337), workers=True)
    
    # Run Dir
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # DataModule
    dm_config = TradeDataModuleConfig(**cfg['datamodule'])
    dm = TradeDataModule(dm_config)
    
    # Model
    from pathograph.models.stmm_gwnet import STMMGraphWaveNet
    model_arch = STMMGraphWaveNet(**cfg['model'])
    model = STMMPLModule.load_from_checkpoint(args.ckpt, model=model_arch)
    model.eval()
    
    # Trainer
    trainer = pl.Trainer(
        default_root_dir=str(run_dir),
        accelerator='auto',
        devices=1,
        logger=pl.loggers.CSVLogger(save_dir=run_dir, name='eval_logs'),
        enable_checkpointing=False
    )
    
    print(f"Evaluating checkpoint: {args.ckpt}")
    trainer.test(model, datamodule=dm)
    
    print("Evaluation complete.")

if __name__ == '__main__':
    main()
