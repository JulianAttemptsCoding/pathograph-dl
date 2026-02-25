"""
ST-MM-GNN Step A Training Entrypoint.

Usage:
    python tools/stmm_stepA_train.py --config config/stmm_stepA.yaml [--fast-dev-run]
"""

import argparse
import inspect
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import torch
import yaml

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathograph.data.trade_datamodule import TradeDataModule, TradeDataModuleConfig
from pathograph.models.stmm_gwnet import STMMGraphWaveNet
from pathograph.pl.stmm_pl_module import STMMPLModule


def filter_kwargs(callable_obj: Callable, kwargs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of kwargs_dict containing only keys accepted by callable_obj.

    Uses inspect.signature so it works for plain functions, classmethods, and
    __init__ methods (the 'self' parameter is automatically excluded).

    Phase 4 policy: if ANY keys are dropped, raises RuntimeError immediately.
    This is intentional — feature kwargs must be accepted by the model signature.
    """
    try:
        sig = inspect.signature(callable_obj)
        allowed = {
            name
            for name, param in sig.parameters.items()
            if name != "self"
            and param.kind
            not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        }
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if has_var_keyword:
            return dict(kwargs_dict)
    except (ValueError, TypeError):
        return dict(kwargs_dict)

    dropped = sorted(set(kwargs_dict) - allowed)
    if dropped:
        msg = (
            f"[filter_kwargs] FATAL: {len(dropped)} unsupported kwarg(s) for "
            f"{callable_obj.__name__}: {dropped}  "
            f"(model accepts {len(allowed)} param(s): {sorted(allowed)})"
        )
        print(msg, file=sys.stderr)
        print(msg)
        raise RuntimeError(msg)
    return {k: v for k, v in kwargs_dict.items() if k in allowed}


def main():
    parser = argparse.ArgumentParser(description='Train ST-MM-GNN Layer A model')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--fast-dev-run', action='store_true', help='Run minimal dev loop')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--run_dir', type=str, default=None, help='Override run directory')
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Deprecated alias for --run_dir (kept for backward compatibility)',
    )
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
    
    # Backward compatibility: prefer --run_dir; fall back to legacy --output_dir.
    if args.run_dir and args.output_dir and args.run_dir != args.output_dir:
        parser.error('--run_dir and --output_dir were both provided with different values')

    effective_run_dir = args.run_dir or args.output_dir

    # Create run directory
    if effective_run_dir:
        run_dir = Path(effective_run_dir)
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
    model_kwargs_raw = cfg.get('model', {}).copy()
    model_kwargs = filter_kwargs(STMMGraphWaveNet, model_kwargs_raw)
    model = STMMGraphWaveNet(**model_kwargs)
    
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

    # Early stopping from config (ignore early_stop_metric CLI for now; use config)
    early_stop_cfg = cfg.get('early_stopping', {})
    if early_stop_cfg:
        early_stop = pl.callbacks.EarlyStopping(**early_stop_cfg)
        callbacks.append(early_stop)
    elif args.early_stop_metric:
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
    # Remove callbacks/logger keys from cfg to avoid duplication
    for _key in ('callbacks', 'logger'):
        trainer_cfg.pop(_key, None)

    # --- Phase 4: force CSVLogger only; TensorBoard/TensorBoardX is disabled ---
    # This prevents tensorboardX import crashes on Vertex AI containers that
    # do not have tensorboard properly installed.
    csv_logger = CSVLogger(save_dir=str(run_dir), name="csv_logs", version=0)
    print(f"[GATE] logger=CSVLogger only (tensorboard disabled), save_dir={run_dir}/csv_logs")

    trainer = pl.Trainer(callbacks=callbacks, logger=csv_logger, **trainer_cfg)

    # Train
    print("Starting training...")
    trainer.fit(pl_module, datamodule=dm)
    
    print(f"Training complete. Logs/checkpoints saved to: {run_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
