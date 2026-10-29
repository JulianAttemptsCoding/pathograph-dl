"""
ST-MM-GNN Step A Training Entry Module.

Importable shim that calls the existing tools/stmm_stepA_train.py launcher.
This module is designed to work in an installed package context without sys.path hacks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """
    Train ST-MM-GNN Step A model.
    
    Args:
        argv: Command-line arguments (uses sys.argv if None)
        
    Returns:
        Exit code (0 for success)
    """
    ap = argparse.ArgumentParser(description='Train ST-MM-GNN Layer A model')
    ap.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    ap.add_argument('--output_dir', type=str, default=None, help='Override run directory')
    ap.add_argument('--fast-dev-run', action='store_true', help='Run minimal dev loop')
    ap.add_argument('--seed', type=int, default=None, help='Random seed')
    ap.add_argument('--max_epochs', type=int, default=None, help='Override max epochs')
    ap.add_argument('--early_stop_metric', type=str, default=None, help='Early stopping metric')
    ap.add_argument('--ckpt_path', type=str, default=None, help='Path to a .ckpt to resume from')
    
    args = ap.parse_args(argv)
    
    # Import the real training function from tools.stmm_stepA_train
    from tools.stmm_stepA_train import main as train_main
    
    # Build argv for the real main (it also uses argparse)
    real_argv = ['--config', args.config]
    
    if args.fast_dev_run:
        real_argv.append('--fast-dev-run')
    
    if args.seed is not None:
        real_argv.extend(['--seed', str(args.seed)])
    
    if args.output_dir is not None:
        real_argv.extend(['--run_dir', args.output_dir])
    
    if args.max_epochs is not None:
        real_argv.extend(['--max_epochs', str(args.max_epochs)])
    
    if args.early_stop_metric is not None:
        real_argv.extend(['--early_stop_metric', args.early_stop_metric])
    
    # Debug: print ckpt_path value for Vertex diagnostics
    print(f"DEBUG ckpt_path raw: {repr(args.ckpt_path)}", file=sys.stderr)
    
    # Only forward non-empty, non-whitespace ckpt_path
    if args.ckpt_path and str(args.ckpt_path).strip():
        ckpt_path = str(args.ckpt_path).strip()
        real_argv.extend(['--ckpt_path', ckpt_path])
    
    # Temporarily replace sys.argv to pass args to the real main
    old_argv = sys.argv
    try:
        sys.argv = ['stepA_train'] + real_argv
        train_main()
        return 0
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(f"Training failed: {e}", file=sys.stderr)
        raise
    finally:
        sys.argv = old_argv


if __name__ == '__main__':
    raise SystemExit(main())
