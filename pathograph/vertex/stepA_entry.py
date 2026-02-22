"""
Vertex AI Entry Point for ST-MM-GNN Step A Training.

This module handles:
1. Downloading config and data from GCS to local disk
2. Setting up the working directory structure
3. Running the Step A training
4. Uploading outputs back to GCS
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and optionally check return code."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}",  file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if check:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    
    return result


def _sync_gcs_to_local(gcs_prefix: str, local_dir: Path) -> None:
    """Sync GCS prefix to local directory using gsutil."""
    local_dir.mkdir(parents=True, exist_ok=True)
    _run_cmd(['gsutil', '-m', 'rsync', '-r', gcs_prefix, str(local_dir)])


def _sync_local_to_gcs(local_dir: Path, gcs_prefix: str) -> None:
    """Sync local directory to GCS prefix using gsutil."""
    if local_dir.exists():
        _run_cmd(['gsutil', '-m', 'rsync', '-r', str(local_dir), gcs_prefix])
    else:
        print(f"Warning: {local_dir} does not exist, skipping upload")


def main(argv: list[str] | None = None) -> int:
    """
    Vertex AI entry point for Step A training.
    
    Args:
        argv: Command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    ap = argparse.ArgumentParser(description='Vertex AI Step A Training Wrapper')
    ap.add_argument('--config_gcs', type=str, required=True,
                    help='GCS path to config YAML, e.g., gs://bucket/configs/stmm_stepA.yaml')
    ap.add_argument('--data_gcs_prefix', type=str, required=True,
                    help='GCS prefix for dataset, e.g., gs://bucket/datasets/stepA/v1')
    ap.add_argument('--output_gcs_prefix', type=str, required=True,
                    help='GCS prefix for outputs, e.g., gs://bucket/runs/stepA/run_001')
    ap.add_argument('--stage_to_local', type=int, default=1, choices=[0, 1],
                    help='Stage dataset to local disk (1=yes, 0=use /gcs mount)')
    ap.add_argument('--fast_dev_run', action='store_true',
                    help='Pass --fast-dev-run to training')
    ap.add_argument('--max_epochs', type=int, default=None,
                    help='Override max epochs from config (optional)')
    ap.add_argument('--seed', type=int, default=None,
                    help='Override seed from config (optional)')
    
    args = ap.parse_args(argv)
    
    print("="*80)
    print("Vertex AI Step A Training Wrapper")
    print("="*80)
    print(f"Config GCS: {args.config_gcs}")
    print(f"Data GCS: {args.data_gcs_prefix}")
    print(f"Output GCS: {args.output_gcs_prefix}")
    print(f"Stage to local: {args.stage_to_local}")
    print("="*80)
    
    # Set up working directory structure
    work_dir = Path('/tmp/work')
    work_dir.mkdir(parents=True, exist_ok=True)
    
    config_dir = work_dir / 'config'
    data_dir = work_dir / 'data' / 'processed'
    output_dir = work_dir / 'runs' / 'stepA'
    
    config_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download config from GCS
    print("\n[1/4] Downloading config from GCS...")
    config_local = config_dir / Path(args.config_gcs).name
    _run_cmd(['gsutil', 'cp', args.config_gcs, str(config_local)])
    
    # Download/stage data from GCS
    if args.stage_to_local == 1:
        print("\n[2/4] Staging dataset to local disk (this may take several minutes)...")
        _sync_gcs_to_local(args.data_gcs_prefix, data_dir)
        # Debug: show staged contents
        print(f"DEBUG: staged data_dir = {data_dir}", file=sys.stderr)
        if data_dir.exists():
            print(f"DEBUG: data_dir contents = {list(data_dir.iterdir())[:10]}", file=sys.stderr)
            # Show first level of subdirs
            for item in list(data_dir.iterdir())[:5]:
                if item.is_dir():
                    try:
                        children = list(item.iterdir())[:5]
                        print(f"DEBUG:   {item.name}/: {[c.name for c in children]}...", file=sys.stderr)
                    except Exception:
                        pass
        else:
            print(f"DEBUG: data_dir DOES NOT EXIST after staging!", file=sys.stderr)
    else:
        print("\n[2/4] Using /gcs mount for data access...")
        # Create symlink from work_dir/data/processed to /gcs/bucket/...
        # Extract bucket and path from GCS URI
        gcs_path = args.data_gcs_prefix.replace('gs://', '/gcs/')
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        if data_dir.exists():
            data_dir.unlink()
        data_dir.symlink_to(gcs_path)
    
    # Change to working directory (so relative paths in config work)
    print(f"\n[3/4] Changing to working directory: {work_dir}")
    os.chdir(work_dir)
    
    # Run training
    print("\n[4/4] Starting training...")
    print(f"Config: {config_local}")
    print(f"Output: {output_dir}")
    
    from pathograph.train.stepA_train import main as stepA_main
    
    train_argv = [
        '--config', str(config_local),
        '--run_dir', str(output_dir),
    ]
    
    if args.fast_dev_run:
        train_argv.append('--fast-dev-run')
    
    if args.max_epochs is not None:
        train_argv.extend(['--max_epochs', str(args.max_epochs)])
        print(f"[GATE] max_epochs override: {args.max_epochs}", file=sys.stderr)
    
    if args.seed is not None:
        train_argv.extend(['--seed', str(args.seed)])
        print(f"[GATE] seed override: {args.seed}", file=sys.stderr)
    
    try:
        rc = stepA_main(train_argv)
        if rc != 0:
            raise RuntimeError(f"Training failed with exit code {rc}")
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(f"Training failed: {e}", file=sys.stderr)
        raise
    
    # Upload outputs to GCS
    print("\n[5/5] Uploading outputs to GCS...")
    _sync_local_to_gcs(output_dir, args.output_gcs_prefix)
    
    print("\n" + "="*80)
    print("✓ Step A training completed successfully")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
