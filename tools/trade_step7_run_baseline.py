"""Trade Step 7 - Run Baseline Orchestrator.

Orchestrates the full baseline training run:
1. Creates run directory with git commit info
2. Verifies artifacts
3. Runs training via trade_step6_train_entrypoint.py
4. Exports metrics breakdown

Usage:
    python tools/trade_step7_run_baseline.py
    python tools/trade_step7_run_baseline.py --smoke  # Quick validation run
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional


ROOT = Path(__file__).resolve().parent.parent


def get_git_info() -> dict:
    """Get git commit hash and status."""
    info = {}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        info["commit"] = result.stdout.strip() if result.returncode == 0 else "UNKNOWN"
        
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        info["branch"] = result.stdout.strip() if result.returncode == 0 else "UNKNOWN"
        
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        info["is_clean"] = len(result.stdout.strip()) == 0 if result.returncode == 0 else False
        info["dirty_files"] = result.stdout.strip().split("\n") if result.stdout.strip() else []
    except Exception as e:
        info["error"] = str(e)
    return info


def get_env_info() -> dict:
    """Get Python, PyTorch, and CUDA version info."""
    info = {
        "python": sys.version,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import pytorch_lightning as pl
        info["pytorch_lightning"] = pl.__version__
    except ImportError:
        pass
    return info


def run_verification(run_dir: Path) -> bool:
    """Run artifact verification and save output."""
    print(">>> Running artifact verification...")
    output = run_dir / "artifact_verification.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "trade_step7_verify_artifacts.py"),
            "--output", str(output),
        ],
        cwd=ROOT,
    )
    return result.returncode == 0


def run_training(config_path: Path, run_dir: Path, smoke: bool = False) -> int:
    """Run training via Step 6 entrypoint."""
    print(">>> Starting training...")
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "trade_step6_train_entrypoint.py"),
        "--config", str(config_path),
        # Override output directory to match orchestrator's run_dir
        "--override", f"logging.out_dir={run_dir}",
    ]
    
    if smoke:
        # Use limited batches instead of fast_dev_run to still get checkpoints
        cmd.extend([
            "--override", "trainer.max_epochs=1",
            "--override", "trainer.limit_train_batches=2",
            "--override", "trainer.limit_val_batches=2",
            "--override", "trainer.limit_test_batches=2",
        ])
    
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode



def run_metrics_export(run_dir: Path, checkpoint_path: Optional[Path] = None) -> bool:
    """Run metrics export if script exists."""
    export_script = ROOT / "tools" / "trade_step7_export_metrics.py"
    if not export_script.exists():
        print(">>> Metrics export script not found, skipping...")
        return True
    
    print(">>> Exporting metrics breakdown...")
    cmd = [
        sys.executable,
        str(export_script),
        "--run-dir", str(run_dir),
    ]
    if checkpoint_path:
        cmd.extend(["--checkpoint", str(checkpoint_path)])
    
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode == 0


def find_best_checkpoint(run_dir: Path) -> Optional[Path]:
    """Find the best checkpoint in the run directory."""
    # Check multiple possible checkpoint locations
    ckpt_dirs = []
    
    # Direct checkpoints directory
    direct_ckpt = run_dir / "checkpoints"
    if direct_ckpt.exists():
        ckpt_dirs.append(direct_ckpt)
    
    # Lightning version directories (sorted to get latest version first)
    for version_dir in sorted(run_dir.glob("version_*"), reverse=True):
        candidate = version_dir / "checkpoints"
        if candidate.exists():
            ckpt_dirs.append(candidate)
    
    # Search all checkpoint directories
    for ckpt_dir in ckpt_dirs:
        # Look for best.ckpt or last.ckpt first
        for name in ["best.ckpt", "last.ckpt"]:
            candidate = ckpt_dir / name
            if candidate.exists():
                return candidate
        
        # Find any .ckpt file
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        if ckpts:
            return ckpts[0]
    
    return None



def copy_resolved_config(run_dir: Path) -> None:
    """Copy resolved config from Lightning output if it exists."""
    # Lightning might save this in the log directory
    candidates = [
        run_dir / "resolved_config.yaml",
        run_dir / "hparams.yaml",
    ]
    for version_dir in run_dir.glob("version_*"):
        candidates.append(version_dir / "resolved_config.yaml")
        candidates.append(version_dir / "hparams.yaml")
    
    for src in candidates:
        if src.exists():
            dst = run_dir / "config_resolved.yaml"
            if not dst.exists():
                shutil.copy(src, dst)
                print(f">>> Copied resolved config to {dst}")
            break


def main():
    parser = argparse.ArgumentParser(description="Trade Step 7 - Run Baseline Orchestrator")
    parser.add_argument(
        "--config",
        default="config/trade_step7.yaml",
        help="Path to config file (default: config/trade_step7.yaml)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run in smoke mode (1 epoch, limited batches, but still checkpoints)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip artifact verification step",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training step (useful for re-exporting metrics)",
    )
    parser.add_argument(
        "--run-dir",
        default="runs/trade_baseline_v1",
        help="Output run directory",
    )
    args = parser.parse_args()

    # Resolve paths
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir

    print("=" * 60)
    print("Trade Step 7 - Baseline Bundle Orchestrator")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Run Dir: {run_dir}")
    print(f"Smoke Mode: {args.smoke}")
    print()

    # 1. Setup run directory
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    # 2. Write git commit info
    git_info = get_git_info()
    env_info = get_env_info()
    
    with open(run_dir / "git_commit.txt", "w") as f:
        f.write(f"commit: {git_info.get('commit', 'UNKNOWN')}\n")
        f.write(f"branch: {git_info.get('branch', 'UNKNOWN')}\n")
        f.write(f"is_clean: {git_info.get('is_clean', False)}\n")
        if git_info.get("dirty_files"):
            f.write(f"dirty_files:\n")
            for df in git_info["dirty_files"]:
                f.write(f"  {df}\n")
    
    with open(run_dir / "environment.json", "w") as f:
        json.dump({"git": git_info, "env": env_info}, f, indent=2)
    
    print(f">>> Git commit: {git_info.get('commit', 'UNKNOWN')[:8]}")
    print(f">>> Environment saved to {run_dir / 'environment.json'}")

    # 3. Verify artifacts
    if not args.skip_verify:
        if not run_verification(run_dir):
            print("ERROR: Artifact verification failed!")
            sys.exit(1)
        print(">>> Artifact verification passed")
    else:
        print(">>> Skipping artifact verification")

    # 4. Run training
    if not args.skip_train:
        retcode = run_training(config_path, run_dir, smoke=args.smoke)
        if retcode != 0:
            print(f"ERROR: Training failed with exit code {retcode}")
            sys.exit(retcode)
        print(">>> Training completed")
    else:
        print(">>> Skipping training")

    # 5. Copy resolved config
    copy_resolved_config(run_dir)

    # 6. Find best checkpoint
    best_ckpt = find_best_checkpoint(run_dir)
    if best_ckpt:
        print(f">>> Best checkpoint: {best_ckpt}")
        # Create symlink or copy as best.ckpt if not already named that
        best_dst = run_dir / "checkpoints" / "best.ckpt"
        if not best_dst.exists() and best_ckpt.name != "best.ckpt":
            shutil.copy(best_ckpt, best_dst)
            print(f">>> Copied checkpoint to {best_dst}")
    else:
        print("WARNING: No checkpoint found!")

    # 7. Export metrics
    run_metrics_export(run_dir, best_ckpt)

    # 8. Final summary
    print()
    print("=" * 60)
    print("Step 7 Baseline Bundle Complete")
    print("=" * 60)
    print(f"Run directory: {run_dir}")
    print("Contents:")
    for f in sorted(run_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(run_dir)
            print(f"  {rel}")

    sys.exit(0)


if __name__ == "__main__":
    main()
