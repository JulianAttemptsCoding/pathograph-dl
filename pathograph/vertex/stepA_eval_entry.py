"""
Vertex AI Step A Evaluation Wrapper

Runs trainer.test() against a checkpoint and uploads eval artifacts to GCS.

Example:
    python -m pathograph.vertex.stepA_eval_entry \
      --config_gcs gs://.../configs/stmm_stepA.yaml \
      --data_gcs_prefix gs://.../datasets/stepA/v1_zarr2 \
      --ckpt_gcs gs://.../runs/stepA/smoke_v11/epoch=8-step=7128-val_loss=0.0790.ckpt \
      --output_gcs_prefix gs://.../runs/stepA/smoke_v11/eval_vertex \
      --stage_to_local 1
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
import pytorch_lightning as pl

from pathograph.data.trade_datamodule import TradeDataModule, TradeDataModuleConfig
from pathograph.pl.stmm_pl_module import STMMPLModule
from pathograph.models.stmm_gwnet import STMMGraphWaveNet


def gcs_cp(src: str, dst: str) -> None:
    """Copy file from GCS using gcloud if available, else gsutil."""
    use_gcloud = shutil.which("gcloud") is not None
    if use_gcloud:
        cmd = ["gcloud", "storage", "cp", src, dst]
    else:
        cmd = ["gsutil", "cp", src, dst]
        
    if os.name == 'nt':
        cmd[0] = cmd[0] + '.cmd'
        
    kwargs = {'shell': True} if os.name == 'nt' else {}
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def gcs_rsync(src: str, dst: str) -> None:
    """Rsync dir from GCS using gcloud if available, else gsutil."""
    use_gcloud = shutil.which("gcloud") is not None
    if use_gcloud:
        cmd = ["gcloud", "storage", "rsync", src, dst, "--recursive"]
    else:
        cmd = ["gsutil", "-m", "rsync", "-r", src, dst]
        
    if os.name == 'nt':
        cmd[0] = cmd[0] + '.cmd'
        
    kwargs = {'shell': True} if os.name == 'nt' else {}
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def main() -> int:
    """Vertex AI evaluation entry point."""
    ap = argparse.ArgumentParser(description='Vertex AI Step A Evaluation Wrapper')
    ap.add_argument("--config_gcs", required=True, help='GCS path to config YAML')
    ap.add_argument("--data_gcs_prefix", required=True, help='GCS prefix for dataset')
    ap.add_argument("--ckpt_gcs", required=True, help='GCS path to checkpoint file')
    ap.add_argument("--output_gcs_prefix", required=True, help='GCS prefix for eval outputs')
    ap.add_argument("--stage_to_local", type=int, default=1, choices=[0, 1],
                    help='Stage dataset to local disk (1=yes, 0=no)')
    args = ap.parse_args()

    work = Path("./eval_work_tmp").absolute()
    cfg_dir = work / "config"
    data_dir = work / "data" / "processed"
    ckpt_dir = work / "ckpt"
    out_dir = work / "out"
    run_dir = work / "runs" / "stepA_eval"

    # Create ALL directories before any gsutil calls
    cfg_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)  # rsync destination must exist
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_local = cfg_dir / "stmm_stepA.yaml"
    ckpt_local = ckpt_dir / "model.ckpt"

    print("=" * 80)
    print("Vertex AI Step A Eval Wrapper")
    print("=" * 80)
    print(f"Config GCS: {args.config_gcs}")
    print(f"Data GCS: {args.data_gcs_prefix}")
    print(f"CKPT GCS: {args.ckpt_gcs}")
    print(f"Output GCS: {args.output_gcs_prefix}")
    print(f"Stage to local: {args.stage_to_local}")
    print("=" * 80)

    try:
        # 1) Download config + ckpt
        print("\n[1/5] Downloading config and checkpoint from GCS...")
        gcs_cp(args.config_gcs, str(cfg_local))
        gcs_cp(args.ckpt_gcs, str(ckpt_local))

        # Gate: verify config and checkpoint downloaded successfully
        if not cfg_local.exists():
            raise RuntimeError(f"GATE FAIL: config not found after download: {cfg_local}")
        if not ckpt_local.exists():
            raise RuntimeError(f"GATE FAIL: checkpoint not found after download: {ckpt_local}")
        print(f"  ✓ Config downloaded: {cfg_local}")
        print(f"  ✓ Checkpoint downloaded: {ckpt_local}")

        # 2) Stage dataset to local so relative paths like data/processed/... resolve
        if args.stage_to_local == 1:
            print("\n[2/5] Staging dataset to local disk (this may take several minutes)...")
            # Normalize paths with trailing slashes for rsync clarity
            src = args.data_gcs_prefix.rstrip("/") + "/"
            dst = str(data_dir).rstrip("/\\") + "/"
            gcs_rsync(src, dst)
            # Gate: verify data_dir is non-empty after rsync
            if not data_dir.exists() or not any(data_dir.iterdir()):
                raise RuntimeError(f"GATE FAIL: data_dir is empty after rsync: {data_dir}")
            print(f"  ✓ Dataset staged to: {data_dir}")
        else:
            # If you ever add a "no-stage" mode, you'd need to mount /gcs or rewrite paths.
            raise ValueError("stage_to_local=0 not supported in this wrapper.")

        # 3) chdir so YAML relative paths resolve (data/processed/...)
        print(f"\n[3/5] Changing to working directory: {work}")
        os.chdir(work)

        # 4) Load config
        print(f"\n[4/5] Loading config from: {cfg_local}")
        with open(cfg_local, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        pl.seed_everything(cfg.get("seed", 1337), workers=True)

        # 5) DataModule
        print("Instantiating DataModule...")
        dm_cfg = TradeDataModuleConfig(**cfg["datamodule"])
        dm = TradeDataModule(dm_cfg)

        # 6) Model
        print("Loading model from checkpoint...")
        if "adaptive_emb_dim" in cfg["model"] or cfg["model"].get("use_adaptive_adj", False):
            from pathograph.models.stmm_gwnet_adaptive import STMMGraphWaveNet as STMMGraphWaveNetAdaptive
            model_arch = STMMGraphWaveNetAdaptive(**cfg["model"])
        else:
            model_arch = STMMGraphWaveNet(**cfg["model"])
        
        model = STMMPLModule.load_from_checkpoint(str(ckpt_local), model=model_arch)
        model.eval()

        # 7) Test (write a durable artifact: metrics.csv)
        print(f"\n[5/5] Running evaluation on test set...")
        logger = pl.loggers.CSVLogger(
            save_dir=str(run_dir),
            name="eval_logs",
            version=0,  # always eval_logs/version_0
        )
        trainer = pl.Trainer(
            default_root_dir=str(run_dir),
            accelerator="auto",
            devices=1,
            logger=logger,
            enable_checkpointing=False,
        )

        print(f"Evaluating checkpoint: {ckpt_local}")
        trainer.test(model, datamodule=dm)
        print("Evaluation complete.")

        # Gate: verify metrics artifact exists (search recursively for metrics.csv or test_metrics.json)
        def find_metrics(search_dir: Path) -> list[Path]:
            patterns = ["**/metrics.csv", "**/test_metrics.json"]
            found = []
            for pattern in patterns:
                found.extend(search_dir.glob(pattern))
            return found

        metrics_files = find_metrics(run_dir)
        if not metrics_files:
            raise RuntimeError(
                f"GATE FAIL: no metrics artifact found under {run_dir}. "
                f"Expected **/metrics.csv or **/test_metrics.json"
            )
        print(f"  ✓ Found {len(metrics_files)} metrics artifact(s): {[str(f) for f in metrics_files]}")

        # 8) Upload eval artifacts to GCS
        print(f"\nUploading eval artifacts to {args.output_gcs_prefix}...")
        src_run = str(run_dir).rstrip("/\\") + "/"
        upload_dst = args.output_gcs_prefix.rstrip("/") + "/"
        gcs_rsync(src_run, upload_dst)
        
        print("\n" + "=" * 80)
        print("✓ Step A eval completed successfully")
        print("=" * 80)
        return 0

    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(f"\nEvaluation failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
