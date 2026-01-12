"""Trade Step 7 - Export Metrics Breakdown.

Loads trained checkpoint and computes granular metrics on val/test sets:
- Base loss/MAE/RMSE split by exports vs imports channel
- Imports further split by is_estimated==1 vs is_estimated==0
- Risk metrics (aggregate and per-commodity-group)

Exports:
- metrics_val.json
- metrics_test.json
- metrics_breakdown.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ChannelMetrics:
    mse: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    count: int = 0


@dataclass
class SplitMetrics:
    total_loss: float = 0.0
    base_exports: ChannelMetrics = field(default_factory=ChannelMetrics)
    base_imports: ChannelMetrics = field(default_factory=ChannelMetrics)
    base_imports_estimated: ChannelMetrics = field(default_factory=ChannelMetrics)
    base_imports_observed: ChannelMetrics = field(default_factory=ChannelMetrics)
    risk_aggregate: ChannelMetrics = field(default_factory=ChannelMetrics)
    num_batches: int = 0


def compute_masked_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> ChannelMetrics:
    """Compute MSE, MAE, RMSE on masked values."""
    mask_bool = mask.astype(bool)
    count = int(mask_bool.sum())
    if count == 0:
        return ChannelMetrics(mse=0.0, mae=0.0, rmse=0.0, count=0)
    
    p = pred[mask_bool]
    t = target[mask_bool]
    
    diff = p - t
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    
    return ChannelMetrics(mse=mse, mae=mae, rmse=rmse, count=count)


def accumulate_metrics(acc: ChannelMetrics, new: ChannelMetrics) -> None:
    """Accumulate metrics (weighted by count)."""
    total_count = acc.count + new.count
    if total_count == 0:
        return
    
    # Weighted average
    if new.count > 0:
        w_acc = acc.count / total_count
        w_new = new.count / total_count
        acc.mse = acc.mse * w_acc + new.mse * w_new
        acc.mae = acc.mae * w_acc + new.mae * w_new
        acc.rmse = float(np.sqrt(acc.mse))
        acc.count = total_count


def evaluate_split(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    split_name: str,
) -> SplitMetrics:
    """Evaluate model on a split and compute metrics breakdown."""
    model.eval()
    metrics = SplitMetrics()
    
    all_base_exports = []
    all_base_imports = []
    all_base_imports_est = []
    all_base_imports_obs = []
    all_risk = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            preds = model(batch)
            
            # Base metrics
            if "y_base" in batch and "y_base_pred" in preds:
                y_base = batch["y_base"].cpu().numpy()
                y_base_pred = preds["y_base_pred"].cpu().numpy()
                y_base_mask = batch.get("y_base_mask", torch.ones_like(batch["y_base"])).cpu().numpy()
                y_base_est = batch.get("y_base_is_estimated", torch.zeros_like(batch["y_base_mask"])).cpu().numpy()
                
                # Channel 0 = exports, Channel 1 = imports
                for b in range(y_base.shape[0]):
                    # Exports (channel 0)
                    exp_m = compute_masked_metrics(
                        y_base_pred[b, :, :, 0],
                        y_base[b, :, :, 0],
                        y_base_mask[b, :, :] if y_base_mask.ndim == 3 else y_base_mask[b, :, :, 0],
                    )
                    all_base_exports.append(exp_m)
                    
                    # Imports (channel 1)
                    imp_mask = y_base_mask[b, :, :] if y_base_mask.ndim == 3 else y_base_mask[b, :, :, 1]
                    imp_m = compute_masked_metrics(
                        y_base_pred[b, :, :, 1],
                        y_base[b, :, :, 1],
                        imp_mask,
                    )
                    all_base_imports.append(imp_m)
                    
                    # Imports split by is_estimated
                    if y_base_est.ndim >= 3:
                        est_flag = y_base_est[b, :, :, 1] if y_base_est.ndim == 4 else y_base_est[b, :, :]
                        
                        # Estimated imports
                        est_mask = (imp_mask.astype(bool)) & (est_flag == 1)
                        est_m = compute_masked_metrics(
                            y_base_pred[b, :, :, 1],
                            y_base[b, :, :, 1],
                            est_mask.astype(np.uint8),
                        )
                        all_base_imports_est.append(est_m)
                        
                        # Non-estimated imports
                        obs_mask = (imp_mask.astype(bool)) & (est_flag == 0)
                        obs_m = compute_masked_metrics(
                            y_base_pred[b, :, :, 1],
                            y_base[b, :, :, 1],
                            obs_mask.astype(np.uint8),
                        )
                        all_base_imports_obs.append(obs_m)
            
            # Risk metrics
            if "y_risk" in batch and "y_risk_pred" in preds:
                y_risk = batch["y_risk"].cpu().numpy()
                y_risk_pred = preds["y_risk_pred"].cpu().numpy()
                y_risk_mask = batch.get("y_risk_mask", torch.ones_like(batch["y_risk"])).cpu().numpy()
                
                # Aggregate over all commodities
                for b in range(y_risk.shape[0]):
                    # Flatten K and channels
                    pred_flat = y_risk_pred[b].flatten()
                    target_flat = y_risk[b].flatten()
                    mask_flat = y_risk_mask[b].flatten()
                    
                    risk_m = compute_masked_metrics(pred_flat, target_flat, mask_flat)
                    all_risk.append(risk_m)
            
            metrics.num_batches += 1
    
    # Aggregate all batch metrics
    def aggregate_list(metrics_list: List[ChannelMetrics]) -> ChannelMetrics:
        if not metrics_list:
            return ChannelMetrics()
        total_count = sum(m.count for m in metrics_list)
        if total_count == 0:
            return ChannelMetrics()
        
        weighted_mse = sum(m.mse * m.count for m in metrics_list) / total_count
        weighted_mae = sum(m.mae * m.count for m in metrics_list) / total_count
        return ChannelMetrics(
            mse=weighted_mse,
            mae=weighted_mae,
            rmse=float(np.sqrt(weighted_mse)),
            count=total_count,
        )
    
    metrics.base_exports = aggregate_list(all_base_exports)
    metrics.base_imports = aggregate_list(all_base_imports)
    metrics.base_imports_estimated = aggregate_list(all_base_imports_est)
    metrics.base_imports_observed = aggregate_list(all_base_imports_obs)
    metrics.risk_aggregate = aggregate_list(all_risk)
    
    # Total loss approximation
    base_mse = (metrics.base_exports.mse + metrics.base_imports.mse) / 2 if metrics.base_exports.count > 0 else 0
    metrics.total_loss = base_mse + metrics.risk_aggregate.mse
    
    return metrics


def serialize_metrics(m: SplitMetrics) -> Dict[str, Any]:
    """Convert SplitMetrics to JSON-serializable dict."""
    return {
        "total_loss": m.total_loss,
        "num_batches": m.num_batches,
        "base": {
            "exports": asdict(m.base_exports),
            "imports": asdict(m.base_imports),
            "imports_estimated": asdict(m.base_imports_estimated),
            "imports_observed": asdict(m.base_imports_observed),
        },
        "risk": {
            "aggregate": asdict(m.risk_aggregate),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Trade Step 7 - Export Metrics Breakdown")
    parser.add_argument("--run-dir", required=True, help="Path to run directory")
    parser.add_argument("--checkpoint", help="Path to checkpoint (auto-detects if not provided)")
    parser.add_argument("--config", default="config/trade_step7.yaml", help="Config file path")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    # Find checkpoint
    ckpt_path = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        # Search for checkpoint
        for candidate in [
            run_dir / "checkpoints" / "best.ckpt",
            run_dir / "checkpoints" / "last.ckpt",
        ]:
            if candidate.exists():
                ckpt_path = candidate
                break
        if ckpt_path is None:
            for ckpt in run_dir.rglob("*.ckpt"):
                ckpt_path = ckpt
                break
    
    if ckpt_path is None or not ckpt_path.exists():
        print("ERROR: No checkpoint found!")
        sys.exit(1)
    
    print(f">>> Loading checkpoint: {ckpt_path}")

    # Load model
    sys.path.insert(0, str(ROOT))
    from pathograph.train.trade_lightning_module import TradeBaselinePL
    from pathograph.data.trade_datamodule import TradeDataModule, TradeDataModuleConfig
    from pathograph.data.trade_dataset import TradeSplit
    import yaml

    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load model from checkpoint
    model = TradeBaselinePL.load_from_checkpoint(str(ckpt_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Setup datamodule
    dm_raw = cfg.get("datamodule", {})
    for p_key in ["base_zarr_path", "risk_zarr_path", "scaler_json_path"]:
        if p_key in dm_raw and isinstance(dm_raw[p_key], str):
            p = Path(dm_raw[p_key])
            if not p.is_absolute():
                dm_raw[p_key] = str(ROOT / p)
    
    # Convert splits
    for key in ["split_train", "split_val", "split_test"]:
        if key in dm_raw and isinstance(dm_raw[key], (list, tuple)):
            vals = dm_raw[key]
            dm_raw[key] = TradeSplit(int(vals[0]), int(vals[1]))
    
    dm_cfg = TradeDataModuleConfig(**dm_raw)
    dm = TradeDataModule(dm_cfg)
    dm.setup()

    # Evaluate on val and test
    print(">>> Evaluating on validation set...")
    val_metrics = evaluate_split(model, dm.val_dataloader(), device, "val")
    
    print(">>> Evaluating on test set...")
    test_metrics = evaluate_split(model, dm.test_dataloader(), device, "test")

    # Save metrics
    with open(run_dir / "metrics_val.json", "w") as f:
        json.dump(serialize_metrics(val_metrics), f, indent=2)
    
    with open(run_dir / "metrics_test.json", "w") as f:
        json.dump(serialize_metrics(test_metrics), f, indent=2)
    
    # Combined breakdown
    breakdown = {
        "checkpoint": str(ckpt_path),
        "config": str(config_path),
        "val": serialize_metrics(val_metrics),
        "test": serialize_metrics(test_metrics),
    }
    with open(run_dir / "metrics_breakdown.json", "w") as f:
        json.dump(breakdown, f, indent=2)

    print(f">>> Metrics saved to {run_dir}")
    print(f"    Val loss: {val_metrics.total_loss:.6f}")
    print(f"    Test loss: {test_metrics.total_loss:.6f}")
    print(f"    Val exports MAE: {val_metrics.base_exports.mae:.6f}")
    print(f"    Val imports MAE: {val_metrics.base_imports.mae:.6f}")
    print(f"    Val imports (estimated) MAE: {val_metrics.base_imports_estimated.mae:.6f}")
    print(f"    Val imports (observed) MAE: {val_metrics.base_imports_observed.mae:.6f}")

    sys.exit(0)


if __name__ == "__main__":
    main()
