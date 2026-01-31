import argparse
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from pathograph.pl.stmm_pl_module import STMMPLModule
from pathograph.data.trade_datamodule import TradeDataModule, TradeDataModuleConfig
from pathograph.calibration.temperature_scaling import TemperatureScaling

def main():
    parser = argparse.ArgumentParser(description="Fit temperature scaling on val set")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None, help="Checkoint path. If None, tries to find 'last.ckpt' or 'best.ckpt' in run_dir.")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Model
    run_dir = Path(args.run_dir)
    ckpt_path = args.ckpt
    
    if ckpt_path is None:
        # Try to find best or last
        candidates = list(run_dir.glob("**/*.ckpt"))
        best = [c for c in candidates if "epoch=" in c.name and "step=" in c.name] # Usually best
        last = [c for c in candidates if "last" in c.name]
        
        if best:
            ckpt_path = best[0]
        elif last:
            ckpt_path = last[0]
        elif candidates:
            ckpt_path = candidates[0]
        else:
            print("No checkpoint found! Cannot calibrate.")
            # For testing without training, we might want to skip or fail.
            # But the task implies we run this AFTER training. 
            # If no checkpoint, maybe we just init random model? 
            # No, that's useless. Fail if not found.
            exit(1)
            
    print(f"Loading checkpoint: {ckpt_path}")
    from pathograph.models.stmm_gwnet import STMMGraphWaveNet
    model_arch = STMMGraphWaveNet(**cfg['model'])
    pl_module = STMMPLModule.load_from_checkpoint(ckpt_path, model=model_arch)
    pl_module.to(device)
    pl_module.eval()
    
    # 2. DataModule
    print("Loading DataModule...")
    dm_config = TradeDataModuleConfig(**cfg['datamodule'])
    dm = TradeDataModule(dm_config)
    dm.setup()
    
    # 3. Collect Validation Logits/Targets
    print("Collecting validation outputs...")
    val_loader = dm.val_dataloader()
    
    all_logits = []
    all_targets = []
    all_masks = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            logits = pl_module(batch) # (B, N, P)
            targets = batch['y_next']
            mask = batch['y_mask']
            
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
            all_masks.append(mask.cpu())
            
    # Cat
    full_logits = torch.cat(all_logits, dim=0)
    full_targets = torch.cat(all_targets, dim=0)
    full_masks = torch.cat(all_masks, dim=0)
    
    print(f"Collected val shape: {full_logits.shape}")
    
    # 4. Fit Temperature
    print("Fitting temperature...")
    scaler = TemperatureScaling()
    scaler.fit(full_logits, full_targets, full_masks)
    
    T = scaler.temperature.item()
    print(f"Fitted Temperature T={T:.4f}")
    
    # 5. Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        "temperature": T,
        "source_run": str(run_dir),
        "ckpt_used": str(ckpt_path),
        "split_indices": cfg['datamodule']['split_val'],
        "sample_count": int(full_logits.shape[0]),
        "timestamp": 123456789 # Placeholder or use actual
    }
    
    with open(out_dir / "temperature.json", "w") as f:
        json.dump(result, f, indent=2)
        
    print(f"Saved calibration to {out_dir / 'temperature.json'}")

if __name__ == "__main__":
    main()
