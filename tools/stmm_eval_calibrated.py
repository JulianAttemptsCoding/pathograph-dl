"""
Evaluate Calibrated ST-MM-GNN.

Loads a model and a temperature scalar, runs test set, applies temperature scaling, and computes metrics.
"""

import argparse
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from pathograph.pl.stmm_pl_module import STMMPLModule
from pathograph.data.trade_datamodule import TradeDataModule, TradeDataModuleConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir", required=True, help="Run dir to save results")
    parser.add_argument("--calib_dir", required=True, help="Directory containing temperature.json")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    # Load Temperature
    calib_path = Path(args.calib_dir) / "temperature.json"
    with open(calib_path, 'r') as f:
        calib_data = json.load(f)
    
    T = calib_data['temperature']
    ckpt_path = calib_data['ckpt_used']
    print(f"Loaded T={T:.4f} from {calib_path}")
    print(f"Using checkpoint: {ckpt_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    from pathograph.models.stmm_gwnet import STMMGraphWaveNet
    model_arch = STMMGraphWaveNet(**cfg['model'])
    pl_module = STMMPLModule.load_from_checkpoint(ckpt_path, model=model_arch)
    pl_module.to(device)
    pl_module.eval()
    
    # DataModule
    dm_config = TradeDataModuleConfig(**cfg['datamodule'])
    dm = TradeDataModule(dm_config)
    dm.setup()
    test_loader = dm.test_dataloader()
    
    # Metrics
    num_pathogens = pl_module.num_pathogens
    auroc_metrics = [BinaryAUROC().to(device) for _ in range(num_pathogens)]
    auprc_metrics = [BinaryAveragePrecision().to(device) for _ in range(num_pathogens)]
    
    metrics = {
        "test_cal_auroc_p": {},
        "test_cal_auprc_p": {},
        "test_cal_pos_total": 0,
        "test_cal_valid_pathogens": 0
    }
    
    print("Running calibrated evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward
            logits = pl_module(batch)
            
            # Calibrate
            scaled_logits = logits / T
            probs = torch.sigmoid(scaled_logits)
            
            targets = batch['y_next']
            mask = batch['y_mask']
            
            # Update counts
            observed = mask > 0.5
            positives = (targets > 0.5) & observed
            metrics["test_cal_pos_total"] += int(positives.sum().item())
            
            # Update metrics per pathogen
            for p in range(num_pathogens):
                probs_p = probs[:, :, p].flatten()
                targets_p = targets[:, :, p].flatten()
                mask_p = mask[:, :, p].flatten()
                
                observed_p = mask_p > 0.5
                if observed_p.sum() > 0:
                    auroc_metrics[p].update(probs_p[observed_p], targets_p[observed_p].long())
                    auprc_metrics[p].update(probs_p[observed_p], targets_p[observed_p].long())

    # Compute
    auroc_scores = []
    auprc_scores = []
    
    for p in range(num_pathogens):
        auroc_val = float('nan')
        auprc_val = float('nan')
        
        try:
            val = auroc_metrics[p].compute()
            if torch.isfinite(val):
                auroc_val = val.item()
        except:
            pass
            
        try:
            val = auprc_metrics[p].compute()
            if torch.isfinite(val):
                auprc_val = val.item()
        except:
            pass
            
        metrics["test_cal_auroc_p"][f"p{p}"] = auroc_val
        metrics["test_cal_auprc_p"][f"p{p}"] = auprc_val
        
        if not np.isnan(auroc_val) and not np.isnan(auprc_val):
            auroc_scores.append(auroc_val)
            auprc_scores.append(auprc_val)
            metrics["test_cal_valid_pathogens"] += 1
            
    # Macro
    if auroc_scores:
        metrics["test_cal_auroc_macro"] = np.mean(auroc_scores)
    else:
        metrics["test_cal_auroc_macro"] = float('nan')
        
    if auprc_scores:
        metrics["test_cal_auprc_macro"] = np.mean(auprc_scores)
    else:
        metrics["test_cal_auprc_macro"] = float('nan')
        
    # Save
    out_dir = Path(args.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "calibrated_metrics.json"
    
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Results saved to {out_path}")
    print(f"Macro AUPRC (Calibrated): {metrics['test_cal_auprc_macro']}")

if __name__ == "__main__":
    main()
