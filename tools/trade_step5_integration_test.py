import sys
from pathlib import Path
import torch
import yaml

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from pathograph.data.trade_datamodule import TradeDataModule, TradeDataModuleConfig, TradeSplit
from pathograph.models.trade_baseline import PersistenceBaseline
from pathograph.train.trade_losses import masked_mse

def main():
    print(">>> Starting Trade Step 5 Integration Test")
    
    # 1. Load defaults from config file, but override for targets
    config_path = ROOT / "config" / "trade_step3.yaml"
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)["trade_step3"]
    
    # Construct config object
    cfg = TradeDataModuleConfig(
        base_zarr_path=str(ROOT / raw_cfg["inputs"]["base_zarr"]),
        risk_zarr_path=str(ROOT / raw_cfg["inputs"]["risk_zarr"]),
        scaler_json_path=str(ROOT / raw_cfg["outputs"]["scaler_json"]),
        lookback=raw_cfg["windowing"]["lookback_months"],
        horizon=raw_cfg["windowing"]["horizon_months"],
        split_train=TradeSplit(**raw_cfg["splits"]["train"]),
        split_val=TradeSplit(**raw_cfg["splits"]["val"]),
        split_test=TradeSplit(**raw_cfg["splits"]["test"]),
        apply_log1p=raw_cfg["transforms"]["apply_log1p"],
        standardize=raw_cfg["transforms"]["standardize"],
        batch_size=2,
        
        # KEY STEP 5 OVERRIDES
        return_targets=True,
        target_kind="both",
        include_target_masks=True,
        
        # ENABLE VALID-INDEX FILTERING (required for non-zero loss)
        require_target_observed=True,
        min_target_observed=1,
        require_target_observed_kind="both",
    )
    
    print(f">>> Config loaded. Base Zarr: {cfg.base_zarr_path}")
    
    # 2. Setup DataModule
    dm = TradeDataModule(cfg)
    dm.setup()
    
    # 3. Get Batch
    dl = dm.train_dataloader()
    batch = next(iter(dl))
    print(">>> Batch fetched.")
    print("Batch keys:", list(batch.keys()))
    
    # Check shapes
    N = 194 # Approx known size, or dynamic
    B = 2
    
    y_base = batch.get("y_base")
    if y_base is None:
        print("FAIL: y_base missing from batch")
        sys.exit(1)
    
    print(f"y_base shape: {y_base.shape}")
    assert y_base.ndim == 4, f"Expected 4 dims (B,N,N,2), got {y_base.ndim}"
    
    if "y_risk" in batch:
        print(f"y_risk shape: {batch['y_risk'].shape}")
    
    # 4. Run Model (Manual Loop)
    model = PersistenceBaseline()
    
    print(">>> Running forward pass (loss computation)...")
    preds = model(batch)
    
    # custom loss calc
    total_loss = 0.0
    if "y_base" in batch:
        l = masked_mse(preds["y_base_pred"], batch["y_base"], batch["y_base_mask"])
        print(f"Base loss: {l.item()}")
        total_loss += l
        
    if "y_risk" in batch:
        l = masked_mse(preds["y_risk_pred"], batch["y_risk"], batch["y_risk_mask"])
        print(f"Risk loss: {l.item()}")
        total_loss += l
        
    print(f">>> Total Loss computed: {total_loss.item()}")
    
    # 5. Backward check
    print(">>> Running backward pass...")
    try:
        total_loss.backward()
        print(">>> Backward successful.")
    except Exception as e:
        print(f"FAIL: Backward pass failed: {e}")
        sys.exit(1)
        
    print(">>> INTEGRATION TEST PASSED")

if __name__ == "__main__":
    main()
