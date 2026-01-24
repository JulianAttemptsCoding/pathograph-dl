import yaml
import torch
import sys
import os
import dataclasses
import glob

# Ensure repo root is in path
sys.path.append(os.getcwd())

from pathograph.data.trade_datamodule import TradeDataModule, TradeDataModuleConfig

def check_loader():
    print("Running Clean Loader Check (Verifying Foot-Gun Fixes)...")
    config_path = "config/stmm_stepA.yaml"
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return

    print(f"Loading config: {config_path}")
    with open(config_path, 'r') as f:
        full_cfg = yaml.safe_load(f)
    
    if 'datamodule' not in full_cfg:
        print("ERROR: 'datamodule' section missing in yaml")
        return
    cfg = full_cfg['datamodule']
    
    # Filter keys valid for dataclass
    valid_keys = {f.name for f in dataclasses.fields(TradeDataModuleConfig)}
    init_kwargs = {k: v for k, v in cfg.items() if k in valid_keys}
    
    # NO MANUAL TradeSplit CONVERSION HERE
    # This verifies that TradeDataModuleConfig.__post_init__ handles the lists

    print("Instantiating TradeDataModuleConfig (expecting auto-conversion of splits)...")
    try:
        dconf = TradeDataModuleConfig(**init_kwargs)
        # Check if the conversion happened
        print(f"dconf.split_train type: {type(dconf.split_train)}")
        if not hasattr(dconf.split_train, 't_min'):
            print("FAIL: split_train does not have t_min attribute. Auto-conversion failed.")
            return
        
        print("Instantiating TradeDataModule...")
        dm = TradeDataModule(dconf)
        
        # Verify setup signature accepts stage
        print("Calling dm.setup(stage='fit')...")
        dm.setup(stage="fit") 
        print("Setup passed.")
        
        loader = dm.train_dataloader()
        print("Fetching batch...")
        batch = next(iter(loader))
        print("Batch fetched successfully.")
        print("Status: FOOT-GUNS FIXED.")

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_loader()
