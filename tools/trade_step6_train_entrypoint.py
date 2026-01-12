import argparse
import sys
import yaml
import torch
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from pathograph.data.trade_dataset import TradeDatasetConfig, TradeSplit
from pathograph.data.trade_datamodule import TradeDataModule, TradeDataModuleConfig
from pathograph.train.trade_lightning_module import TradeBaselinePL

def convert_splits(cfg_dict):
    """Convert list-based split definitions to TradeSplit objects in place."""
    for key in ["split_train", "split_val", "split_test"]:
        if key in cfg_dict and isinstance(cfg_dict[key], (list, tuple)):
            vals = cfg_dict[key]
            if len(vals) != 2:
                raise ValueError(f"Split {key} must have exactly 2 elements (min, max). Got: {vals}")
            cfg_dict[key] = TradeSplit(int(vals[0]), int(vals[1]))

class PLDataModuleWrapper(pl.LightningDataModule):
    def __init__(self, dm: TradeDataModule):
        super().__init__()
        self.dm = dm
        
    def setup(self, stage=None):
        self.dm.setup()
        
    def train_dataloader(self):
        return self.dm.train_dataloader()
        
    def val_dataloader(self):
        return self.dm.val_dataloader()
        
    def test_dataloader(self):
        return self.dm.test_dataloader()

def main():
    parser = argparse.ArgumentParser(description="Trade Step 6 Training Entrypoint")
    parser.add_argument("--config", default="config/trade_step6.yaml", help="Path to YAML config")
    parser.add_argument("--override", action="append", help="Override config keys (dotted path), e.g. run.fast_dev_run=True")
    args = parser.parse_args()

    # 1. Load YAML
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    
    if not config_path.exists():
        print(f"ERROR: Config file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. Apply Overrides
    if args.override:
        for ov in args.override:
            if "=" not in ov:
                print(f"WARNING: Ignoring invalid override '{ov}' (expected key=value)")
                continue
            k, v = ov.split("=", 1)
            # Try to cast v to int/float/bool
            if v.lower() == "true": v = True
            elif v.lower() == "false": v = False
            else:
                try: v = int(v)
                except ValueError:
                    try: v = float(v)
                    except ValueError: pass
            
            # recursive set
            parts = k.split(".")
            d = cfg
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = v
    
    print(f">>> Configuration loaded from {config_path}")
    
    # 3. set seed
    if "seed" in cfg:
        pl.seed_everything(cfg["seed"], workers=True)

    # 4. Construct DataModule
    dm_raw = cfg.get("datamodule", {})
    # resolve paths relative to ROOT if they are strings and look like paths
    for p_key in ["base_zarr_path", "risk_zarr_path", "scaler_json_path"]:
        if p_key in dm_raw and isinstance(dm_raw[p_key], str):
            p = Path(dm_raw[p_key])
            if not p.is_absolute():
                dm_raw[p_key] = str(ROOT / p)
    
    convert_splits(dm_raw)
    
    try:
        dm_cfg = TradeDataModuleConfig(**dm_raw)
    except TypeError as e:
        print(f"ERROR: Failed to instantiate TradeDataModuleConfig. Check yaml keys against class definition.\n{e}")
        sys.exit(1)
        
    dm_impl = TradeDataModule(dm_cfg)
    dm = PLDataModuleWrapper(dm_impl)
    
    # Verify Dataset Config (optional but good sanity check, reading 'dataset' section if present)
    # The datamodule creates dataset configs internally, but if we wanted to validate the 'dataset' block:
    if "dataset" in cfg:
        ds_raw = cfg["dataset"].copy()
        for p_key in ["base_zarr_path", "risk_zarr_path", "scaler_json_path"]:
            if p_key in ds_raw and isinstance(ds_raw[p_key], str):
                p = Path(ds_raw[p_key])
                if not p.is_absolute():
                    ds_raw[p_key] = str(ROOT / p)
        convert_splits(ds_raw)
        try:
             TradeDatasetConfig(**ds_raw)
        except TypeError as e:
            print(f"WARNING: 'dataset' section in config matches TradeDatasetConfig, but has issues: {e}")

    # 5. Construct Model
    model_raw = cfg.get("model", {})
    lr = float(model_raw.get("lr", 1e-3))
    wd = float(model_raw.get("weight_decay", 0.0))
    model = TradeBaselinePL(lr=lr, weight_decay=wd)

    # 6. Checkpoint & Logging
    run_cfg = cfg.get("run", {})
    log_cfg = cfg.get("logging", {})
    ckpt_cfg = cfg.get("checkpoint", {})
    
    out_dir = Path(log_cfg.get("out_dir", "runs"))
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    
    logger = TensorBoardLogger(
        save_dir=str(out_dir.parent),
        name=out_dir.name,
        version=log_cfg.get("name", "version") if log_cfg.get("name") != "baseline" else None
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir + "/checkpoints",
        filename="{epoch}-" + f"{{{ckpt_cfg.get('monitor', 'val_loss')}:.4f}}",
        monitor=ckpt_cfg.get("monitor", "val_loss"),
        mode=ckpt_cfg.get("mode", "min"),
        save_top_k=ckpt_cfg.get("save_top_k", 1),
        save_last=ckpt_cfg.get("save_last", True),
        save_on_train_epoch_end=ckpt_cfg.get("save_on_train_epoch_end", False),
    )
    
    # 7. Trainer
    trainer_kwargs = cfg.get("trainer", {}).copy()
    
    # handle fast_dev_run override from run_cfg if set
    if run_cfg.get("fast_dev_run", False):
        trainer_kwargs["fast_dev_run"] = True
        
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        **trainer_kwargs
    )
    
    # Save resolved config
    if trainer.is_global_zero:
        Path(logger.log_dir).mkdir(parents=True, exist_ok=True)
        # We can't dump TradeSplit objects easily to JSON, so let's stick to yaml dump of raw dict
        # or just dump the cfg dictionary after some cleanup if needed.
        # Simple string dump of args for provenance
        with open(Path(logger.log_dir) / "resolved_config.yaml", "w") as f:
            yaml.dump(cfg, f)

    # 8. Fit
    print(">>> Starting Training...")
    trainer.fit(model, datamodule=dm)
    
    # 9. Test
    if run_cfg.get("do_test", False) and not trainer_kwargs.get("fast_dev_run"):
        print(">>> Starting Testing...")
        trainer.test(model, datamodule=dm, ckpt_path="best")

if __name__ == "__main__":
    main()
