from __future__ import annotations

from dataclasses import asdict


def _load_yaml(path: str) -> dict:
    try:
        import yaml
    except Exception as e:
        raise RuntimeError("PyYAML not installed; install pyyaml") from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = _load_yaml("config/trade_step4.yaml")["trade_step4"]

    from pathograph.data.trade_datamodule import TradeDataModule, TradeDataModuleConfig

    dm_cfg = TradeDataModuleConfig(
        base_zarr_path=cfg["inputs"]["base_zarr"],
        risk_zarr_path=cfg["inputs"]["risk_zarr"],
        scaler_json_path=cfg["inputs"]["scaler_json"],
        lookback=int(cfg["windowing"]["lookback_months"]),
        horizon=int(cfg["windowing"]["horizon_months"]),
        apply_log1p=bool(cfg["transforms"]["apply_log1p"]),
        standardize=bool(cfg["transforms"]["standardize"]),
        split_train=(__import__("pathograph.data.trade_dataset", fromlist=["TradeSplit"]).TradeSplit)(**cfg["splits"]["train"]),
        split_val=(__import__("pathograph.data.trade_dataset", fromlist=["TradeSplit"]).TradeSplit)(**cfg["splits"]["val"]),
        split_test=(__import__("pathograph.data.trade_dataset", fromlist=["TradeSplit"]).TradeSplit)(**cfg["splits"]["test"]),
        batch_size=int(cfg["dataloader"]["batch_size"]),
        num_workers=int(cfg["dataloader"]["num_workers"]),
        pin_memory=bool(cfg["dataloader"]["pin_memory"]),
        persistent_workers=bool(cfg["dataloader"]["persistent_workers"]),
        prefetch_factor=int(cfg["dataloader"]["prefetch_factor"]),
        drop_last_train=bool(cfg["dataloader"]["drop_last_train"]),
    )

    dm = TradeDataModule(dm_cfg)
    dm.setup()

    for name, dl in [("train", dm.train_dataloader()), ("val", dm.val_dataloader()), ("test", dm.test_dataloader())]:
        batch = next(iter(dl))
        print(f"\n[{name}] keys:", sorted(batch.keys()))
        for k, v in batch.items():
            try:
                shape = tuple(v.shape)
                print(f"  {k}: shape={shape} dtype={v.dtype}")
            except Exception:
                print(f"  {k}: type={type(v)}")

    print("\nOK: Trade Step 4 smoketest completed.")


if __name__ == "__main__":
    main()
