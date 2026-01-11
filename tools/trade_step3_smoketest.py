from __future__ import annotations

import time

from pathograph.data.trade_dataset import TradeDatasetConfig, TradeDatasetZarr


if __name__ == "__main__":
    cfg = TradeDatasetConfig(
        base_zarr_path="data/processed/trade/imf_imts_step1/trade_fob_tensor.zarr",
        risk_zarr_path="data/processed/trade/faostat_step2/trade_risk_tensor.zarr",
        lookback=24,
        horizon=1,
        split="train",
        apply_log1p=True,
        standardize=True,
        scaler_json_path="data/processed/trade/trade_step3_scaler.json",
        return_mode="separate",
    )

    ds = TradeDatasetZarr(cfg)
    print("LEN:", len(ds))

    t0 = time.time()
    for i in [0, 1, 2, len(ds)//2, len(ds)-1]:
        ex = ds[i]
        print("\nIDX", i, "t", int(ex["t"]), "t_y", int(ex["t_y"]))
        print("time_feat", ex["time_feat"].shape, ex["time_feat"])
        print("base_trade", ex["base_trade"].shape, ex["base_trade"].dtype)
        print("risk_trade", ex["risk_trade"].shape, ex["risk_trade"].dtype)
        print("base_mask", ex["base_mask"].shape, ex["base_mask"].dtype)
        print("risk_mask", ex["risk_mask"].shape, ex["risk_mask"].dtype)
    t1 = time.time()

    print("\nSmoketest seconds:", round(t1 - t0, 3))
