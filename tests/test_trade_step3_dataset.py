import os

import numpy as np

from pathograph.data.trade_dataset import TradeDatasetConfig, TradeDatasetZarr


def test_trade_step3_dataset_shapes_and_alignment():
    base = "data/processed/trade/imf_imts_step1/trade_fob_tensor.zarr"
    risk = "data/processed/trade/faostat_step2/trade_risk_tensor.zarr"
    scaler = "data/processed/trade/trade_step3_scaler.json"

    assert os.path.exists(base)
    assert os.path.exists(risk)
    assert os.path.exists(scaler)

    cfg = TradeDatasetConfig(
        base_zarr_path=base,
        risk_zarr_path=risk,
        lookback=24,
        horizon=1,
        split="train",
        apply_log1p=True,
        standardize=True,
        scaler_json_path=scaler,
        return_mode="separate",
    )

    ds = TradeDatasetZarr(cfg)
    ex = ds[0]

    assert ex["base_trade"].shape == (24, 194, 194, 2)
    assert ex["risk_trade"].shape == (24, 194, 194, 8, 2)
    assert ex["base_mask"].shape == (24, 194, 194, 2)
    assert ex["risk_mask"].shape == (24, 194, 194, 8, 2)

    # basic numeric sanity: standardized values should be finite
    assert np.isfinite(ex["base_trade"]).all()
    assert np.isfinite(ex["risk_trade"]).all()


def test_trade_step3_concat_mode_feature_dim():
    base = "data/processed/trade/imf_imts_step1/trade_fob_tensor.zarr"
    risk = "data/processed/trade/faostat_step2/trade_risk_tensor.zarr"
    scaler = "data/processed/trade/trade_step3_scaler.json"

    cfg = TradeDatasetConfig(
        base_zarr_path=base,
        risk_zarr_path=risk,
        lookback=12,
        horizon=1,
        split="val",
        apply_log1p=True,
        standardize=True,
        scaler_json_path=scaler,
        return_mode="concat",
    )

    ds = TradeDatasetZarr(cfg)
    ex = ds[0]
    feat = ex["edge_feat"]

    # F = base_val(2) + risk_val(16) + base_mask(2) + risk_mask(16) + base_est(2) + risk_est(16) = 54
    assert feat.shape == (12, 194, 194, 54)
    assert np.isfinite(feat).all()
