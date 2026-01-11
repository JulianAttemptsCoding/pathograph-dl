from __future__ import annotations

import numpy as np


def test_trade_step4_datamodule_shapes():
    try:
        import torch  # noqa: F401
    except Exception:
        # If torch isn't installed in this environment, skip.
        return

    from pathograph.data.trade_datamodule import TradeDataModule, TradeDataModuleConfig
    from pathograph.data.trade_dataset import TradeSplit

    cfg = TradeDataModuleConfig(
        base_zarr_path="data/processed/trade/imf_imts_step1/trade_fob_tensor.zarr",
        risk_zarr_path="data/processed/trade/faostat_step2/trade_risk_tensor.zarr",
        scaler_json_path="data/processed/trade/trade_step3_scaler.json",
        lookback=24,
        horizon=1,
        split_train=TradeSplit(0, 815),
        split_val=TradeSplit(816, 851),
        split_test=TradeSplit(852, 907),
        batch_size=1,
        num_workers=0,
    )

    dm = TradeDataModule(cfg)
    dm.setup()

    batch = next(iter(dm.train_dataloader()))

    # Required keys
    for k in [
        "t", "t_y", "time_feat",
        "base_trade", "base_mask", "base_is_estimated",
        "risk_trade", "risk_mask", "risk_is_estimated",
    ]:
        assert k in batch

    B = batch["t"].shape[0]
    assert batch["time_feat"].shape == (B, 2)

    # Base: (B,L,N,N,2)
    assert batch["base_trade"].ndim == 5
    assert batch["base_trade"].shape[-1] == 2
    assert batch["base_mask"].shape == batch["base_trade"].shape

    # Risk: (B,L,N,N,K,2)
    assert batch["risk_trade"].ndim == 6
    assert batch["risk_trade"].shape[-1] == 2
    assert batch["risk_mask"].shape == batch["risk_trade"].shape

    # Masking contract: masked positions must be exactly 0 in trade tensors
    base_trade = batch["base_trade"].detach().cpu().numpy()
    base_mask = batch["base_mask"].detach().cpu().numpy().astype(bool)
    if np.any(~base_mask):
        assert np.all(base_trade[~base_mask] == 0.0)

    risk_trade = batch["risk_trade"].detach().cpu().numpy()
    risk_mask = batch["risk_mask"].detach().cpu().numpy().astype(bool)
    if np.any(~risk_mask):
        assert np.all(risk_trade[~risk_mask] == 0.0)
