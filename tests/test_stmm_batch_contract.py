"""Test multimodal batch contract for ST-MM-GNN.

This test ensures the DataModule emits all required modalities:
trade+risk+climate+climate_anoms+meta with correct shapes.
"""

from pathlib import Path
import torch
import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from pathograph.data.trade_datamodule import TradeDataModule, TradeDataModuleConfig
from pathograph.data.trade_dataset import TradeSplit
from conftest import require_local_zarr, TRADE_BASE_ZARR, TRADE_RISK_ZARR


def convert_splits(cfg_dict):
    """Convert list-based split definitions to TradeSplit objects in place."""
    for key in ["split_train", "split_val", "split_test"]:
        if key in cfg_dict and isinstance(cfg_dict[key], (list, tuple)):
            vals = cfg_dict[key]
            if len(vals) != 2:
                raise ValueError(f"Split {key} must have exactly 2 elements (min, max). Got: {vals}")
            cfg_dict[key] = TradeSplit(int(vals[0]), int(vals[1]))


def test_multimodal_batch_contract():
    """Test that DataModule emits all required multimodal keys with correct shapes.
    
    Required modalities for ST-MM-GNN:
    - base_trade: (B, L, 194, 194, 2)  - Trade FOB tensor
    - risk_trade: (B, L, 194, 194, 8, 2) - Risk-conditioned trade tensor
    - climate: (B, L, 194, 10) - Climate features
    - climate_anoms: (B, L, 194, 10) - Climate anomalies
    - distance_km: (194, 194) - Geographic distance matrix
    - adjacency_border: (194, 194) - Border adjacency matrix
    - y_next: (B, 194, 8) - Pathogen status targets
    - y_mask: (B, 194, 8) - Pathogen status mask
    - t, t_y: Time indices
    """
    require_local_zarr(TRADE_BASE_ZARR, TRADE_RISK_ZARR)
    cfg_path = Path("config/stmm_stepA.yaml")
    assert cfg_path.exists(), f"Missing {cfg_path}"
    
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    
    # Get datamodule config
    dm_raw = cfg.get("datamodule", {})
    ROOT = Path.cwd()
    
    # Resolve relative paths
    for p_key in [
        "base_zarr_path",
        "risk_zarr_path",
        "scaler_json_path",
        "pathogen_zarr_path",
        "climate_zarr_path",
        "climate_anoms_zarr_path",
        "meta_distance_path",
        "meta_adjacency_path",
    ]:
        if p_key in dm_raw and isinstance(dm_raw[p_key], str):
            p = Path(dm_raw[p_key])
            if not p.is_absolute():
                dm_raw[p_key] = str(ROOT / p)
    
    convert_splits(dm_raw)
    dm_cfg = TradeDataModuleConfig(**dm_raw)
    
    # Instantiate and setup datamodule
    dm = TradeDataModule(dm_cfg)
    dm.setup()
    
    # Get first batch
    batch = next(iter(dm.train_dataloader()))
    
    # Assert batch is dict
    assert isinstance(batch, dict), f"Batch must be dict, got {type(batch)}"
    
    # Assert required keys present
    required_keys = [
        "base_trade",
        "risk_trade",
        "climate",
        "climate_anoms",
        "distance_km",
        "adjacency_border",
        "y_next",
        "y_mask",
        "t",
        "t_y",
    ]
    
    for k in required_keys:
        assert k in batch, f"Missing required key: {k}; batch keys={sorted(batch.keys())}"
    
    # Get batch size
    B = batch["t"].shape[0]
    
    # Assert shapes
    assert batch["base_trade"].shape == (B, 24, 194, 194, 2), (
        f"base_trade shape mismatch: {tuple(batch['base_trade'].shape)}"
    )
    assert batch["risk_trade"].shape == (B, 24, 194, 194, 8, 2), (
        f"risk_trade shape mismatch: {tuple(batch['risk_trade'].shape)}"
    )
    assert batch["climate"].shape == (B, 24, 194, 10), (
        f"climate shape mismatch: {tuple(batch['climate'].shape)}"
    )
    assert batch["climate_anoms"].shape == (B, 24, 194, 10), (
        f"climate_anoms shape mismatch: {tuple(batch['climate_anoms'].shape)}"
    )
    assert batch["distance_km"].shape == (194, 194), (
        f"distance_km shape mismatch: {tuple(batch['distance_km'].shape)}"
    )
    assert batch["adjacency_border"].shape == (194, 194), (
        f"adjacency_border shape mismatch: {tuple(batch['adjacency_border'].shape)}"
    )
    assert batch["y_next"].shape == (B, 194, 8), (
        f"y_next shape mismatch: {tuple(batch['y_next'].shape)}"
    )
    assert batch["y_mask"].shape == (B, 194, 8), (
        f"y_mask shape mismatch: {tuple(batch['y_mask'].shape)}"
    )
    
    # Assert y_mask has positives
    y_mask_nonzero = int((batch["y_mask"] > 0).sum().item())
    assert y_mask_nonzero > 0, (
        f"y_mask has no positive values; would cause loss=0. "
        f"Dataset valid-index filtering may be broken."
    )
    
    # Print summary for documentation
    print(f"\n[MULTIMODAL BATCH CONTRACT TEST PASSED]")
    print(f"Batch size: {B}")
    print(f"All required keys present with correct shapes:")
    print(f"  base_trade: {tuple(batch['base_trade'].shape)}")
    print(f"  risk_trade: {tuple(batch['risk_trade'].shape)}")
    print(f"  climate: {tuple(batch['climate'].shape)}")
    print(f"  climate_anoms: {tuple(batch['climate_anoms'].shape)}")
    print(f"  distance_km: {tuple(batch['distance_km'].shape)}")
    print(f"  adjacency_border: {tuple(batch['adjacency_border'].shape)}")
    print(f"  y_next: {tuple(batch['y_next'].shape)}")
    print(f"  y_mask: {tuple(batch['y_mask'].shape)} (nonzero={y_mask_nonzero})")


if __name__ == "__main__":
    test_multimodal_batch_contract()
