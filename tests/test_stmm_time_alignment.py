"""Test time alignment for ST-MM-GNN datamodule.

This test proves that y_next corresponds to the next month relative to 
the input window, without guessing or print-only verification.
"""

from pathlib import Path
import torch
import yaml
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathograph.data.trade_datamodule import TradeDataModule, TradeDataModuleConfig
from pathograph.data.trade_dataset import TradeSplit


def convert_splits(cfg_dict):
    """Convert list-based split definitions to TradeSplit objects in place."""
    for key in ["split_train", "split_val", "split_test"]:
        if key in cfg_dict and isinstance(cfg_dict[key], (list, tuple)):
            vals = cfg_dict[key]
            if len(vals) != 2:
                raise ValueError(f"Split {key} must have exactly 2 elements (min, max). Got: {vals}")
            cfg_dict[key] = TradeSplit(int(vals[0]), int(vals[1]))


def test_time_alignment_and_target_shapes():
    """Prove that t_y = t + horizon and y_next has correct shape (B,194,8).
    
    Time Indexing Convention (discovered from trade_dataset.py line 231-241):
    - t: Last index of input window
    - t0 = t - (L - 1): First index of input window
    - t1 = t + 1: End of input window (exclusive slice)
    - t_y = t + H: Target index, where H is horizon
    - Input window: [t0, t1) = [t-(L-1), t+1) covers L months ending at t
    - Target: t_y = t + H, so for horizon=1, t_y = t + 1 (next month after window)
    
    This test asserts:
    1. t_y == t + horizon for each batch sample
    2. y_next.shape == (B, 194, 8)
    3. y_mask.shape == (B, 194, 8)
    4. y_mask has at least one positive value (non-empty targets)
    """
    cfg_path = Path("config/stmm_stepA.yaml")
    assert cfg_path.exists(), f"Missing {cfg_path}"
    
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    
    # Get datamodule config
    dm_raw = cfg.get("datamodule", {})
    ROOT = Path.cwd()
    
    # Resolve relative paths
    for p_key in ["base_zarr_path", "risk_zarr_path", "scaler_json_path", "pathogen_zarr_path"]:
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
    required_keys = ["t", "t_y", "y_next", "y_mask"]
    for k in required_keys:
        assert k in batch, f"Missing key {k}; batch keys={sorted(batch.keys())}"
    
    t = batch["t"]
    t_y = batch["t_y"]
    y_next = batch["y_next"]
    y_mask = batch["y_mask"]
    
    # Get horizon from config (default 1)
    horizon = dm_cfg.horizon
    
    # Assert time alignment: t_y == t + horizon
    # Both t and t_y should be (B,) tensors of indices
    assert t.shape == t_y.shape, f"t shape {t.shape} != t_y shape {t_y.shape}"
    assert t.dim() == 1, f"t must be 1D, got shape {t.shape}"
    
    # Check elementwise: t_y[i] == t[i] + horizon for all i
    expected_t_y = t + horizon
    assert torch.equal(t_y, expected_t_y), (
        f"Time alignment broken: t_y != t + {horizon}. "
        f"t={t.tolist()}, t_y={t_y.tolist()}, expected_t_y={expected_t_y.tolist()}"
    )
    
    # Assert target shapes
    B = t.shape[0]  # Batch size
    assert y_next.shape == (B, 194, 8), (
        f"y_next must be (B,194,8), got {tuple(y_next.shape)}"
    )
    assert y_mask.shape == (B, 194, 8), (
        f"y_mask must be (B,194,8), got {tuple(y_mask.shape)}"
    )
    
    # Assert y_mask has positives (non-empty targets)
    nonzero = int((y_mask > 0).sum().item())
    assert nonzero > 0, (
        f"y_mask has no positive values; would cause loss=0. "
        f"Dataset valid-index filtering may be broken."
    )
    
    # Print summary for documentation
    print(f"\n[TIME ALIGNMENT TEST PASSED]")
    print(f"Horizon: {horizon}")
    print(f"Invariant verified: t_y = t + {horizon}")
    print(f"Batch size: {B}")
    print(f"Sample t[0]={int(t[0].item())}, t_y[0]={int(t_y[0].item())}")
    print(f"y_next shape: {tuple(y_next.shape)}")
    print(f"y_mask shape: {tuple(y_mask.shape)}")
    print(f"y_mask nonzero: {nonzero}")


if __name__ == "__main__":
    test_time_alignment_and_target_shapes()
