import numpy as np
import pytest

from pathograph.data.trade_dataset import TradeDatasetZarr, TradeDatasetConfig


class DummyTradeDatasetZarr(TradeDatasetZarr):
    """
    Dummy wrapper to override __init__ to bypass Zarr disk loads entirely for pure numeric testing.
    """
    def __init__(self, cfg):
        # We manually bypass the constructor to prevent disk IO.
        self.cfg = cfg
        self.t_start = 0
        self.t_end = 1
        self._filtering_enabled = False
        self._scaler = {
            "base": {"mean": [0.1, 0.1], "std": [2.0, 2.0]},
            "risk": {"mean": [0.1] * 16, "std": [2.0] * 16}
        }

def test_trade_transforms_log1p_and_standardize_sequence():
    """
    Test that when both `apply_log1p` and `standardize` are true,
    the dataset applies log1p first, and standardization second.
    """
    cfg = TradeDatasetConfig(
        base_zarr_path="dummy",
        risk_zarr_path="dummy",
        apply_log1p=True,
        standardize=True,
        scaler_json_path="dummy"  # Scaler manually bypassed in Dummy wrapper
    )
    
    # Initialize the dummy dataset
    ds = DummyTradeDatasetZarr(cfg)
    
    # Generate mock data replicating astronomically huge trade sums
    # 1.0e11 = 100 billion dollars
    raw_input = np.array([1.0e11, 0.0, 1.0e5], dtype=np.float32)
    
    # Simulate the scaler dictionary for base_trade (mapped via `_apply_transforms`)
    mean = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    std = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    
    # Hard mathematical calculation of expected bounds:
    # 1. np.log1p(1.0e11) ≈ 25.328
    # 2. (25.328 - 0.1) / 2.0 ≈ 12.614
    expected_log = np.log1p(np.maximum(raw_input, 0.0))
    expected_final = (expected_log - mean) / (std + 1e-8)
    
    # Manually execute the exact snippet identical to `__getitem__`
    x_test = raw_input.copy()
    if ds.cfg.standardize:
        x_test = ds._apply_transforms(x_test, mean, std)
    else:
        if ds.cfg.apply_log1p:
            x_test = np.log1p(np.maximum(x_test, 0.0))
    
    # Assert standard bounds mapping matches mathematical expectations perfectly
    np.testing.assert_allclose(x_test, expected_final, rtol=1e-5)
    
    # Further assert that the `vmax > 1e4` guard will NOT trip
    assert np.abs(x_test).max() < 1e4, f"Scale exploded to {np.abs(x_test).max()}"
    
    # Assert sanity
    assert x_test[0] < 15.0, "The 100 billion dollar input should compress cleanly under 15."

if __name__ == "__main__":
    test_trade_transforms_log1p_and_standardize_sequence()
    print("Test passed successfully.")
