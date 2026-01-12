import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from pathograph.data.trade_dataset import TradeDatasetZarr, TradeDatasetConfig, TradeSplit
from pathograph.data.trade_collate import trade_collate_separate
from pathograph.train.trade_losses import masked_mse

@pytest.fixture
def mock_zarr():
    # Mocking the Zarr hierarchy object returned by open_trade_zarr
    h = MagicMock()
    h.T = 100
    h.N = 10
    h.K = 2
    
    # Create dummy data using numpy
    # Shapes:
    # base_trade: (T, N, N, 2)
    # base_mask: (T, N, N, 2)
    # base_est: (T, N, N, 2)
    h.base_trade = np.random.randn(100, 10, 10, 2).astype(np.float32)
    h.base_mask = np.ones((100, 10, 10, 2), dtype=np.uint8) # 1=Observed (Code != 0)
    h.base_is_estimated = np.zeros((100, 10, 10, 2), dtype=np.uint8)
    
    h.risk_trade = np.random.randn(100, 10, 10, 2, 2).astype(np.float32)
    # risk_mask "observed_mask": 1=observed
    # Shape matching trade: (T, N, N, K, 2)
    h.risk_mask = np.ones((100, 10, 10, 2, 2), dtype=np.uint8) 
    h.risk_is_estimated = np.zeros((100, 10, 10, 2), dtype=np.uint8)
    
    return h

@pytest.fixture
def base_config(tmp_path):
    # create dummy scaler json
    scaler_path = tmp_path / "scaler.json"
    import json
    with open(scaler_path, "w") as f:
        json.dump({
            "base": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
            "risk": {"mean": [0.0]*4, "std": [1.0]*4}
        }, f)
        
    return TradeDatasetConfig(
        base_zarr_path="dummy_base",
        risk_zarr_path="dummy_risk",
        scaler_json_path=str(scaler_path),
        lookback=5,
        horizon=1,
        split_train=TradeSplit(0, 50),
        split_val=TradeSplit(51, 80),
        split_test=TradeSplit(81, 99),
        return_targets=True,
        target_kind="both",
        include_target_masks=True
    )

def test_getitem_shapes(mock_zarr, base_config):
    with patch("pathograph.data.trade_dataset.open_trade_zarr", return_value=mock_zarr):
        ds = TradeDatasetZarr(base_config)
        item = ds[0]
        
        # Check targets
        assert "y_base" in item
        assert item["y_base"].shape == (10, 10, 2)
        assert item["y_base"].dtype == np.float32
        
        assert "y_risk" in item
        assert item["y_risk"].shape == (10, 10, 2, 2)

def test_collate_structure(mock_zarr, base_config):
    with patch("pathograph.data.trade_dataset.open_trade_zarr", return_value=mock_zarr):
        ds = TradeDatasetZarr(base_config)
        batch = [ds[0], ds[1]]
        
        collated = trade_collate_separate(batch)
        
        assert "y_base" in collated
        assert collated["y_base"].shape == (2, 10, 10, 2)
        assert isinstance(collated["y_base"], torch.Tensor)
        
        assert "y_risk" in collated
        assert collated["y_risk"].shape == (2, 10, 10, 2, 2)

def test_masked_loss():
    # Simple case
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 5.0])
    # mask out the error
    mask = torch.tensor([1, 1, 0])
    
    loss = masked_mse(pred, target, mask)
    # (0 + 0 + ignored) / 2 = 0
    assert loss.item() == 0.0
    
    # Case with error
    pred = torch.tensor([0.0, 0.0])
    target = torch.tensor([1.0, 1.0])
    mask = torch.tensor([1, 1])
    loss = masked_mse(pred, target, mask)
    # (1+1)/2 = 1.0
    assert abs(loss.item() - 1.0) < 1e-6


def test_filtered_dataset_nonempty_target_masks(mock_zarr, base_config):
    """Verify that with require_target_observed=True, dataset yields non-empty target masks."""
    # Update config to enable filtering
    from dataclasses import replace
    cfg = replace(
        base_config,
        require_target_observed=True,
        min_target_observed=1,
        require_target_observed_kind="both",
    )
    
    with patch("pathograph.data.trade_dataset.open_trade_zarr", return_value=mock_zarr):
        ds = TradeDatasetZarr(cfg)
        
        # Dataset should be non-empty (fixtures have full coverage)
        assert len(ds) > 0, "Filtered dataset should not be empty"
        
        # Check first few indices have non-empty masks
        for i in range(min(3, len(ds))):
            item = ds[i]
            y_base_mask = item.get("y_base_mask")
            y_risk_mask = item.get("y_risk_mask")
            
            if y_base_mask is not None:
                base_count = np.count_nonzero(y_base_mask)
                assert base_count >= 1, f"Index {i}: y_base_mask should be non-empty, got {base_count}"
            
            if y_risk_mask is not None:
                risk_count = np.count_nonzero(y_risk_mask)
                assert risk_count >= 1, f"Index {i}: y_risk_mask should be non-empty, got {risk_count}"

