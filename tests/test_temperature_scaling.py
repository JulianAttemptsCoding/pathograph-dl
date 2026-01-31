"""
Unit tests for temperature scaling calibration.

Tests:
- Temperature fitting on validation logits
- Calibrated probability output
- Masked sample handling
"""

import pytest
import torch

from pathograph.calibration import TemperatureScaling


def test_temperature_scaling_init():
    """Test TemperatureScaling initializes with T=1."""
    ts = TemperatureScaling()
    
    assert hasattr(ts, 'temperature')
    assert ts.temperature.item() == pytest.approx(1.0, abs=1e-5)


def test_temperature_scaling_forward():
    """Test forward pass applies temperature scaling."""
    ts = TemperatureScaling()
    
    # Set temperature to 2.0
    ts.temperature.data.fill_(2.0)
    
    # Create logits
    logits = torch.tensor([0.0, 1.0, -1.0, 2.0])
    
    # Apply temperature scaling
    probs = ts(logits)
    
    # Should be sigmoid(logits / 2.0)
    expected = torch.sigmoid(logits / 2.0)
    
    assert probs.shape == logits.shape
    assert torch.allclose(probs, expected, atol=1e-5)


def test_temperature_scaling_fit():
    """Test temperature fitting on synthetic data."""
    ts = TemperatureScaling()
    
    # Create synthetic overconfident logits
    # Model predicts high confidence but should be calibrated
    logits = torch.tensor([5.0, 5.0, -5.0, -5.0, 3.0, -3.0])
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    mask = torch.ones_like(logits)
    
    # Fit temperature
    ts.fit(logits, targets, mask, max_iter=50)
    
    # Temperature should be in reasonable range (can be < 1 or > 1)
    assert 0.1 <= ts.temperature.item() <= 10.0


def test_temperature_scaling_fit_with_mask():
    """Test temperature fitting respects mask."""
    ts = TemperatureScaling()
    
    # Create data with some masked samples
    logits = torch.tensor([5.0, 5.0, -5.0, -5.0, 100.0, -100.0])
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])  # Mask out outliers
    
    # Fit temperature
    ts.fit(logits, targets, mask, max_iter=50)
    
    # Should fit only on first 4 samples, temperature in valid range
    assert 0.1 <= ts.temperature.item() <= 10.0


def test_temperature_scaling_multidimensional():
    """Test temperature scaling with (B, N, P) shaped tensors."""
    ts = TemperatureScaling()
    
    B, N, P = 2, 3, 4
    
    logits = torch.randn(B, N, P) * 2  # Some variance
    targets = torch.randint(0, 2, (B, N, P)).float()
    mask = torch.ones(B, N, P)
    
    # Fit temperature
    ts.fit(logits, targets, mask, max_iter=20)
    
    # Should produce valid temperature
    assert 0.1 <= ts.temperature.item() <= 10.0
    
    # Forward pass should work
    probs = ts(logits)
    assert probs.shape == logits.shape
    assert (probs >= 0).all() and (probs <= 1).all()


def test_temperature_scaling_no_observed_samples():
    """Test that fit raises error when mask is all zeros."""
    ts = TemperatureScaling()
    
    logits = torch.randn(10)
    targets = torch.randint(0, 2, (10,)).float()
    mask = torch.zeros(10)  # No observed samples
    
    with pytest.raises(ValueError, match="No observed samples"):
        ts.fit(logits, targets, mask)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
