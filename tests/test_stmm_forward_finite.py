"""
Unit test: STMM forward pass must produce finite logits even with NaN inputs.
"""
import torch
import pytest
from pathograph.models.stmm_gwnet import STMMGraphWaveNet


def test_forward_with_nan_climate():
    """Test that model forward handles NaN climate inputs without producing NaN logits."""
    B, L, N, P = 2, 24, 8, 8  # Small batch
    
    # Create model
    model = STMMGraphWaveNet(
        num_nodes=N,
        num_pathogens=P,
        residual_channels=16,
        dilation_channels=16,
        skip_channels=32,
        end_channels=64,
        dilations=[1, 2],  # Just 2 layers for speed
    )
    model.eval()
    
    # Create synthetic batch with NaN climate data
    batch = {
        'base_trade': torch.randn(B, L, N, N, 2),
        'risk_trade': torch.randn(B, L, N, N, 8, 2),
        'climate_anoms': torch.full((B, L, N, 10), float('nan')),  # ALL NaN!
        'adjacency_border': torch.randint(0, 2, (N, N), dtype=torch.uint8),
    }
    
    # Forward pass (with no_grad since this is a test)
    with torch.no_grad():
        logits = model(batch)
    
    # Check shape
    assert logits.shape == (B, N, P), f"Expected shape ({B}, {N}, {P}), got {logits.shape}"
    
    # Check finite
    assert torch.isfinite(logits).all(), \
        f"Logits contain NaN/Inf even though NaN inputs were sanitized! " \
        f"NaN count: {torch.isnan(logits).sum()}, Inf count: {torch.isinf(logits).sum()}"
    
    print("✅ Model forward produces finite logits despite NaN climate inputs")


def test_forward_with_mixed_nan():
    """Test forward with some NaN values (not all) in climate."""
    B, L, N, P = 2, 24, 8, 8
    
    model = STMMGraphWaveNet(
        num_nodes=N,
        num_pathogens=P,
        residual_channels=16,
        dilation_channels=16,
        skip_channels=32,
        end_channels=64,
        dilations=[1, 2],
    )
    model.eval()
    
    # Create batch with partial NaN
    climate_anoms = torch.randn(B, L, N, 10)
    climate_anoms[0, :, :5, :] = float('nan')  # Some NaN regions
    
    batch = {
        'base_trade': torch.randn(B, L, N, N, 2),
        'risk_trade': torch.randn(B, L, N, N, 8, 2),
        'climate_anoms': climate_anoms,
        'adjacency_border': torch.eye(N, dtype=torch.uint8),  # Identity (all self-loops)
    }
    
    with torch.no_grad():
        logits = model(batch)
    
    assert torch.isfinite(logits).all(), "Logits contain NaN/Inf with partial NaN inputs"
    
    print("✅ Model handles partial NaN inputs correctly")


if __name__ == "__main__":
    test_forward_with_nan_climate()
    test_forward_with_mixed_nan()
