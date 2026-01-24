"""
Test: ST-MM-GNN Forward Contract

Verifies that the model produces correct output shapes given a synthetic batch.
"""

import torch

from pathograph.models.stmm_gwnet import STMMGraphWaveNet


def test_stmm_forward_contract():
    """Test that model forward pass produces correct shape."""
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Synthetic config (small N for speed)
    B, L, N, P = 2, 24, 8, 8
    K, C, F = 8, 2, 10
    
    # Create synthetic batch
    batch = {
        'base_trade': torch.randn(B, L, N, N, C),
        'risk_trade': torch.randn(B, L, N, N, K, C),
        'climate_anoms': torch.randn(B, L, N, F),
        'adjacency_border': torch.randint(0, 2, (N, N), dtype=torch.uint8),
    }
    
    # Instantiate model
    model = STMMGraphWaveNet(
        residual_channels=16,
        dilation_channels=16,
        skip_channels=32,
        end_channels=64,
        kernel_size=2,
        dilations=[1, 2, 4],
        diffusion_K=2,
        dropout=0.1,
        num_pathogens=P,
        num_nodes=N,
    )
    
    # Forward pass
    logits = model(batch)
    
    # Assertions
    assert logits.ndim == 3, f"Expected 3D tensor, got {logits.ndim}D"
    assert logits.shape[0] == B, f"Batch size mismatch: {logits.shape[0]} vs {B}"
    assert logits.shape[1] == N, f"Node count mismatch: {logits.shape[1]} vs {N}"
    assert logits.shape[2] == P, f"Pathogen count mismatch: {logits.shape[2]} vs {P}"
    
    # Check dtype and finiteness
    assert logits.dtype == torch.float32, f"Expected float32, got {logits.dtype}"
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"
    
    print(f"✓ Forward contract test passed: logits.shape = {logits.shape}")


if __name__ == '__main__':
    test_stmm_forward_contract()
