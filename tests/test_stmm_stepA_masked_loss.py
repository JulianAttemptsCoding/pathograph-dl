"""
Test: ST-MM-GNN Masked Loss

Verifies that masked loss is finite and positive for sparse masks.
"""

import torch

from pathograph.models.stmm_gwnet import STMMGraphWaveNet
from pathograph.pl.stmm_pl_module import STMMPLModule


def test_stmm_masked_loss_sparse():
    """Test masked loss with sparse mask (few observed entries)."""
    torch.manual_seed(42)
    
    # Synthetic config
    B, N, P = 4, 8, 8
    
    # Create sparse mask (only 1% of entries observed)
    mask = torch.zeros(B, N, P)
    num_observed = max(1, int(0.01 * B * N * P))  # At least 1
    indices = torch.randperm(B * N * P)[:num_observed]
    mask.view(-1)[indices] = 1.0
    
    # Synthetic logits and targets
    logits = torch.randn(B, N, P)
    targets = torch.randint(0, 2, (B, N, P)).float()
    
    # Create dummy model for PL module
    model = torch.nn.Linear(1, 1)  # Dummy
    pl_module = STMMPLModule(model, lr=0.001)
    
    # Compute masked loss
    loss = pl_module._masked_bce_with_logits(logits, targets, mask)
    
    # Assertions
    mask_sum = mask.sum()
    assert mask_sum > 0, "Mask must have at least one observed entry"
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    assert loss.item() > 0, f"Loss should be positive: {loss.item()}"
    
    print(f"✓ Masked loss test passed: loss = {loss.item():.6f}, mask_sum = {mask_sum.item()}")


def test_stmm_masked_loss_empty_mask_guard():
    """Test that empty mask guard raises RuntimeError."""
    import pytest
    
    torch.manual_seed(42)
    
    B, N, P = 2, 4, 8
    
    # Empty mask
    mask = torch.zeros(B, N, P)
    logits = torch.randn(B, N, P, requires_grad=True)
    targets = torch.randint(0, 2, (B, N, P)).float()
    
    # Create dummy model
    model = torch.nn.Linear(1, 1)
    pl_module = STMMPLModule(model, lr=0.001)
    
    # Compute loss - should raise
    with pytest.raises(RuntimeError) as excinfo:
        loss = pl_module._masked_bce_with_logits(logits, targets, mask)
    
    # Check error message contains expected text
    assert '[STMM Loss] All' in str(excinfo.value)
    assert 'zero observed labels' in str(excinfo.value)
    
    print("✓ Empty mask guard test passed: correctly raises RuntimeError")


if __name__ == '__main__':
    test_stmm_masked_loss_sparse()
    test_stmm_masked_loss_empty_mask_guard()
