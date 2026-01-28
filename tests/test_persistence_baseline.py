"""
Unit tests for persistence baseline.

Tests:
- Persistence prediction logic
- Masked BCE loss computation
- AUROC/AUPRC metric updates
- Validation and test steps
"""

import pytest
import torch

from pathograph.baselines import PersistenceBaseline


def test_persistence_baseline_init():
    """Test PersistenceBaseline initializes correctly."""
    model = PersistenceBaseline(num_pathogens=8)
    
    assert model.num_pathogens == 8
    assert len(model.val_auroc) == 8
    assert len(model.val_auprc) == 8
    assert len(model.test_auroc) == 8
    assert len(model.test_auprc) == 8


def test_persistence_prediction():
    """Test persistence prediction returns last observed value."""
    model = PersistenceBaseline(num_pathogens=3)
    
    B, L, N, P = 2, 4, 5, 3
    
    # Create simple history: all 0s except last time step = 1
    y_hist = torch.zeros(B, L, N, P)
    y_hist[:, -1, :, :] = 1.0
    
    y_hist_mask = torch.ones(B, L, N, P)
    
    batch = {
        'y_hist': y_hist,
        'y_hist_mask': y_hist_mask,
    }
    
    preds = model._predict_persistence(batch)
    
    # Should predict 1.0 (last observed value)
    assert preds.shape == (B, N, P)
    assert torch.allclose(preds, torch.ones(B, N, P))


def test_persistence_prediction_with_gaps():
    """Test persistence handles missing observations correctly."""
    model = PersistenceBaseline(num_pathogens=2)
    
    B, L, N, P = 1, 6, 3, 2
    
    # Create history with gaps
    y_hist = torch.zeros(B, L, N, P)
    y_hist[0, 2, :, :] = 1.0  # Observe 1 at t=2
    y_hist[0, 5, :, :] = 0.5  # Should not be seen (masked)
    
    y_hist_mask = torch.ones(B, L, N, P)
    y_hist_mask[0, 3:, :, :] = 0.0  # Mask out t>=3
    
    batch = {
        'y_hist': y_hist,
        'y_hist_mask': y_hist_mask,
    }
    
    preds = model._predict_persistence(batch)
    
    # Should predict 1.0 (last observed at t=2, not t=5 which is masked)
    assert preds.shape == (B, N, P)
    assert torch.allclose(preds, torch.ones(B, N, P))


def test_persistence_prediction_no_history():
    """Test persistence predicts 0 when no history is observed."""
    model = PersistenceBaseline(num_pathogens=2)
    
    B, L, N, P = 1, 4, 3, 2
    
    # All history masked out
    y_hist = torch.ones(B, L, N, P)  # Values don't matter
    y_hist_mask = torch.zeros(B, L, N, P)  # All masked
    
    batch = {
        'y_hist': y_hist,
        'y_hist_mask': y_hist_mask,
    }
    
    preds = model._predict_persistence(batch)
    
    # Should predict 0.0 (negative class default)
    assert preds.shape == (B, N, P)
    assert torch.allclose(preds, torch.zeros(B, N, P))


def test_persistence_masked_bce():
    """Test masked BCE loss computation."""
    model = PersistenceBaseline(num_pathogens=2)
    
    B, N, P = 2, 3, 2
    
    preds = torch.tensor([
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    ])
    
    targets = torch.tensor([
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],  # Perfect match
        [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],  # Perfect match
    ])
    
    mask = torch.ones(B, N, P)
    
    loss = model._masked_bce(preds, targets, mask)
    
    # Perfect predictions should give very low loss
    assert loss.item() < 0.1


def test_persistence_validation_step():
    """Test validation step runs without errors."""
    model = PersistenceBaseline(num_pathogens=3)
    
    B, L, N, P = 2, 6, 5, 3
    
    batch = {
        'y_hist': torch.randint(0, 2, (B, L, N, P)).float(),
        'y_hist_mask': torch.ones(B, L, N, P),
        'y_next': torch.randint(0, 2, (B, N, P)).float(),
        'y_mask': torch.ones(B, N, P),
    }
    
    # Run validation step
    loss = model.validation_step(batch, batch_idx=0)
    
    # Loss should be scalar
    assert loss.ndim == 0
    assert loss.item() >= 0
    
    # Run epoch end
    model.on_validation_epoch_end()


def test_persistence_test_step():
    """Test test step runs without errors."""
    model = PersistenceBaseline(num_pathogens=3)
    
    B, L, N, P = 2, 6, 5, 3
    
    batch = {
        'y_hist': torch.randint(0, 2, (B, L, N, P)).float(),
        'y_hist_mask': torch.ones(B, L, N, P),
        'y_next': torch.randint(0, 2, (B, N, P)).float(),
        'y_mask': torch.ones(B, N, P),
    }
    
    # Run test step
    loss = model.test_step(batch, batch_idx=0)
    
    # Loss should be scalar
    assert loss.ndim == 0
    assert loss.item() >= 0
    
    # Run epoch end
    model.on_test_epoch_end()


def test_persistence_configure_optimizers():
    """Test that persistence baseline returns None for optimizer."""
    model = PersistenceBaseline(num_pathogens=3)
    
    opt = model.configure_optimizers()
    
    assert opt is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
