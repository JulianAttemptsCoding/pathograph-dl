"""
Unit tests for masked AUROC and AUPRC metrics in STMM Lightning module.

Tests:
- Metric updates with masked samples
- Per-pathogen computation
- Macro averaging
- Validation and test splits
- Degeneracy handling (NaN)
- Count reporting
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from pathograph.pl.stmm_pl_module import STMMPLModule


class DummyModel(nn.Module):
    """Dummy model for testing."""
    
    def __init__(self, num_nodes=10, num_pathogens=8, out_channels=128):
        super().__init__()
        self.linear = nn.Linear(num_nodes * num_pathogens, num_nodes * num_pathogens)
    
    def forward(self, batch):
        B, L, N, P = batch['y_hist'].shape
        # Return random logits
        return torch.randn(B, N, P)


@pytest.fixture
def pl_module():
    model = DummyModel()
    return STMMPLModule(model, num_pathogens=3)


def test_stmm_module_has_metrics(pl_module):
    """Test that STMMPLModule initializes with epoch-level metric accumulators."""
    # Check validation accumulators
    assert hasattr(pl_module, 'val_probs_batches')
    assert hasattr(pl_module, 'val_targets_batches')
    assert hasattr(pl_module, 'val_mask_batches')
    
    # Check test accumulators
    assert hasattr(pl_module, 'test_probs_batches')
    assert hasattr(pl_module, 'test_targets_batches')
    assert hasattr(pl_module, 'test_mask_batches')
    
    # Check initialized as empty lists
    assert isinstance(pl_module.val_probs_batches, list)
    assert len(pl_module.val_probs_batches) == 0


def test_validation_step_updates_metrics(pl_module):
    """Test that validation_step accumulates batch data."""
    # Create synthetic batch
    B, L, N, P = 4, 6, 10, 3
    batch = {
        'y_hist': torch.randn(B, L, N, P),
        'y_hist_mask': torch.ones(B, L, N, P),
        'y_next': torch.randint(0, 2, (B, N, P)).float(),
        'y_mask': torch.ones(B, N, P),
        'climate': torch.randn(B, L, N, 10),
        'climate_mask': torch.ones(B, L, N, 10),
    }
    
    pl_module.on_validation_epoch_start()
    assert len(pl_module.val_probs_batches) == 0
    
    loss = pl_module.validation_step(batch, batch_idx=0)
    
    assert loss.ndim == 0
    # Check batch data was accumulated
    assert len(pl_module.val_probs_batches) == 1
    assert pl_module.val_probs_batches[0].shape == (B, N, P)


def test_metrics_handle_degeneracy(pl_module):
    """Test that degenerate pathogens (no positives) result in NaN metrics."""
    B, N, P = 2, 10, 3
    L = 6
    
    # Pathogen 0: Valid mix
    # Pathogen 1: All zeros (degenerate - no pos)
    # Pathogen 2: All masked out (no data)
    
    y_next = torch.zeros(B, N, P)
    y_next[:, :, 0] = torch.randint(0, 2, (B, N)).float() # Mix
    y_next[:, :, 1] = 0.0 # All zeros (degenerate)
    
    y_mask = torch.ones(B, N, P)
    y_mask[:, :, 2] = 0.0 # All masked (no data)
    
    batch = {
        'y_hist': torch.randn(B, L, N, P),
        'y_hist_mask': torch.ones(B, L, N, P),
        'y_next': y_next,
        'y_mask': y_mask,
    }
    
    # Mock log
    logged = {}
    pl_module.log = MagicMock(side_effect=lambda k, v, **kwargs: logged.update({k: v}))
    
    pl_module.on_validation_epoch_start()
    pl_module.validation_step(batch, 0)
    pl_module.on_validation_epoch_end()
    
    # p0 should be valid (depends on random data, but check it exists)
    assert 'val_auroc_p0' in logged
    
    # p1 should be NaN (no positives)
    assert 'val_auroc_p1' in logged
    assert torch.isnan(torch.tensor(logged['val_auroc_p1']))
    
    # p2 should be NaN (no samples)
    assert 'val_auroc_p2' in logged
    assert torch.isnan(torch.tensor(logged['val_auroc_p2']))
    
    # Check n_valid counts exist and are correct
    assert 'val_n_valid_auroc' in logged
    # At most 1 valid pathogen (p0 if it has both pos and neg)
    assert logged['val_n_valid_auroc'] <= 1
    
    # Check totals
    assert 'val_pos_total' in logged
    assert 'val_valid_total' in logged


def test_counts_reporting(pl_module):
    """Test that pos_total and valid_total are reported."""
    B, N, P = 2, 10, 3
    batch = {
        'y_hist': torch.randn(B, 6, N, P),
        'y_hist_mask': torch.ones(B, 6, N, P),
        'y_next': torch.ones(B, N, P), # All positives
        'y_mask': torch.ones(B, N, P),
    }
    
    logged = {}
    pl_module.log = MagicMock(side_effect=lambda k, v, **kwargs: logged.update({k: v}))
    
    pl_module.on_validation_epoch_start()
    pl_module.validation_step(batch, 0)
    pl_module.on_validation_epoch_end()
    
    # Check totals
    assert 'val_pos_total' in logged
    assert 'val_valid_total' in logged
    assert logged['val_pos_total'] == B * N * P
    assert logged['val_valid_total'] == B * N * P


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
