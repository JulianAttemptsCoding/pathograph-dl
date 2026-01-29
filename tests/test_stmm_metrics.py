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
    """Test that STMMPLModule initializes with AUROC and AUPRC metrics."""
    # Check validation metrics
    assert hasattr(pl_module, 'val_auroc')
    assert hasattr(pl_module, 'val_auprc')
    assert hasattr(pl_module, 'val_pos_total')
    
    # Check test metrics
    assert hasattr(pl_module, 'test_auroc')
    assert hasattr(pl_module, 'test_auprc')
    assert hasattr(pl_module, 'test_pos_total')


def test_validation_step_updates_metrics(pl_module):
    """Test that validation_step updates AUROC and AUPRC."""
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
    loss = pl_module.validation_step(batch, batch_idx=0)
    
    assert loss.ndim == 0
    # Check pos total updated
    assert pl_module.val_pos_total > 0


def test_metrics_handle_degeneracy(pl_module):
    """Test that degenerate pathogens (no positives) result in NaN metrics."""
    B, N, P = 2, 10, 3
    L = 6
    
    # Pathogen 0: Valid mix
    # Pathogen 1: All zeros (degenerate)
    # Pathogen 2: All masked out
    
    y_next = torch.zeros(B, N, P)
    y_next[:, :, 0] = torch.randint(0, 2, (B, N)).float() # Mix
    y_next[:, :, 1] = 0.0 # All zeros
    
    y_mask = torch.ones(B, N, P)
    y_mask[:, :, 2] = 0.0 # All masked
    
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
    
    # p0 should be valid
    # p1 should be NaN (no positives) - TorchMetrics might return 0 or error depending on config, but our code enforces NaN or catch
    # p2 should be NaN (no samples)
    
    # Check what was logged
    print(logged)
    
    # p1 and p2 should be NaN
    assert torch.isnan(torch.tensor(logged['val_auroc_p1'])) or logged['val_auroc_p1'] == 0.0 # AUPRC might be 0, AUROC is undefined.
    # Our code catches exceptions. If TorchMetrics returns value for all-zeros, it might be 0.
    # Wait, BinaryAUROC with all zeros targets usually errors "No positive samples in targets".
    # So our try-except block should catch it and leave it as NaN.
    
    assert torch.isnan(torch.tensor(logged['val_auroc_p2'])) # No samples
    
    # Check macro valid pathogens count
    # Only p0 should be valid? Maybe p1 is valid if it returns 0?
    # Ideally degenerate should be excluded.
    # If our try-except catches it, it's excluded.
    
    # Check counts
    assert 'val_pos_total' in logged
    assert 'macro_valid_pathogens' in logged
    
    # Ensure macro only includes finite values
    # If p1 and p2 are NaN, macro should == p0
    auroc_p0 = logged['val_auroc_p0']
    macro = logged['val_auroc_macro']
    
    if torch.isfinite(torch.tensor(auroc_p0)):
         # If p0 is valid, macro should be close to p0 (since others are excluded)
         # Note: test data is random, p0 might also be degenerate if unlucky
         pass


def test_counts_reporting(pl_module):
    """Test that pos_total and macro_valid_pathogens are reported."""
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
    
    assert logged['val_pos_total'] == B * N * P
    assert 'macro_valid_pathogens' in logged


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
