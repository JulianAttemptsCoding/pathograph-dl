"""
Tests for masked classification metrics with degenerate cases.

Validates that safe_auroc and safe_auprc return NaN for degenerate inputs
and that macro aggregation correctly excludes NaNs.
"""

import numpy as np
import pytest
import torch

from pathograph.metrics import (
    count_pos_neg,
    flatten_masked,
    macro_nanmean,
    per_pathogen_metrics,
    safe_auprc,
    safe_auroc,
)


def test_flatten_masked_filters_correctly():
    """Test that flatten_masked correctly filters to observed samples."""
    probs = torch.tensor([[0.1, 0.9], [0.3, 0.7]])  # (B=2, N=2)
    targets = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    mask = torch.tensor([[1.0, 1.0], [0.0, 1.0]])  # First pathogen masked out in batch 1
    
    p, y = flatten_masked(probs, targets, mask)
    
    assert p.numel() == 3  # Only 3 observed samples
    assert y.numel() == 3
    assert torch.allclose(p, torch.tensor([0.1, 0.9, 0.7]))
    assert torch.allclose(y, torch.tensor([0.0, 1.0, 1.0]))


def test_count_pos_neg_basic():
    """Test counting positives and negatives."""
    y = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
    pos, neg = count_pos_neg(y)
    assert pos == 3
    assert neg == 2


def test_safe_auroc_all_zeros_returns_nan():
    """AUROC should return NaN when all targets are 0 (no positives)."""
    preds = torch.tensor([0.1, 0.3, 0.5, 0.7])
    targets = torch.tensor([0.0, 0.0, 0.0, 0.0])
    
    result = safe_auroc(preds, targets)
    assert np.isnan(result)


def test_safe_auroc_all_ones_returns_nan():
    """AUROC should return NaN when all targets are 1 (no negatives)."""
    preds = torch.tensor([0.1, 0.3, 0.5, 0.7])
    targets = torch.tensor([1.0, 1.0, 1.0, 1.0])
    
    result = safe_auroc(preds, targets)
    assert np.isnan(result)


def test_safe_auroc_empty_returns_nan():
    """AUROC should return NaN for empty inputs."""
    preds = torch.tensor([])
    targets = torch.tensor([])
    
    result = safe_auroc(preds, targets)
    assert np.isnan(result)


def test_safe_auroc_valid_case():
    """AUROC should return valid value for non-degenerate case."""
    # Perfect separation
    preds = torch.tensor([0.1, 0.2, 0.8, 0.9])
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0])
    
    result = safe_auroc(preds, targets)
    assert not np.isnan(result)
    assert 0.0 <= result <= 1.0
    assert result > 0.9  # Should be near-perfect


def test_safe_auprc_no_positives_returns_nan():
    """AUPRC should return NaN when no positives."""
    preds = torch.tensor([0.1, 0.3, 0.5, 0.7])
    targets = torch.tensor([0.0, 0.0, 0.0, 0.0])
    
    result = safe_auprc(preds, targets)
    assert np.isnan(result)


def test_safe_auprc_empty_returns_nan():
    """AUPRC should return NaN for empty inputs."""
    preds = torch.tensor([])
    targets = torch.tensor([])
    
    result = safe_auprc(preds, targets)
    assert np.isnan(result)


def test_safe_auprc_valid_case():
    """AUPRC should return valid value for non-degenerate case."""
    preds = torch.tensor([0.1, 0.2, 0.8, 0.9])
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0])
    
    result = safe_auprc(preds, targets)
    assert not np.isnan(result)
    assert 0.0 <= result <= 1.0


def test_per_pathogen_metrics_mixed_degeneracy():
    """Test per-pathogen metrics with some pathogens degenerate."""
    B, N, P = 2, 3, 4
    
    # Create synthetic data where:
    # p0: valid, mixed
    # p1: all zeros (no positives)
    # p2: all ones (no negatives)
    # p3: all masked out (no valid)
    
    probs = torch.rand(B, N, P)
    targets = torch.zeros(B, N, P)
    mask = torch.ones(B, N, P)
    
    # p0: mixed targets
    targets[:, :, 0] = torch.tensor([[0, 1, 0], [1, 0, 1]])
    
    # p1: all zeros (already set)
    targets[:, :, 1] = 0.0
    
    # p2: all ones
    targets[:, :, 2] = 1.0
    
    # p3: mask out completely
    mask[:, :, 3] = 0.0
    
    result = per_pathogen_metrics(probs, targets, mask)
    
    # Check counts
    assert result['valid'][0] == B * N  # p0: all observed
    assert result['pos'][0] == 3  # p0: 3 ones
    assert result['neg'][0] == 3  # p0: 3 zeros
    
    assert result['valid'][1] == B * N  # p1: all observed
    assert result['pos'][1] == 0  # p1: no positives
    assert result['neg'][1] == B * N  # p1: all negatives
    
    assert result['valid'][2] == B * N  # p2: all observed
    assert result['pos'][2] == B * N  # p2: all positives
    assert result['neg'][2] == 0  # p2: no negatives
    
    assert result['valid'][3] == 0  # p3: none observed
    assert result['pos'][3] == 0
    assert result['neg'][3] == 0
    
    # Check metrics
    assert not np.isnan(result['auroc'][0])  # p0: valid
    assert not np.isnan(result['auprc'][0])
    
    assert np.isnan(result['auroc'][1])  # p1: degenerate (no pos)
    assert np.isnan(result['auprc'][1])
    
    assert np.isnan(result['auroc'][2])  # p2: degenerate (no neg)
    # AUPRC might be valid (only needs pos), but our impl returns NaN for all-ones too
    
    assert np.isnan(result['auroc'][3])  # p3: no data
    assert np.isnan(result['auprc'][3])


def test_macro_nanmean_excludes_nans():
    """Test that macro_nanmean correctly excludes NaNs and counts them."""
    values = np.array([0.8, np.nan, 0.9, np.nan, 0.85])
    
    mean, n_used = macro_nanmean(values)
    
    assert n_used == 3
    assert np.isclose(mean, (0.8 + 0.9 + 0.85) / 3)


def test_macro_nanmean_all_nans():
    """Test macro_nanmean with all NaN input."""
    values = np.array([np.nan, np.nan, np.nan])
    
    mean, n_used = macro_nanmean(values)
    
    assert n_used == 0
    assert np.isnan(mean)


def test_macro_nanmean_no_nans():
    """Test macro_nanmean with no NaNs."""
    values = np.array([0.1, 0.2, 0.3, 0.4])
    
    mean, n_used = macro_nanmean(values)
    
    assert n_used == 4
    assert np.isclose(mean, 0.25)
