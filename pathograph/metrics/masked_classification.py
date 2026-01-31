"""
Masked classification metrics with explicit degeneracy handling.

Functions compute AUROC/AUPRC on masked data with NaN returns for degenerate cases
(no positives, no negatives, or no valid samples).
"""

from typing import Dict, Tuple

import numpy as np
import torch


def flatten_masked(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten and filter predictions and targets to observed samples only.
    
    Args:
        preds: (B, N, P) or (N, P) probabilities
        targets: Same shape as preds, binary {0, 1}
        mask: Same shape as preds, observation mask {0, 1}
    
    Returns:
        (p, y): 1D tensors filtered to mask==1
    """
    p_flat = preds.flatten()
    y_flat = targets.flatten()
    m_flat = mask.flatten()
    
    observed = m_flat > 0.5
    return p_flat[observed], y_flat[observed]


def count_pos_neg(y: torch.Tensor) -> Tuple[int, int]:
    """
    Count positives and negatives in binary tensor.
    
    Args:
        y: 1D binary tensor
    
    Returns:
        (pos_count, neg_count)
    """
    pos = (y > 0.5).sum().item()
    neg = (y < 0.5).sum().item()
    return int(pos), int(neg)


def safe_auroc(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute AUROC with degeneracy handling.
    
    Args:
        preds: 1D tensor of probabilities
        targets: 1D binary tensor
    
    Returns:
        AUROC as float, or NaN if degenerate (no pos or no neg or empty)
    """
    if preds.numel() == 0 or targets.numel() == 0:
        return float('nan')
    
    pos, neg = count_pos_neg(targets)
    
    # Degenerate if no positives or no negatives
    if pos == 0 or neg == 0:
        return float('nan')
    
    # Use torchmetrics functional if available, otherwise simple fallback
    try:
        from torchmetrics.functional.classification import binary_auroc
        result = binary_auroc(preds, targets.long())
        return float(result.item())
    except ImportError:
        # Fallback: simple rank-based AUROC
        # Sort by prediction score descending
        sorted_indices = torch.argsort(preds, descending=True)
        sorted_targets = targets[sorted_indices]
        
        # Compute TPR and FPR at each threshold
        tp = torch.cumsum(sorted_targets, dim=0)
        fp = torch.cumsum(1 - sorted_targets, dim=0)
        
        tpr = tp / pos
        fpr = fp / neg
        
        # Trapez rule for AUC
        auroc = torch.trapz(tpr, fpr).item()
        return float(auroc)


def safe_auprc(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute AUPRC (Average Precision) with degeneracy handling.
    
    Args:
        preds: 1D tensor of probabilities
        targets: 1D binary tensor
    
    Returns:
        AUPRC as float, or NaN if degenerate (no pos or empty)
    """
    if preds.numel() == 0 or targets.numel() == 0:
        return float('nan')
    
    pos, neg = count_pos_neg(targets)
    
    # Degenerate if no positives
    if pos == 0:
        return float('nan')
    
    # Use torchmetrics functional if available
    try:
        from torchmetrics.functional.classification import binary_average_precision
        result = binary_average_precision(preds, targets.long())
        return float(result.item())
    except ImportError:
        # Fallback: simple precision-recall curve
        sorted_indices = torch.argsort(preds, descending=True)
        sorted_targets = targets[sorted_indices]
        
        tp = torch.cumsum(sorted_targets, dim=0)
        fp = torch.cumsum(1 - sorted_targets, dim=0)
        
        precision = tp / (tp + fp)
        recall = tp / pos
        
        # Average precision via trapz
        auprc = torch.trapz(precision, recall).item()
        return float(auprc)


def per_pathogen_metrics(
    probs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """
    Compute per-pathogen AUROC, AUPRC, and counts.
    
    Args:
        probs: (B, N, P) probabilities
        targets: (B, N, P) binary targets
        mask: (B, N, P) observation mask
    
    Returns:
        dict with keys:
            'auroc': (P,) array of AUROC per pathogen (NaN if degenerate)
            'auprc': (P,) array of AUPRC per pathogen (NaN if degenerate)
            'valid': (P,) array of observed sample counts
            'pos': (P,) array of positive counts
            'neg': (P,) array of negative counts
    """
    if probs.ndim == 2:
        # Add batch dimension
        probs = probs.unsqueeze(0)
        targets = targets.unsqueeze(0)
        mask = mask.unsqueeze(0)
    
    B, N, P = probs.shape
    
    auroc_array = np.full(P, np.nan)
    auprc_array = np.full(P, np.nan)
    valid_array = np.zeros(P, dtype=int)
    pos_array = np.zeros(P, dtype=int)
    neg_array = np.zeros(P, dtype=int)
    
    for p in range(P):
        # Extract pathogen p across all batches and nodes
        probs_p = probs[:, :, p]
        targets_p = targets[:, :, p]
        mask_p = mask[:, :, p]
        
        # Flatten and filter
        p_flat, y_flat = flatten_masked(probs_p, targets_p, mask_p)
        
        valid_array[p] = int(p_flat.numel())
        
        if p_flat.numel() > 0:
            pos, neg = count_pos_neg(y_flat)
            pos_array[p] = pos
            neg_array[p] = neg
            
            # Compute metrics with degeneracy handling
            auroc_array[p] = safe_auroc(p_flat, y_flat)
            auprc_array[p] = safe_auprc(p_flat, y_flat)
    
    return {
        'auroc': auroc_array,
        'auprc': auprc_array,
        'valid': valid_array,
        'pos': pos_array,
        'neg': neg_array,
    }


def macro_nanmean(values: np.ndarray) -> Tuple[float, int]:
    """
    Compute NaN-excluding mean of an array.
    
    Args:
        values: 1D array possibly containing NaNs
    
    Returns:
        (mean, n_used): mean excluding NaNs, and count of non-NaN values
                        If all NaN, returns (NaN, 0)
    """
    valid_mask = ~np.isnan(values)
    n_used = int(valid_mask.sum())
    
    if n_used == 0:
        return float('nan'), 0
    
    mean = float(np.mean(values[valid_mask]))
    return mean, n_used
