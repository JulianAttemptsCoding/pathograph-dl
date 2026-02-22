import pytest
import torch
import numpy as np

def test_incident_label_logic_mock():
    # prev 0, future 1 -> incident=1
    # prev 1, future 1 -> incident=0
    # masks missing at either endpoint -> scored mask=0
    
    # Let's write the exact numpy logic from trade_dataset.py
    prev = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
    future = np.array([1, 1, 0, 0, 1], dtype=np.uint8)
    
    prev_m = np.array([1, 1, 1, 1, 0], dtype=np.uint8)
    status_m = np.array([1, 1, 1, 1, 1], dtype=np.uint8)
    
    y_incident = ((future == 1) & (prev == 0)).astype(np.float32)
    mask_incident = status_m & prev_m
    
    # Expected answers
    # idx 0: prev=0, fut=1 -> y_inc=1. masks=1,1 -> mask_inc=1
    # idx 1: prev=1, fut=1 -> y_inc=0. masks=1,1 -> mask_inc=1
    # idx 2: prev=0, fut=0 -> y_inc=0. masks=1,1 -> mask_inc=1
    # idx 3: prev=1, fut=0 -> y_inc=0. masks=1,1 -> mask_inc=1
    # idx 4: prev=0, fut=1 -> y_inc=1. masks=0,1 -> mask_inc=0 (missing prev)
    
    assert y_incident[0] == 1.0
    assert y_incident[1] == 0.0
    assert y_incident[2] == 0.0
    assert y_incident[3] == 0.0
    assert y_incident[4] == 1.0
    
    assert mask_incident[0] == 1
    assert mask_incident[1] == 1
    assert mask_incident[2] == 1
    assert mask_incident[3] == 1
    assert mask_incident[4] == 0


def test_degenerate_metrics():
    # Force a batch where masked labels are all zeros to ensure metric returns NaN and module doesn't throw
    from pathograph.metrics.masked_classification import per_pathogen_metrics, macro_nanmean
    
    # 1 batch, 2 nodes, 2 pathogens
    probs = torch.tensor([[[0.1, 0.9], [0.2, 0.8]]])
    targets = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])  # ALL ZEROS
    mask = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])     # All observed
    
    result = per_pathogen_metrics(probs, targets, mask)
    
    # Pathogen 0 and 1 have 0 positives, so auroc/auprc should be NaN
    assert np.isnan(result['auroc'][0])
    assert np.isnan(result['auroc'][1])
    assert np.isnan(result['auprc'][0])
    assert np.isnan(result['auprc'][1])
    
    assert result['pos'][0] == 0
    assert result['neg'][0] == 2
    
    macro_auroc, n_auroc = macro_nanmean(result['auroc'])
    assert np.isnan(macro_auroc)
    assert n_auroc == 0
