"""
Tests for FAOSTAT Step 2: Weight Computation

Tests:
1. Weight normalization (sum_k W[y,i,j,k] == 1)
2. Unmapped item handling (drop vs OTHER)
3. Coverage and QC counters
"""

import numpy as np
import pandas as pd
import pytest
from pathograph.trade.faostat_step2 import (
    load_group_mapping,
    apply_group_mapping,
    compute_corridor_year_weights
)


def test_weight_normalization():
    """Test that weights sum to 1.0 for corridors with positive denominator."""
    # Create synthetic FAOSTAT data
    # 2 corridors, 2 years, 3 items mapped to 2 groups
    df = pd.DataFrame({
        'year': [2020, 2020, 2020, 2021, 2021, 2021],
        'reporter_iso3': ['USA', 'USA', 'USA', 'USA', 'USA', 'USA'],
        'partner_iso3': ['CAN', 'CAN', 'CAN', 'CAN', 'CAN', 'CAN'],
        'item_code': [15, 27, 44, 15, 27, 44],
        'value': [100.0, 200.0, 300.0, 150.0, 250.0, 350.0],
        'group_id': ['CEREALS', 'LIVESTOCK', 'VEGETABLES', 'CEREALS', 'LIVESTOCK', 'VEGETABLES']
    })
    
    # Create minimal node mapping
    iso3_to_id = {'USA': 0, 'CAN': 1}
    group_order = ['CEREALS', 'LIVESTOCK', 'VEGETABLES']
    N = 2
    K = 3
    
    # Compute weights
    W, weight_mask, stats = compute_corridor_year_weights(df, iso3_to_id, group_order, N, K)
    
    # Check shape
    assert W.shape == (2, 2, 2, 3)  # (Y=2, N=2, N=2, K=3)
    assert weight_mask.shape == (2, 2, 2)
    
    # Check normalization for corridor (USA -> CAN) in both years
    i, j = 0, 1  # USA -> CAN
    
    for y in range(2):
        if weight_mask[y, i, j]:
            weight_sum = W[y, i, j, :].sum()
            assert np.isclose(weight_sum, 1.0, atol=1e-5), f"Weights don't sum to 1 at y={y}, i={i}, j={j}: sum={weight_sum}"
    
    # Check specific values for year 2020
    # Total for USA->CAN in 2020: 100 + 200 + 300 = 600
    # Expected weights: CEREALS=100/600, LIVESTOCK=200/600, VEGETABLES=300/600
    expected_2020 = np.array([100/600, 200/600, 300/600])
    np.testing.assert_allclose(W[0, i, j, :], expected_2020, atol=1e-5)
    
    # Check specific values for year 2021
    # Total for USA->CAN in 2021: 150 + 250 + 350 = 750
    expected_2021 = np.array([150/750, 250/750, 350/750])
    np.testing.assert_allclose(W[1, i, j, :], expected_2021, atol=1e-5)
    
    # Check stats
    assert stats['year_min'] == 2020
    assert stats['year_max'] == 2021
    assert stats['Y'] == 2
    assert stats['corridors_with_weights'] > 0


def test_unmapped_item_handling_drop():
    """Test that unmapped items are dropped when OTHER group doesn't exist."""
    # Create synthetic data with unmapped item
    df = pd.DataFrame({
        'year': [2020, 2020, 2020],
        'reporter_iso3': ['USA', 'USA', 'USA'],
        'partner_iso3': ['CAN', 'CAN', 'CAN'],
        'item_code': [15, 27, 999],  # 999 is unmapped
        'value': [100.0, 200.0, 300.0]
    })
    
    # Create mapping without OTHER group
    item_to_group = {15: 'CEREALS', 27: 'LIVESTOCK'}
    group_order = ['CEREALS', 'LIVESTOCK']
    
    # Apply mapping
    df_mapped, stats = apply_group_mapping(df, item_to_group, group_order)
    
    # Check that unmapped item was dropped
    assert len(df_mapped) == 2
    assert stats['unmapped_items'] == 1
    assert stats['unmapped_value'] == 300.0
    assert stats['mapped_rows'] == 2
    
    # Check that only mapped items remain
    assert set(df_mapped['item_code']) == {15, 27}


def test_unmapped_item_handling_other():
    """Test that unmapped items are assigned to OTHER when it exists."""
    # Create synthetic data with unmapped item
    df = pd.DataFrame({
        'year': [2020, 2020, 2020],
        'reporter_iso3': ['USA', 'USA', 'USA'],
        'partner_iso3': ['CAN', 'CAN', 'CAN'],
        'item_code': [15, 27, 999],  # 999 is unmapped
        'value': [100.0, 200.0, 300.0]
    })
    
    # Create mapping with OTHER group
    item_to_group = {15: 'CEREALS', 27: 'LIVESTOCK', 999: 'OTHER'}
    group_order = ['CEREALS', 'LIVESTOCK', 'OTHER']
    
    # Apply mapping
    df_mapped, stats = apply_group_mapping(df, item_to_group, group_order)
    
    # Check that all items were kept
    assert len(df_mapped) == 3
    assert stats['unmapped_items'] == 0
    assert stats['mapped_rows'] == 3
    
    # Check that unmapped item was assigned to OTHER
    other_rows = df_mapped[df_mapped['item_code'] == 999]
    assert len(other_rows) == 1
    assert other_rows.iloc[0]['group_id'] == 'OTHER'


def test_multi_corridor_multi_year():
    """Test weight computation with multiple corridors and years."""
    # Create synthetic data for 2 corridors, 2 years, 2 groups
    df = pd.DataFrame({
        'year': [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2021],
        'reporter_iso3': ['USA', 'USA', 'CAN', 'CAN', 'USA', 'USA', 'CAN', 'CAN'],
        'partner_iso3': ['CAN', 'CAN', 'USA', 'USA', 'CAN', 'CAN', 'USA', 'USA'],
        'item_code': [15, 27, 15, 27, 15, 27, 15, 27],
        'value': [100.0, 200.0, 150.0, 250.0, 110.0, 210.0, 160.0, 260.0],
        'group_id': ['CEREALS', 'LIVESTOCK', 'CEREALS', 'LIVESTOCK', 'CEREALS', 'LIVESTOCK', 'CEREALS', 'LIVESTOCK']
    })
    
    iso3_to_id = {'USA': 0, 'CAN': 1}
    group_order = ['CEREALS', 'LIVESTOCK']
    N = 2
    K = 2
    
    W, weight_mask, stats = compute_corridor_year_weights(df, iso3_to_id, group_order, N, K)
    
    # Check that both corridors have weights in both years
    assert weight_mask[0, 0, 1] == 1  # USA->CAN, year 2020
    assert weight_mask[0, 1, 0] == 1  # CAN->USA, year 2020
    assert weight_mask[1, 0, 1] == 1  # USA->CAN, year 2021
    assert weight_mask[1, 1, 0] == 1  # CAN->USA, year 2021
    
    # Check normalization for all corridors
    for y in range(2):
        for i in range(N):
            for j in range(N):
                if weight_mask[y, i, j]:
                    weight_sum = W[y, i, j, :].sum()
                    assert np.isclose(weight_sum, 1.0, atol=1e-5)


def test_zero_trade_corridor():
    """Test that corridors with zero trade don't get weights."""
    # Create synthetic data with one corridor having zero trade
    df = pd.DataFrame({
        'year': [2020, 2020],
        'reporter_iso3': ['USA', 'USA'],
        'partner_iso3': ['CAN', 'CAN'],
        'item_code': [15, 27],
        'value': [100.0, 200.0],
        'group_id': ['CEREALS', 'LIVESTOCK']
    })
    
    iso3_to_id = {'USA': 0, 'CAN': 1, 'MEX': 2}
    group_order = ['CEREALS', 'LIVESTOCK']
    N = 3
    K = 2
    
    W, weight_mask, stats = compute_corridor_year_weights(df, iso3_to_id, group_order, N, K)
    
    # USA->CAN should have weights
    assert weight_mask[0, 0, 1] == 1
    
    # USA->MEX should NOT have weights (no trade)
    assert weight_mask[0, 0, 2] == 0
    
    # CAN->USA should NOT have weights (no trade)
    assert weight_mask[0, 1, 0] == 0


def test_single_group_corridor():
    """Test corridor with only one group (should have weight=1.0 for that group)."""
    # Create synthetic data with single group
    df = pd.DataFrame({
        'year': [2020],
        'reporter_iso3': ['USA'],
        'partner_iso3': ['CAN'],
        'item_code': [15],
        'value': [100.0],
        'group_id': ['CEREALS']
    })
    
    iso3_to_id = {'USA': 0, 'CAN': 1}
    group_order = ['CEREALS', 'LIVESTOCK']
    N = 2
    K = 2
    
    W, weight_mask, stats = compute_corridor_year_weights(df, iso3_to_id, group_order, N, K)
    
    # Check that CEREALS has weight 1.0, LIVESTOCK has weight 0.0
    expected = np.array([1.0, 0.0])
    np.testing.assert_allclose(W[0, 0, 1, :], expected, atol=1e-5)
    
    # Check normalization
    assert np.isclose(W[0, 0, 1, :].sum(), 1.0, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
