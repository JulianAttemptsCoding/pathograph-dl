"""
Tests for FAOSTAT Step 2: Pseudo-Flow Generation

Tests:
1. Lag application (month in year Y uses weights from Y-1)
2. Pseudo-flow multiplication (E = F * W)
3. Sum preservation (sum_k E ≈ F when W sums to 1)
4. Mask propagation (observed_risk = observed_base AND has_weight)
5. Year boundary behavior
"""

import numpy as np
import pytest
from pathograph.trade.faostat_step2 import (
    build_month_to_year_mapping,
    apply_lag_and_generate_pseudoflows
)


def test_lag_application():
    """Test that month t in year Y uses weights from year Y-1."""
    # Create synthetic base tensor for 24 months (2 years)
    # Months 0-11: year 2020 (t_min=840 = Jan 2020)
    # Months 12-23: year 2021
    T = 24
    N = 2
    K = 2
    
    # Base tensor: simple constant values
    base_tensor = np.ones((T, N, N, 2), dtype=np.float32) * 100.0
    base_mask = np.ones((T, N, N, 2), dtype=np.uint8)
    base_is_estimated = np.zeros((T, N, N, 2), dtype=np.uint8)
    
    # Time index: Jan 2020 to Dec 2021
    # t = (year - 1950) * 12 + (month - 1)
    # Jan 2020: t = (2020 - 1950) * 12 + 0 = 840
    t_min = 840
    time_index = np.arange(t_min, t_min + T, dtype=np.int32)
    
    # Weights for years 2019 and 2020 (Y=2)
    # Year 2019 (y=0): CEREALS=0.3, LIVESTOCK=0.7
    # Year 2020 (y=1): CEREALS=0.4, LIVESTOCK=0.6
    W = np.zeros((2, N, N, K), dtype=np.float32)
    W[0, 0, 1, :] = [0.3, 0.7]  # 2019
    W[1, 0, 1, :] = [0.4, 0.6]  # 2020
    
    weight_mask = np.zeros((2, N, N), dtype=np.uint8)
    weight_mask[0, 0, 1] = 1
    weight_mask[1, 0, 1] = 1
    
    weight_year_min = 2019
    lag = 1
    
    # Generate pseudo-flows
    E, observed_risk, is_estimated_risk, backoff_risk, stats = apply_lag_and_generate_pseudoflows(
        base_tensor, base_mask, base_is_estimated,
        W, weight_mask, None,
        time_index, t_min, weight_year_min, lag, N, K
    )
    
    # Check that months in 2020 (t=0..11) use weights from 2019 (y=0)
    # E[t, 0, 1, k, ch] = 100.0 * W[0, 0, 1, k]
    for t in range(12):
        expected_cereals = 100.0 * 0.3
        expected_livestock = 100.0 * 0.7
        
        assert np.isclose(E[t, 0, 1, 0, 0], expected_cereals, atol=1e-3), f"Month {t} (2020) should use 2019 weights"
        assert np.isclose(E[t, 0, 1, 1, 0], expected_livestock, atol=1e-3)
    
    # Check that months in 2021 (t=12..23) use weights from 2020 (y=1)
    # E[t, 0, 1, k, ch] = 100.0 * W[1, 0, 1, k]
    for t in range(12, 24):
        expected_cereals = 100.0 * 0.4
        expected_livestock = 100.0 * 0.6
        
        assert np.isclose(E[t, 0, 1, 0, 0], expected_cereals, atol=1e-3), f"Month {t} (2021) should use 2020 weights"
        assert np.isclose(E[t, 0, 1, 1, 0], expected_livestock, atol=1e-3)


def test_pseudoflow_multiplication():
    """Test that E[t,i,j,k,ch] = F[t,i,j,ch] * W[y,i,j,k]."""
    # Create simple synthetic data
    T = 12
    N = 2
    K = 2
    
    # Base tensor with varying values
    base_tensor = np.zeros((T, N, N, 2), dtype=np.float32)
    base_tensor[:, 0, 1, 0] = 200.0  # USA->CAN, exports
    base_tensor[:, 0, 1, 1] = 300.0  # USA->CAN, imports
    
    base_mask = np.ones((T, N, N, 2), dtype=np.uint8)
    base_is_estimated = np.zeros((T, N, N, 2), dtype=np.uint8)
    
    # Time index: Jan 2020 to Dec 2020
    t_min = 840
    time_index = np.arange(t_min, t_min + T, dtype=np.int32)
    
    # Weights for year 2019
    W = np.zeros((1, N, N, K), dtype=np.float32)
    W[0, 0, 1, :] = [0.25, 0.75]  # CEREALS=0.25, LIVESTOCK=0.75
    
    weight_mask = np.ones((1, N, N), dtype=np.uint8)
    weight_mask[0, 0, 1] = 1
    
    weight_year_min = 2019
    lag = 1
    
    # Generate pseudo-flows
    E, observed_risk, is_estimated_risk, backoff_risk, stats = apply_lag_and_generate_pseudoflows(
        base_tensor, base_mask, base_is_estimated,
        W, weight_mask, None,
        time_index, t_min, weight_year_min, lag, N, K
    )
    
    # Check multiplication for exports (ch=0)
    expected_cereals_exports = 200.0 * 0.25
    expected_livestock_exports = 200.0 * 0.75
    
    np.testing.assert_allclose(E[:, 0, 1, 0, 0], expected_cereals_exports, atol=1e-3)
    np.testing.assert_allclose(E[:, 0, 1, 1, 0], expected_livestock_exports, atol=1e-3)
    
    # Check multiplication for imports (ch=1)
    expected_cereals_imports = 300.0 * 0.25
    expected_livestock_imports = 300.0 * 0.75
    
    np.testing.assert_allclose(E[:, 0, 1, 0, 1], expected_cereals_imports, atol=1e-3)
    np.testing.assert_allclose(E[:, 0, 1, 1, 1], expected_livestock_imports, atol=1e-3)


def test_sum_preservation():
    """Test that sum_k E ≈ F when weights sum to 1."""
    T = 12
    N = 2
    K = 3
    
    # Base tensor
    base_tensor = np.random.rand(T, N, N, 2).astype(np.float32) * 1000.0
    base_mask = np.ones((T, N, N, 2), dtype=np.uint8)
    base_is_estimated = np.zeros((T, N, N, 2), dtype=np.uint8)
    
    # Time index
    t_min = 840
    time_index = np.arange(t_min, t_min + T, dtype=np.int32)
    
    # Weights that sum to 1
    W = np.random.rand(1, N, N, K).astype(np.float32)
    # Normalize to sum to 1
    W = W / W.sum(axis=3, keepdims=True)
    
    weight_mask = np.ones((1, N, N), dtype=np.uint8)
    
    weight_year_min = 2019
    lag = 1
    
    # Generate pseudo-flows
    E, observed_risk, is_estimated_risk, backoff_risk, stats = apply_lag_and_generate_pseudoflows(
        base_tensor, base_mask, base_is_estimated,
        W, weight_mask, None,
        time_index, t_min, weight_year_min, lag, N, K
    )
    
    # Check that sum_k E ≈ F for all corridors and channels
    for t in range(T):
        for i in range(N):
            for j in range(N):
                for ch in range(2):
                    base_value = base_tensor[t, i, j, ch]
                    risk_sum = E[t, i, j, :, ch].sum()
                    
                    assert np.isclose(risk_sum, base_value, atol=1e-2), \
                        f"Sum preservation failed at t={t}, i={i}, j={j}, ch={ch}: {risk_sum} != {base_value}"


def test_mask_propagation():
    """Test that observed_risk = observed_base AND has_weight."""
    T = 12
    N = 2
    K = 2
    
    # Base tensor with some missing values
    base_tensor = np.ones((T, N, N, 2), dtype=np.float32) * 100.0
    base_mask = np.ones((T, N, N, 2), dtype=np.uint8)
    base_mask[:, 0, 0, :] = 0  # Diagonal is missing
    base_mask[:6, 0, 1, 0] = 0  # First 6 months of USA->CAN exports are missing
    
    base_is_estimated = np.zeros((T, N, N, 2), dtype=np.uint8)
    
    # Time index
    t_min = 840
    time_index = np.arange(t_min, t_min + T, dtype=np.int32)
    
    # Weights
    W = np.ones((1, N, N, K), dtype=np.float32) * 0.5
    weight_mask = np.ones((1, N, N), dtype=np.uint8)
    weight_mask[0, 1, 1] = 0  # No weights for CAN->CAN (diagonal)
    
    weight_year_min = 2019
    lag = 1
    
    # Generate pseudo-flows
    E, observed_risk, is_estimated_risk, backoff_risk, stats = apply_lag_and_generate_pseudoflows(
        base_tensor, base_mask, base_is_estimated,
        W, weight_mask, None,
        time_index, t_min, weight_year_min, lag, N, K
    )
    
    # Check mask propagation
    for t in range(T):
        for i in range(N):
            for j in range(N):
                for ch in range(2):
                    for k in range(K):
                        expected_mask = base_mask[t, i, j, ch] and weight_mask[0, i, j]
                        actual_mask = observed_risk[t, i, j, k, ch]
                        
                        assert actual_mask == expected_mask, \
                            f"Mask propagation failed at t={t}, i={i}, j={j}, k={k}, ch={ch}"


def test_year_boundary():
    """Test correct behavior at year boundaries."""
    # Create data spanning Dec 2020 to Jan 2021
    T = 2
    N = 2
    K = 2
    
    base_tensor = np.ones((T, N, N, 2), dtype=np.float32) * 100.0
    base_mask = np.ones((T, N, N, 2), dtype=np.uint8)
    base_is_estimated = np.zeros((T, N, N, 2), dtype=np.uint8)
    
    # Time index: Dec 2020 (t=851) and Jan 2021 (t=852)
    # Dec 2020: t = (2020 - 1950) * 12 + 11 = 851
    # Jan 2021: t = (2021 - 1950) * 12 + 0 = 852
    time_index = np.array([851, 852], dtype=np.int32)
    t_min = 851
    
    # Weights for years 2019 and 2020
    W = np.zeros((2, N, N, K), dtype=np.float32)
    W[0, 0, 1, :] = [0.3, 0.7]  # 2019
    W[1, 0, 1, :] = [0.4, 0.6]  # 2020
    
    weight_mask = np.ones((2, N, N), dtype=np.uint8)
    weight_mask[:, 0, 1] = 1
    
    weight_year_min = 2019
    lag = 1
    
    # Generate pseudo-flows
    E, observed_risk, is_estimated_risk, backoff_risk, stats = apply_lag_and_generate_pseudoflows(
        base_tensor, base_mask, base_is_estimated,
        W, weight_mask, None,
        time_index, t_min, weight_year_min, lag, N, K
    )
    
    # Dec 2020 (t=0) should use 2019 weights (y=0)
    expected_dec = [100.0 * 0.3, 100.0 * 0.7]
    np.testing.assert_allclose(E[0, 0, 1, :, 0], expected_dec, atol=1e-3)
    
    # Jan 2021 (t=1) should use 2020 weights (y=1)
    expected_jan = [100.0 * 0.4, 100.0 * 0.6]
    np.testing.assert_allclose(E[1, 0, 1, :, 0], expected_jan, atol=1e-3)


def test_missing_weight_year():
    """Test behavior when weight year is out of range."""
    T = 12
    N = 2
    K = 2
    
    base_tensor = np.ones((T, N, N, 2), dtype=np.float32) * 100.0
    base_mask = np.ones((T, N, N, 2), dtype=np.uint8)
    base_is_estimated = np.zeros((T, N, N, 2), dtype=np.uint8)
    
    # Time index: Jan 2020 to Dec 2020
    t_min = 840
    time_index = np.arange(t_min, t_min + T, dtype=np.int32)
    
    # Weights only for year 2020 (not 2019)
    # So months in 2020 need weights from 2019, which don't exist
    W = np.ones((1, N, N, K), dtype=np.float32) * 0.5
    weight_mask = np.ones((1, N, N), dtype=np.uint8)
    
    weight_year_min = 2020  # Weights start at 2020, but we need 2019
    lag = 1
    
    # Generate pseudo-flows
    E, observed_risk, is_estimated_risk, backoff_risk, stats = apply_lag_and_generate_pseudoflows(
        base_tensor, base_mask, base_is_estimated,
        W, weight_mask, None,
        time_index, t_min, weight_year_min, lag, N, K
    )
    
    # All months should have no risk flows (weight year out of range)
    assert np.all(E == 0.0)
    assert np.all(observed_risk == 0)
    
    # Stats should show months without weights
    assert stats['months_without_weights'] == T
    assert stats['months_with_weights'] == 0


def test_month_to_year_mapping():
    """Test month index to year conversion."""
    # Test a few known conversions
    # Jan 1950: t=0 -> year=1950
    # Dec 1950: t=11 -> year=1950
    # Jan 1951: t=12 -> year=1951
    # Jan 2020: t=840 -> year=2020
    
    time_index = np.array([0, 11, 12, 840], dtype=np.int32)
    t_min = 0
    
    years = build_month_to_year_mapping(time_index, t_min)
    
    expected = np.array([1950, 1950, 1951, 2020], dtype=np.int32)
    np.testing.assert_array_equal(years, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
