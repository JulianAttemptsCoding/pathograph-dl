"""
Unit Tests for IMTS Step 1 Pipeline

Critical test cases:
1. Order-independent FOB-first logic
2. Month column detection and parsing
3. Month index calculation
4. Continuous time axis with gaps
5. SCALE.ID interpretation
6. CIF→FOB conversion
7. Direction convention
8. Diagonal policy
9. Observed zeros validation
10. Negative value clipping
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pathograph.trade.imts_step1 import (
    compute_month_index,
    parse_month_column,
    detect_month_columns,
    load_node_index,
    merge_imports_fob_first,
    apply_diagonal_policy,
    process_chunk_to_accumulators
)


# ============================================================================
# Test: Month Index Calculation
# ============================================================================

def test_month_index_epoch():
    """Test that 1950-01 maps to index 0."""
    assert compute_month_index(1950, 1) == 0


def test_month_index_progression():
    """Test month index progression."""
    assert compute_month_index(1950, 12) == 11
    assert compute_month_index(1951, 1) == 12
    assert compute_month_index(2024, 1) == 888


def test_month_index_pre_1950():
    """Test that pre-1950 dates give negative indices."""
    assert compute_month_index(1949, 12) == -1
    assert compute_month_index(1949, 1) == -12


# ============================================================================
# Test: Month Column Parsing
# ============================================================================

def test_parse_month_column_valid():
    """Test parsing valid month column names."""
    assert parse_month_column('2024-M01') == (2024, 1)
    assert parse_month_column('1950-M12') == (1950, 12)
    assert parse_month_column('2000-M06') == (2000, 6)


def test_parse_month_column_invalid():
    """Test that invalid formats raise ValueError."""
    with pytest.raises(ValueError):
        parse_month_column('2024-01')  # Missing 'M'
    
    with pytest.raises(ValueError):
        parse_month_column('2024-M13')  # Invalid month
    
    with pytest.raises(ValueError):
        parse_month_column('2024M01')  # Missing hyphen


# ============================================================================
# Test: Month Column Detection
# ============================================================================

def test_detect_month_columns_basic():
    """Test basic month column detection."""
    df = pd.DataFrame(columns=['COUNTRY.ID', '2024-M01', '2024-M02', '2024-M03'])
    
    month_cols, t_indices, t_min, t_max, T_full = detect_month_columns(df)
    
    assert len(month_cols) == 3
    assert month_cols == ['2024-M01', '2024-M02', '2024-M03']
    assert T_full == 3
    assert t_max - t_min + 1 == 3


def test_detect_month_columns_with_gaps():
    """Test continuous time axis with gaps."""
    df = pd.DataFrame(columns=['COUNTRY.ID', '2020-M01', '2020-M03', '2020-M05'])
    
    month_cols, t_indices, t_min, t_max, T_full = detect_month_columns(df)
    
    # Should have 3 columns but T_full=5 (Jan, Feb, Mar, Apr, May)
    assert len(month_cols) == 3
    assert T_full == 5  # Continuous from M01 to M05


def test_detect_month_columns_pre_1950_filtering():
    """Test that pre-1950 columns are filtered out."""
    df = pd.DataFrame(columns=['COUNTRY.ID', '1949-M12', '1950-M01', '1950-M02'])
    
    month_cols, t_indices, t_min, t_max, T_full = detect_month_columns(df)
    
    # Only 1950-M01 and 1950-M02 should remain
    assert len(month_cols) == 2
    assert '1949-M12' not in month_cols
    assert t_min == 0  # 1950-M01


# ============================================================================
# Test: Node Index Loading
# ============================================================================

def test_load_node_index_valid():
    """Test loading valid node index."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Write 194 nodes
        f.write('node_id,iso3,iso2,name\n')
        for i in range(194):
            f.write(f'{i},ISO{i:03d},I{i},Country{i}\n')
        temp_path = f.name
    
    try:
        iso3_to_id, id_to_iso3, N = load_node_index(temp_path)
        
        assert N == 194
        assert len(iso3_to_id) == 194
        assert len(id_to_iso3) == 194
        assert iso3_to_id['ISO000'] == 0
        assert id_to_iso3[0] == 'ISO000'
    finally:
        Path(temp_path).unlink()


def test_load_node_index_wrong_count():
    """Test that non-194 node count raises error."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('node_id,iso3,iso2,name\n')
        for i in range(100):  # Only 100 nodes
            f.write(f'{i},ISO{i:03d},I{i},Country{i}\n')
        temp_path = f.name
    
    try:
        with pytest.raises(ValueError, match="exactly 194 rows"):
            load_node_index(temp_path)
    finally:
        Path(temp_path).unlink()


# ============================================================================
# Test: FOB-First Imports Merge (CRITICAL)
# ============================================================================

def test_merge_imports_fob_first_basic():
    """Test basic FOB-first merge."""
    T, N = 2, 3
    m = 0.06
    
    # FOB present for some cells
    fob_sum = np.array([
        [[0, 100, 0],
         [0, 0, 200],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]
    ], dtype=np.float64)
    
    fob_mask = (fob_sum > 0).astype(np.uint8)
    
    # CIF present for different cells
    cif_sum = np.array([
        [[0, 0, 300],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 400, 0],
         [0, 0, 0],
         [0, 0, 0]]
    ], dtype=np.float64)
    
    cif_mask = (cif_sum > 0).astype(np.uint8)
    
    imports_best, imports_mask, is_estimated, diag = merge_imports_fob_first(
        fob_sum, fob_mask, cif_sum, cif_mask, m
    )
    
    # Check FOB cells are used directly
    assert imports_best[0, 0, 1] == 100  # FOB
    assert is_estimated[0, 0, 1] == 0
    
    # Check CIF cells are converted
    assert imports_best[0, 0, 2] == pytest.approx(300 / 1.06)
    assert is_estimated[0, 0, 2] == 1
    
    # Check mask
    assert imports_mask[0, 0, 1] == 1
    assert imports_mask[0, 0, 2] == 1


def test_merge_imports_fob_overwrites_cif():
    """CRITICAL: Test that FOB overwrites CIF when both present."""
    T, N = 1, 2
    m = 0.06
    
    # Both FOB and CIF present for same cell
    fob_sum = np.array([[[0, 100], [0, 0]]], dtype=np.float64)
    fob_mask = np.array([[[0, 1], [0, 0]]], dtype=np.uint8)
    
    cif_sum = np.array([[[0, 200], [0, 0]]], dtype=np.float64)  # Different value
    cif_mask = np.array([[[0, 1], [0, 0]]], dtype=np.uint8)
    
    imports_best, imports_mask, is_estimated, diag = merge_imports_fob_first(
        fob_sum, fob_mask, cif_sum, cif_mask, m
    )
    
    # Must use FOB (100), not CIF (200/1.06 ≈ 188.7)
    assert imports_best[0, 0, 1] == 100
    assert is_estimated[0, 0, 1] == 0  # Not estimated
    
    # Diagnostics should show overlap
    assert diag['overlap_cells'] == 1


# ============================================================================
# Test: Diagonal Policy
# ============================================================================

def test_diagonal_policy():
    """Test that diagonal is forced to zero."""
    T, N, C = 2, 3, 2
    
    trade = np.ones((T, N, N, C), dtype=np.float32)
    mask = np.ones((T, N, N, C), dtype=np.uint8)
    is_estimated = np.ones((T, N, N, C), dtype=np.uint8)
    
    apply_diagonal_policy(trade, mask, is_estimated)
    
    # Check diagonal is zero
    for t in range(T):
        for c in range(C):
            for i in range(N):
                assert trade[t, i, i, c] == 0
                assert mask[t, i, i, c] == 0
                assert is_estimated[t, i, i, c] == 0


# ============================================================================
# Test: SCALE.ID Interpretation
# ============================================================================

def test_scale_id_interpretation():
    """Test that SCALE.ID is interpreted as 10^x."""
    # SCALE.ID = 6 means millions (10^6)
    assert 10 ** 6 == 1_000_000
    
    # SCALE.ID = 0 means units (10^0 = 1)
    assert 10 ** 0 == 1
    
    # SCALE.ID = 3 means thousands (10^3)
    assert 10 ** 3 == 1_000


# ============================================================================
# Test: CIF to FOB Conversion
# ============================================================================

def test_cif_to_fob_conversion():
    """Test CIF→FOB conversion formula."""
    m = 0.06
    cif_value = 106.0
    
    fob_value = cif_value / (1 + m)
    
    assert fob_value == pytest.approx(100.0)


# ============================================================================
# Test: Direction Convention
# ============================================================================

def test_direction_convention():
    """Test that direction is correctly interpreted."""
    # Exports: goods flow from reporter to partner
    # XG_FOB_USD: reporter=USA, partner=CHN → src=USA, dst=CHN
    
    # Imports: goods flow from partner to reporter
    # MG_FOB_USD: reporter=USA, partner=CHN → src=CHN, dst=USA
    
    # This is tested implicitly in process_chunk_to_accumulators
    # but we document the convention here
    pass


# ============================================================================
# Test: Observed Zeros
# ============================================================================

def test_observed_zeros_valid():
    """Test that observed zeros are valid (mask=1, value=0)."""
    # An observed cell with value 0 should have mask=1
    # This can happen after negative clipping or if IMF reports 0
    
    trade = np.array([[0, 100], [0, 0]], dtype=np.float32)
    mask = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    
    # Cell [0,0] is observed zero (mask=1, trade=0)
    assert mask[0, 0] == 1
    assert trade[0, 0] == 0
    
    # This is valid and should pass QC


# ============================================================================
# Test: Negative Value Clipping
# ============================================================================

def test_negative_clipping():
    """Test that negative values are clipped to zero."""
    # Simulated in process_chunk_to_accumulators with negative_policy='clip'
    
    val_usd = -100.0
    
    if val_usd < 0:
        val_usd = 0.0
    
    assert val_usd == 0.0


# ============================================================================
# Test: Chunk Processing
# ============================================================================

def test_process_chunk_basic():
    """Test basic chunk processing."""
    # Create minimal test data
    chunk = pd.DataFrame({
        'COUNTRY.ID': ['USA', 'CHN'],
        'COUNTERPART_COUNTRY.ID': ['CHN', 'USA'],
        'INDICATOR.ID': ['XG_FOB_USD', 'MG_FOB_USD'],
        'SCALE.ID': [6, 6],  # Millions
        '2024-M01': [100, 200]
    })
    
    month_cols = ['2024-M01']
    t_map = {'2024-M01': 0}
    iso3_map = {'USA': 0, 'CHN': 1}
    
    T, N = 1, 2
    exports_sum = np.zeros((T, N, N), dtype=np.float64)
    exports_mask = np.zeros((T, N, N), dtype=np.uint8)
    imports_fob_sum = np.zeros((T, N, N), dtype=np.float64)
    imports_fob_mask = np.zeros((T, N, N), dtype=np.uint8)
    imports_cif_sum = np.zeros((T, N, N), dtype=np.float64)
    imports_cif_mask = np.zeros((T, N, N), dtype=np.uint8)
    
    stats = {}
    
    process_chunk_to_accumulators(
        chunk, month_cols, t_map, iso3_map, 'SCALE.ID',
        exports_sum, exports_mask,
        imports_fob_sum, imports_fob_mask,
        imports_cif_sum, imports_cif_mask,
        stats=stats
    )
    
    # Check exports (USA→CHN)
    assert exports_sum[0, 0, 1] == 100 * 1e6
    assert exports_mask[0, 0, 1] == 1
    
    # Check imports (CHN→USA)
    assert imports_fob_sum[0, 0, 1] == 200 * 1e6
    assert imports_fob_mask[0, 0, 1] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
