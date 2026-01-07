"""
IMTS Step 1: IMF IMTS CSV → FOB Trade Tensor

This module implements the complete pipeline for ingesting IMF IMTS monthly bilateral
trade data into a (T×194×194×2) tensor with FOB-first imports logic.

Key Features:
- Order-independent FOB-first: separate FOB/CIF accumulators, merge at end
- Continuous time axis: t_min to t_max with gaps filled as missing
- Robust validation: schema checks, entity mapping, scale interpretation
- Comprehensive QC: coverage, top corridors, estimated share, trace samples
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import zarr


# ============================================================================
# Constants
# ============================================================================

MONTH_COL_REGEX = r'^\d{4}-M\d{2}$'
EPOCH_YEAR = 1950
EPOCH_MONTH = 1


# ============================================================================
# Node Index & Entity Mapping
# ============================================================================

def load_node_index(path: str) -> Tuple[Dict[str, int], Dict[int, str], int]:
    """
    Load node_index.csv and validate 194-node universe.
    
    Returns:
        (iso3_to_id, id_to_iso3, N)
    
    Raises:
        ValueError: if validation fails
    """
    df = pd.read_csv(path)
    
    # Validate required columns
    if not {'iso3', 'node_id'}.issubset(df.columns):
        raise ValueError(f"node_index must have columns 'iso3' and 'node_id'; got {df.columns.tolist()}")
    
    # Validate 194 nodes
    if len(df) != 194:
        raise ValueError(f"node_index must have exactly 194 rows; got {len(df)}")
    
    # Validate node_id range
    node_ids = sorted(df['node_id'].unique())
    if node_ids != list(range(194)):
        raise ValueError(f"node_id must be 0..193; got range {min(node_ids)}..{max(node_ids)}")
    
    # Build mappings
    iso3_to_id = dict(zip(df['iso3'], df['node_id']))
    id_to_iso3 = dict(zip(df['node_id'], df['iso3']))
    
    logging.info(f"Loaded node index: 194 nodes, node_id 0..193")
    logging.info(f"Sample nodes: {list(iso3_to_id.keys())[:5]}")
    
    return iso3_to_id, id_to_iso3, 194


# ============================================================================
# Month Column Detection & Time Axis
# ============================================================================

def compute_month_index(year: int, month: int) -> int:
    """
    Compute month index since 1950-01 epoch.
    
    Formula: t = (year - 1950) * 12 + (month - 1)
    
    Examples:
        1950-01 → 0
        1950-12 → 11
        1951-01 → 12
        2024-01 → 888
    """
    return (year - EPOCH_YEAR) * 12 + (month - EPOCH_MONTH)


def parse_month_column(col: str) -> Tuple[int, int]:
    """
    Parse month column name to (year, month).
    
    Expected format: 'YYYY-M##' (e.g., '2024-M01')
    
    Returns:
        (year, month) as integers
    
    Raises:
        ValueError: if format is invalid
    """
    match = re.match(r'^(\d{4})-M(\d{2})$', col)
    if not match:
        raise ValueError(f"Invalid month column format: {col}")
    
    year = int(match.group(1))
    month = int(match.group(2))
    
    if not (1 <= month <= 12):
        raise ValueError(f"Invalid month in column {col}: month must be 1..12")
    
    return year, month


def detect_month_columns(
    df_header: pd.DataFrame,
    regex: str = MONTH_COL_REGEX
) -> Tuple[List[str], List[int], int, int, int]:
    """
    Detect and validate month columns, build continuous time axis.
    
    Args:
        df_header: DataFrame with header row (nrows=0)
        regex: Regex pattern for month columns
    
    Returns:
        (month_cols, t_indices, t_min, t_max, T_full)
        - month_cols: sorted list of month column names
        - t_indices: corresponding month indices (same length as month_cols)
        - t_min: minimum month index
        - t_max: maximum month index
        - T_full: t_max - t_min + 1 (continuous time axis length)
    
    Raises:
        ValueError: if no valid month columns found or all pre-1950
    """
    pattern = re.compile(regex)
    all_cols = df_header.columns.tolist()
    
    # Find month columns
    month_cols_raw = [c for c in all_cols if pattern.match(c)]
    
    if not month_cols_raw:
        raise ValueError(f"No month columns found matching regex {regex}")
    
    # Parse and compute indices
    month_data = []
    for col in month_cols_raw:
        try:
            year, month = parse_month_column(col)
            t = compute_month_index(year, month)
            month_data.append((col, year, month, t))
        except ValueError as e:
            logging.warning(f"Skipping invalid month column {col}: {e}")
    
    if not month_data:
        raise ValueError("No valid month columns after parsing")
    
    # Filter out pre-1950 (t < 0)
    pre_1950_count = sum(1 for _, _, _, t in month_data if t < 0)
    month_data = [(col, y, m, t) for col, y, m, t in month_data if t >= 0]
    
    if pre_1950_count > 0:
        logging.warning(f"Dropped {pre_1950_count} month columns with year < 1950")
    
    if not month_data:
        raise ValueError("No month columns remain after filtering pre-1950 data")
    
    # Sort by (year, month)
    month_data.sort(key=lambda x: (x[1], x[2]))
    
    month_cols = [col for col, _, _, _ in month_data]
    t_indices = [t for _, _, _, t in month_data]
    
    t_min = min(t_indices)
    t_max = max(t_indices)
    T_full = t_max - t_min + 1
    
    logging.info(f"Detected {len(month_cols)} month columns")
    logging.info(f"Time range: t={t_min}..{t_max} (T_full={T_full})")
    logging.info(f"First month: {month_cols[0]}, Last month: {month_cols[-1]}")
    
    return month_cols, t_indices, t_min, t_max, T_full


# ============================================================================
# Schema Validation
# ============================================================================

def validate_schema(
    df_header: pd.DataFrame,
    required_cols: List[str],
    optional_cols: List[str]
) -> Dict[str, any]:
    """
    Validate CSV schema against required and optional columns.
    
    Returns:
        dict with validation results and warnings
    
    Raises:
        ValueError: if required columns are missing
    """
    all_cols = set(df_header.columns)
    required_set = set(required_cols)
    optional_set = set(optional_cols)
    
    missing_required = required_set - all_cols
    if missing_required:
        raise ValueError(f"Missing required columns: {sorted(missing_required)}")
    
    present_optional = optional_set & all_cols
    missing_optional = optional_set - all_cols
    
    result = {
        'required_present': sorted(required_set & all_cols),
        'optional_present': sorted(present_optional),
        'optional_missing': sorted(missing_optional),
        'all_columns': sorted(all_cols)
    }
    
    logging.info(f"Schema validation passed: {len(result['required_present'])} required columns present")
    if missing_optional:
        logging.warning(f"Optional columns missing: {sorted(missing_optional)}")
    
    return result


# ============================================================================
# Chunk Processing (Two-Accumulator Approach)
# ============================================================================

def process_chunk_to_accumulators(
    chunk: pd.DataFrame,
    month_cols: List[str],
    t_map: Dict[str, int],  # month_col -> tensor_index (relative to t_min)
    iso3_map: Dict[str, int],
    scale_col: str,
    exports_sum: np.ndarray,
    exports_mask: np.ndarray,
    imports_fob_sum: np.ndarray,
    imports_fob_mask: np.ndarray,
    imports_cif_sum: np.ndarray,
    imports_cif_mask: np.ndarray,
    negative_policy: str = 'clip',
    stats: Optional[Dict] = None
) -> None:
    """
    Process a chunk and update accumulators in-place.
    
    Uses strict mask semantics: mask |= observed (never +=)
    Handles observed zeros correctly (mask=1 even if value=0)
    
    Args:
        chunk: DataFrame chunk
        month_cols: list of month column names
        t_map: mapping from month_col to tensor index [0, T_full-1]
        iso3_map: mapping from ISO3 to node_id
        scale_col: name of SCALE.ID column
        exports_sum, exports_mask: (T, N, N) accumulators for exports
        imports_fob_sum, imports_fob_mask: (T, N, N) accumulators for FOB imports
        imports_cif_sum, imports_cif_mask: (T, N, N) accumulators for CIF imports
        negative_policy: 'clip' or 'fail'
        stats: optional dict to accumulate statistics
    """
    if stats is None:
        stats = {}
    
    # Extract metadata columns
    reporter_iso3 = chunk['COUNTRY.ID'].values
    partner_iso3 = chunk['COUNTERPART_COUNTRY.ID'].values
    indicator = chunk['INDICATOR.ID'].values
    
    # Parse SCALE.ID safely - DROP invalid rows, never coerce to 0
    scale_raw = chunk[scale_col]
    try:
        # Try to convert to float then int
        scale_float = pd.to_numeric(scale_raw, errors='coerce')
        valid_scale_mask = ~scale_float.isna()
        
        # Drop rows with invalid SCALE.ID
        invalid_count = (~valid_scale_mask).sum()
        if invalid_count > 0:
            stats['dropped_invalid_scale'] = stats.get('dropped_invalid_scale', 0) + invalid_count
            
            # Log examples of invalid SCALE.ID (up to 20)
            invalid_examples = chunk[~valid_scale_mask][['COUNTRY.ID', 'COUNTERPART_COUNTRY.ID', 'INDICATOR.ID', scale_col]].head(20)
            if 'invalid_scale_examples' not in stats:
                stats['invalid_scale_examples'] = []
            stats['invalid_scale_examples'].extend(invalid_examples.to_dict('records'))
            
            # Filter chunk to valid rows only
            chunk = chunk[valid_scale_mask].copy()
            reporter_iso3 = chunk['COUNTRY.ID'].values
            partner_iso3 = chunk['COUNTERPART_COUNTRY.ID'].values
            indicator = chunk['INDICATOR.ID'].values
            scale_raw = chunk[scale_col]
        
        scale_id = scale_raw.astype(float).astype(int).values
    except Exception as e:
        logging.error(f"Unexpected error parsing SCALE.ID: {e}")
        # Drop entire chunk if parsing fails catastrophically
        stats['dropped_invalid_scale'] = stats.get('dropped_invalid_scale', 0) + len(chunk)
        return
    
    if len(chunk) == 0:
        return  # All rows had invalid SCALE.ID
    
    scale_mult = 10.0 ** scale_id
    
    # Get month values as matrix
    month_values = chunk[month_cols].values.astype(float)  # (n_rows, n_months)
    
    # Process each row
    for i in range(len(chunk)):
        rep_iso = reporter_iso3[i]
        par_iso = partner_iso3[i]
        indic = indicator[i]
        mult = scale_mult[i]
        
        # Map to node IDs
        if rep_iso not in iso3_map or par_iso not in iso3_map:
            stats['dropped_nonuniverse'] = stats.get('dropped_nonuniverse', 0) + 1
            continue
        
        rep_id = iso3_map[rep_iso]
        par_id = iso3_map[par_iso]
        
        # Determine direction and target accumulators
        if indic == 'XG_FOB_USD':
            # Exports: src=reporter, dst=partner
            src, dst = rep_id, par_id
            target_sum = exports_sum
            target_mask = exports_mask
        elif indic == 'MG_FOB_USD':
            # Imports FOB: src=partner, dst=reporter
            src, dst = par_id, rep_id
            target_sum = imports_fob_sum
            target_mask = imports_fob_mask
        elif indic == 'MG_CIF_USD':
            # Imports CIF: src=partner, dst=reporter
            src, dst = par_id, rep_id
            target_sum = imports_cif_sum
            target_mask = imports_cif_mask
        else:
            stats['dropped_unknown_indicator'] = stats.get('dropped_unknown_indicator', 0) + 1
            continue
        
        # Process month values
        row_values = month_values[i, :]
        
        for j, month_col in enumerate(month_cols):
            val = row_values[j]
            
            # Check if observed (non-null)
            if pd.isna(val):
                continue
            
            # Apply scaling
            val_usd = val * mult
            
            # Handle negatives
            if val_usd < 0:
                stats['negative_cells'] = stats.get('negative_cells', 0) + 1
                stats['negative_usd_total'] = stats.get('negative_usd_total', 0.0) + abs(val_usd)
                
                if negative_policy == 'fail':
                    raise ValueError(f"Negative value encountered: {val_usd} USD")
                elif negative_policy == 'clip':
                    val_usd = 0.0
            
            # Get tensor index
            t_idx = t_map[month_col]
            
            # Update accumulators (strict mask semantics)
            target_sum[t_idx, src, dst] += val_usd
            target_mask[t_idx, src, dst] = 1  # Observed (even if zero after clipping)
        
        stats['processed_rows'] = stats.get('processed_rows', 0) + 1


# ============================================================================
# Imports Merge (FOB-First Logic)
# ============================================================================

def merge_imports_fob_first(
    imports_fob_sum: np.ndarray,
    imports_fob_mask: np.ndarray,
    imports_cif_sum: np.ndarray,
    imports_cif_mask: np.ndarray,
    m: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Merge FOB and CIF imports with FOB-first policy.
    
    Logic:
        - If FOB observed: use FOB
        - Else if CIF observed: use CIF/(1+m), mark as estimated
        - Else: missing (value=0, mask=0)
    
    Returns:
        (imports_best, imports_mask, is_estimated, diagnostics)
        - imports_best: (T, N, N) float64 array
        - imports_mask: (T, N, N) uint8 array
        - is_estimated: (T, N, N) uint8 array
        - diagnostics: dict with overlap counts
    """
    # FOB-first merge
    imports_best = np.where(
        imports_fob_mask == 1,
        imports_fob_sum,
        np.where(
            imports_cif_mask == 1,
            imports_cif_sum / (1.0 + m),
            0.0
        )
    )
    
    # Mask: observed if either FOB or CIF present
    imports_mask = (imports_fob_mask | imports_cif_mask).astype(np.uint8)
    
    # Estimated: CIF-derived only (FOB absent, CIF present)
    is_estimated = ((imports_fob_mask == 0) & (imports_cif_mask == 1)).astype(np.uint8)
    
    # Diagnostics
    overlap_cells = np.sum((imports_fob_mask == 1) & (imports_cif_mask == 1))
    fob_only_cells = np.sum((imports_fob_mask == 1) & (imports_cif_mask == 0))
    cif_only_cells = np.sum((imports_fob_mask == 0) & (imports_cif_mask == 1))
    
    diagnostics = {
        'overlap_cells': int(overlap_cells),
        'fob_only_cells': int(fob_only_cells),
        'cif_only_cells': int(cif_only_cells),
        'total_imports_observed': int(np.sum(imports_mask)),
        'estimated_cells': int(np.sum(is_estimated))
    }
    
    logging.info(f"Imports merge: {diagnostics['total_imports_observed']} observed cells")
    logging.info(f"  FOB-only: {fob_only_cells}, CIF-only: {cif_only_cells}, Overlap: {overlap_cells}")
    logging.info(f"  Estimated (CIF-derived): {diagnostics['estimated_cells']}")
    
    return imports_best, imports_mask, is_estimated, diagnostics


# ============================================================================
# Diagonal Policy
# ============================================================================

def apply_diagonal_policy(
    trade: np.ndarray,
    mask: np.ndarray,
    is_estimated: np.ndarray
) -> None:
    """
    Force diagonal to zero with mask=0 (in-place).
    
    Diagonal represents self-trade (i==j), which is structurally invalid.
    """
    T, N, _, C = trade.shape
    
    for t in range(T):
        for c in range(C):
            for i in range(N):
                trade[t, i, i, c] = 0.0
                mask[t, i, i, c] = 0
                is_estimated[t, i, i, c] = 0
    
    logging.info("Applied diagonal policy: forced all (i,i) cells to zero with mask=0")


# ============================================================================
# File Hashing
# ============================================================================

def compute_file_sha256(path: str, chunk_size: int = 8192) -> str:
    """Compute SHA256 hash of file (streaming)."""
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


# ============================================================================
# Zarr Output
# ============================================================================

def write_zarr_outputs(path, trade, mask, is_estimated, time_index, attrs):
    import numpy as np
    import zarr

    # --- pick a filesystem store compatible with zarr v2/v3 ---
    try:
        store = zarr.DirectoryStore(path)  # zarr v2
    except AttributeError:
        from zarr.storage import LocalStore  # zarr v3
        store = LocalStore(path)

    root = zarr.open_group(store=store, mode="w")
    if attrs:
        root.attrs.update(attrs)

    # reasonable default chunking
    chunks_4d = (min(12, trade.shape[0]), 64, 64, 2)
    chunks_1d = (min(1024, trade.shape[0]),)

    def _create(name, data, chunks):
        data = np.asarray(data)
        # create_dataset exists in v2; v3 may prefer create_array
        try:
            arr = root.create_dataset(name, shape=data.shape, dtype=data.dtype, chunks=chunks)
        except Exception:
            arr = root.create_array(name, shape=data.shape, dtype=data.dtype, chunks=chunks)
        arr[...] = data

    _create("trade", trade.astype(np.float32), chunks_4d)
    _create("mask", mask.astype(np.uint8), chunks_4d)
    _create("is_estimated", is_estimated.astype(np.uint8), chunks_4d)
    _create("time_index", time_index.astype(np.int32), chunks_1d)


# ============================================================================
# QC Metrics
# ============================================================================

def compute_qc_metrics(
    trade: np.ndarray,
    mask: np.ndarray,
    is_estimated: np.ndarray,
    time_index: np.ndarray,
    id_to_iso3: Dict[int, str],
    t_min: int
) -> Dict:
    """
    Compute comprehensive QC metrics.
    
    Returns dict with:
        - coverage_overall
        - coverage_by_channel
        - estimated_share_imports
        - observed_zeros_count
        - top_corridors_exports
        - top_corridors_imports
        - nonnegativity_check
        - mask_invariant_check
    """
    T, N, _, C = trade.shape
    
    # Overall coverage (excluding diagonal)
    total_cells = T * N * (N - 1) * C  # Exclude diagonal
    observed_cells = np.sum(mask) - np.sum(mask[:, range(N), range(N), :])
    coverage_overall = float(observed_cells) / total_cells if total_cells > 0 else 0.0
    
    # Coverage by channel
    coverage_exports = float(np.sum(mask[..., 0])) / (T * N * N) if T * N * N > 0 else 0.0
    coverage_imports = float(np.sum(mask[..., 1])) / (T * N * N) if T * N * N > 0 else 0.0
    
    # Estimated share (imports only)
    imports_observed = np.sum(mask[..., 1])
    imports_estimated = np.sum(is_estimated[..., 1])
    estimated_share = float(imports_estimated) / imports_observed if imports_observed > 0 else 0.0
    
    # Observed zeros
    observed_zeros = np.sum((mask == 1) & (trade == 0))
    
    # Top corridors (exports)
    exports_total = np.sum(trade[..., 0], axis=0)  # (N, N)
    top_exports_idx = np.argsort(exports_total.flatten())[-50:][::-1]
    top_exports = []
    for idx in top_exports_idx:
        src = idx // N
        dst = idx % N
        if src == dst:
            continue  # Skip diagonal
        total_usd = float(exports_total[src, dst])
        if total_usd > 0:
            top_exports.append({
                'src_iso3': id_to_iso3[src],
                'dst_iso3': id_to_iso3[dst],
                'total_usd': total_usd
            })
    
    # Top corridors (imports)
    imports_total = np.sum(trade[..., 1], axis=0)  # (N, N)
    top_imports_idx = np.argsort(imports_total.flatten())[-50:][::-1]
    top_imports = []
    for idx in top_imports_idx:
        src = idx // N
        dst = idx % N
        if src == dst:
            continue
        total_usd = float(imports_total[src, dst])
        if total_usd > 0:
            top_imports.append({
                'src_iso3': id_to_iso3[src],
                'dst_iso3': id_to_iso3[dst],
                'total_usd': total_usd
            })
    
    # Validation checks
    nonnegativity_ok = bool(np.all(trade >= 0))
    mask_zero_implies_trade_zero = bool(np.all(trade[mask == 0] == 0))
    mask_zero_implies_estimated_zero = bool(np.all(is_estimated[mask == 0] == 0))
    exports_never_estimated = bool(np.all(is_estimated[..., 0] == 0))
    
    metrics = {
        'coverage_overall': coverage_overall,
        'coverage_exports': coverage_exports,
        'coverage_imports': coverage_imports,
        'estimated_share_imports': estimated_share,
        'observed_zeros_count': int(observed_zeros),
        'top_corridors_exports': top_exports[:20],
        'top_corridors_imports': top_imports[:20],
        'validation': {
            'nonnegativity_ok': nonnegativity_ok,
            'mask_zero_implies_trade_zero': mask_zero_implies_trade_zero,
            'mask_zero_implies_estimated_zero': mask_zero_implies_estimated_zero,
            'exports_never_estimated': exports_never_estimated
        }
    }
    
    logging.info(f"QC Metrics:")
    logging.info(f"  Coverage: {coverage_overall:.2%} overall, {coverage_exports:.2%} exports, {coverage_imports:.2%} imports")
    logging.info(f"  Estimated share (imports): {estimated_share:.2%}")
    logging.info(f"  Observed zeros: {observed_zeros}")
    logging.info(f"  Validation: all checks passed = {all(metrics['validation'].values())}")
    
    return metrics


# ============================================================================
# Trace Samples (Audit Artifact)
# ============================================================================

def generate_trace_samples(
    trade: np.ndarray,
    mask: np.ndarray,
    is_estimated: np.ndarray,
    time_index: np.ndarray,
    id_to_iso3: Dict[int, str],
    t_min: int,
    imports_fob_sum: np.ndarray,
    imports_fob_mask: np.ndarray,
    imports_cif_sum: np.ndarray,
    imports_cif_mask: np.ndarray,
    m: float,
    max_samples: int = 200
) -> pd.DataFrame:
    """
    Generate trace samples for audit trail.
    
    Samples observed cells stratified by time and channel, with oversampling
    of estimated imports when present.
    
    Returns DataFrame with columns:
        t, year, month, src_iso3, dst_iso3, src_id, dst_id, ch,
        value_usd, mask, is_estimated, has_fob, has_cif,
        fob_value_usd, cif_value_usd, cif_derived_usd
    """
    T, N, _, C = trade.shape
    
    samples = []
    
    # Build candidate indices for each channel
    for ch in range(C):
        observed_idx = np.argwhere(mask[..., ch] == 1)
        
        if len(observed_idx) == 0:
            continue
        
        # Stratify by time (10 bins)
        n_bins = min(10, T)
        bin_size = T // n_bins
        
        samples_per_bin = max(1, max_samples // (n_bins * C))
        
        for bin_idx in range(n_bins):
            bin_start = bin_idx * bin_size
            bin_end = (bin_idx + 1) * bin_size if bin_idx < n_bins - 1 else T
            
            # Filter to this time bin
            bin_mask = (observed_idx[:, 0] >= bin_start) & (observed_idx[:, 0] < bin_end)
            bin_indices = observed_idx[bin_mask]
            
            if len(bin_indices) == 0:
                continue
            
            # For imports channel, oversample estimated cells
            if ch == 1:
                estimated_mask = np.array([is_estimated[t, i, j, ch] == 1 
                                          for t, i, j in bin_indices])
                n_estimated = estimated_mask.sum()
                n_not_estimated = (~estimated_mask).sum()
                
                # Target 25-40% estimated if available
                if n_estimated > 0:
                    n_est_samples = min(n_estimated, int(samples_per_bin * 0.4))
                    n_reg_samples = samples_per_bin - n_est_samples
                    
                    est_idx = bin_indices[estimated_mask]
                    reg_idx = bin_indices[~estimated_mask]
                    
                    if len(est_idx) > n_est_samples:
                        est_idx = est_idx[np.random.choice(len(est_idx), n_est_samples, replace=False)]
                    if len(reg_idx) > n_reg_samples:
                        reg_idx = reg_idx[np.random.choice(len(reg_idx), n_reg_samples, replace=False)]
                    
                    selected = np.vstack([est_idx, reg_idx]) if len(reg_idx) > 0 else est_idx
                else:
                    # No estimated cells, sample uniformly
                    n_sample = min(len(bin_indices), samples_per_bin)
                    selected = bin_indices[np.random.choice(len(bin_indices), n_sample, replace=False)]
            else:
                # Exports: sample uniformly
                n_sample = min(len(bin_indices), samples_per_bin)
                selected = bin_indices[np.random.choice(len(bin_indices), n_sample, replace=False)]
            
            # Build sample rows
            for t_idx, i, j in selected:
                t_abs = time_index[t_idx]
                year = EPOCH_YEAR + t_abs // 12
                month = (t_abs % 12) + 1
                
                sample = {
                    't': int(t_abs),
                    'year': int(year),
                    'month': int(month),
                    'src_iso3': id_to_iso3[i],
                    'dst_iso3': id_to_iso3[j],
                    'src_id': int(i),
                    'dst_id': int(j),
                    'ch': int(ch),
                    'value_usd': float(trade[t_idx, i, j, ch]),
                    'mask': int(mask[t_idx, i, j, ch]),
                    'is_estimated': int(is_estimated[t_idx, i, j, ch])
                }
                
                # Add FOB/CIF details for imports
                if ch == 1:
                    sample['has_fob'] = int(imports_fob_mask[t_idx, i, j])
                    sample['has_cif'] = int(imports_cif_mask[t_idx, i, j])
                    sample['fob_value_usd'] = float(imports_fob_sum[t_idx, i, j]) if sample['has_fob'] else 0.0
                    sample['cif_value_usd'] = float(imports_cif_sum[t_idx, i, j]) if sample['has_cif'] else 0.0
                    sample['cif_derived_usd'] = float(imports_cif_sum[t_idx, i, j] / (1 + m)) if sample['has_cif'] else 0.0
                else:
                    sample['has_fob'] = 0
                    sample['has_cif'] = 0
                    sample['fob_value_usd'] = 0.0
                    sample['cif_value_usd'] = 0.0
                    sample['cif_derived_usd'] = 0.0
                
                samples.append(sample)
    
    df = pd.DataFrame(samples)
    
    # Limit to max_samples
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    
    df = df.sort_values(['t', 'ch', 'src_id', 'dst_id']).reset_index(drop=True)
    
    logging.info(f"Generated {len(df)} trace samples")
    
    return df


# ============================================================================
# Long-Form Output
# ============================================================================

def write_long_output(
    output_path: str,
    chunk: pd.DataFrame,
    month_cols: List[str],
    t_map: Dict[str, int],
    iso3_map: Dict[str, int],
    t_min: int
) -> None:
    """
    Write observed long-form rows to partitioned parquet.
    
    This is called during chunk processing to incrementally build
    the long dataset. Rows are partitioned by year.
    
    Columns:
        t, year, month, src_id, dst_id, ch, value_usd, mask, is_estimated,
        indicator_id, reporter_iso3, partner_iso3, scale_id, source_indicator
    """
    # This function would be called from within process_chunk_to_accumulators
    # For now, we'll implement a simpler version that writes all observed rows
    # with a 'source_indicator' column to document FOB vs CIF
    
    # Implementation deferred - would require significant refactoring of chunk processing
    # to buffer long rows. For Step 1 completion, we'll document this as a known limitation.
    raise NotImplementedError(
        "Long-form output (--write-long) is not yet implemented. "
        "This feature requires buffering observed rows during chunk processing. "
        "Exit code 4: write-long requested but not implemented."
    )
