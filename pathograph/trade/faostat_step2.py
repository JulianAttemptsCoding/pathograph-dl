"""
FAOSTAT Step 2: Annual Bilateral Trade -> Risk-Weighted Monthly Pseudo-Flows

This module implements the complete pipeline for integrating FAOSTAT annual detailed
bilateral trade data with Step 1 IMTS monthly base flows to generate risk-weighted
monthly pseudo-flow tensors.

Key Features:
- Chunked FAOSTAT ingestion to avoid memory issues
- Template generator for groups mapping (--emit-groups-template)
- Corridor-year commodity/risk-group shares W[y,i,j,k]
- 1-year lag enforcement to prevent leakage
- Optional backoff policy for missing weights (default: OFF)
- Comprehensive QC and traceability
"""

import hashlib
import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set
import numpy as np
import pandas as pd
import zarr

_M49_DIGITS_RE = re.compile(r"(\d+)")
_YEAR_COL_RE = re.compile(r"^Y\d{4}$")
_YEAR4 = re.compile(r"(\d{4})")


YEAR_EPOCH = 1950


def _norm_colname(c: object) -> str:
    return str(c).strip().lstrip('\ufeff')


def _is_year_col(c: object) -> bool:
    return bool(re.fullmatch(r'Y\d{4}', _norm_colname(c)))


def parse_year_label(col: str) -> int:
    """Robustly parse 4-digit year from string (e.g. 'Y1986', '\ufeffY1986')."""
    s = _norm_colname(col)
    m = _YEAR4.search(s)
    if not m:
        raise ValueError(f"Could not parse 4-digit year from column: {col!r}")
    y = int(m.group(1))
    if y < 1900 or y > 2100:
        raise ValueError(f"Parsed implausible year {y} from column: {col!r}")
    return y


def _clean_m49_series(s):
    """Return 3-digit M49 strings (e.g., '004'). Robust to quotes/mixed types."""
    x = s.astype(str).str.replace("'", "", regex=False)
    x = x.str.extract(_M49_DIGITS_RE, expand=False)
    x = x.fillna("")
    return x.str.zfill(3)


def scan_faostat_items_present_streaming(
    faostat_path: str,
    iso3_set: set,
    m49_to_iso3_csv: str,
    chunksize: int = 50000,
    zip_member: str | None = None,
    logger: logging.Logger | None = None,
) :
    """Stream-scan FAOSTAT (wide format) to produce per-item totals without melt.

    Output columns:
      item_code (int)
      item_name (str)
      total_value (float)   # sum over all year columns for retained rows
      row_count (int)

    Filters (matching intended Step2 template semantics):
      - Element contains 'export' AND 'value'
      - Unit contains 'usd' OR 'us$'
      - Map Reporter/Partner M49 -> ISO3 using m49_to_iso3_csv
      - Filter to iso3_set (canonical 194)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Load crosswalk
    cw = __import__('pandas').read_csv(m49_to_iso3_csv)
    if 'm49' not in cw.columns or 'iso3' not in cw.columns:
        raise ValueError(f"Crosswalk missing required columns m49, iso3: {m49_to_iso3_csv}")
    cw_m49 = cw['m49'].astype(str).str.replace("'", "", regex=False)
    cw_m49 = cw_m49.str.extract(_M49_DIGITS_RE, expand=False).fillna("").str.zfill(3)
    cw_iso3 = cw['iso3'].astype(str).str.upper().str.strip()
    m49_to_iso3 = dict(zip(cw_m49, cw_iso3))

    # Open file / zip member
    if faostat_path.lower().endswith('.zip'):
        zf = zipfile.ZipFile(faostat_path)
        members = set(zf.namelist())
        if zip_member is None:
            # Prefer NOFLAG if present
            cand1 = [m for m in members if m.endswith('All_Data_NOFLAG.csv')]
            cand2 = [m for m in members if m.endswith('All_Data.csv')]
            if cand1:
                zip_member = sorted(cand1)[0]
            elif cand2:
                zip_member = sorted(cand2)[0]
            else:
                raise ValueError(f"No All_Data*.csv member found in zip: {faostat_path}")
        if zip_member not in members:
            raise ValueError(f"zip_member not found in zip: {zip_member}")
        fh_for_header = zf.open(zip_member)
    else:
        zf = None
        fh_for_header = open(faostat_path, 'rb')

    import pandas as pd

    # Read header
    header_df = pd.read_csv(fh_for_header, nrows=0)
    cols = list(header_df.columns)

    # Required columns (exact names from FAOSTAT TM bulk)
    reporter_m49_col = 'Reporter Country Code (M49)'
    partner_m49_col = 'Partner Country Code (M49)'
    item_code_col = 'Item Code'
    item_name_col = 'Item'
    element_col = 'Element'
    unit_col = 'Unit'

    missing = [c for c in [reporter_m49_col, partner_m49_col, item_code_col, element_col, unit_col] if c not in cols]
    if missing:
        raise ValueError(f"FAOSTAT main file missing required columns: {missing}. Found columns={cols[:30]}...")

    year_cols = []
    for c in cols:
        try:
            parse_year_label(c)
            year_cols.append(c)
        except ValueError:
            continue

    if not year_cols:
        raise ValueError("No year columns like Y1986 found; not wide-format TM export")
    year_cols = sorted(year_cols)

    usecols = [reporter_m49_col, partner_m49_col, item_code_col]
    if item_name_col in cols:
        usecols.append(item_name_col)
    usecols += [element_col, unit_col] + year_cols

    # dtype=str to avoid DtypeWarning; numeric coercion happens per chunk
    dtype_map = {c: 'string' for c in usecols}

    def _iter_chunks():
        if zf is not None:
            fh = zf.open(zip_member)
            yield from pd.read_csv(fh, usecols=usecols, dtype=dtype_map, chunksize=chunksize, low_memory=True)
        else:
            yield from pd.read_csv(faostat_path, usecols=usecols, dtype=dtype_map, chunksize=chunksize, low_memory=True)

    totals = {}  # item_code -> {total_value,row_count,item_name}
    chunk_idx = 0

    for chunk in _iter_chunks():
        chunk_idx += 1

        # Filter: export value
        el = chunk[element_col].astype(str).str.lower()
        chunk = chunk[el.str.contains('export', na=False) & el.str.contains('value', na=False)]
        if len(chunk) == 0:
            continue

        # Filter: USD unit
        un = chunk[unit_col].astype(str).str.lower()
        chunk = chunk[un.str.contains('usd', na=False) | un.str.contains('us$', na=False)]
        if len(chunk) == 0:
            continue

        # Map M49 -> ISO3
        rm49 = _clean_m49_series(chunk[reporter_m49_col])
        pm49 = _clean_m49_series(chunk[partner_m49_col])
        r_iso3 = rm49.map(m49_to_iso3)
        p_iso3 = pm49.map(m49_to_iso3)

        ok = r_iso3.notna() & p_iso3.notna()
        chunk = chunk[ok]
        r_iso3 = r_iso3[ok]
        p_iso3 = p_iso3[ok]
        if len(chunk) == 0:
            continue

        in_univ = r_iso3.isin(iso3_set) & p_iso3.isin(iso3_set)
        chunk = chunk[in_univ]
        if len(chunk) == 0:
            continue

        # Sum numeric values across year columns without melt
        block = chunk[year_cols].apply(lambda col: pd.to_numeric(col, errors='coerce'))
        row_sum = block.sum(axis=1, skipna=True)

        item_code = pd.to_numeric(chunk[item_code_col], errors='coerce')
        if item_name_col in chunk.columns:
            item_name = chunk[item_name_col].astype(str)
        else:
            item_name = pd.Series([''] * len(chunk), index=chunk.index)

        tmp = pd.DataFrame({'item_code': item_code, 'row_sum': row_sum, 'item_name': item_name})
        tmp = tmp.dropna(subset=['item_code'])
        if len(tmp) == 0:
            continue

        gsum = tmp.groupby('item_code')['row_sum'].sum()
        gcount = tmp.groupby('item_code').size()
        gname = tmp.groupby('item_code')['item_name'].agg(lambda s: next((x for x in s if x and x != 'nan'), ''))

        for k in gsum.index:
            ik = int(k)
            d = totals.get(ik)
            if d is None:
                totals[ik] = {
                    'total_value': float(gsum.loc[k]),
                    'row_count': int(gcount.loc[k]),
                    'item_name': str(gname.loc[k])
                }
            else:
                d['total_value'] += float(gsum.loc[k])
                d['row_count'] += int(gcount.loc[k])
                if (not d.get('item_name')) and str(gname.loc[k]):
                    d['item_name'] = str(gname.loc[k])

        if chunk_idx % 50 == 0:
            logger.info(f"scan_faostat_items_present_streaming: processed {chunk_idx} chunks; items_so_far={len(totals)}")

    out = pd.DataFrame([{'item_code': k, **v} for k, v in totals.items()])
    if len(out) == 0:
        raise ValueError("No valid FAOSTAT rows found after filters + mapping + universe filter")

    out = out.sort_values('total_value', ascending=False).reset_index(drop=True)

    if zf is not None:
        zf.close()

    return out



# ============================================================================
# Constants
# ============================================================================

EPOCH_YEAR = 1950
EPOCH_MONTH = 1

_M49_DIGITS = re.compile(r"\d+")

def clean_m49(x) -> str | None:
    """
    Convert FAOSTAT M49 strings like \"'004\" or \"004\" to \"004\".
    Returns None if not parseable.
    """
    if pd.isna(x):
        return None
    s = str(x).strip().replace("'", "")
    m = _M49_DIGITS.search(s)
    if not m:
        return None
    return m.group(0).zfill(3)
# ============================================================================
# Preflight: Step 1 Validation
# ============================================================================

def load_step1_manifest(path: str) -> Dict:
    """
    Load and validate Step 1 preprocessing manifest.
    
    Returns:
        Manifest dict
    
    Raises:
        ValueError: if manifest is invalid or missing required fields
    """
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise ValueError(f"Step 1 manifest not found: {path}")
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Validate required fields
    required_fields = ['outputs', 'time_axis', 'inputs']
    missing = [f for f in required_fields if f not in manifest]
    if missing:
        raise ValueError(f"Step 1 manifest missing required fields: {missing}")
    
    # Validate outputs section
    if 'zarr_path' not in manifest['outputs']:
        raise ValueError("Step 1 manifest missing outputs.zarr_path")
    
    if 'tensor_shape' not in manifest['outputs']:
        raise ValueError("Step 1 manifest missing outputs.tensor_shape")
    
    # Validate time axis
    if 't_min' not in manifest['time_axis'] or 't_max' not in manifest['time_axis']:
        raise ValueError("Step 1 manifest missing time_axis.t_min or t_max")
    
    # Validate node index
    if 'node_index' not in manifest['inputs']:
        raise ValueError("Step 1 manifest missing inputs.node_index")
    
    logging.info(f"Loaded Step 1 manifest from {path}")
    logging.info(f"  Tensor shape: {manifest['outputs']['tensor_shape']}")
    logging.info(f"  Time range: t={manifest['time_axis']['t_min']}..{manifest['time_axis']['t_max']}")
    
    return manifest


def validate_step1_artifacts(manifest: Dict, allow_noncanonical: bool = False) -> None:
    """
    Validate that Step 1 artifacts exist and have expected shapes.
    
    Args:
        manifest: Step 1 manifest dict
        allow_noncanonical: If True, allow N != 194 (TEST/DEV ONLY). Default: False (strict).
    
    Raises:
        ValueError: if artifacts are missing or invalid
    """
    zarr_path = manifest['outputs']['zarr_path']
    
    if not Path(zarr_path).exists():
        raise ValueError(f"Step 1 Zarr tensor not found: {zarr_path}")
    
    # Open Zarr and validate shape
    try:
        store = zarr.DirectoryStore(zarr_path)
    except AttributeError:
        from zarr.storage import LocalStore
        store = LocalStore(zarr_path)
    
    root = zarr.open_group(store=store, mode='r')
    
    if 'trade' not in root:
        raise ValueError(f"Step 1 Zarr missing 'trade' array")
    
    trade = root['trade']
    expected_shape = tuple(manifest['outputs']['tensor_shape'])
    
    if trade.shape != expected_shape:
        raise ValueError(f"Step 1 trade tensor shape mismatch: expected {expected_shape}, got {trade.shape}")
    
    # Validate it's (T, N, N, 2)
    if len(trade.shape) != 4 or trade.shape[1] != trade.shape[2] or trade.shape[3] != 2:
        raise ValueError(f"Step 1 trade tensor has invalid shape: {trade.shape} (expected (T, N, N, 2))")
    
    N = trade.shape[1]
    
    # Strict mode: require N == 194
    if not allow_noncanonical:
        if N != 194:
            raise ValueError(f"Step 1 trade tensor must have N=194 nodes; got N={N}. Use --allow-noncanonical-n for test/dev mode.")
    else:
        # Noncanonical mode: require N >= 2
        if N < 2:
            raise ValueError(f"Step 1 trade tensor has too few nodes: N={N} (expected N >= 2)")
        if N != 194:
            logging.warning(f"NON-CANONICAL NODE UNIVERSE ENABLED (TEST/DEV ONLY): N={N} (canonical is 194)")
    
    logging.info(f"Validated Step 1 Zarr tensor: {zarr_path}")
    logging.info(f"  Shape: {trade.shape}, dtype: {trade.dtype}")


def load_node_index(path: str, allow_noncanonical: bool = False) -> Tuple[Dict[str, int], Dict[int, str], Set[str], int]:
    """
    Load node_index.csv and validate node universe.
    
    Args:
        path: Path to node_index.csv
        allow_noncanonical: If True, allow N != 194 (TEST/DEV ONLY). Default: False (strict).
    
    Returns:
        (iso3_to_id, id_to_iso3, iso3_set, N)
    
    Raises:
        ValueError: if validation fails
    """
    df = pd.read_csv(path)
    
    # Validate required columns
    if not {'iso3', 'node_id'}.issubset(df.columns):
        raise ValueError(f"node_index must have columns 'iso3' and 'node_id'; got {df.columns.tolist()}")
    
    N = len(df)
    
    # Strict mode: require N == 194
    if not allow_noncanonical:
        if N != 194:
            raise ValueError(f"node_index must have exactly 194 rows; got {N}. Use --allow-noncanonical-n for test/dev mode.")
    else:
        # Noncanonical mode: require N >= 2
        if N < 2:
            raise ValueError(f"node_index must have at least 2 nodes; got {N}")
        if N != 194:
            logging.warning(f"NON-CANONICAL NODE UNIVERSE ENABLED (TEST/DEV ONLY): N={N} (canonical is 194)")
    
    # Validate node_id range
    node_ids = sorted(df['node_id'].unique())
    if node_ids != list(range(N)):
        raise ValueError(f"node_id must be 0..{N-1}; got range {min(node_ids)}..{max(node_ids)}")
    
    # Build mappings
    iso3_to_id = dict(zip(df['iso3'], df['node_id']))
    id_to_iso3 = dict(zip(df['node_id'], df['iso3']))
    iso3_set = set(df['iso3'])
    
    logging.info(f"Loaded node index: {N} nodes, node_id 0..{N-1}")
    logging.info(f"Sample nodes: {list(iso3_to_id.keys())[:5]}")
    
    return iso3_to_id, id_to_iso3, iso3_set, N


# ============================================================================
# FAOSTAT File Detection and Schema Validation
# ============================================================================

def detect_faostat_file(path: str) -> Tuple[str, Optional[str]]:
    """
    Detect FAOSTAT file format and return path to CSV.
    
    Args:
        path: Path to .zip or .csv file
    
    Returns:
        (csv_path, zip_member_name)
        - If input is CSV: (path, None)
        - If input is ZIP: (path_to_zip, member_name)
    
    Raises:
        ValueError: if file not found or invalid format
    """
    file_path = Path(path)
    
    if not file_path.exists():
        raise ValueError(f"FAOSTAT file not found: {path}")
    
    if file_path.suffix.lower() == '.csv':
        logging.info(f"FAOSTAT file is CSV: {path}")
        return str(file_path), None
    
    elif file_path.suffix.lower() == '.zip':
        # Open zip and find largest CSV
        with zipfile.ZipFile(file_path, 'r') as zf:
            csv_members = [m for m in zf.namelist() if m.lower().endswith('.csv')]
            
            if not csv_members:
                raise ValueError(f"No CSV files found in ZIP: {path}")
            
            # Select NOFLAG if present, else default
            members_set = set(zf.namelist())
            if 'Trade_DetailedTradeMatrix_E_All_Data_NOFLAG.csv' in members_set:
                selected_member = 'Trade_DetailedTradeMatrix_E_All_Data_NOFLAG.csv'
            elif 'Trade_DetailedTradeMatrix_E_All_Data.csv' in members_set:
                selected_member = 'Trade_DetailedTradeMatrix_E_All_Data.csv'
            else:
                # Fallback to largest CSV
                member_sizes = [(m, zf.getinfo(m).file_size) for m in csv_members]
                member_sizes.sort(key=lambda x: x[1], reverse=True)
                selected_member = member_sizes[0][0]
            
            selected_size = zf.getinfo(selected_member).file_size
            logging.info(f"FAOSTAT file is ZIP: {path}")
            logging.info(f"  Selected CSV member: {selected_member} ({selected_size / (1024**2):.1f} MB)")
            
            return str(file_path), selected_member
    
    else:
        raise ValueError(f"FAOSTAT file must be .csv or .zip; got {file_path.suffix}")


def validate_faostat_schema(df_header: pd.DataFrame) -> Dict[str, Any]:
    cols = df_header.columns.tolist()
    cols_lower = [str(c).lower() for c in cols]

    # ------------------------------------------------------------
    # FAOSTAT TM bulk "wide" format detector (Y1986, Y1987, ...)
    # ------------------------------------------------------------
    year_cols = sorted([c for c in cols if _is_year_col(c)])
    
    # Flag columns usually match Y\d{4}F
    flag_cols = sorted([c for c in cols if re.match(r"^Y\d{4}F$", _norm_colname(c))])

    if year_cols:
        # This matches your ZIP columns exactly
        required = [
            "Reporter Country Code",
            "Partner Country Code",
            "Item Code",
            "Item",
            "Element",
            "Unit",
        ]
        missing_req = [c for c in required if c not in cols]
        if missing_req:
            raise ValueError(
                f"FAOSTAT wide schema missing required columns: {missing_req}. "
                f"Available columns: {cols}"
            )

        return {
            "format": "wide",
            "reporter_col": "Reporter Country Code",
            "partner_col": "Partner Country Code",
            "item_code_col": "Item Code",
            "item_name_col": "Item",
            "element_col": "Element",
            "unit_col": "Unit",
            "reporter_m49_col": "Reporter Country Code (M49)",
            "partner_m49_col": "Partner Country Code (M49)",
            "year_cols": [str(c) for c in year_cols],
            "flag_cols": [str(c) for c in flag_cols],
        }
    # ------------------------------------------------------------
    # Continue with your existing long-format logic below
    # ------------------------------------------------------------

    # Build mapping
    mapping = {}
    
    # Reporter (country, reporter, reporter country, etc.)
    reporter_candidates = ['reporter', 'reporter country', 'country', 'reporter countries']
    for cand in reporter_candidates:
        if cand in cols_lower:
            mapping['reporter'] = cols[cols_lower.index(cand)]
            break
    
    # Partner (partner, partner country, etc.)
    partner_candidates = ['partner', 'partner country', 'partner countries']
    for cand in partner_candidates:
        if cand in cols_lower:
            mapping['partner'] = cols[cols_lower.index(cand)]
            break
    
    # Year
    year_candidates = ['year']
    for cand in year_candidates:
        if cand in cols_lower:
            mapping['year'] = cols[cols_lower.index(cand)]
            break
    
    # Item Code
    item_candidates = ['item code', 'itemcode', 'item']
    for cand in item_candidates:
        if cand in cols_lower:
            mapping['item_code'] = cols[cols_lower.index(cand)]
            break
    
    # Value
    value_candidates = ['value']
    for cand in value_candidates:
        if cand in cols_lower:
            mapping['value'] = cols[cols_lower.index(cand)]
            break
    
    # Element/Flow (optional)
    element_candidates = ['element', 'flow']
    for cand in element_candidates:
        if cand in cols_lower:
            mapping['element'] = cols[cols_lower.index(cand)]
            break
    
    # Validate required fields
    required = ['reporter', 'partner', 'year', 'item_code', 'value']
    missing = [f for f in required if f not in mapping]
    
    if missing:
        raise ValueError(f"FAOSTAT schema missing required columns: {missing}. Available columns: {cols}")
    
    logging.info(f"FAOSTAT schema validated:")
    for canonical, actual in mapping.items():
        logging.info(f"  {canonical} -> {actual}")
    
    return mapping


# ============================================================================
# FAOSTAT Ingestion (Chunked)
# ============================================================================

def ingest_faostat_chunked(
    file_path: str,
    zip_member: Optional[str],
    schema_mapping: Dict[str, str],
    iso3_set: Set[str],
    chunksize: int = 100000
) -> pd.DataFrame:
    """
    Ingest FAOSTAT data in chunks, filter to 194-node universe, and return aggregated DataFrame.
    
    This function reads the FAOSTAT file in chunks to avoid loading the entire dataset into memory.
    It filters rows to only include those where both reporter and partner are in the canonical
    194 ISO3 node universe.
    
    Args:
        file_path: Path to FAOSTAT file (.csv or .zip)
        zip_member: If .zip, the member name to extract
        schema_mapping: Column name mapping from validate_faostat_schema
        iso3_set: Set of valid ISO3 codes (194 nodes)
        chunksize: Number of rows per chunk
    
    Returns:
        DataFrame with columns: year, reporter_iso3, partner_iso3, item_code, value
        Aggregated by (year, reporter_iso3, partner_iso3, item_code)
    """
    stats = {
        'total_rows': 0,
        'filtered_rows': 0,
        'dropped_non_iso3': 0,
        'dropped_not_in_universe': 0,
        'year_min': None,
        'year_max': None
    }
    
    # Determine how to read the file
    if zip_member:
        # Read from ZIP
        import io
        with zipfile.ZipFile(file_path, 'r') as zf:
            with zf.open(zip_member) as csv_file:
                # Read header first to get column names
                header_df = pd.read_csv(io.TextIOWrapper(csv_file, encoding='utf-8'), nrows=0)
        
        # Now read in chunks
        def chunk_reader():
            with zipfile.ZipFile(file_path, 'r') as zf:
                with zf.open(zip_member) as csv_file:
                    yield from pd.read_csv(
                        io.TextIOWrapper(csv_file, encoding='utf-8'),
                        chunksize=chunksize
                    )
    else:
        # Read from CSV directly
        def chunk_reader():
            yield from pd.read_csv(file_path, chunksize=chunksize)
    
    # Accumulate filtered data
    accumulated_chunks = []
    
    # Load M49 -> ISO3 crosswalk (REQUIRED for FAOSTAT)
    crosswalk_path = Path("config/m49_to_iso3.csv")
    if not crosswalk_path.exists():
        raise FileNotFoundError(
            f"M49-to-ISO3 crosswalk not found: {crosswalk_path}. "
            "This file is REQUIRED to map FAOSTAT country codes to ISO3. "
            "See docs/reports/faostat_step2_readiness_report.md for instructions."
        )
    cw = pd.read_csv(crosswalk_path, dtype={"m49": str, "iso3": str})
    cw["m49"] = cw["m49"].astype(str).str.replace("'", "", regex=False).str.zfill(3)
    cw["iso3"] = cw["iso3"].astype(str).str.upper().str.strip()
    m49_to_iso3 = dict(zip(cw["m49"], cw["iso3"]))
    logging.info(f"Loaded M49-to-ISO3 crosswalk: {len(m49_to_iso3)} mappings")
    
    for chunk in chunk_reader():
    
        stats['total_rows'] += len(chunk)

    
        # Canonicalize FAOSTAT chunk to long format columns: reporter, partner, year, item_code, value
    
        # Supports both 'long' (year/value columns) and TM bulk 'wide' (Y1986.. columns).
    
        if schema_mapping.get('format') == 'wide':
    
                        # Accept either key style:
            # - wide schema detector may return reporter_col/partner_col/item_code_col
            # - or may return reporter/partner/item_code
            reporter_col = schema_mapping.get("reporter") or schema_mapping.get("reporter_col")
            partner_col  = schema_mapping.get("partner")  or schema_mapping.get("partner_col")
            item_code_col = schema_mapping.get("item_code") or schema_mapping.get("item_code_col")

            item_name_col = (
                schema_mapping.get("item")
                or schema_mapping.get("item_name_col")
                or "Item"
            )
            element_col = (
                schema_mapping.get("element")
                or schema_mapping.get("element_col")
                or "Element"
            )
            unit_col = (
                schema_mapping.get("unit")
                or schema_mapping.get("unit_col")
                or "Unit"
            )

            year_cols = schema_mapping.get("year_cols", [])

            if not (reporter_col and partner_col and item_code_col and year_cols):
                raise ValueError(f"Wide FAOSTAT schema mapping incomplete: {schema_mapping}")

    
            id_cols = [reporter_col, partner_col, item_code_col]
            
            # Add M49 columns to id_cols so they're preserved during melt
            reporter_m49_col = schema_mapping.get("reporter_m49_col")
            partner_m49_col = schema_mapping.get("partner_m49_col")
            if reporter_m49_col and reporter_m49_col in chunk.columns:
                id_cols.append(reporter_m49_col)
            if partner_m49_col and partner_m49_col in chunk.columns:
                id_cols.append(partner_m49_col)
    
            for c in (item_name_col, element_col, unit_col):
    
                if c in chunk.columns and c not in id_cols:
    
                    id_cols.append(c)

    
            # Use _is_year_col to perform strict year column selection locally
            year_cols_raw = [c for c in chunk.columns if _is_year_col(c)]
            if not year_cols_raw:
                # Fallback: if somehow chunk has no matching columns, maybe they were lost or renamed earlier?
                # But we expect them here. raise if empty.
                if year_cols:
                   # Try to use what we knew from schema but verify
                   year_cols_raw = [c for c in year_cols if c in chunk.columns]
                
                if not year_cols_raw:
                     # Show normalized cols for debug
                     norm_cols = [_norm_colname(c) for c in chunk.columns]
                     raise RuntimeError(f"No year columns found in chunk. First 30 normalized: {norm_cols[:30]}")

            year_map = {c: parse_year_label(_norm_colname(c)) for c in year_cols_raw}
            assert len(set(year_map.values())) == len(year_map), 'duplicate year targets after rename (possible flag columns or bad year selection)'
            
            wide = chunk[id_cols + year_cols_raw]
            wide2 = wide.rename(columns=year_map)

            long = wide2.melt(
                id_vars=id_cols,
                value_vars=sorted(set(year_map.values())),  # integers now
                var_name="year",
                value_name="value",
            )

            # Year is already integer from year_map.values()
            long['year'] = pd.to_numeric(long['year'], errors='coerce')
            long = long.dropna(subset=['year'])
            long['year'] = long['year'].astype(int)

            long['value'] = pd.to_numeric(long['value'], errors='coerce')
    
            long = long.dropna(subset=['value'])

    
            rename_map = {
    
                reporter_col: 'reporter',
    
                partner_col: 'partner',
    
                item_code_col: 'item_code',
    
            }
    
            if item_name_col in long.columns:
    
                rename_map[item_name_col] = 'item'
    
            if element_col in long.columns:
    
                rename_map[element_col] = 'element'
    
            if unit_col in long.columns:
    
                rename_map[unit_col] = 'unit'

    
            chunk = long.rename(columns=rename_map)

    
        else:
    
            # Existing long-format path: rename detected columns to canonical names
    
            rename_map = {
    
                schema_mapping['reporter']: 'reporter',
    
                schema_mapping['partner']: 'partner',
    
                schema_mapping['year']: 'year',
    
                schema_mapping['item_code']: 'item_code',
    
                schema_mapping['value']: 'value',
    
            }
    
            # Optional fields if present in long schema
    
            if 'element' in schema_mapping:
    
                rename_map[schema_mapping['element']] = 'element'
    
            if 'unit' in schema_mapping:
    
                rename_map[schema_mapping['unit']] = 'unit'
    
            if 'item' in schema_mapping:
    
                rename_map[schema_mapping['item']] = 'item'
    
            chunk = chunk.rename(columns=rename_map)

        # --- FAOSTAT numeric/M49 -> ISO3 mapping (REQUIRED) ---
        # Do this BEFORE any filtering, using original column names from schema_mapping
        
        reporter_m49_col = schema_mapping.get("reporter_m49_col")
        partner_m49_col = schema_mapping.get("partner_m49_col")
        if reporter_m49_col and partner_m49_col:
            # M49 columns exist (wide format), use them
            chunk["reporter_m49"] = chunk[reporter_m49_col].map(clean_m49)
            chunk["partner_m49"] = chunk[partner_m49_col].map(clean_m49)
            
            chunk["reporter_iso3"] = chunk["reporter_m49"].map(m49_to_iso3)
            chunk["partner_iso3"] = chunk["partner_m49"].map(m49_to_iso3)
            
            # Drop rows we cannot map deterministically
            unmapped = chunk["reporter_iso3"].isna() | chunk["partner_iso3"].isna()
            stats['dropped_unmapped_m49'] = stats.get('dropped_unmapped_m49', 0) + unmapped.sum()
            chunk = chunk.dropna(subset=["reporter_iso3", "partner_iso3"])
        else:
            # Fallback: assume reporter/partner are already ISO3 (long format)
            chunk['reporter_iso3'] = chunk['reporter'].astype(str).str.upper().str.strip()
            chunk['partner_iso3'] = chunk['partner'].astype(str).str.upper().str.strip()

    
        # Filter by element/flow if available (prefer exports)
    
        if 'element' in chunk.columns:
    
            export_mask = chunk['element'].astype(str).str.lower().str.contains('export', na=False)
    
            chunk = chunk[export_mask]
    
        # Enforce canonical 194-node universe
        in_universe = chunk["reporter_iso3"].isin(iso3_set) & chunk["partner_iso3"].isin(iso3_set)
        dropped_not_in_universe = len(chunk) - in_universe.sum()
        stats['dropped_not_in_universe'] += dropped_not_in_universe
        
        chunk = chunk[in_universe]
        
        if len(chunk) == 0:
            continue
        
        # Keep only needed columns
        chunk = chunk[['year', 'reporter_iso3', 'partner_iso3', 'item_code', 'value']]
        
        # Convert types
        chunk['year'] = pd.to_numeric(chunk['year'], errors='coerce')
        chunk['item_code'] = pd.to_numeric(chunk['item_code'], errors='coerce')
        chunk['value'] = pd.to_numeric(chunk['value'], errors='coerce')
        
        # Drop rows with invalid data
        chunk = chunk.dropna(subset=['year', 'item_code', 'value'])
        chunk['year'] = chunk['year'].astype(int)
        chunk['item_code'] = chunk['item_code'].astype(int)
        
        stats['filtered_rows'] += len(chunk)
        
        # Update year range
        if len(chunk) > 0:
            chunk_year_min = chunk['year'].min()
            chunk_year_max = chunk['year'].max()
            
            if stats['year_min'] is None or chunk_year_min < stats['year_min']:
                stats['year_min'] = chunk_year_min
            if stats['year_max'] is None or chunk_year_max > stats['year_max']:
                stats['year_max'] = chunk_year_max
        
        accumulated_chunks.append(chunk)
    
    # Concatenate all chunks
    if not accumulated_chunks:
        raise ValueError("No valid FAOSTAT rows found after filtering to 194-node universe")
    
    df = pd.concat(accumulated_chunks, ignore_index=True)
    
    # Aggregate by (year, reporter_iso3, partner_iso3, item_code)
    df = df.groupby(['year', 'reporter_iso3', 'partner_iso3', 'item_code'], as_index=False)['value'].sum()
    
    logging.info(f"FAOSTAT ingestion complete:")
    logging.info(f"  Total rows read: {stats['total_rows']:,}")
    logging.info(f"  Dropped (non-ISO3): {stats['dropped_non_iso3']:,}")
    logging.info(f"  Dropped (not in 194 universe): {stats['dropped_not_in_universe']:,}")
    logging.info(f"  Filtered rows kept: {stats['filtered_rows']:,}")
    logging.info(f"  Aggregated rows: {len(df):,}")
    logging.info(f"  Year range: {stats['year_min']}..{stats['year_max']}")
    logging.info(f"  Distinct items: {df['item_code'].nunique()}")
    
    return df


# ============================================================================
# Template Generator
# ============================================================================

def generate_groups_template(
    df: pd.DataFrame,
    output_dir: str
) -> Tuple[str, str]:
    """
    Generate template files for FAOSTAT groups mapping.
    
    Creates two files:
    1. faostat_items_present.csv: All observed item codes with totals
    2. faostat_groups.template.csv: Template for user to fill in group mappings
    
    Args:
        df: FAOSTAT DataFrame with 'item_code' and 'value' columns
        output_dir: Directory to write template files
    
    Returns:
        (items_path, template_path)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Compute item statistics
    item_stats = df.groupby('item_code').agg({
        'value': 'sum',
        'year': 'count'  # Use year as proxy for row count
    }).reset_index()
    
    item_stats.columns = ['item_code', 'total_value', 'row_count']
    item_stats = item_stats.sort_values('total_value', ascending=False)
    
    # Write items present
    items_path = output_path / 'faostat_items_present.csv'
    item_stats.to_csv(items_path, index=False)
    
    logging.info(f"Wrote FAOSTAT items present: {items_path}")
    logging.info(f"  {len(item_stats)} distinct item codes")
    
    # Generate template
    template_df = pd.DataFrame({
        'item_code': item_stats['item_code'],
        'group_id': 'OTHER',  # Default to OTHER
        'group_name': ''
    })
    
    template_path = output_path / 'faostat_groups.template.csv'
    template_df.to_csv(template_path, index=False)
    
    logging.info(f"Wrote FAOSTAT groups template: {template_path}")
    logging.info(f"  User should edit this file to assign group_id and group_name")
    
    return str(items_path), str(template_path)


# ============================================================================
# Group Mapping
# ============================================================================

def load_group_mapping(path: str) -> Tuple[Dict[int, str], Dict[str, str], int, List[str]]:
    """
    Load FAOSTAT groups mapping from CSV.
    
    Expected columns: item_code, group_id, group_name (optional)
    
    Returns:
        (item_to_group, group_names, K, group_order)
        - item_to_group: dict mapping item_code (int) to group_id (str)
        - group_names: dict mapping group_id to group_name
        - K: number of groups
        - group_order: sorted list of group_ids
    
    Raises:
        ValueError: if file invalid or missing required columns
    """
    df = pd.read_csv(path)
    
    if 'item_code' not in df.columns or 'group_id' not in df.columns:
        raise ValueError(f"Groups mapping must have 'item_code' and 'group_id' columns; got {df.columns.tolist()}")
    
    # Build item -> group mapping
    item_to_group = dict(zip(df['item_code'].astype(int), df['group_id'].astype(str)))
    
    # Build group names
    if 'group_name' in df.columns:
        group_names = dict(zip(df['group_id'].astype(str), df['group_name'].astype(str)))
    else:
        group_names = {gid: gid for gid in df['group_id'].unique()}
    
    # Determine group order (sorted)
    group_order = sorted(df['group_id'].unique())
    K = len(group_order)
    
    logging.info(f"Loaded FAOSTAT groups mapping from {path}")
    logging.info(f"  {len(item_to_group)} items mapped to {K} groups")
    logging.info(f"  Groups: {group_order}")
    
    return item_to_group, group_names, K, group_order


def apply_group_mapping(
    df: pd.DataFrame,
    item_to_group: Dict[int, str],
    group_order: List[str]
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply group mapping to FAOSTAT DataFrame.
    
    Args:
        df: FAOSTAT DataFrame with 'item_code' column
        item_to_group: Mapping from item_code to group_id
        group_order: Sorted list of group_ids
    
    Returns:
        (df_mapped, stats)
        - df_mapped: DataFrame with 'group_id' column added
        - stats: Dict with mapping statistics
    """
    stats = {
        'total_rows': len(df),
        'unmapped_items': 0,
        'unmapped_value': 0.0,
        'mapped_rows': 0
    }
    
    # Map items to groups
    df = df.copy()
    df['group_id'] = df['item_code'].map(item_to_group)
    
    # Identify unmapped rows
    unmapped_mask = df['group_id'].isna()
    stats['unmapped_items'] = df[unmapped_mask]['item_code'].nunique()
    stats['unmapped_value'] = df[unmapped_mask]['value'].sum()
    
    # Drop unmapped rows
    df = df[~unmapped_mask]
    stats['mapped_rows'] = len(df)
    
    if stats['unmapped_items'] > 0:
        logging.warning(f"Dropped {stats['unmapped_items']} unmapped items ({stats['unmapped_value']:,.0f} total value)")
    
    logging.info(f"Applied group mapping: {stats['mapped_rows']:,} rows kept")
    
    return df, stats


# ============================================================================
# Weight Computation
# ============================================================================

def compute_corridor_year_weights(
    df: pd.DataFrame,
    iso3_to_id: Dict[str, int],
    group_order: List[str],
    N: int,
    K: int
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Compute corridor-year commodity/risk-group shares W[y,i,j,k].
    
    Args:
        df: FAOSTAT DataFrame with columns: year, reporter_iso3, partner_iso3, group_id, value
        iso3_to_id: Mapping from ISO3 to node_id
        group_order: Sorted list of group_ids
        N: Number of nodes (194)
        K: Number of groups
    
    Returns:
        (W, weight_mask, stats)
        - W: (Y, N, N, K) array of weights
        - weight_mask: (Y, N, N) boolean array indicating where weights are available
        - stats: Dict with computation statistics
    """
    # Determine year range
    year_min = df['year'].min()
    year_max = df['year'].max()
    Y = int(year_max - YEAR_EPOCH + 1)
    
    logging.info(f"Computing corridor-year weights:")
    logging.info(f"  FAOSTAT calendar years: {year_min}..{year_max}")
    logging.info(f"  Weights year index (1950-based): {year_min - YEAR_EPOCH}..{year_max - YEAR_EPOCH} (Y={Y})")
    logging.info(f"  Nodes: N={N}, Groups: K={K}")
    
    # Initialize arrays
    V = np.zeros((Y, N, N, K), dtype=np.float64)  # Volumes
    W = np.zeros((Y, N, N, K), dtype=np.float32)  # Weights
    weight_mask = np.zeros((Y, N, N), dtype=np.uint8)  # Has weights
    
    # Build group_id -> k mapping
    group_to_k = {gid: k for k, gid in enumerate(group_order)}
    
    # Aggregate volumes V[y,i,j,k]
    logging.info("  Aggregating values (vectorized)...")
    
    y_vals = df['year'].astype(int).values - YEAR_EPOCH
    i_vals = df['reporter_iso3'].map(iso3_to_id).fillna(-1).astype(np.int32).values
    j_vals = df['partner_iso3'].map(iso3_to_id).fillna(-1).astype(np.int32).values
    k_vals = df['group_id'].map(group_to_k).fillna(-1).astype(np.int32).values
    v_vals = df['value'].astype(np.float64).values
    
    valid_mask = (y_vals >= 0) & (y_vals < Y) & (i_vals >= 0) & (j_vals >= 0) & (k_vals >= 0)
    
    if not valid_mask.all():
         logging.warning(f"  Dropped {len(df) - valid_mask.sum()} rows during aggregation")
         
    np.add.at(V, (y_vals[valid_mask], i_vals[valid_mask], j_vals[valid_mask], k_vals[valid_mask]), v_vals[valid_mask])
    
    # Compute corridor-year denominators D[y,i,j] = sum_k V[y,i,j,k]
    D = V.sum(axis=3)  # (Y, N, N)
    
    # Compute weights W[y,i,j,k] = V[y,i,j,k] / D[y,i,j]
    # Only where D > 0
    positive_denom = D > 0
    
    with np.errstate(divide='ignore', invalid='ignore'):
         W[:] = np.divide(V, D[..., None], out=np.zeros_like(W), where=D[..., None] > 0)
    
    weight_mask[:] = (D > 0).astype(np.uint8)
    
    # Compute statistics
    total_corridors_years = Y * N * N
    corridors_with_weights = weight_mask.sum()
    coverage = float(corridors_with_weights) / total_corridors_years
    
    stats = {
        'year_min': int(year_min),
        'year_max': int(year_max),
        'Y': int(Y),
        'total_corridors_years': int(total_corridors_years),
        'corridors_with_weights': int(corridors_with_weights),
        'coverage': float(coverage)
    }
    
    logging.info(f"Computed corridor-year weights:")
    logging.info(f"  Corridors with weights: {corridors_with_weights:,} / {total_corridors_years:,} ({coverage:.2%})")
    
    # Validate normalization
    for y in range(Y):
        for i in range(N):
            for j in range(N):
                if weight_mask[y, i, j]:
                    weight_sum = W[y, i, j, :].sum()
                    if not np.isclose(weight_sum, 1.0, atol=1e-5):
                        logging.warning(f"Weight normalization issue at y={y}, i={i}, j={j}: sum={weight_sum}")
    
    return W, weight_mask, stats


def apply_backoff_policy(
    W: np.ndarray,
    weight_mask: np.ndarray,
    V: np.ndarray,
    N: int,
    K: int
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Apply backoff policy to fill missing corridor-year weights.
    
    Hierarchy:
    1. Corridor-year (direct)
    2. Exporter-year (aggregate over all partners)
    3. World-year (aggregate over all corridors)
    4. Missing
    
    Args:
        W: (Y, N, N, K) weights array (will be modified in-place)
        weight_mask: (Y, N, N) mask indicating where weights are available
        V: (Y, N, N, K) volumes array
        N: Number of nodes
        K: Number of groups
    
    Returns:
        (W_filled, backoff_code, stats)
        - W_filled: (Y, N, N, K) weights with backoff applied
        - backoff_code: (Y, N, N) array with backoff level used
        - stats: Dict with backoff statistics
    """
    Y = W.shape[0]
    
    # Initialize backoff code
    # 0 = direct, 1 = exporter-year, 2 = world-year, 3 = missing
    backoff_code = np.full((Y, N, N), 3, dtype=np.uint8)
    
    # Mark direct weights as backoff_code=0
    backoff_code[weight_mask == 1] = 0
    
    # Compute exporter-year shares: aggregate over all partners
    exporter_year_V = V.sum(axis=2)  # (Y, N, K)
    exporter_year_D = exporter_year_V.sum(axis=2)  # (Y, N)
    exporter_year_W = np.zeros((Y, N, K), dtype=np.float32)
    
    for y in range(Y):
        for i in range(N):
            if exporter_year_D[y, i] > 0:
                exporter_year_W[y, i, :] = exporter_year_V[y, i, :] / exporter_year_D[y, i]
    
    # Compute world-year shares: aggregate over all corridors
    world_year_V = V.sum(axis=(1, 2))  # (Y, K)
    world_year_D = world_year_V.sum(axis=1)  # (Y,)
    world_year_W = np.zeros((Y, K), dtype=np.float32)
    
    for y in range(Y):
        if world_year_D[y] > 0:
            world_year_W[y, :] = world_year_V[y, :] / world_year_D[y]
    
    # Apply backoff
    stats = {
        'direct': int((backoff_code == 0).sum()),
        'exporter_year': 0,
        'world_year': 0,
        'missing': 0
    }
    
    for y in range(Y):
        for i in range(N):
            for j in range(N):
                if weight_mask[y, i, j] == 0:
                    # Try exporter-year
                    if exporter_year_D[y, i] > 0:
                        W[y, i, j, :] = exporter_year_W[y, i, :]
                        backoff_code[y, i, j] = 1
                        weight_mask[y, i, j] = 1
                        stats['exporter_year'] += 1
                    # Try world-year
                    elif world_year_D[y] > 0:
                        W[y, i, j, :] = world_year_W[y, :]
                        backoff_code[y, i, j] = 2
                        weight_mask[y, i, j] = 1
                        stats['world_year'] += 1
                    else:
                        # Remains missing
                        stats['missing'] += 1
    
    logging.info(f"Applied backoff policy:")
    logging.info(f"  Direct: {stats['direct']:,}")
    logging.info(f"  Exporter-year: {stats['exporter_year']:,}")
    logging.info(f"  World-year: {stats['world_year']:,}")
    logging.info(f"  Missing: {stats['missing']:,}")
    
    return W, backoff_code, stats


# ============================================================================
# Lag Application and Pseudo-Flow Generation
# ============================================================================

def build_month_to_year_mapping(time_index: np.ndarray, t_min: int) -> np.ndarray:
    """
    Build mapping from month index t to calendar year Y.
    
    Args:
        time_index: Array of month indices (t_min..t_max)
        t_min: Minimum month index
    
    Returns:
        Array of years corresponding to each month index
    """
    # Convert month index to year
    # t = (year - 1950) * 12 + (month - 1)
    # year = 1950 + t // 12
    years = EPOCH_YEAR + time_index // 12
    
    return years.astype(np.int32)


def apply_lag_and_generate_pseudoflows(
    base_tensor: np.ndarray,
    base_mask: np.ndarray,
    base_is_estimated: np.ndarray,
    W: np.ndarray,
    weight_mask: np.ndarray,
    backoff_code: Optional[np.ndarray],
    time_index: np.ndarray,
    t_min: int,
    weight_year_min: int,
    lag: int,
    N: int,
    K: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Dict]:
    """
    Apply lag and generate risk-weighted monthly pseudo-flows.
    
    For each month t in year Y, use weights from year (Y - lag).
    Compute E[t,i,j,k,ch] = F_base[t,i,j,ch] * W[Y-lag,i,j,k]
    
    Args:
        base_tensor: (T, N, N, 2) base trade tensor from Step 1
        base_mask: (T, N, N, 2) observed mask from Step 1
        base_is_estimated: (T, N, N, 2) is_estimated flag from Step 1
        W: (Y_w, N, N, K) weights array
        weight_mask: (Y_w, N, N) weight availability mask
        backoff_code: (Y_w, N, N) backoff code (or None if no backoff)
        time_index: (T,) array of month indices
        t_min: Minimum month index
        weight_year_min: First year in weights array
        lag: Lag in years (default: 1)
        N: Number of nodes
        K: Number of groups
    
    Returns:
        (E, observed_risk, is_estimated_risk, backoff_risk, stats)
        - E: (T, N, N, K, 2) risk tensor
        - observed_risk: (T, N, N, K, 2) observed mask
        - is_estimated_risk: (T, N, N, K, 2) is_estimated flag
        - backoff_risk: (T, N, N, K, 2) backoff code (or None)
        - stats: Dict with generation statistics
    """
    T = base_tensor.shape[0]
    
    logging.info(f"Generating risk-weighted pseudo-flows:")
    logging.info(f"  Base tensor shape: {base_tensor.shape}")
    logging.info(f"  Weights shape: {W.shape}")
    logging.info(f"  Lag: {lag} year(s)")
    
    # Build month -> year mapping
    month_years = build_month_to_year_mapping(time_index, t_min)
    
    # Initialize risk tensor
    E = np.zeros((T, N, N, K, 2), dtype=np.float32)
    observed_risk = np.zeros((T, N, N, K, 2), dtype=np.uint8)
    is_estimated_risk = np.zeros((T, N, N, K, 2), dtype=np.uint8)
    
    if backoff_code is not None:
        backoff_risk = np.zeros((T, N, N, K, 2), dtype=np.uint8)
    else:
        backoff_risk = None
    
    stats = {
        'months_processed': 0,
        'months_with_weights': 0,
        'months_without_weights': 0,
        'cells_generated': 0
    }
    
    # Process each month
    for t in range(T):
        stats["months_processed"] += 1
        calendar_year = YEAR_EPOCH + (t // 12)
        weight_year = calendar_year - lag
        y_w = weight_year - YEAR_EPOCH
        
        # Check if weight index is in range
        if y_w < 0 or y_w >= W.shape[0]:
            # Weight year out of range
            stats['months_without_weights'] += 1
            continue
        
        stats['months_with_weights'] += 1
        
        # Apply weights
        for i in range(N):
            for j in range(N):
                if weight_mask[y_w, i, j]:
                    for ch in range(2):
                        if base_mask[t, i, j, ch]:
                            # E[t,i,j,k,ch] = F[t,i,j,ch] * W[y_w,i,j,k]
                            E[t, i, j, :, ch] = base_tensor[t, i, j, ch] * W[y_w, i, j, :]
                            observed_risk[t, i, j, :, ch] = 1
                            is_estimated_risk[t, i, j, :, ch] = base_is_estimated[t, i, j, ch]
                            
                            if backoff_code is not None:
                                backoff_risk[t, i, j, :, ch] = backoff_code[y_w, i, j]
                            
                            stats['cells_generated'] += K
        
    logging.info(f"Risk tensor generation complete:")
    logging.info(f"  Months processed: {stats['months_processed']}")
    logging.info(f"  Months with weights: {stats['months_with_weights']}")
    logging.info(f"  Months without weights: {stats['months_without_weights']}")
    logging.info(f"  Risk cells generated: {stats['cells_generated']:,}")
    
    return E, observed_risk, is_estimated_risk, backoff_risk, stats


# ============================================================================
# Zarr Output
# ============================================================================

def write_weights_zarr(
    W: np.ndarray,
    weight_mask: np.ndarray,
    backoff_code: Optional[np.ndarray],
    year_min: int,
    group_order: List[str],
    path: str
) -> None:
    """
    Write corridor-year weights to Zarr.
    
    Args:
        W: (Y, N, N, K) weights array
        weight_mask: (Y, N, N) weight availability mask
        backoff_code: (Y, N, N) backoff code (or None)
        year_min: First year in weights array
        group_order: List of group IDs
        path: Output Zarr path
    """
    try:
        store = zarr.DirectoryStore(path)
    except AttributeError:
        from zarr.storage import LocalStore
        store = LocalStore(path)
    
    root = zarr.open_group(store=store, mode='w')
    
    # Set attributes
    root.attrs['year_min'] = int(year_min)
    root.attrs['year_max'] = int(year_min + W.shape[0] - 1)
    root.attrs['Y'] = int(W.shape[0])
    root.attrs['N'] = int(W.shape[1])
    root.attrs['K'] = int(W.shape[3])
    root.attrs['groups'] = group_order
    
    # Chunking
    chunks_W = (1, 64, 64, W.shape[3])
    chunks_mask = (1, 64, 64)
    
    def _create(name, data, chunks):
        data = np.asarray(data)
        try:
            arr = root.create_dataset(name, shape=data.shape, dtype=data.dtype, chunks=chunks)
        except Exception:
            arr = root.create_array(name, shape=data.shape, dtype=data.dtype, chunks=chunks)
        arr[...] = data
    
    _create('weights', W, chunks_W)
    _create('weight_mask', weight_mask, chunks_mask)
    
    if backoff_code is not None:
        _create('backoff_code', backoff_code, chunks_mask)
    
    logging.info(f"Wrote weights Zarr: {path}")


def write_risk_tensor_zarr(
    E: np.ndarray,
    observed_risk: np.ndarray,
    is_estimated_risk: np.ndarray,
    backoff_risk: Optional[np.ndarray],
    time_index: np.ndarray,
    group_order: List[str],
    path: str
) -> None:
    """
    Write risk tensor to Zarr.
    
    Args:
        E: (T, N, N, K, 2) risk tensor
        observed_risk: (T, N, N, K, 2) observed mask
        is_estimated_risk: (T, N, N, K, 2) is_estimated flag
        backoff_risk: (T, N, N, K, 2) backoff code (or None)
        time_index: (T,) array of month indices
        group_order: List of group IDs
        path: Output Zarr path
    """
    try:
        store = zarr.DirectoryStore(path)
    except AttributeError:
        from zarr.storage import LocalStore
        store = LocalStore(path)
    
    root = zarr.open_group(store=store, mode='w')
    
    # Set attributes
    root.attrs['T'] = int(E.shape[0])
    root.attrs['N'] = int(E.shape[1])
    root.attrs['K'] = int(E.shape[3])
    root.attrs['channels'] = ['exports_fob_usd', 'imports_fob_best_usd']
    root.attrs['groups'] = group_order
    
    # Chunking: (time=12, i=64, j=64, k=K, ch=2)
    chunks = (min(12, E.shape[0]), 64, 64, E.shape[3], 2)
    chunks_1d = (min(1024, E.shape[0]),)
    
    def _create(name, data, chunks):
        data = np.asarray(data)
        try:
            arr = root.create_dataset(name, shape=data.shape, dtype=data.dtype, chunks=chunks)
        except Exception:
            arr = root.create_array(name, shape=data.shape, dtype=data.dtype, chunks=chunks)
        arr[...] = data
    
    _create('trade_risk', E, chunks)
    _create('observed_mask', observed_risk, chunks)
    _create('is_estimated', is_estimated_risk, chunks)
    
    if backoff_risk is not None:
        _create('backoff_code', backoff_risk, chunks)
    
    _create('time_index', time_index, chunks_1d)
    
    logging.info(f"Wrote risk tensor Zarr: {path}")


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
# QC Metrics
# ============================================================================

def compute_qc_metrics(
    E: np.ndarray,
    observed_risk: np.ndarray,
    base_tensor: np.ndarray,
    base_mask: np.ndarray,
    W: np.ndarray,
    weight_mask: np.ndarray,
    backoff_code: Optional[np.ndarray],
    group_order: List[str],
    id_to_iso3: Dict[int, str]
) -> Dict:
    """
    Compute comprehensive QC metrics for Step 2.
    
    Returns dict with:
        - time_coverage: Fraction of base months with weights available
        - base_to_risk_coverage: Fraction of base observed cells that also have weights
        - top_corridors: Top corridors by base flow with weight availability
        - group_distributions: Per-group global share distribution
        - backoff_distribution: Distribution of backoff codes (if applicable)
    """
    T, N, _, K, C = E.shape
    
    # Time coverage: fraction of months with any weights
    month_has_weights = np.zeros(T, dtype=bool)
    for t in range(T):
        if np.any(observed_risk[t, :, :, :, :]):
            month_has_weights[t] = True
    
    time_coverage = float(month_has_weights.sum()) / T
    
    # Base to risk coverage: fraction of base observed cells with weights
    base_observed_cells = base_mask.sum()
    
    # For each base observed cell, check if corresponding risk cells have weights
    base_with_risk = 0
    for t in range(T):
        for i in range(N):
            for j in range(N):
                for ch in range(C):
                    if base_mask[t, i, j, ch]:
                        if np.any(observed_risk[t, i, j, :, ch]):
                            base_with_risk += 1
    
    base_to_risk_coverage = float(base_with_risk) / base_observed_cells if base_observed_cells > 0 else 0.0
    
    # Top corridors by base flow
    base_total = base_tensor.sum(axis=(0, 3))  # (N, N)
    top_idx = np.argsort(base_total.flatten())[-20:][::-1]
    
    top_corridors = []
    for idx in top_idx:
        i = idx // N
        j = idx % N
        if i == j:
            continue
        
        total_base = float(base_total[i, j])
        if total_base == 0:
            continue
        
        # Check weight availability across all years
        has_weights_any_year = weight_mask[:, i, j].any()
        
        top_corridors.append({
            'src_iso3': id_to_iso3[i],
            'dst_iso3': id_to_iso3[j],
            'total_base_usd': total_base,
            'has_weights': bool(has_weights_any_year)
        })
    
    # Group distributions (weighted by corridor-year denominator)
    # Compute global share for each group across all corridor-years with weights
    group_totals = np.zeros(K, dtype=np.float64)
    
    for y in range(W.shape[0]):
        for i in range(N):
            for j in range(N):
                if weight_mask[y, i, j]:
                    # Add weighted contribution
                    group_totals += W[y, i, j, :]
    
    total_weight_sum = group_totals.sum()
    group_shares = (group_totals / total_weight_sum) if total_weight_sum > 0 else np.zeros(K)
    
    group_distributions = [
        {'group_id': group_order[k], 'global_share': float(group_shares[k])}
        for k in range(K)
    ]
    
    # Backoff distribution
    backoff_distribution = None
    if backoff_code is not None:
        backoff_counts = {
            'direct': int((backoff_code == 0).sum()),
            'exporter_year': int((backoff_code == 1).sum()),
            'world_year': int((backoff_code == 2).sum()),
            'missing': int((backoff_code == 3).sum())
        }
        total_corridors_years = backoff_code.size
        backoff_distribution = {
            'counts': backoff_counts,
            'fractions': {k: v / total_corridors_years for k, v in backoff_counts.items()}
        }
    
    metrics = {
        'time_coverage': time_coverage,
        'base_to_risk_coverage': base_to_risk_coverage,
        'base_observed_cells': int(base_observed_cells),
        'base_with_risk_cells': int(base_with_risk),
        'top_corridors': top_corridors,
        'group_distributions': group_distributions,
        'backoff_distribution': backoff_distribution
    }
    
    logging.info(f"QC Metrics:")
    logging.info(f"  Time coverage: {time_coverage:.2%}")
    logging.info(f"  Base->Risk coverage: {base_to_risk_coverage:.2%}")
    logging.info(f"  Base observed cells: {base_observed_cells:,}")
    logging.info(f"  Base cells with risk weights: {base_with_risk:,}")
    
    return metrics


# ============================================================================
# Trace Samples
# ============================================================================

def generate_trace_samples(
    E: np.ndarray,
    observed_risk: np.ndarray,
    base_tensor: np.ndarray,
    base_mask: np.ndarray,
    W: np.ndarray,
    weight_mask: np.ndarray,
    backoff_code: Optional[np.ndarray],
    time_index: np.ndarray,
    month_years: np.ndarray,
    weight_year_min: int,
    lag: int,
    group_order: List[str],
    id_to_iso3: Dict[int, str],
    max_samples: int = 50
) -> List[Dict]:
    """
    Generate trace samples for audit trail.
    
    Each sample links:
    - Base flow F[t,i,j,ch]
    - Selected weight year (Y-lag)
    - Weights vector W[Y-lag,i,j,:]
    - Resulting risk flows E[t,i,j,:,ch]
    
    Returns:
        List of dicts with trace information
    """
    T, N, _, K, C = E.shape
    
    samples = []
    
    # Sample observed risk cells
    observed_indices = []
    for t in range(T):
        for i in range(N):
            for j in range(N):
                for ch in range(C):
                    if observed_risk[t, i, j, 0, ch]:  # Check first group (all groups have same mask)
                        observed_indices.append((t, i, j, ch))
    
    if not observed_indices:
        logging.warning("No observed risk cells found for trace sampling")
        return []
    
    # Sample uniformly
    n_samples = min(len(observed_indices), max_samples)
    sampled_indices = np.random.choice(len(observed_indices), n_samples, replace=False)
    
    for idx in sampled_indices:
        t, i, j, ch = observed_indices[idx]
        
        year_t = int(month_years[t])
        month_t = int((time_index[t] % 12) + 1)
        weight_year = year_t - lag
        y_w = weight_year - weight_year_min
        
        sample = {
            't': int(time_index[t]),
            'year': year_t,
            'month': month_t,
            'exporter_iso3': id_to_iso3[i],
            'importer_iso3': id_to_iso3[j],
            'channel': int(ch),
            'base_flow_usd': float(base_tensor[t, i, j, ch]),
            'selected_weight_year': weight_year,
            'weights': {group_order[k]: float(W[y_w, i, j, k]) for k in range(K)},
            'risk_flows': {group_order[k]: float(E[t, i, j, k, ch]) for k in range(K)},
            'has_weight': bool(weight_mask[y_w, i, j])
        }
        
        if backoff_code is not None:
            sample['backoff_code'] = int(backoff_code[y_w, i, j])
        
        samples.append(sample)
    
    logging.info(f"Generated {len(samples)} trace samples")
    
    return samples

