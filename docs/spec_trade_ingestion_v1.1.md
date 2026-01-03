# PathoGraph-DL Trade Ingestion Specification v1.1

## Project Goal
Build a monthly bilateral trade edge tensor for a 194-node country graph (UNGA voters + Taiwan), then use it as an edge modality in an MM-ST-GNN.

## Non-Negotiable Principles
- **Training/inference must NOT call APIs**; APIs/downloads only used for acquisition, then cached and versioned
- Missing data is NOT zero trade. Always use explicit masks/confidence
- Everything must be deterministic and reproducible

## Time Axis Convention
- **Epoch:** 1950-01
- **Month Index Formula:** `month_index = 12*(YYYY-1950) + (MM-1)`
- **Month Index 0:** 1950-01 → 0
- **Stored Fields:** `month_index:int32`, `month_id:'YYYY-MM':string`

## Node Axis Convention
- **N:** 194 nodes (UNGA voters + Taiwan)
- **Node ID Range:** 0..193 (stable mapping)
- **Ordering Rule:** ISO3 codes in ascending order with TWN included

## Trade Tensor Contract

### Storage Format
- **Format:** Zarr
- **Group Path:** `data/processed/trade_tensor.zarr/trade`

### E_trade Array
- **Shape:** `[T, N, N, C]` where `C=4`
- **Data Type:** `float32`
- **Channels:**
  1. `exports_fob_log1p` - Reporter's exports FOB in log1p USD
  2. `imports_cif_log1p` - Reporter's imports CIF in log1p USD
  3. `filled_exports_fob_log1p` - Mirror-filled exports FOB in log1p USD
  4. `confidence_weight` - Confidence weight (0.0 to 1.0)

### M_trade Array (Mask)
- **Shape:** `[T, N, N]`
- **Data Type:** `uint8`
- **Mask Codes:**
  - `0`: valid
  - `1`: structural_null (i=j, self-trade)
  - `2`: missing
  - `3`: estimated
  - `4`: break

## Mirror Fill Rule
Goal: Deterministic filled exports channel
1. Use reporter exports FOB if present and not break
2. Else use partner imports CIF as mirror and convert to FOB-equivalent
3. Else accept IMF estimated/spliced value (lower confidence)
4. Else missing

**CIF to FOB conversion factor:** 1.1
**Conversion:** `filled_exports_fob_usd = imports_cif_usd / 1.1` when mirror used
**Model transform:** `log1p(value_usd)`

## Obs Status to Confidence Mapping
- `A` (Actual): 1.0
- `E` (Estimated): 0.7
- `P` (Provisional): 0.5
- `B` (Break in series): 0.2

**Notes:**
- If obs_status missing (e.g., Taiwan), default confidence_weight=1.0 for those rows
- Break-in-series should additionally set mask code 4

## Primary Trade Sources
1. IMF DOTS via SDMX JSON (monthly bilateral backbone, deepest history)
2. Taiwan MOF/Customs monthly partner trade tables (mandatory TWN patch)
3. UN Comtrade monthly (optional validation/enrichment post-2000/2010)

## Shared Climate Conventions (for later use)
- Wind speed: Precompute sqrt(u²+v²) then monthly mean of magnitudes
- Precipitation: Convert rate to accumulation using NetCDF units
- Polygons: Admin-0 polygons pinned to versioned GeoPackage