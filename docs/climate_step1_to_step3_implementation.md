# Climate Step 1–3 Implementation Report

**Generated:** 2026-01-15T21:10:00-08:00
**Environment:** `conda run -n pathograph-pre` Python 3.11.14

## Summary
Complete Climate preprocessing pipeline implemented:
1. **Step 1**: ERA5 monthly NetCDF download with SHA256 manifest
2. **Step 2**: Polygon aggregation using `exactextract` with deterministic expver selection and unit validation
3. **Step 3**: Tensorization aligned exactly to `time_index_master.npy`

## Prerequisites (Must Complete Once)

### 1. Accept CDS Licences (Required)
The ERA5 dataset requires accepting licences on the CDS website.
- **Action:** Log in and accept terms at the URL below.
- **URL:** [https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=download#manage-licences](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=download#manage-licences)

### 2. Install pyarrow (Required for Parquet)
Parquet support via `pyarrow` is mandatory.
- **Command:** `conda install -n pathograph-pre -c conda-forge pyarrow -y`

## Files Created

| File | Description |
|------|-------------|
| `config/climate_step1.yaml` | Locked spec with expver policy, unit rules, feature order |
| `tools/climate_step1_download_era5.py` | Year-batched CDS download with idempotency |
| `tools/climate_step2_aggregate_country_month.py` | Polygon aggregation with exactextract |
| `tools/climate_step3_tensorize.py` | Zarr v3 tensorization aligned to master axes |
| `tests/test_climate_config_schema.py` | Config schema validation |
| `tests/test_climate_tensor_contract.py` | Tensor contract validation |

## CLI Commands

### Step 1: Download ERA5 NetCDF
```powershell
# Download single year
conda run -n pathograph-pre python tools/climate_step1_download_era5.py --config config/climate_step1.yaml --years 2024:2024

# Download all years (1950-2025)
conda run -n pathograph-pre python tools/climate_step1_download_era5.py --config config/climate_step1.yaml --years all
```

### Step 2: Aggregate to Country-Month Parquet
```powershell
conda run -n pathograph-pre python tools/climate_step2_aggregate_country_month.py --config config/climate_step1.yaml --years 2024:2024
```

### Step 3: Tensorize to Zarr
```powershell
conda run -n pathograph-pre python tools/climate_step3_tensorize.py --config config/climate_step1.yaml
```

### Run Tests
```powershell
conda run -n pathograph-pre python -m pytest -q tests/test_climate_config_schema.py
conda run -n pathograph-pre python -m pytest -q tests/test_climate_tensor_contract.py
```

## Expected File Tree After Full Pipeline

```
data/
├── raw/
│   └── era5_monthly_netcdf/
│       ├── era5_sl_monthly_1950.nc
│       ├── era5_sl_monthly_1951.nc
│       └── ... (76 files)
└── processed/
    └── climate/
        ├── country_month/
        │   └── source=ERA5/
        │       ├── year=1950/country_month_1950.parquet
        │       ├── year=1951/country_month_1951.parquet
        │       └── ... (76 partitions)
        ├── climate_tensor.zarr/
        │   ├── climate/
        │   ├── mask/
        │   ├── time_index/
        │   └── feature_names/
        └── manifests/
            ├── era5_download_manifest.json
            ├── era5_country_month_manifest_1950.json
            └── climate_step3_tensor_manifest.json
```

## Output Schemas

### Parquet (17 columns)
| Column | Type |
|--------|------|
| node_id | int |
| iso3 | str |
| year | int |
| month | int |
| month_index | int |
| t2m_mean_c | float |
| d2m_mean_c | float |
| sp_mean_pa | float |
| msl_mean_pa | float |
| u10_mean | float |
| v10_mean | float |
| wind10_speed_mean | float |
| rh_mean | float |
| vpd_mean_kpa | float |
| tp_mean_mm_month | float |
| is_missing_any | bool |
| qc_flags | str |

### Zarr Arrays (4 arrays)
| Array | Shape | Dtype | Chunks |
|-------|-------|-------|--------|
| climate | (908, 194, 10) | float32 | (24, 194, 10) |
| mask | (908, 194, 10) | uint8 | (24, 194, 10) |
| time_index | (908,) | int32 | (908,) |
| feature_names | (10,) | U32 | (10,) |

## Locked Rules Implemented

### expver Selection
- If `expver` dimension exists: select `expver=1` if present, else select max numeric expver
- STOP if expver is non-numeric and ambiguous

### Unit Validation
- Temperature: Accept `K`/`kelvin`/`Kelvin` → convert to Celsius
- Pressure: Accept `Pa`/`pascal`/`pascals` → keep as Pa
- Precipitation: Accept `m` (depth) → mm×1000; or `*s-1` (rate) → mm = rate × seconds_in_month × 1000
- STOP on any other units

### Centroid Fallback
- Only when ALL base variables are NaN for a polygon
- Flag with `qc_flags=FALLBACK_CENTROID`

## Verification Results
| Test | Status |
|------|--------|
| `test_climate_config_schema` | ✅ PASSED |
| `test_climate_tensor_contract` | ✅ PASSED (skipped: no output yet) |
| Step 1 CLI `--help` | ✅ OK |
| Step 2 CLI `--help` | ✅ OK |
| Step 3 CLI `--help` | ✅ OK |

## Next Steps
1. Run smoke test with single year: `--years 2024:2024`
2. **Verify Parquet output:**
   ```powershell
   conda run -n pathograph-pre python -c "import pandas as pd; df=pd.read_parquet('data/processed/climate/country_month/source=ERA5/year=2024/country_month_2024.parquet'); need={'node_id','iso3','year','month','month_index'}; missing=sorted(list(need - set(map(str, df.columns)))); print('ROWS', len(df)); print('MISSING', missing); assert not missing; exp=194*12; print('EXPECT', exp); assert len(df)==exp"
   ```
3. After smoke test passes, run full pipeline: `--years all`
4. After Step 3 completes, re-run tensor contract test to validate output
