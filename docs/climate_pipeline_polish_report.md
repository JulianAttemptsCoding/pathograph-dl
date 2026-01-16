# Climate Pipeline Polish Report

**Date:** 2026-01-15
**Status:** ✅ Polished & Verified

## Summary
The Climate Step 1-3 pipeline has been hardened to eliminate recurring blockers using zero-guessing logic.

### 1. Pyarrow Prerequisite
- **Action:** Installed `pyarrow` into `pathograph-pre` environment.
- **Verification:** `tests/test_climate_prereqs_imports.py` passed.

### 2. CDS Licence Gate
- **Tool Created:** `tools/check_climate_prereqs.py`
  - Probes CDS with a tiny request (2024-01, 1 var).
  - Deterministically catches HTTP 403 "required licences not accepted".
  - Prints explicit actionable instructions.
- **Verification:** Script run captured correct failure mode:
  ```
  CDS_LICENCE_NOT_ACCEPTED
  Action Required: Log in to CDS and accept the required licences.
  URL: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=download#manage-licences
  ```

### 3. Downloader Patch
- **File:** `tools/climate_step1_download_era5.py`
- **Change:** Wrapped `c.retrieve()` to catch specific 403 errors and exit gracefully with instructions (exit code 2) instead of crashing with a traceback.

### 4. Documentation & Tests
- Updated `docs/climate_step1_to_step3_implementation.md` with explicit Prerequisites section.
- Added regression tests:
  - `tests/test_climate_prereqs_imports.py` (Pass)
  - `tests/test_climate_step1_licence_detection.py` (Pass)

## Required User Action
The pipeline is code-complete and verified. The **ONLY** remaining blocker is external:
**You must accept the ERA5 dataset licences.**

**Link:** [https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=download#manage-licences](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=download#manage-licences)

Once accepted, run the smoke test:
```powershell
conda run -n pathograph-pre python tools/climate_step1_download_era5.py --config config/climate_step1.yaml --years 2024:2024
```
