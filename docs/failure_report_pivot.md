# Failure Report: granular SDMX API (IMTS/DOTS)

## 1. Executive Summary
We rigorously probed the IMF's SDMX endpoints (`sdmxcentral.imf.org` and `api.imf.org`) to fetch a "Canary" trade data series (US Exports to World/China).
- **Result**: **No real data observations** were retrieved.
- **SDMX Central**: Returned `501 Not Implemented` for all data queries.
- **IMF Data Portal (`api.imf.org`)**: Returned `200 OK` for `IMTS` queries but with **Empty Payloads** (no `Obs` elements). This persists across multiple key variants, time ranges (2023, 2024, All-Time), and header configurations.

## 2. Detailed Findings

### A. Endpoint: `sdmxcentral.imf.org` (REST 1.x)
- **Status**: **Dead for Data**.
- **Evidence**:
  - Structure queries (Dataflow) work.
  - Data queries (e.g., `DOTS/M.US.TXG_FOB_USD.CN`) return `501 Not Implemented`.

### B. Endpoint: `api.imf.org` (SDMX 2.1)
- **Status**: **Alive but Returns Empty Data**.
- **Evidence**:
  - `DOTS`: Returns `404 Not Found` (implies DOTS is deprecated/removed on this endpoint).
  - `IMTS`: Returns `200 OK` for keys like `US.TXG_FOB_USD.W00.M`.
  - **Payload**: XML Header is present, but `DataSet` contains no observations.
  - **Diagnostics**: Tested `2023`, `2024`, and `Last1`. Tested `W00` and `CN`. Tested `TXG_FOB_USD`. All returned empty 200s.
  - **Conclusion**: The keys are syntactically valid (hence no 400/404) but match no published data. This likely indicates a mismatch in **Codelist Values** (e.g., Indicator code is valid in generic dictionary but not used in IMTS data cube) or **Access Restrictions** (data not public via this API).

## 3. Pivot Plan (Step 5 Recommendation)

Since the granular API is effectively unusable for automated ingestion without a valid "Rosetta Stone" of codes (which we cannot discover via the API itself due to opaque DSD linking), we must pivot.

### Recommendation: Bulk Download Ingestion
The IMF provides bulk SDMX/CSV downloads for entire datasets. This is the standard robust path for high-volume ingestion.

**Next Actions:**
1.  **Locate Bulk URL**: Find the stable URL for "IMTS" or "DOTS" full history (usually `downloads.imf.org` or similar).
2.  **Ingest Strategy**:
    - Download the full zip (~500MB+).
    - Parse locally using `pandas` (Parquet conversion).
    - Filter to 194-node universe.
    - Build Tensor.

**Pros**:
- Bypasses 429/501/404 API errors.
- Guarantees full history (2005-Present).
- faster than 194 * 12 * 2 requests.

**Cons**:
- Requires handling large file in memory/disk. (Feasible).
