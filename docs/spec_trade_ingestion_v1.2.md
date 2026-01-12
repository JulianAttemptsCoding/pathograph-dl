# PathoGraph-DL Trade Ingestion Specification v1.2

> [!CAUTION]
> This specification is superseded by **v1.3**. Please refer to `docs/spec_trade_ingestion_v1.3.md` for the latest contract, especially regarding Step 7 Baseline Bundle.


## Project Goal
Build a monthly bilateral trade edge tensor for a 194-node country graph (UNGA voters + Taiwan), then use it as an edge modality in an MM-ST-GNN.

## Changes from v1.1
- Explicitly split definition into "Storage Contract" (Step 1 + Step 2 Zarrs) and "Runtime Contract" (Tensor shapes and masks).
- Clarified mask semantics: **observed iff code != 0** (typically Code 1 = Observed, Code 0 = Missing/Null).
- Added details on supervised targets ($y_{t+H}$).

## User Principles
- **Training/inference must NOT call APIs**; APIs/downloads only used for acquisition, then cached and versioned
- Missing data is NOT zero trade. Always use explicit masks/confidence
- Everything must be deterministic and reproducible

## Time Axis Convention
- **Epoch:** 1950-01
- **Month Index Formula:** `month_index = 12*(YYYY-1950) + (MM-1)`
- **Stored Fields:** `month_index:int32`, `month_id:'YYYY-MM':string`
- **Range:** 1986-01 (t=432) to present (e.g. 2024=t~900)

## Node Axis Convention
- **N:** 194 nodes (UNGA voters + Taiwan)
- **Node ID Range:** 0..193 (stable mapping)
- **Ordering Rule:** ISO3 codes in ascending order with TWN included

## Storage Contract (Zarr)

### Step 1: Base Trade (IMTS)
- **Path:** `data/processed/trade/imf_imts_step1/trade_fob_tensor.zarr`
- **Keys:**
  - `trade`: `[T, N, N, 2]` (float32). Channels: `[Exports, Imports]`.
  - `mask`: `[T, N, N, 2]` (uint8).
    - **Code 1 (and non-zero):** Valid/Observed.
    - **Code 0:** Missing/Null/Noise.
  - `is_estimated`: `[T, N, N, 2]` (uint8). 1 if mirror-filled or estimated.
  - `time_index`: `[T]` (int32).

### Step 2: Risk Trade (FAOSTAT)
- **Path:** `data/processed/trade/faostat_step2/trade_risk_tensor.zarr`
- **Keys:**
  - `trade_risk`: `[T, N, N, K, 2]` (float32). K=8 Commodity Groups.
  - `observed_mask`: `[T, N, N, K, 2]` (uint8).
    - **Code 1:** Observed.
    - **Code 0:** Missing.
  - `is_estimated`: `[T, N, N, K, 2]` (uint8).
  - `time_index`: `[T]` (int32).

## Runtime Contract (Dataset/DataModule)

### Input Features ($x_t$)
- **Base Trade:** `[B, L, N, N, 2]`. Log1p + Standardized.
- **Base Mask:** `[B, L, N, N, 2]`. **Binary: 1=Observed, 0=Missing.**
  - *Note:* Passes through non-zero codes from storage as 1.
- **Risk Trade:** `[B, L, N, N, K, 2]`. Log1p + Standardized.
- **Risk Mask:** `[B, L, N, N, K, 2]`. **Binary: 1=Observed, 0=Missing.**

### Timestamps
- `t`: Current time index (end of lookback).
- `t_y`: Target time index ($t + H$).
- `time_feat`: `[B, 2]`. Sin/Cos of month for $t_y$.

### Targets ($y_{t+H}$)
Optional, controlled by `return_targets=True`.

- **Base Target:**
  - `y_base`: `[B, N, N, 2]`.
  - `y_base_mask`: `[B, N, N, 2]`. (1=Observed).
- **Risk Target:**
  - `y_risk`: `[B, N, N, K, 2]`.
  - `y_risk_mask`: `[B, N, N, K, 2]`. (1=Observed).

### Loss Calculation
Loss should be masked MSE:
$$ L = \frac{\sum ((\hat{y} - y)^2 \odot M)}{\sum M + \epsilon} $$
Where $M$ is the binary target mask.
