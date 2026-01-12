# PathoGraph-DL Trade Ingestion Specification v1.3
## Overview
This document specifies the data ingestion and baseline training pipeline for the Trade component of PathoGraph-DL.
It supersedes v1.2 by adding the **Step 7 Baseline Bundle** contract.

## Project Goal
Build a monthly bilateral trade edge tensor for a 194-node country graph (UNGA voters + Taiwan), then use it as an edge modality in an MM-ST-GNN.

## Changes from v1.2
- **Added Step 7:** Definition of the reproducible baseline bundle, including orchestration, verification, and metrics contracts.
- **Clarified Smoke Test Strategy:** Step 7 smoke tests use `limit_*_batches` + `max_epochs=1` (not `fast_dev_run`) to ensure checklist correctness (checkpoint creation).

## User Principles
- **Training/inference must NOT call APIs**; APIs/downloads only used for acquisition, then cached and versioned
- **Missing data is NOT zero trade.** Always use explicit masks/confidence
- **Everything must be deterministic and reproducible** via fixed seeds and versioned artifacts

## Time Axis Convention
- **Epoch:** 1950-01
- **Month Index Formula:** `month_index = 12*(YYYY-1950) + (MM-1)`
- **Stored Fields:** `month_index:int32`, `month_id:'YYYY-MM':string`
- **Range:** 1986-01 (t=432) to present (e.g. 2024=t~900)

## Node Axis Convention
- **N:** 194 nodes (UNGA voters + Taiwan)
- **Node ID Range:** 0..193 (stable mapping)
- **Ordering Rule:** ISO3 codes in ascending order with TWN included

---

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

---

## Runtime Contract (Dataset/DataModule)

### Input Features ($x_t$)
- **Base Trade:** `[B, L, N, N, 2]`. Log1p + Standardized.
- **Base Mask:** `[B, L, N, N, 2]`. **Binary: 1=Observed, 0=Missing.**
- **Risk Trade:** `[B, L, N, N, K, 2]`. Log1p + Standardized.
- **Risk Mask:** `[B, L, N, N, K, 2]`. **Binary: 1=Observed, 0=Missing.**

### Timestamps
- `t`: Current time index (end of lookback).
- `t_y`: Target time index ($t + H$).
- `time_feat`: `[B, 2]`. Sin/Cos of month for $t_y$.

### Targets ($y_{t+H}$)
Optional, controlled by `return_targets=True`.
- `y_base`: `[B, N, N, 2]`.
- `y_base_mask`: `[B, N, N, 2]`. (1=Observed).
- `y_risk`: `[B, N, N, K, 2]`.
- `y_risk_mask`: `[B, N, N, K, 2]`. (1=Observed).

### Loss Calculation
Loss constitutes masked MSE:
$$ L = \frac{\sum ((\hat{y} - y)^2 \odot M)}{\sum M + \epsilon} $$

---

## Step 7: Baseline Bundle (Reproducible Trade-Only Baseline)
This step defines the official baseline bundle for trade prediction. It wraps Step 6 mechanics with production hardening (seeds, logging, verification).

### Run Directory Structure
A Step 7 run directory acts as the unit of work/reproducibility:
```
runs/trade_baseline_v1/
├── git_commit.txt                # Git hash, branch, dirty status
├── environment.json              # Python/Torch/CUDA versions
├── artifact_verification.json    # Input artifact SHA256s and schema checks
├── config_resolved.yaml          # Effective config used
├── checkpoints/
│   ├── best.ckpt                 # Best model by val_loss
│   └── last.ckpt                 # Last epoch state
├── metrics_breakdown.json        # Full breakdown of metrics
├── metrics_val.json              # Validation set summary
└── metrics_test.json             # Test set summary
```

### Configuration Defaults
Production settings in `config/trade_step7.yaml`:
- **Seed:** 1337 (Fixed)
- **Batch Size:** 4
- **Max Epochs:** 100
- **Splits:** Train [0, 815], Val [816, 851], Test [852, 907]
- **Standardization:** `data/processed/trade/trade_step3_scaler.json`

### Commands
| Action | Command | Notes |
| :--- | :--- | :--- |
| **Full Run** | `python tools/trade_step7_run_baseline.py` | Runs verify -> train -> export |
| **Smoke Run** | `python tools/trade_step7_run_baseline.py --smoke --run-dir runs/_test_smoke` | Uses `max_epochs=1`, `limit_*_batches=2` |
| **Verify Only** | `python tools/trade_step7_verify_artifacts.py --output check.json` | Checks artifacts without running training |
| **Metrics Only** | `python tools/trade_step7_export_metrics.py --run-dir runs/...` | Re-computes metrics from checkpoint |

### Metrics Schema
Metrics are exported with detailed breakdowns to account for data quality issues (mirror statistics).
- **Channels:** Exports (0) vs Imports (1)
- **Quality Split:** Imported values are split by `is_estimated` flag from Step 1.
  - `imports_estimated`: Original data was missing/null, filled via mirror exports.
  - `imports_observed`: Original data was reported by the importer.

**JSON Schema:**
```json
{
  "total_loss": float,
  "base": {
    "exports": { "mse": float, "mae": float, "count": int },
    "imports": { "mse": float, "mae": float, "count": int },
    "imports_estimated": { "mse": float, "mae": float, "count": int },
    "imports_observed": { "mse": float, "mae": float, "count": int }
  },
  "risk": { "aggregate": { "mse": float, ... } }
}
```
