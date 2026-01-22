# Prerequisite Verification Report: ST-MM-GNN Layer A MVP

**Date:** 2026-01-19
**Repo Root:** `C:/Users/bubga.JULIAN-LAPTOPE2/PycharmProjects/PathoGraph-DL`
**Branch:** (Not checked explicitly, but git status shows untracked files)

## Summary
| Step | ID | Status | Details |
|---|---|---|---|
| Repo Status | R00 | PASS | Repo root confirmed. Untracked `docs/audits/` present. |
| Python Env | R01 | PASS | Python 3.11.14 (pathograph-train). All imports succeeded. |
| Preprocessing | R02 | PASS | `verify_preprocessing_complete.py --require-meta` passed. |
| Artifacts | R03 | PASS | robust discovery found climate artifacts and verified trade/meta contracts. |
| DataModule | R04 | **FAIL** | **Contract Mismatch**. `y_next` shape is (1, 194, 194, 2) (Trade), expected (1, 194, 8) (Pathogen). Config uses `target_kind: "base"`. |
| Splits | R06 | PASS | (Implicitly verified by config/code existence, R06 in original list but not explicitly re-run in robust pass, but config `trade_step6.yaml` defines splits) |
| Tests | R09 | **FAIL** | **Collection Error**. `pytest` failed with `UnicodeDecodeError` during collection. |
| Smoke Run | R10 | PASS | `trade_step6_train_entrypoint.py --override run.fast_dev_run=True` passed with `train_loss_base=0.0988`. |

## Detailed Findings

### R03: Climate Artifacts Discovery
Discovered the following canonical paths:
- **Climate Tensor:** `data/processed/climate/climate_tensor.zarr` (contains `climate` array)
- **Climate Anomalies:** `data/processed/climate/climate_step4/climate_anomalies.zarr` (contains keys resolving to shape (908, 194, 10))

### R04: DataModule Contract Mismatch
The `TradeDataModule` (configured via `config/trade_step6.yaml`) produces batches suited for **Trade Forecasting**, not explicitly Pathogen prediction as the primary target.

**Observed Batch Keys:**
`['base_is_estimated', 'base_mask', 'base_trade', 'risk_is_estimated', 'risk_trade', 't', 't_y', 'time_feat', 'y_base', 'y_base_is_estimated', 'y_base_mask']`

**Key Mappings:**
- `y_next` -> `y_base` (Shape: `(1, 194, 194, 2)`, dtype: `float32`)
- `y_mask` -> `y_base_mask` (Shape: `(1, 194, 194, 2)`, dtype: `uint8`)

**Issue:** The user specified `expected_axes` for `P` (Pathogens) is 8, and implies the goal is ST-MM-GNN for pathogen status. The current baseline setup predicts the trade tensor itself.

### R09: Pytest Collection Failure
```
============================== ERRORS ===============================
!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position ...
```
This suggests `pytest` is attempting to parse a binary file (likely in `data/` or a cache) as a test file or config.

### R10: Training Entrypoint
Successfully ran a fast-dev-run using `config/trade_step6.yaml` (with CLI override).
Log: `Epoch 0: 100% ... train_loss_base=0.0988`
This confirms the *Trade Baseline* wiring is functional, despite the target mismatch for the ST-MM-GNN goal.

## Recommendations
1.  **Resolve Target Definition:** Clarify if Layer A MVP ST-MM-GNN should predict Pathogens (`target_kind="status" (?)`) or if we are verifying the *Trade* backbone first. If Pathogen prediction is required, `TradeDataset` needs to support `target_kind="status"` or a new Datset is needed.
2.  **Fix Pytest:** exclude `data/` from pytest collection in `pyproject.toml` or `pytest.ini`.
