# STMM Evaluation Metrics Implementation

**Date**: 2026-01-27  
**Status**: ✅ Complete  
**Git Commit**: (to be recorded)

## Objective

Implement AUROC and AUPRC metrics for STMM Step-A model evaluation, add temperature scaling calibration, and create persistence baseline for comparison.

## Implementation Summary

### 1. Masked AUROC and AUPRC in STMM Lightning Module

**File**: `pathograph/pl/stmm_pl_module.py`

**Changes**:
- Added `torchmetrics.BinaryAUROC` and `torchmetrics.BinaryAveragePrecision` imports
- Initialized per-pathogen metrics for validation and test splits (P=8 pathogens)
- Updated `validation_step` to:
  - Compute probabilities via `sigmoid(logits)`
  - Update per-pathogen metrics with masked samples only (`y_mask > 0.5`)
  - Flatten and filter (B, N, P) tensors to observed samples
- Added `on_validation_epoch_end` to:
  - Compute per-pathogen AUROC and AUPRC
  - Log individual pathogen metrics (`val_auroc_p{i}`, `val_auprc_p{i}`)
  - Compute macro average across pathogens
  - Log `val_auroc_macro` and `val_auprc_macro` (progress bar)
  - Reset metrics for next epoch
- Added `test_step` and `on_test_epoch_end` with identical logic for test split
- Updated `__init__` signature to accept `num_pathogens=8` parameter

**Metrics Contract**:
- Binary classification per pathogen
- Masked evaluation: only samples with `y_mask == 1` are included
- Macro averaging: unweighted mean across P=8 pathogens
- Logged on validation and test only (not training)

**Test Coverage**: `tests/test_stmm_metrics.py`
- ✅ 5 tests passing
- Verifies metric initialization, updates, masking, and macro averaging

---

### 2. Temperature Scaling Calibration

**File**: `pathograph/calibration/temperature_scaling.py`

**Implementation**:
- Single scalar temperature parameter `T`
- Fitted on validation logits to minimize masked NLL
- Uses LBFGS optimizer (max 50 iterations)
- Clamped to range [0.1, 10.0] to prevent extreme values
- Supports multidimensional inputs (B, N, P)
- Respects observation masks (`y_mask`)

**Usage**:
```python
from pathograph.calibration import TemperatureScaling

# After validation epoch, collect logits
ts = TemperatureScaling()
ts.fit(val_logits, val_targets, val_mask, max_iter=50)

# Apply calibration
calibrated_probs = ts(test_logits)
```

**Test Coverage**: `tests/test_temperature_scaling.py`
- ✅ 6 tests passing
- Verifies initialization, forward pass, fitting with masks, multidimensional inputs

---

### 3. Persistence Baseline

**File**: `pathograph/baselines/persistence.py`

**Implementation**:
- Predicts `y(t+1) = y(t)` using last observed value in input window
- For each (node, pathogen), searches backwards through `y_hist` to find last observed (`y_hist_mask == 1`)
- If no history observed, predicts 0.0 (negative class)
- Implements identical metric contract as STMM:
  - Masked BCE loss
  - Per-pathogen AUROC and AUPRC
  - Macro averaging
  - Validation and test splits
- No optimizer needed (no trainable parameters)

**Evaluation Entrypoint**: `tools/eval_persistence_baseline.py`

**Usage**:
```bash
python tools/eval_persistence_baseline.py \
  --config config/stmm_stepA.yaml \
  --run_dir runs/persistence_baseline/run1 \
  --seed 42
```

**Features**:
- Uses identical `STMMDataModule` and splits as STMM training
- Logs results to CSV
- Saves config snapshot and run manifest with git commit SHA
- Supports `--fast_dev_run` for testing

**Test Coverage**: `tests/test_persistence_baseline.py`
- ✅ 8 tests passing
- Verifies persistence prediction logic with gaps and masking
- Tests loss computation and metric updates

---

## Test Results Summary

| Test Suite | Status | Tests Passed |
|------------|--------|--------------|
| `test_stmm_metrics.py` | ✅ PASS | 5/5 |
| `test_temperature_scaling.py` | ✅ PASS | 6/6 |
| `test_persistence_baseline.py` | ✅ PASS | 8/8 |
| **Total** | **✅ PASS** | **19/19** |

All tests executed with `pathograph-train` environment.

---

## Integration with Existing Training

The updated `STMMPLModule` is **backward compatible** with existing training scripts:

**Old instantiation** (still works):
```python
pl_module = STMMPLModule(model, lr=0.001, weight_decay=0.0)
```

**New instantiation** (recommended):
```python
pl_module = STMMPLModule(model, lr=0.001, weight_decay=0.0, num_pathogens=8)
```

**New logged metrics** (automatically available):
- `val_auroc_p0` through `val_auroc_p7` (per-pathogen validation AUROC)
- `val_auprc_p0` through `val_auprc_p7` (per-pathogen validation AUPRC)
- `val_auroc_macro` (macro-averaged AUROC, progress bar)
- `val_auprc_macro` (macro-averaged AUPRC, progress bar)
- Same for test split: `test_auroc_p*`, `test_auprc_p*`, `test_auroc_macro`, `test_auprc_macro`

---

## Files Created

### New Modules
- `pathograph/calibration/__init__.py`
- `pathograph/calibration/temperature_scaling.py`
- `pathograph/baselines/__init__.py`
- `pathograph/baselines/persistence.py`

### New Tools
- `tools/eval_persistence_baseline.py`

### New Tests
- `tests/test_stmm_metrics.py`
- `tests/test_temperature_scaling.py`
- `tests/test_persistence_baseline.py`

---

## Next Steps

1. **Run STMM training with new metrics**:
   ```bash
   python tools/train_stmm_stepA.py --config config/stmm_stepA.yaml --max_epochs 10
   ```
   - Verify `val_auroc_macro` and `val_auprc_macro` are logged
   - Check TensorBoard/CSV logs for per-pathogen metrics

2. **Evaluate persistence baseline**:
   ```bash
   python tools/eval_persistence_baseline.py \
     --config config/stmm_stepA.yaml \
     --run_dir runs/persistence_baseline/$(date +%Y%m%d_%H%M%S) \
     --seed 42
   ```
   - Compare baseline AUROC/AUPRC to STMM model
   - Persistence is expected to be a strong baseline for rare events

3. **Calibration workflow** (post-training):
   - Load best STMM checkpoint
   - Run validation epoch to collect logits
   - Fit `TemperatureScaling` on validation set
   - Evaluate calibrated probabilities on test set
   - Compare calibration metrics (ECE, reliability diagrams)

4. **Optional: Add calibration to training pipeline**:
   - Add `on_validation_epoch_end` hook to fit temperature scaling
   - Log calibrated metrics alongside raw metrics
   - Save temperature parameter to checkpoint

---

## References

- **AUROC/AUPRC**: Standard metrics for imbalanced binary classification
- **Temperature Scaling**: Guo et al. 2017, "On Calibration of Modern Neural Networks"
- **Persistence Baseline**: Last-observation-carried-forward, common benchmark for time series
- **Macro Averaging**: Treats all pathogens equally regardless of prevalence

---

## Verification Checklist

- [x] AUROC and AUPRC implemented with torchmetrics
- [x] Per-pathogen metrics (P=8)
- [x] Macro averaging across pathogens
- [x] Respects `y_mask == 1` for observed samples only
- [x] Logged on validation and test splits only
- [x] Temperature scaling calibration module
- [x] Persistence baseline with identical metric contract
- [x] Persistence baseline evaluation entrypoint
- [x] All unit tests passing (19/19)
- [x] Backward compatibility with existing training code
