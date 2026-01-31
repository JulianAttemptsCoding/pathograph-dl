# STMM Layer-A Evaluation & Reporting Gate

This document describes the Layer-A evaluation and reporting gate for the STMM model. This gate ensures:
- Data quality (label sanity)
- Mask-correct, degenerate-safe metrics
- Baseline comparison
- Temperature scaling for calibration
- Auditable artifact generation

## Components

### 1. Label Sanity Checker
**Tool:** `tools/stmm_stepA_label_sanity.py`

Verifies that validation and test splits contain non-degenerate label distributions for all pathogens.

**Usage:**
```bash
python tools/stmm_stepA_label_sanity.py \
  --config config/stmm_stepA.yaml \
  --split all \
  --max_batches 50 \
  --strict
```

**Outputs:**
- JSON with per-pathogen valid/pos/neg counts
- Flags degenerate cases (no positives OR no negatives)
- Exit code 1 in strict mode if invariants fail

**Invariants:**
- Each pathogen must have `valid > 0` in scanned batches
- Each non-degenerate pathogen has both `pos > 0` AND `neg > 0`

---

### 2. Masked Metrics Library
**Module:** `pathograph/metrics/masked_classification.py`

Degenerate-safe AUROC/AUPRC computation with explicit NaN handling.

**Key Functions:**
- `flatten_masked(preds, targets, mask)` - Filter to observed samples
- `safe_auroc(p, y)` - Returns NaN if no pos OR no neg
- `safe_auprc(p, y)` - Returns NaN if no pos
- `per_pathogen_metrics(probs, y, mask)` - Compute for all pathogens
- `macro_nanmean(values)` - NaN-excluding mean with count

**Degeneracy Handling:**
- AUROC requires both positives and negatives → returns NaN otherwise
- AUPRC requires positives → returns NaN otherwise
- Checks happen **before** calling torchmetrics to avoid warnings
- Macro aggregation explicitly excludes NaN and reports `n_used`

---

### 3. STMM Module Refactor (Epoch-Level Metrics)
**File:** `pathograph/pl/stmm_pl_module.py`

**Change:** Moved from batch-level TorchMetrics updates to epoch-level accumulation.

**Rationale:**
- Batch-level updates can trigger spurious degeneracy warnings
- Validation/test splits have few unique time indices
- Epoch-level computation is stable and aligns with final reported metrics

**Implementation:**
- `validation_step`: Accumulate `(probs, targets, mask)` as CPU tensors
- `on_validation_epoch_end`: Concatenate, call `per_pathogen_metrics`, log macro + per-pathogen
- Same pattern for `test_step` / `on_test_epoch_end`
- No stateful TorchMetrics objects for val/test

---

### 4. Unified Eval/Report Tool
**Tool:** `tools/stmm_stepA_eval_report.py`

Performs baseline and model evaluation with temperature scaling.

**Usage:**
```bash
# Baseline only
python tools/stmm_stepA_eval_report.py \
  --config config/stmm_stepA.yaml \
  --split all \
  --max_batches 10 \
  --device cpu

# With model checkpoint
python tools/stmm_stepA_eval_report.py \
  --config config/stmm_stepA.yaml \
  --ckpt runs/xxx/last.ckpt \
  --split all \
  --device cpu
```

**Workflow:**
1. Evaluate persistence baseline on VAL and/or TEST
2. If model provided:
   - Evaluate raw model on VAL and TEST
   - Fit temperature scaling on VAL logits
   - Apply temperature to TEST logits
   - Compute calibrated metrics on TEST
3. Generate artifacts (see below)
4. Apply GO/NO-GO decision rule

**Outputs (in `runs/stmm_eval/<timestamp>/`):**
- `val_metrics.json` - Val split results
- `test_metrics.json` - Test split results
- `per_pathogen_metrics.csv` - Combined per-pathogen table
- `temperature_scaling.json` - Temperature parameter + NLL before/after
- `report.md` - Human-readable summary with GO/NO-GO decision

**GO/NO-GO Rule:**
- **GO:** Model macro AUPRC ≥ Baseline AUPRC + 0.01
- **NO-GO:** Model < Baseline
- **INCONCLUSIVE:** 0 ≤ delta < 0.01 or NaN metrics

---

### 5. Gate Runner
**Tool:** `tools/gate_layerA_eval_report.py`

Runs the full gate sequence in one command.

**Usage:**
```bash
python tools/gate_layerA_eval_report.py \
  --config config/stmm_stepA.yaml \
  --max_batches 10
```

**Optional:** Set `STMM_CKPT=path/to/checkpoint.ckpt` to also run model evaluation.

**Sequence:**
1. Label sanity check (strict, bounded)
2. Pytest (full suite)
3. Eval report (baseline-only, bounded)
4. Eval report (with model, if `STMM_CKPT` set)

**Exit:** Nonzero on first failure.

---

## NaN Handling & Macro Aggregation

**Why NaN?**
- AUROC is undefined when all labels are 0 or all labels are 1
- AUPRC is undefined when all labels are 0
- Explicit NaN is better than silent 0/1 or exceptions

**Macro Aggregation:**
- Compute per-pathogen metric (may be NaN)
- Exclude NaN values from mean
- Log `n_used` to show how many pathogens contributed
- Example: `macro_auroc=0.85 (n=7/8)` means 7 pathogens valid, 1 degenerate

**Logged Metrics:**
- `val_auroc_macro`, `val_auprc_macro` - NaN-excluding means
- `val_n_valid_auroc`, `val_n_valid_auprc` - Count of non-NaN pathogens
- `val_auroc_p0` through `val_auroc_p7` - Per-pathogen (may be NaN)
- Same for test split

---

## Recommended Actions

### If Label Sanity Fails
- Check data pipeline and mask generation
- Verify split indices in config
- Increase `--max_batches` to scan more data

### If Baseline Eval Fails
- Check persistence baseline implementation
- Verify data module setup
- Inspect per-pathogen counts in output JSON

### If Model Eval Shows NO-GO
- Review training curves (loss, AUROC, AUPRC)
- Check for overfitting (val vs train gap)
- Consider data augmentation or regularization
- Retrain with different hyperparameters

### If Temperature Scaling Fails
- Check VAL split logits distribution
- Verify mask correctness in NLL calculation
- Temperature should be in range [0.1, 10.0]

---

## Testing

**Unit Tests:**
- `tests/test_masked_metrics_degenerate.py` - 13 tests for degeneracy handling

**Smoke Test:**
Run bounded gate:
```bash
python tools/gate_layerA_eval_report.py --max_batches 5
```

Expected: All steps pass, artifacts written to `runs/gate_eval_baseline/`.

---

## Artifacts

All artifacts use ASCII-only output for portability.

**JSON Schema (Metrics):**
```json
{
  "split": "val|test",
  "baseline": {
    "per_pathogen": {
      "p0": {"auroc": float, "auprc": float, "valid": int, "pos": int, "neg": int},
      ...
    },
    "macro": {"auroc": float, "auprc": float, "n_valid_auroc": int, "n_valid_auprc": int}
  },
  "model": { /* same structure, plus _raw and _cal variants */ }
}
```

**CSV Columns:**
- `pathogen` (p0-p7)
- `valid`, `pos`, `neg`
- `baseline_auroc`, `baseline_auprc`
- `model_auroc_raw`, `model_auprc_raw` (if model)
- `model_auroc_cal`, `model_auprc_cal` (if model + TEST split)

**Report Markdown:**
- Config metadata
- Per-split baseline and model macro metrics
- GO/NO-GO decision with reasoning
- Recommended next actions

---

## Integration with CI/CD

The gate runner can be integrated into CI pipelines:

```yaml
# .github/workflows/gate.yml
- name: Run Layer-A Gate
  run: |
    python tools/gate_layerA_eval_report.py --max_batches 10
  env:
    STMM_CKPT: ${{ secrets.STMM_CHECKPOINT_PATH }}  # optional
```

Exit code indicates pass/fail for automation.
