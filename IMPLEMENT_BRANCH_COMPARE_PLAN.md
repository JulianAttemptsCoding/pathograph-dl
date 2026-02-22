# Branch Compare Plan (Step A)

## Goal

Run a controlled experiment matrix across model families and random seeds, then aggregate:
- Macro AUROC / AUPRC (equal-weight across pathogens)
- Calibration: Brier macro, ECE macro (where available)

---

## Model Families

| Family | Config suffix | Notes |
|--------|---------------|-------|
| `adaptive` | `stmm_stepA_adaptive.yaml` | **Champion** — FiLM-gated adaptive pooling |
| `film` | `stmm_stepA_film.yaml` | FiLM-only variant |
| `diag0` | `stmm_stepA_diag0.yaml` | Diagnostic baseline |
| `baseline` | `stmm_stepA.yaml` | Unmodified base |

> Update this table to match current `configs/` directory contents.

---

## Seeds

Use 5 seeds for significance: `[1338, 1339, 1340, 1341, 1342]`  
Seed 1338 is the Phase 2 champion seed (`adaptive_s1338`).

---

## Protocol

1. **Stage dataset** locally or point directly to GCS prefix.
2. **Train** each family × seed with identical data split + trainer budget.
3. **Evaluate** each run; write `metrics.csv` + `calibration_bins_test.json`.
4. **Aggregate** across seeds → mean/std/min/max via `tools/reporting/pull_phase3_aggregate.py`.
5. **Compare** families by macro AUROC mean ± std.

---

## Gating Criteria

- **Integrity gate**: `python -m pytest -q` passes with skip for data.
- **Repro gate**: seed, config URI, and package URI logged per run.
- **Shape gate**: `y_mask` has positives; loss is finite (not NaN) on first batch.

---

## Output Layout (GCS)

```
gs://pathograph-057a2273fe-data/runs/stepA/
  phase3/
    adaptive_s1338/
    adaptive_s1339/
    film_s1338/
    ...
    _aggregate/
      phase3_summary_stats.csv
      phase3_cal_final_summary_stats.csv
    _reports/
```
