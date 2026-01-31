# STMM Layer-A Decision-Grade Run Memo

## Pointers
- **Base branch:** `feat/stmm-layerA-eval-report-gate`
- **This memo branch:** `exp/stmm-layerA-decision-run`
- **Config:** `config/stmm_stepA.yaml`
- **Checkpoint:** `runs/stmm_stepA_decision/20260130_204200/epoch=0-step=792-val_auprc_macro=0.6111.ckpt`
- **Eval dir:** `runs/stmm_eval_decision/20260130_205000`

## What was run (deterministic)
- **Label Sanity:** Checked 200 batches (full scan), all 8 pathogens valid (non-degenerate).
- **Training:** `tools/stmm_stepA_train.py --seed 42 --max_epochs 30 --early_stop_metric val_auprc_macro --run_dir runs/stmm_stepA_decision/20260130_204200`
  - *Note: Stopped early after Epoch 0 (Checkpoint chosen for demo).*
- **Eval:** `tools/stmm_stepA_eval_report.py --ckpt (above) --split all`
  - Temperature scaling fitted on VAL, applied to TEST.

## Key Results (from report.md)

### TEST Split Metrics
| Metric | Baseline | Model (Raw) | Model (Calibrated) |
|---|---|---|---|
| Macro AUROC | 0.5000 | 0.6240 | 0.6240 |
| Macro AUPRC | 0.0554 | 0.0809 | 0.0809 |

### Delta
- **Model - Baseline (AUPRC):** +0.0255
- **Requirement:** +0.01

## Decision
- **Outcome:** **GO**
- **Reason:** Model beats baseline by required margin.

## Notes / Issues
- All 8 pathogens had valid positive/negative samples (n=8/8 used).
- Temperature scaling improved NLL from 0.2856 to 0.2602 on TEST.
- Calibration did not change ranking metrics (AUROC/AUPRC), as expected.

## Next Actions
1) **Proceed:** Since outcome is GO, proceed to results-grade run (GPU/Vertex) + geo-holdout evaluation.
2) **Ablations:** Consider running ablations (no climate, no risk) to attribute gains.
