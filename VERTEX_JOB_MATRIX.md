# Vertex Job Matrix (Step A)

## Canonical Python Entry Points

| Job | Module |
|-----|--------|
| **Training** | `pathograph.vertex.stepA_entry` |
| **Evaluation** | `pathograph.vertex.stepA_eval_entry` |

---

## Train Job Args

Pass each as a separate arg-line in the Vertex UI (one per line):

```
--config_gcs=gs://pathograph-057a2273fe-data/configs/stmm_stepA_adaptive.yaml
--data_gcs_prefix=gs://pathograph-057a2273fe-data/datasets/stepA/v1
--output_gcs_prefix=gs://pathograph-057a2273fe-data/runs/stepA/<RUN_NAME>
--stage_to_local=1
--seed=<SEED>
```

## Eval Job Args

```
--config_gcs=gs://pathograph-057a2273fe-data/configs/stmm_stepA_adaptive.yaml
--data_gcs_prefix=gs://pathograph-057a2273fe-data/datasets/stepA/v1
--ckpt_gcs=gs://pathograph-057a2273fe-data/runs/stepA/<RUN_NAME>/<CKPT_FILENAME>.ckpt
--output_gcs_prefix=gs://pathograph-057a2273fe-data/runs/stepA/<RUN_NAME>/eval_vertex
--stage_to_local=1
```

---

## Vertex Console Setup

1. **Training method**: Pre-built container (PyTorch 2.4)
2. **No inference container** needed
3. **Package location**: `gs://pathograph-057a2273fe-data/packages/pathograph-0.1.5.tar.gz`
4. **Region**: `us-central1`
5. **Machine type**: e.g. `n1-standard-8` + 1× T4 GPU, or `a2-highgpu-1g` for A100

## Package Upload (build + push)

```powershell
# Build
python -m build --sdist

# Upload
gcloud storage cp dist/pathograph-*.tar.gz gs://pathograph-057a2273fe-data/packages/
```

---

## Naming Conventions

| Resource | Pattern |
|----------|---------|
| Run prefix | `gs://pathograph-057a2273fe-data/runs/stepA/<family>_s<seed>/` |
| Eval output | `.../eval_vertex/` |
| Champion ckpt | `.../epoch=N-step=S-val_auroc_macro=<score>.ckpt` |

---

## Champion (Adaptive, Phase 2)

| Item | URI |
|------|-----|
| Config | `gs://pathograph-057a2273fe-data/configs/stmm_stepA_adaptive.yaml` |
| Checkpoint | `gs://pathograph-057a2273fe-data/runs/stepA/phase2/adaptive_s1338/epoch=8-step=7128-val_auroc_macro=0.9771.ckpt` |
| Run prefix | `gs://pathograph-057a2273fe-data/runs/stepA/phase2/adaptive_s1338/` |
