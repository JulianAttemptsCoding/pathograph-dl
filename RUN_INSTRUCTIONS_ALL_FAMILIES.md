# Run Instructions (All Families)

## Prerequisites

- `gcloud` authenticated: `gcloud auth login; gcloud auth application-default login`
- Conda env: `conda env create -f environment.yml; conda activate pathograph-train`
- Repo installed: `pip install -e .`

---

## 1. Stage Local Data from GCS

```powershell
# Stage all processed v1 data (Trade, Pathogen, Climate, Meta)
.\tools\stage_stepA_v1_from_gcs.ps1

# Optionally include pytest gate:
.\tools\stage_stepA_v1_from_gcs.ps1 -RunPytestGate
```

Dataset GCS prefix: `gs://pathograph-057a2273fe-data/datasets/stepA/v1`

---

## 2. Run Unit Tests

```powershell
python -m pytest -q
# Expected: all pass or skip (data-dependent skip cleanly without local zarr)
```

---

## 3. Local Fast-Dev Train (Sanity Check)

```powershell
python tools\stmm_stepA_train.py --config config\stmm_stepA.yaml --fast_dev_run
```

---

## 4. Vertex Training (All Families × Seeds)

Build and upload the sdist first:

```powershell
python -m build --sdist
gcloud storage cp dist/pathograph-*.tar.gz gs://pathograph-057a2273fe-data/packages/
```

Then for each `<family>` × `<seed>`:

**Module**: `pathograph.vertex.stepA_entry`

**Args** (one per line in Vertex UI):
```
--config_gcs=gs://pathograph-057a2273fe-data/configs/stmm_stepA_<family>.yaml
--data_gcs_prefix=gs://pathograph-057a2273fe-data/datasets/stepA/v1
--output_gcs_prefix=gs://pathograph-057a2273fe-data/runs/stepA/phase3/<family>_s<seed>
--stage_to_local=1
--seed=<seed>
```

See `VERTEX_JOB_MATRIX.md` for full Vertex console setup details.

---

## 5. Vertex Evaluation

After each training job completes, for each run:

**Module**: `pathograph.vertex.stepA_eval_entry`

**Args**:
```
--config_gcs=gs://pathograph-057a2273fe-data/configs/stmm_stepA_<family>.yaml
--data_gcs_prefix=gs://pathograph-057a2273fe-data/datasets/stepA/v1
--ckpt_gcs=gs://pathograph-057a2273fe-data/runs/stepA/phase3/<family>_s<seed>/<best_ckpt>.ckpt
--output_gcs_prefix=gs://pathograph-057a2273fe-data/runs/stepA/phase3/<family>_s<seed>/eval_vertex
--stage_to_local=1
```

---

## 6. Aggregate Results

```powershell
python tools/reporting/pull_phase3_aggregate.py `
  --phase3_aggregate_prefix gs://pathograph-057a2273fe-data/runs/stepA/phase3/_aggregate `
  --out_dir reports/phase3/_local_cache
```
