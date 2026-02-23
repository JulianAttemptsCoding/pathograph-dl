# STATE REPORT: Empty Zarr Path Fix (2026-02-22)

### 1. The Real Root Cause
The previous job `3051299712318570496` failed with:
`zarr.errors.PathNotFoundError: nothing found at path ''`

Initial hypotheses suggested that `cfg.base_zarr_path` evaluated completely into an empty string `""` due to dictionary merge precedence. By evaluating `config_effective_paths_dump.json`, we proved **this was not the case**. The Python dictionary parsed perfectly into:
`data/processed/trade/imf_imts_step1/trade_fob_tensor.zarr`

The error surfaced because **this path simply does not exist on disk** inside your GCS bucket `datasets/stepA/v1`. The `zarr.open(path)` library strictly throws `PathNotFoundError: nothing found at path ''` when the root directory passed as the `store` argument is completely absent! The GCS pilot dataset actually holds the tensor at `data/processed/trade/trade_tensor_pilot.zarr`.

### 2. Namespace Findings
To be thoroughly deterministic, the dump revealed the pipeline handles precedence linearly:
`trade_datamodule.setup` **strictly uses the `datamodule` dictionary namespace** to instantiate `TradeDatasetConfig`. It never merges `dataset` on top of `datamodule` inside `setup()`.

### 3. Resolution (v2 Config + Guard)
1. **Config Patch (`v2`)**: We directly created a `stmm_stepA_baseline_incident_overfit_v2.yaml`.
   - `dataset.base_zarr_path` = `data/processed/trade/trade_tensor_pilot.zarr`
   - `datamodule.base_zarr_path` = `data/processed/trade/trade_tensor_pilot.zarr`
   Uploaded to `gs://pathograph-057a2273fe-data/configs/incident_overfit/stmm_stepA_baseline_incident_overfit_v2.yaml`
2. **Missing Files Guard**: In `pathograph/data/trade_zarr.py`, I injected a direct existence checker before `zarr.open`. Now it will crash deterministically with `ValueError("base_zarr_path does not exist on disk")` if it ever happens again.
3. **New Pipeline Prefix**: Output switched to `baseline_incident_s1337_v4b` to stay cleanly separated from partial data.


### 4. Rerun Instructions (v4b)
**Rerun Job Spec JSON:** 
`vertex_specs/incident_overfit/stepA_overfit_baseline_s1337_train_v4b_pytorch.json`

**Deploy deterministically via CLI:**
```bash
gcloud ai custom-jobs create `
  --region=us-central1 `
  --project=727252250786 `
  --display-name=stepA_phase2_incident_overfit_baseline_s1337_train_v4b_pytorch `
  --config=vertex_specs/incident_overfit/stepA_overfit_baseline_s1337_train_v4b_pytorch.json
```
