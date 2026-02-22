# Vertex UI Entry: Incident Overfit Pilot
**Model Name:** `stepA_phase2_incident_overfit_baseline_s1337_train`

Copy and paste these exact arguments into your Vertex Custom Job settings:

### Package
**Python module:** `pathograph.vertex.stepA_entry`
**Package URI:** `gs://pathograph-057a2273fe-data/packages/pathograph_incident_overfit_20260222.tar.gz`

### Arguments
(Add these one by one using the "Add Argument" button)
```text
--config_gcs=gs://pathograph-057a2273fe-data/configs/incident_overfit/stmm_stepA_baseline_incident_overfit.yaml
--data_gcs_prefix=gs://pathograph-057a2273fe-data/datasets/stepA/v1
--output_gcs_prefix=gs://pathograph-057a2273fe-data/runs/stepA/phase2_incident_overfit/baseline_incident_s1337
--stage_to_local=1
--seed=1337
```

### Compute 
**Machine Type:** Same as Phase 2 (e.g. `n1-standard-8` with `1x NVIDIA_TESLA_T4` or your preferred standard).
**Inference Container:** *No inference container needed.*

> [!WARNING]
> **Manual Stop Required!**
> This run has early-stopping intentionally disabled (max 200 epochs). Monitor Tensorboard and manually CANCEL the run when train loss decreases but validation loss increases steadily (for 3+ consecutive epochs).
>
> NEVER evaluate the test set during this training phase. Wait until the job stops, pick your 3 target epochs locally, and run the `stepA_eval_entry` module against them in separate evaluating-only jobs.
