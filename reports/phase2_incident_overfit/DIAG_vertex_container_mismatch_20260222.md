# DIAGNOSIS: Vertex Container Mismatch (2026-02-22)

### Incident
The `stepA_phase2_incident_overfit_baseline_s1337_train_v3-custom-job` failed during the package installation phase. 

**Root Cause:** The job was launched using a legacy TensorFlow 2.1 container built running Python 3.7. 
`us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-1:latest`

**Error Signature:** Since the `pathograph` package strictly requires Python >= 3.9 (per its `setup.cfg`), `pip` evaluated the container environment and threw:
`ERROR: Package 'pathograph' requires a different Python: 3.7.16 not in '>=3.9'`

### Resolution
The job logically assumes the PyTorch 2.4 / Python 3.10 container, which Vertex provides as a standard prebuilt image. To fix this, simply recreate the job specifying the exact correct PyTorch container image:
**Correct `executorImageUri`:** `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest`

### Validated Specs for Rerun
**Package URI:** `gs://pathograph-057a2273fe-data/packages/pathograph_incident_overfit_20260222_v3.tar.gz`
**Python Module:** `pathograph.vertex.stepA_entry`
**Arguments:**
```text
  "--config_gcs=gs://pathograph-057a2273fe-data/configs/incident_overfit/stmm_stepA_baseline_incident_overfit.yaml",
  "--data_gcs_prefix=gs://pathograph-057a2273fe-data/datasets/stepA/v1",
  "--output_gcs_prefix=gs://pathograph-057a2273fe-data/runs/stepA/phase2_incident_overfit/baseline_incident_s1337",
  "--stage_to_local=1",
  "--seed=1337"
```
