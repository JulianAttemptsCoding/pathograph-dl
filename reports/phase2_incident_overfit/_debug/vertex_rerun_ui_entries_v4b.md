# Vertex UI Entry: Incident Overfit Pilot (v4b - Empty Path Fix)

**Model Name:** `stepA_phase2_incident_overfit_baseline_s1337_train_v4b_pytorch`

### 1. Container Settings (CRITICAL)
Choose **Pre-built container**:
- **Framework:** PyTorch
- **Framework version:** 2.4
- **Python version:** Python 3.10
- **Accelerator:** GPU

### 2. Package Settings
**Python module:** `pathograph.vertex.stepA_entry`
**Package URI:** `gs://pathograph-057a2273fe-data/packages/pathograph_incident_overfit_20260222_v4.tar.gz`

### 3. Arguments
(Add these one by one using the "Add Argument" button)
```text
--config_gcs=gs://pathograph-057a2273fe-data/configs/incident_overfit/stmm_stepA_baseline_incident_overfit_v2.yaml
--data_gcs_prefix=gs://pathograph-057a2273fe-data/datasets/stepA/v1
--output_gcs_prefix=gs://pathograph-057a2273fe-data/runs/stepA/phase2_incident_overfit/baseline_incident_s1337_v4b
--stage_to_local=1
--seed=1337
```
