# Vertex UI Entry: Incident Overfit Pilot (v4 - Arguments Fix)

**Model Name:** `stepA_phase2_incident_overfit_baseline_s1337_train_v4_pytorch`

If you are using the UI to submit the job, **you must explicitly avoid the TensorFlow container**. 

### 1. Container Settings (CRITICAL)
Choose **Pre-built container**:
- **Framework:** PyTorch
- **Framework version:** 2.4
- **Python version:** Python 3.10
- **Accelerator:** GPU

> [!CAUTION]  
> If you omit this step, Vertex AI defaults to `tf-gpu.2-1:latest` (TensorFlow 2.1 / Python 3.7) which explicitly lacks `Python >= 3.9` causing an instant installation failure.

### 2. Package Settings
**Python module:** `pathograph.vertex.stepA_entry`
**Package URI:** `gs://pathograph-057a2273fe-data/packages/pathograph_incident_overfit_20260222_v4.tar.gz`

### 3. Arguments
(Add these one by one using the "Add Argument" button)
```text
--config_gcs=gs://pathograph-057a2273fe-data/configs/incident_overfit/stmm_stepA_baseline_incident_overfit.yaml
--data_gcs_prefix=gs://pathograph-057a2273fe-data/datasets/stepA/v1
--output_gcs_prefix=gs://pathograph-057a2273fe-data/runs/stepA/phase2_incident_overfit/baseline_incident_s1337
--stage_to_local=1
--seed=1337
```

---

### Alternative: Deterministic `gcloud` command
Instead of clicking through the UI, you can deterministically launch the corrected job from this repo using the exact JSON specification we generated:

```bash
gcloud ai custom-jobs create `
  --region=us-central1 `
  --project=727252250786 `
  --display-name=stepA_phase2_incident_overfit_baseline_s1337_train_v4_pytorch `
  --config=vertex_specs/incident_overfit/stepA_overfit_baseline_s1337_train_v4_pytorch.json
```
