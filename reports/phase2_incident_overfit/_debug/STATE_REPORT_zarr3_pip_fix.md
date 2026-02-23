# STATE REPORT: Zarr V3 PyPI Dependency Fix (2026-02-22)

### 1. The Real Root Cause
The previous jobs (`v4` and `v4b`) continued to fail with `zarr.errors.PathNotFoundError: nothing found at path ''` even after the YAML paths were fixed and proven to point to the correct GCS directories. 

Investigation into the exact Vertex worker logs revealed:
1. The `v1` datasets on GCS (e.g., `trade_tensor_pilot.zarr/zarr.json`) are **Zarr Format 3** arrays.
2. The PyTorch Vertex Container `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest` runs **Python 3.10.18**.
3. Because `pathograph`'s `pyproject.toml` only specified `"zarr"`, the `pip` resolver running in Vertex evaluated the Python version and other preinstalled constraints and **downgraded the installation to `zarr-2.18.3`**.
4. Zarr 2.18.3 does NOT natively recognize Zarr V3 storage layers without explicit experimental flags. It looks strictly for `.zgroup` files. When it couldn't find them, it wrongly raised `PathNotFoundError` as if the directory simply didn't exist.

(Locally, you did not experience this bug because your laptop uses conda environment with `zarr 3.1.5` installed).

### 2. Resolution (v5 Payload + Env Lock)
1. **Dependency Pinning (`v5`)**: I explicitly modified `pyproject.toml` inside the `pathograph-dl` project to force `pip` onto Zarr V3:
   ```toml
   dependencies = [
     "numpy",
     "pyyaml",
     "zarr>=3.0.0",
     "torchmetrics",
     "lightning",
     # ...
   ]
   ```
2. **Built Payload Payload**: Recompiled the source directory and uploaded `gs://pathograph-057a2273fe-data/packages/pathograph_incident_overfit_20260222_v5.tar.gz` to ensure the container receives the new constraints.
3. **New Pipeline Prefix**: Output switched to `baseline_incident_s1337_v5` to stay cleanly separated from partial data.

### 3. Execution (v5)
Job `8151476756734279680` is currently running:
```json
"executorImageUri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest",
"packageUris": ["gs://pathograph-057a2273fe-data/packages/pathograph_incident_overfit_20260222_v5.tar.gz"],
"config_gcs": "gs://pathograph-057a2273fe-data/configs/incident_overfit/stmm_stepA_baseline_incident_overfit_v2.yaml"
```

If it drops again, we know it's not the path strings or the Zarr library version. But theoretically, this solves the exact symptom.
