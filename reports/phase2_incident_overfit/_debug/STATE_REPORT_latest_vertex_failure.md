# START OF STATE REPORT
# Vertex Failure Debug State (2026-02-22)

### 1. Repository Reality Check
**Branch:** `fix/vertex-stepA-train-import`
**HEAD SHA:** `17a950c Bump Vertex tracking files to v4 due to uncommitted train_stmm args fix`
**Working Tree:** Clean (no untracked modifications to `pathograph/` files were found)

### 2. Intended Configuration
- **Executor Image URI:** `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest`
- **Package URI:** `gs://pathograph-057a2273fe-data/packages/pathograph_incident_overfit_20260222_v4.tar.gz`

### 3. Actual Vertex Failure Fields (Job ID: `3051299712318570496`)
- **Executor Image URI:** `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest`
- **Package URIs:** `['gs://pathograph-057a2273fe-data/packages/pathograph_incident_overfit_20260222_v4.tar.gz']`
- **Python Module:** `pathograph.vertex.stepA_entry`
- **Args:**
  ```json
  [
    "--config_gcs=gs://pathograph-057a2273fe-data/configs/incident_overfit/stmm_stepA_baseline_incident_overfit.yaml",
    "--data_gcs_prefix=gs://pathograph-057a2273fe-data/datasets/stepA/v1",
    "--output_gcs_prefix=gs://pathograph-057a2273fe-data/runs/stepA/phase2_incident_overfit/baseline_incident_s1337",
    "--stage_to_local=1",
    "--seed=1337"
  ]
  ```

### 4. Failure Evidence
The job successfully downloaded the `_v4` package, successfully staged the dataset to `/tmp/work/data/processed`, correctly parsed all CLI paths, and entered the PyTorch Lightning Datamodule setup phase:

```text
The replica workerpool0-0 exited with a non-zero status of 1. Termination reason: Error. 
Traceback (most recent call last):
  [...]
  File "/root/.local/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 1039, in _run
    call._call_setup_hook(self)  # allow user to set up LightningModule in accelerator environment
  File "/root/.local/lib/python3.10/site-packages/pathograph/data/trade_datamodule.py", line 115, in setup
    self._train = TradeDatasetZarr(TradeDatasetConfig(**base, split="train"))
  File "/root/.local/lib/python3.10/site-packages/pathograph/data/trade_dataset.py", line 86, in __init__
    self.h = open_trade_zarr(cfg.base_zarr_path, cfg.risk_zarr_path)
  File "/root/.local/lib/python3.10/site-packages/pathograph/data/trade_zarr.py", line 42, in open_trade_zarr
    base = zarr.open(base_zarr_path, mode="r")
  File "/root/.local/lib/python3.10/site-packages/zarr/convenience.py", line 137, in open
    raise PathNotFoundError(path)
zarr.errors.PathNotFoundError: nothing found at path ''
```

### 5. Conclusion
**Mismatch Category:** `other` (Empty data path config / Dataset instantiation failure).
The container, arguments, and module parsing succeeded perfectly. The job crashed purely because `cfg.base_zarr_path` evaluated to an empty string `""` inside the `stmm_stepA_baseline_incident_overfit.yaml` config when `trade_dataset.py` attempted to open it via `zarr`.
