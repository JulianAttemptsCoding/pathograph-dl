# Trade Step 5: Model-Integration Discovery Report

## 1. Artifact Inventory

| Component | Path | Status | Details |
| :--- | :--- | :---: | :--- |
| **Step 1 Manifest** | `data/processed/trade/imf_imts_step1/manifest.json` | Found | T=908, Start=1950-01, N=194, CIF->FOB=0.06 |
| **Step 1 Zarr** | `data/processed/trade/imf_imts_step1/trade_fob_tensor.zarr` | Found | Arrays: `trade`, `mask`, `is_estimated`, `time_index` |
| **Step 2 Manifest** | `data/processed/trade/faostat_step2/preprocessing_manifest.json` | Found | K=8 groups, Weight range 1986-2024 |
| **Step 2 Zarr** | `data/processed/trade/faostat_step2/trade_risk_tensor.zarr` | Found | Arrays: `trade_risk`, `observed_mask`, `is_estimated`, `time_index` |
| **Step 3 Scaler** | `data/processed/trade/trade_step3_scaler.json` | Found | log1p + Standard scaling. Train-only fit (t=0..815) |
| **DataModule** | `pathograph/data/trade_datamodule.py` | Found | Implements `TradeDataModule` with splitting logic |
| **Dataset** | `pathograph/data/trade_dataset.py` | Found | Implements `TradeDatasetZarr` with windowing and scaling |
| **Collate** | `pathograph/data/trade_collate.py` | Found | Implements `trade_collate_separate` with zero-out masking |

## 2. Zarr Array Inventory & Shapes

### Step 1: Base Trade (IMF IMTS)
- **Path**: `data/processed/trade/imf_imts_step1/trade_fob_tensor.zarr`
- **Main Array**: `trade`
  - **Shape**: `(908, 194, 194, 2)`
  - **Dtype**: `float32`
  - **Channels**: `[exports_fob_usd, imports_fob_best_usd]` (per `trade_step3_scaler.json`)
  - **Chunks**: `(12, 64, 64, 2)`
- **Mask Array**: `mask`
  - **Shape**: `(908, 194, 194, 2)`
  - **Dtype**: `uint8` (0=Missing/Unobserved, 1=Observed)

### Step 2: Risk Trade (FAOSTAT)
- **Path**: `data/processed/trade/faostat_step2/trade_risk_tensor.zarr`
- **Main Array**: `trade_risk`
  - **Shape**: `(908, 194, 194, 8, 2)`
  - **Dtype**: `float32`
  - **Channels**: `8 groups x 2 flow types` (Total 16 dimensions when flattened)
  - **Chunks**: `(12, 64, 64, 8, 2)`
- **Mask Array**: `observed_mask`
  - **Shape**: `(908, 194, 194, 8, 2)`
  - **Dtype**: `uint8`

## 3. Batch Contract Observed at Runtime

Verified via `tools/trade_step4_smoketest.py`:

| Key | Shape (B=1, L=24, N=194, K=8) | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `t` | `(1,)` | `int32` | Anchor time index |
| `t_y` | `(1,)` | `int32` | Target time index (t + H) |
| `time_feat` | `(1, 2)` | `float32` | `[sin, cos]` of target month |
| `base_trade` | `(1, 24, 194, 194, 2)` | `float32` | Scaled & log1p base trade values |
| `base_mask` | `(1, 24, 194, 194, 2)` | `uint8` | Mask for base trade |
| `risk_trade` | `(1, 24, 194, 194, 8, 2)` | `float32` | Scaled & log1p risk trade values |
| `risk_mask` | `(1, 24, 194, 194, 8, 2)` | `uint8` | Mask for risk trade |

**Note**: In `trade_collate_separate`, `trade` arrays are multiplied by their masks, ensuring unobserved entries are exactly `0.0`.

## 4. Scaler Semantics & Pipeline

- **Transform Path**: `Input -> log1p -> Standardization -> Output`.
- **Zeros/NaNs**: `np.maximum(x, 0.0)` is applied before `log1p` to handle negative noise. Diagonal is typically zero and remains zero.
- **Masking**: Scaler statistics were computed using **mask-aware aggregation** (`_masked_welford_update`), meaning unobserved cells did not bias the mean/std toward zero.
- **Fitting**: Strictly fitted on `t=0..815` (Train split).

## 5. Compatibility Gaps & Integration Obstacles

### BLOCKERS
- **Missing Target Labels**: The `TradeDatasetZarr.__getitem__` currently returns input features (windows) but **does not return the labels (y)** for time `t_y`. The model has nothing to calculate loss against.
- **Missing Model Consumer**: There is no PyTorch model (e.g., GNN) or training entrypoint (e.g., `train.py`) in the codebase that references these trade inputs.

### DISCREPANCIES
- **Spec v1.1 Divergence**: `docs/spec_trade_ingestion_v1.1.md` specifies 4 channels (`exports_fob`, `imports_cif`, `filled_exports`, `confidence`), but the implementation uses 2 primary base channels (`exports_fob`, `imports_fob_best`) and 16 risk channels. **The implementation is more advanced but the spec is outdated.**

## 6. Synthesis & Next Actions

### Memory & Throughput Estimates
- **Batch Size 1**: ~65 MB per sample.
- **Batch Size 8**: ~520 MB (Safety margin for gradients/activations: ~2-4 GB total).
- **Recommendation**: Safe to run on 8GB+ GPUs with Batch size 8-16.

### Recommended Next Actions (Step 5 Action Plan)
1.  **Update `TradeDatasetZarr`**: Add `return_y: bool` toggle to `TradeDatasetConfig`. If true, pull `base_trade[t_y]` from Zarr to use as the target.
2.  **Harmonize Spec**: Update `docs/spec_trade_ingestion_v1.1.md` to reflect the actual 2+16 channel structure and separation of masks into distinct arrays.
3.  **Implement Integration Test**:
    ```python
    # Placeholder for tools/trade_step5_integration_test.py
    module = TradeDataModule(cfg)
    model = TradeEncoderGNN(in_channels=54) # Example
    batch = next(iter(module.train_dataloader()))
    out = model(batch["base_trade"], batch["risk_trade"])
    loss = mse_loss(out, batch["y"])
    loss.backward()
    ```
4.  **Define Loss Scenarios**: Decide if Step 5 should predict the **total base trade** or **group-specific risk flows**.
