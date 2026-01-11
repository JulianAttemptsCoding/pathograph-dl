# Trade Step 3 (Model Integration Packaging) - Discovery Report

## 1. Artifact Inventory

### Step 1: IMTS Trade Preprocessing
- **Canonical Manifest**: `data/processed/trade/imf_imts_step1/manifest.json`
- **Main Tensor**: `data/processed/trade/imf_imts_step1/trade_fob_tensor.zarr`
- **Node Index**: `data/processed/meta/node_index.csv` (194 nodes)

### Step 2: FAOSTAT Risk Attribution
- **Canonical Manifest**: `data/processed/trade/faostat_step2/preprocessing_manifest.json`
- **Risk Tensor**: `data/processed/trade/faostat_step2/trade_risk_tensor.zarr`
- **Weights Tensor**: `data/processed/trade/faostat_step2/weights_corridor_year.zarr`
- **QC Report**: `data/processed/trade/faostat_step2/qc_report.json`
- **Trace Samples**: `data/processed/trade/faostat_step2/trace_samples.jsonl`

## 2. Canonical Paths & Schemas

### IMTS Base Tensor (Step 1)
- **Path**: `data/processed/trade/imf_imts_step1/trade_fob_tensor.zarr`
- **Arrays**:
  - `trade`: `(908, 194, 194, 2)` | `float32` | Base trade values (USD)
  - `mask`: `(908, 194, 194, 2)` | `uint8` | 1 if observed/estimated, 0 if missing
  - `is_estimated`: `(908, 194, 194, 2)` | `uint8` | 1 if CIF-to-FOB estimated, 0 if direct FOB
  - `time_index`: `(908,)` | `int32` | Mapping `t` to months starting 1950-01 (index 0)

### FAOSTAT Risk Tensor (Step 2)
- **Path**: `data/processed/trade/faostat_step2/trade_risk_tensor.zarr`
- **Arrays**:
  - `trade_risk`: `(908, 194, 194, 8, 2)` | `float32` | Attributed trade (USD) per group
  - `observed_mask`: `(908, 194, 194, 8, 2)` | `uint8` | Coverage from base tensor
  - `is_estimated`: `(908, 194, 194, 8, 2)` | `uint8` | Inherited estimation flag
  - `time_index`: `(908,)` | `int32` | Identical 908 timestamps (1950-01 to 2025-08)

### FAOSTAT Weights Tensor
- **Path**: `data/processed/trade/faostat_step2/weights_corridor_year.zarr`
- **Arrays**:
  - `weights`: `(75, 194, 194, 8)` | `float32` | Share of corridor trade for each group
  - `weight_mask`: `(75, 194, 194)` | `uint8` | Presence of weights for that corridor/year

## 3. Time Axis & Node Alignment

- **T Alignment**: Both Zarr stores share exactly $T=908$ months.
- **Epoch**: $t=0$ is `1950-M01`. Max $t=907$ is `2025-M08`.
- **Node Alignment**: Both tensors use $N=194$ nodes.
- **Node Index**: `data/processed/meta/node_index.csv` defines the mapping.
  - Direction: `tensor[t, i, j, ...]` represents flow from node `i` (exporter) to node `j` (importer).
- **Channels**:
  - Index 0: `exports_fob_usd` (Reporter exports to Partner, direction reporter->partner)
  - Index 1: `imports_fob_best_usd` (Reporter imports from Partner, direction partner->reporter)
  - *Note*: `imports[j, i]` (channel 1) is effectively another observation of flow `i -> j`.

## 4. Mask Semantics

- **Step 1 `mask`**: Indicates that data was present in IMF DOTS.
- **Step 2 `observed_mask`**: Inherited from Step 1.
- **`is_estimated`**: Indicates if the value was derived from a CIF report (CIF / 1.06).

## 5. Existing Code & Config Contract

- **Data Loaders**: None found in the current codebase.
- **Models**: None found.
- **Configs**: `config/trade_ingest.yaml` exists but is focused on ingestion source parameters.
- **Contract**: Step 3 is an empty slate and should implement `TradeDataset` and `TradeDataModule`.

## 6. Gaps & Decisions Needed

1.  **Normalization Strategy**: Recommend `log1p(x)`.
2.  **Train/Val/Test Splits**: Temporal splitting is recommended.
3.  **Lookback/Horizon**: Need to be configurable in a new `config/trade_modeling.yaml`.
4.  **Device Training**: Artifacts are small enough (~170MB total) for local GPU memory.

## 7. JSON Summary Artifact

```json
{
  "base_zarr_path": "data/processed/trade/imf_imts_step1/trade_fob_tensor.zarr",
  "risk_zarr_path": "data/processed/trade/faostat_step2/trade_risk_tensor.zarr",
  "weights_zarr_path": "data/processed/trade/faostat_step2/weights_corridor_year.zarr",
  "node_index_path": "data/processed/meta/node_index.csv",
  "T": 908,
  "N": 194,
  "K": 8,
  "crop_groups": ["BANANA", "BRASSICA", "CASSAVA", "CITRUS", "OLIVE_GRAPE", "OTHER", "PRUNUS", "WHEAT"],
  "channels": ["exports_fob_usd", "imports_fob_best_usd"],
  "start_date": "1950-01-01",
  "end_date": "2025-08-01",
  "tensor_shapes": {
    "base": [908, 194, 194, 2],
    "risk": [908, 194, 194, 8, 2]
  }
}
```
