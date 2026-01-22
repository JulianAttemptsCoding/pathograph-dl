# PathoGraph ST-MM-GNN Architecture & Repository Compatibility Audit

**Date**: 2026-01-16  
**Auditor**: IDE Agent  
**Status**: COMPLETE

---

## 1. Executive Verdict

**VERDICT: COMPATIBLE WITH GAPS**

The PathoGraph-DL repository has robust preprocessing pipelines and clean data artifacts that are **fully compatible** with the proposed ST-MM-GNN architecture. However, the **model layer is embryonic**—only a simple persistence baseline exists. Significant implementation work is required to realize the target multi-modal spatiotemporal GNN.

### Key Findings

| Finding | Status | Criticality |
|---------|--------|-------------|
| Preprocessing artifacts aligned | ✅ PASS | — |
| Temporal split logic exists | ✅ PASS | — |
| Geographic split logic | ❌ MISSING | HIGH |
| Combined temporal+geographic split | ❌ MISSING | HIGH |
| Multimodal model | ❌ MISSING | HIGH |
| Spatial GNN operator | ❌ MISSING | HIGH |
| Temporal operator | ❌ MISSING | MEDIUM |
| FiLM fusion | ❌ MISSING | MEDIUM |
| Multi-pathogen heads | ❌ MISSING | HIGH |
| Calibration pipeline | ❌ MISSING | MEDIUM |
| Focal loss | ❌ MISSING | LOW |
| Data leakage in current splits | ⚠️ LOW RISK | — |

> **UNSUPPORTED**: No empirical evidence yet exists in this codebase for failure mode claims. All architecture recommendations below are derived from cited literature.

---

## 2. What Exists in Repo Today

### 2.1 Repository Structure

```
PathoGraph-DL/
├── config/              # 18 YAML configs (trade, climate, pathogen steps)
├── data/processed/      # All preprocessed artifacts
├── docs/                # Reports, specs, QC docs
├── pathograph/
│   ├── data/            # TradeDataset, TradeDataModule, collate
│   ├── models/          # PersistenceBaseline only
│   └── train/           # Lightning module, MSE loss
├── tests/               # 57+ unit tests
└── tools/               # 50+ CLI tools for preprocessing
```

### 2.2 Entrypoints

| Purpose | File | Notes |
|---------|------|-------|
| Training | [trade_step6_train_entrypoint.py](file:///c:/Users/bubga.JULIAN-LAPTOPE2/PycharmProjects/PathoGraph-DL/tools/trade_step6_train_entrypoint.py) | PyTorch Lightning, trade-only |
| Preprocessing Verifier | [verify_preprocessing_complete.py](file:///c:/Users/bubga.JULIAN-LAPTOPE2/PycharmProjects/PathoGraph-DL/tools/verify_preprocessing_complete.py) | Checks all modalities |
| Trade Step 7 Run | [trade_step7_run_baseline.py](file:///c:/Users/bubga.JULIAN-LAPTOPE2/PycharmProjects/PathoGraph-DL/tools/trade_step7_run_baseline.py) | Baseline training run |

### 2.3 Tests

- **57 tests collected** (4 require torch, skipped in collection due to env)
- Test coverage: trade steps 1-7, climate config/tensor contract, meta spatial matrices, preprocessing verifier
- Tests exist at [tests/](file:///c:/Users/bubga.JULIAN-LAPTOPE2/PycharmProjects/PathoGraph-DL/tests/)

---

## 3. Batch Contracts and Tensor Shapes

### 3.1 Master Dimensions (from [preprocessing_acceptance_report.json](file:///c:/Users/bubga.JULIAN-LAPTOPE2/PycharmProjects/PathoGraph-DL/data/processed/preprocessing_acceptance_report.json))

| Axis | Value | Description |
|------|-------|-------------|
| T | 908 | Time steps (Jan 1950 – Aug 2025) |
| N | 194 | Countries (node_index) |
| K | 8 | Risk crop products (FAOSTAT groups) |
| F | 10 | Climate features |
| P | 8 | Pathogens |

### 3.2 Trade Batch (from [trade_dataset.py](file:///c:/Users/bubga.JULIAN-LAPTOPE2/PycharmProjects/PathoGraph-DL/pathograph/data/trade_dataset.py))

| Key | Shape | Dtype | Notes |
|-----|-------|-------|-------|
| `t` | scalar | int32 | Window end time index |
| `t_y` | scalar | int32 | Target time index (t + horizon) |
| `time_feat` | (2,) | float32 | sin/cos month encoding |
| `base_trade` | (L, N, N, 2) | float32 | Exports_FOB, Imports_FOB |
| `base_mask` | (L, N, N) | uint8 | 1=observed, 0=missing |
| `base_is_estimated` | (L, N, N) | uint8 | CIF-derived flag |
| `risk_trade` | (L, N, N, K, 2) | float32 | Per-crop risk flows |
| `risk_mask` | (L, N, N, K) | uint8 | 1=observed |
| `risk_is_estimated` | (L, N, N, K) | uint8 | CIF-derived flag |
| `y_base` | (N, N, 2) | float32 | Base target (if return_targets) |
| `y_base_mask` | (N, N) | uint8 | Target mask |
| `y_risk` | (N, N, K, 2) | float32 | Risk target (if return_targets) |
| `y_risk_mask` | (N, N, K) | uint8 | Risk target mask |

**Default lookback L=24, horizon H=1.**

### 3.3 Climate Tensor (from [climate_tensor.zarr](file:///c:/Users/bubga.JULIAN-LAPTOPE2/PycharmProjects/PathoGraph-DL/data/processed/climate/climate_tensor.zarr))

| Array | Shape | Notes |
|-------|-------|-------|
| `climate` | (908, 194, 10) | 10 features (t2m, d2m, sp, msl, u10, v10, wind_speed, rh, vpd, tp) |
| `mask` | (908, 194, 10) | Observation mask |
| `time_index` | (908,) | Aligned to master |

### 3.4 Climate Anomalies (from [climate_anomalies.zarr](file:///c:/Users/bubga.JULIAN-LAPTOPE2/PycharmProjects/PathoGraph-DL/data/processed/climate/climate_step4/climate_anomalies.zarr))

| Array | Shape |
|-------|-------|
| `anomaly` | (908, 194, 10) |
| `zscore` | (908, 194, 10) |
| `mask` | (908, 194, 10) |

### 3.5 Pathogen Status (from [status_tensor.zarr](file:///c:/Users/bubga.JULIAN-LAPTOPE2/PycharmProjects/PathoGraph-DL/data/processed/pathogen/status_tensor.zarr))

| Array | Shape | Notes |
|-------|-------|-------|
| `status` | (908, 194, 8) | Binary presence/endemic |
| `status_mask` or `mask` | (908, 194, 8) | Observation mask |

**Pathogens (from [pathogen_step1.yaml](file:///c:/Users/bubga.JULIAN-LAPTOPE2/PycharmProjects/PathoGraph-DL/config/pathogen_step1.yaml))**: BBTD, Cassava, CitrusGreening, Clubroot, PPV, TR4, WheatBlast, XylellaFastidiosa

**Label policy**: Forward-filled (monotone once observed = 1).

### 3.6 Meta Spatial (from [data/processed/meta/](file:///c:/Users/bubga.JULIAN-LAPTOPE2/PycharmProjects/PathoGraph-DL/data/processed/meta))

| File | Shape | Notes |
|------|-------|-------|
| `distance_km.npy` | (194, 194) | Centroid distances, diagonal ≈0 |
| `adjacency_border.npy` | (194, 194) | Border adjacency {0,1} |
| `time_index_master.npy` | (908,) | Month indices 0-907 |
| `node_index.csv` | 194 rows | ISO3 codes |
| `node_geometry.gpkg` | 194 features | Country polygons |

---

## 4. Split/Leakage Audit

### 4.1 Current Split Logic (Temporal Only)

From [trade_step6.yaml](file:///c:/Users/bubga.JULIAN-LAPTOPE2/PycharmProjects/PathoGraph-DL/config/trade_step6.yaml):

| Split | t_min | t_max | Date Range |
|-------|-------|-------|------------|
| train | 0 | 815 | Jan 1950 – Dec 2017 |
| val | 816 | 851 | Jan 2018 – Dec 2020 |
| test | 852 | 907 | Jan 2021 – Aug 2025 |

**Temporal leakage check**: ✅ PASS. Splits are strictly sequential; no future months leak into train.

### 4.2 Geographic Split Logic

❌ **MISSING**: No geographic holdout mechanism exists.

- `TradeDatasetConfig` supports only temporal t_min/t_max splits
- No country-level masking or held-out node lists
- Node dimension (N=194) is always fully included

### 4.3 Combined Temporal + Geographic Split

❌ **MISSING**: No combined split infrastructure.

### 4.4 Scaler/Normalizer Leakage

From [trade_step3_scaler.json](file:///c:/Users/bubga.JULIAN-LAPTOPE2/PycharmProjects/PathoGraph-DL/data/processed/trade/trade_step3_scaler.json):

⚠️ **POTENTIAL ISSUE**: Scaler is fit on the **entire trade tensor**, not train-only. This could leak future statistics.

**Recommendation**: Recompute scaler using only t ∈ [0, 815] (train range).

### 4.5 Adaptive Adjacency Leakage

N/A — no learned adjacency exists yet.

---

## 5. Module-by-Module Architecture Mapping

### 5.1 Gap Matrix

| Target Module | Status | File Path | Notes |
|---------------|--------|-----------|-------|
| **Spatial GNN operator** | ❌ MISSING | — | Need DCRNN/STGCN/Graph WaveNet-style conv (CIT: DCRNN_LI_YU_2017, STGCN_YU_YIN_ZHU_2018, GRAPH_WAVENET_WU_2019) |
| **Temporal operator** | ❌ MISSING | — | Need TCN/GRU for (B,L,N,d)→(B,N,d') (CIT: STGCN_YU_YIN_ZHU_2018, GRAPH_WAVENET_WU_2019) |
| **Multi-modal fusion** | ❌ MISSING | — | Recommend FiLM conditioning (CIT: GNN_FiLM_BROCKSCHMIDT_2020) |
| **Multi-pathogen heads** | ❌ MISSING | — | Need 8 parallel heads with uniform loss weighting |
| **Current model** | ⚠️ BASELINE | [trade_baseline.py](file:///c:/Users/bubga.JULIAN-LAPTOPE2/PycharmProjects/PathoGraph-DL/pathograph/models/trade_baseline.py) | PersistenceBaseline: y_{t+H} = x_t (no learning) |
| **Current loss** | ⚠️ MSE only | [trade_losses.py](file:///c:/Users/bubga.JULIAN-LAPTOPE2/PycharmProjects/PathoGraph-DL/pathograph/train/trade_losses.py) | `masked_mse` over trade; no pathogen loss |
| **Calibration** | ❌ MISSING | — | Need temperature scaling post-hoc (CIT: CALIBRATION_GUO_2017) |
| **Focal loss** | ❌ MISSING | — | Optional for class imbalance (CIT: FOCAL_LOSS_LIN_2017) |
| **MoE (optional)** | ❌ MISSING | — | ST-MoGE-style mixture of graph experts (CIT: ST_MoGE_WU_2024) |
| **Learned adjacency** | ❌ MISSING | — | Graph WaveNet/MTGNN adaptive A (CIT: GRAPH_WAVENET_WU_2019, MTGNN_WU_KDD_2020) |
| **Hawkes/event kernel** | ❌ MISSING | — | UNSUPPORTED: Forward-filled labels may not suit event supervision (CIT: ASTROLOGER_DU_2021, TPP_SURVEY_ZHOU_2025) |

### 5.2 Data Loaders

| Modality | Status | File |
|----------|--------|------|
| Trade (base + risk) | ✅ EXISTS | [trade_dataset.py](file:///c:/Users/bubga.JULIAN-LAPTOPE2/PycharmProjects/PathoGraph-DL/pathograph/data/trade_dataset.py) |
| Climate | ❌ MISSING loader | Tensor exists, no Dataset class |
| Climate anomalies | ❌ MISSING loader | Tensor exists, no Dataset class |
| Pathogen | ❌ MISSING loader | Tensor exists, no Dataset class |
| Meta (static) | ⚠️ MANUAL | Loaded as numpy, not integrated into batch |

---

## 6. Failure Modes and Hardening Recommendations

### 6.1 Leakage Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Scaler fit on full data | MEDIUM | Recompute scaler on train split only |
| No geographic holdout | HIGH | Implement country_mask in dataset config |
| Future climate in train | LOW | Climate is input, not label; acceptable if used properly |

### 6.2 Class Imbalance

Pathogen status is highly imbalanced:
- Most countries × most months = 0 (no pathogen)
- Positive class is sparse (first observations are rare events)

**Recommendation**: 
1. Compute per-pathogen positive rate by decade (CIT: required analysis)
2. If any pathogen has <5% positive rate, consider focal loss (CIT: FOCAL_LOSS_LIN_2017).  But maintain **uniform weighting across pathogens** per non-negotiable.

### 6.3 Calibration

No calibration infrastructure exists.

**Recommendation**: Add post-hoc temperature scaling on validation set (CIT: CALIBRATION_GUO_2017). Minimal implementation:

```python
# Proposed: pathograph/train/calibration.py
class TemperatureScaling:
    def __init__(self): self.temperature = 1.0
    def fit(self, logits, labels): ...  # Optimize T on val
    def calibrate(self, logits): return logits / self.temperature
```

### 6.4 Loss Mask Compliance

Current `masked_mse` respects observation masks. ✅

For pathogen prediction:
- Must implement `masked_bce` or `masked_focal_loss`
- Must respect `status_mask` for label validity

### 6.5 Missing Modality Handling

No explicit handling for missing climate/trade data in forward pass.

**Recommendation**: Implement mask-aware encoders that zero out unobserved values before propagation (already done for trade; needs extension to climate).

---

## 7. Minimal Patch Plan

### 7.1 Priority Order

1. **P0: Geographic split infrastructure** — Critical for proper evaluation
2. **P1: Multi-modal dataloader** — Combine trade + climate + pathogen in single batch
3. **P2: Pathogen prediction heads** — Add classification heads for P=8 pathogens
4. **P3: Spatial GNN operator** — Implement at least one graph conv (e.g., DiffConv)
5. **P4: Temporal operator** — TCN or GRU backbone
6. **P5: Calibration** — Temperature scaling post-hoc

### 7.2 Proposed File Changes

#### [NEW] `pathograph/data/multimodal_dataset.py`
- Combine `TradeDatasetZarr` logic with climate + pathogen loading
- Return unified batch dict with all modalities
- Accept `held_out_countries: List[str]` for geographic splits

#### [NEW] `pathograph/data/multimodal_datamodule.py`
- Wrapper for train/val/test with temporal + geographic splits
- Support combined holdout modes

#### [NEW] `pathograph/models/stmm_gnn.py`
- `STMMGNNBackbone`: shared spatial + temporal encoder
- `SpatialBlock`: DiffConv or ChebConv over adjacency
- `TemporalBlock`: TCN or GRU over time dimension
- `FiLMFusion`: optional conditioning layer

#### [NEW] `pathograph/models/pathogen_heads.py`
- `PathogenClassifier`: P parallel heads with sigmoid output
- Equal-weight BCE loss across pathogens

#### [MODIFY] `pathograph/train/trade_losses.py`
- Add `masked_bce`, `masked_focal_loss`
- Add `multi_pathogen_loss` with uniform weighting

#### [NEW] `pathograph/train/calibration.py`
- `TemperatureScaling` class

#### [MODIFY] `config/stmm_gnn.yaml` (new)
- Full config for multimodal training

### 7.3 Test Plan

| Test | Command | Coverage |
|------|---------|----------|
| Existing trade tests | `.venv\Scripts\python.exe -m pytest tests/test_trade*.py -v` | Trade contract |
| Existing preprocessing | `.venv\Scripts\python.exe -m pytest tests/test_verify_preprocessing*.py -v` | Artifact alignment |
| New multimodal dataset | `pytest tests/test_multimodal_dataset.py -v` | Batch shapes |
| New STMM-GNN smoke | `pytest tests/test_stmm_gnn_smoke.py -v` | Forward pass |
| Preprocessing verifier | `.venv\Scripts\python.exe tools/verify_preprocessing_complete.py --require-meta` | Full alignment |

---

## 8. Appendix: Command Logs

### 8.1 Environment

```
Python: 3.12.4
PyTorch: NOT INSTALLED in venv (tests require torch)
```

### 8.2 Test Collection

```
57 tests collected, 4 errors (torch import)
```

### 8.3 Preprocessing Acceptance (Previous Run)

```json
{
  "summary": {
    "all_time_indices_aligned": true,
    "N_nodes": 194,
    "T_timesteps": 908,
    "K_risk_products": 8,
    "F_climate_features": 10,
    "P_pathogens": 8,
    "spatial_matrices_present": true
  },
  "created_at": "2026-01-16T23:30:10Z"
}
```

---

## 9. Bibliography Reference (Inline Citations Used)

| ID | Reference |
|----|-----------|
| DCRNN_LI_YU_2017 | Li et al., "Diffusion Convolutional RNN" |
| STGCN_YU_YIN_ZHU_2018 | Yu et al., "Spatio-Temporal Graph Convolutional Networks" |
| GRAPH_WAVENET_WU_2019 | Wu et al., "Graph WaveNet for Deep Spatial-Temporal Forecasting" |
| MTGNN_WU_KDD_2020 | Wu et al., "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks" |
| GNN_FiLM_BROCKSCHMIDT_2020 | Brockschmidt, "GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation" |
| ST_MoGE_WU_2024 | Wu et al., "ST-MoGE: Mixture of Graph Experts for Robust Spatiotemporal Learning" |
| CALIBRATION_GUO_2017 | Guo et al., "On Calibration of Modern Neural Networks" |
| FOCAL_LOSS_LIN_2017 | Lin et al., "Focal Loss for Dense Object Detection" |
| ASTROLOGER_DU_2021 | Du et al., "Astrologer: Graph Neural Hawkes Process for Event Sequence Prediction" |
| TPP_SURVEY_ZHOU_2025 | Zhou et al., "Temporal Point Processes Survey" |

---

**END OF AUDIT**
