# PathoGraph-DL

**Spatio-Temporal Multi-Modal Graph Neural Network (ST-MM-GNN)** for global pathogen risk forecasting using trade, climate, and epidemiological data.

## For reviewers: what this is, and how to check it

**Question:** given the last 24 months of trade flows and climate anomalies for 194 countries, can a
graph neural network predict next month's presence status for 8 crop pathogens?

The 8 pathogens are BBTD, cassava disease, citrus greening, clubroot, plum pox (PPV), banana Fusarium
TR4, wheat blast and *Xylella fastidiosa*.

| | |
|---|---|
| **Model** | Graph WaveNet–style ([`pathograph/models/stmm_gwnet.py`](pathograph/models/stmm_gwnet.py)): gated dilated causal TCN blocks (dilations 1–16) interleaved with **diffusion graph convolutions** (K=2) over static country graphs (border adjacency, distance) plus a **learned adaptive adjacency** (10-d node embeddings, top-20 edges) |
| **Inputs per country-month** | 46 features: 4 IMF trade-flow channels, 32 FAOSTAT pathogen-risk-weighted trade channels, 10 climate anomalies |
| **Target** | monthly presence status per (country, pathogen), forward-filled once first reported |
| **Splits** | chronological on a monthly index from 1950-01: train ≤ 2017-12, validation 2018–2020, **test 2021-01 → 2025-08** ([`configs/stmm_stepA_adaptive.yaml`](configs/stmm_stepA_adaptive.yaml)) |
| **Calibration** | post-hoc temperature scaling, fitted on validation only ([`pathograph/calibration/`](pathograph/calibration/)) |
| **Training** | PyTorch Lightning on Vertex AI. Best checkpoint at epoch 8, validation macro AUROC 0.977 |

**Headline test metrics** (macro over pathogens, calibrated, across the Phase 3 runs; from
[`FINAL_RECORD.md`](FINAL_RECORD.md)):

| AUROC | AUPRC | Brier | ECE |
|---|---|---|---|
| **0.960** ± 0.010 | 0.785 ± 0.058 | 0.021 ± 0.003 | 0.020 ± 0.004 |

### Verify without the data (~1 minute)

```bash
pip install -e .
pytest -q          # 96 passed, 15 skipped: data-dependent tests skip cleanly without the zarr stores
```

The tests cover input/target time alignment, mask-correct metrics that are safe on degenerate splits,
the persistence baseline, temperature scaling, the data contracts between pipeline steps, and finite
forward/loss smoke runs of the model.

### Read these

1. [`FINAL_RECORD.md`](FINAL_RECORD.md): frozen champion package, config, checkpoint and test metrics.
2. [`docs/stmm_layerA_eval_gate.md`](docs/stmm_layerA_eval_gate.md): label-sanity checks, masked metrics,
   baseline comparison and calibration gate.
3. [`docs/audits/`](docs/audits/) and [`docs/decisions/`](docs/decisions/): audit trail and decision memos.

### Honest limitations

- **The data isn't in git.** The processed tensors (zarr) live in a GCS bucket, so a reviewer can run the
  tests but can't retrain or re-score without bucket access.
- **Persistence is the baseline that matters.** Status is forward-filled: once a pathogen is reported in a
  country, it stays 1. A "predict last month's status" model therefore already scores well on AUROC. The
  baseline is implemented in [`pathograph/baselines/persistence.py`](pathograph/baselines/persistence.py),
  but its score on the Phase 3 test split isn't recorded here. Until it is, 0.960 should be read as an
  upper bound on the value the model adds. A stricter target would be new-onset prediction, scoring only
  (country, pathogen) pairs not yet positive.

## Quick Start

```bash
# Install in editable mode
pip install -e .

# Run unit tests (works without local data — data-dependent tests skip cleanly)
pytest -q

# Build sdist (for Vertex AI packaging)
python -m build --sdist
```

## Quick Start: Staging Data

To stage the required v1 datasets locally from GCS (`gs://pathograph-057a2273fe-data/datasets/stepA/v1`):

```powershell
.\tools\stage_stepA_v1_from_gcs.ps1
```
## Project Structure

| Directory | Purpose |
|-----------|---------|
| `pathograph/` | Core package: models, data, metrics, calibration, Vertex entrypoints |
| `pathograph/vertex/` | Vertex AI entrypoints (`stepA_entry.py`, `stepA_eval_entry.py`) |
| `tools/` | Pipeline scripts (data ingestion, training, evaluation, auditing) |
| `config/` | YAML configs for pipeline steps |
| `configs/` | Vertex-specific configs (e.g. `stmm_stepA_adaptive.yaml`) |
| `tests/` | Unit + integration tests (32 test files) |
| `data/` | Raw and processed datasets (zarr stores not tracked in git) |
| `docs/` | Documentation, audits, and reports |

## GCS Reference URIs

| Resource | URI |
|----------|-----|
| GCS Bucket | `gs://pathograph-057a2273fe-data` |
| Packages | `gs://pathograph-057a2273fe-data/packages/` |
| Configs | `gs://pathograph-057a2273fe-data/configs/` |
| Phase 2 Best Run | `gs://pathograph-057a2273fe-data/runs/stepA/phase2/adaptive_s1338/` |
| Best Checkpoint | `gs://pathograph-057a2273fe-data/runs/stepA/phase2/adaptive_s1338/epoch=8-step=7128-val_auroc_macro=0.9771.ckpt` |

## Vertex AI Entrypoints

- **Training**: `python -m pathograph.vertex.stepA_entry`
- **Evaluation**: `python -m pathograph.vertex.stepA_eval_entry`
