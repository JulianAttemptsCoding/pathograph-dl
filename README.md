# PathoGraph-DL

**Spatio-Temporal Multi-Modal Graph Neural Network (ST-MM-GNN)** for global pathogen risk forecasting using trade, climate, and epidemiological data.

## Quick Start

```bash
# Install in editable mode
pip install -e .

# Run unit tests (works without local data — data-dependent tests skip cleanly)
pytest -q

# Build sdist (for Vertex AI packaging)
python -m build --sdist
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
