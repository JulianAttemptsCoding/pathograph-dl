import os
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pathograph.pl.stmm_pl_module import STMMPLModule


class DummyModel(nn.Module):
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Just return zeros of shape (B, N, P)
        B = batch["t"].shape[0]
        return torch.zeros(B, 194, 8)


class DummyDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {
            "t": torch.tensor(idx),
            "t_y": torch.tensor(idx + 1), # month_id essentially
            "y_next": torch.zeros(194, 8),
            "y_mask": torch.ones(194, 8),
            # Add other required keys used by STMM (even if unused by DummyModel)
            "base_trade": torch.zeros(24, 194, 194, 2),
            "risk_trade": torch.zeros(24, 194, 194, 8, 2),
            "climate": torch.zeros(24, 194, 10),
            "climate_anoms": torch.zeros(24, 194, 10),
            "distance_km": torch.zeros(194, 194),
            "adjacency_border": torch.zeros(194, 194),
        }

def test_eval_prediction_dump_contract(tmp_path: Path):
    """
    Test that STMM Lightning Module writes test_predictions.parquet
    with correct schema and row counts.
    """
    model = DummyModel()
    pl_module = STMMPLModule(model=model, num_pathogens=8)

    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=2)

    logger = pl.loggers.CSVLogger(save_dir=str(tmp_path), name="eval_logs", version=0)
    trainer = pl.Trainer(
        default_root_dir=str(tmp_path),
        accelerator="cpu",
        devices=1,
        logger=logger,
        enable_checkpointing=False,
    )

    trainer.test(pl_module, dataloaders=dataloader)

    # 1. Assert file exists
    log_dir = Path(logger.log_dir)
    parquet_path = log_dir / "predictions_test.parquet"
    assert parquet_path.exists(), f"Expected prediction dump at {parquet_path}"

    # 2. Assert schema
    df = pd.read_parquet(parquet_path)
    expected_cols = {"month_id", "country_id", "pathogen_id", "y_true", "y_prob", "mask", "split"}
    assert set(df.columns) == expected_cols, f"Schema mismatch. Got {df.columns}"

    # 3. Assert row count equals (B * N * P) where B=2, N=194, P=8
    assert len(df) == 2 * 194 * 8, f"Row count mismatch. Got {len(df)}"

if __name__ == "__main__":
    pytest.main([__file__])
