import sys
from pathlib import Path

import torch
import pytest
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

# Add repo root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from pathograph.train.trade_lightning_module import TradeBaselinePL


def _make_item(N: int = 5, C: int = 2, L: int = 4) -> dict:
    """Return a single dataset item (no batch dimension)."""
    base_trade = torch.randn(L, N, N, C)
    y_base = base_trade[-1].clone()  # persistence target matches last time step
    y_base_mask = torch.ones_like(y_base)
    return {
        "base_trade": base_trade,
        "y_base": y_base,
        "y_base_mask": y_base_mask,
    }


def test_val_loss_logging_and_checkpoint(tmp_path: Path):
    """
    Verify validation_step logs 'val_loss' and ModelCheckpoint can monitor it
    without crashing (save_on_train_epoch_end=False).
    """
    model = TradeBaselinePL(lr=1e-3)

    # Two distinct items (avoid shared dict/tensor refs)
    dataset = [_make_item(), _make_item()]
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0)

    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        save_on_train_epoch_end=False,
    )

    logger = CSVLogger(save_dir=str(tmp_path), name="logs")

    trainer = pl.Trainer(
        default_root_dir=str(tmp_path),
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        callbacks=[ckpt],
        enable_checkpointing=True,
        logger=logger,
        accelerator="cpu",
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    metrics = trainer.callback_metrics
    assert "val_loss" in metrics, "val_loss must be logged"
    assert "train_loss" in metrics, "train_loss must be logged"

    # Strong checkpoint assertions
    assert ckpt.best_model_path, "best_model_path should be set"
    assert Path(ckpt.best_model_path).exists(), f"Best checkpoint missing: {ckpt.best_model_path}"
