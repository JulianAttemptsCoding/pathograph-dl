from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from torch.utils.data import DataLoader

from .trade_collate import trade_collate_separate
from .trade_dataset import TradeDatasetConfig, TradeDatasetZarr, TradeSplit


@dataclass(frozen=True)
class TradeDataModuleConfig:
    base_zarr_path: str
    risk_zarr_path: str
    scaler_json_path: str

    lookback: int = 24
    horizon: int = 1

    split_train: TradeSplit = TradeSplit(0, 815)
    split_val: TradeSplit = TradeSplit(816, 851)
    split_test: TradeSplit = TradeSplit(852, 907)

    apply_log1p: bool = True
    standardize: bool = True

    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2
    drop_last_train: bool = True
    
    # Target config
    return_targets: bool = False
    target_kind: Literal["base", "risk", "both"] = "base"
    include_target_masks: bool = True
    
    # Valid-index filtering
    require_target_observed: bool = False
    min_target_observed: int = 1
    require_target_observed_kind: Optional[Literal["base", "risk", "both"]] = None
    valid_t_cache_dir: Optional[str] = None


class TradeDataModule:
    """Training-time wrapper for the trade dataset.

    This is intentionally implemented without requiring Lightning.
    If you later adopt Lightning, you can trivially wrap/extend this.
    """

    def __init__(self, cfg: TradeDataModuleConfig):
        self.cfg = cfg
        self._train = None
        self._val = None
        self._test = None

    def setup(self) -> None:
        c = self.cfg
        base = dict(
            base_zarr_path=c.base_zarr_path,
            risk_zarr_path=c.risk_zarr_path,
            lookback=c.lookback,
            horizon=c.horizon,
            split_train=c.split_train,
            split_val=c.split_val,
            split_test=c.split_test,
            apply_log1p=c.apply_log1p,
            standardize=c.standardize,
            scaler_json_path=c.scaler_json_path,
            return_mode="separate",
            return_targets=c.return_targets,
            target_kind=c.target_kind,
            include_target_masks=c.include_target_masks,
            require_target_observed=c.require_target_observed,
            min_target_observed=c.min_target_observed,
            require_target_observed_kind=c.require_target_observed_kind,
            valid_t_cache_dir=c.valid_t_cache_dir,
        )
        self._train = TradeDatasetZarr(TradeDatasetConfig(**base, split="train"))
        self._val = TradeDatasetZarr(TradeDatasetConfig(**base, split="val"))
        self._test = TradeDatasetZarr(TradeDatasetConfig(**base, split="test"))

    @property
    def train_dataset(self):
        if self._train is None:
            self.setup()
        return self._train

    @property
    def val_dataset(self):
        if self._val is None:
            self.setup()
        return self._val

    @property
    def test_dataset(self):
        if self._test is None:
            self.setup()
        return self._test

    def train_dataloader(self) -> DataLoader:
        c = self.cfg
        return DataLoader(
            self.train_dataset,
            batch_size=c.batch_size,
            shuffle=True,
            num_workers=c.num_workers,
            pin_memory=c.pin_memory,
            persistent_workers=c.persistent_workers,
            prefetch_factor=c.prefetch_factor if c.num_workers > 0 else None,
            drop_last=c.drop_last_train,
            collate_fn=trade_collate_separate,
        )

    def val_dataloader(self) -> DataLoader:
        c = self.cfg
        return DataLoader(
            self.val_dataset,
            batch_size=c.batch_size,
            shuffle=False,
            num_workers=c.num_workers,
            pin_memory=c.pin_memory,
            persistent_workers=c.persistent_workers,
            prefetch_factor=c.prefetch_factor if c.num_workers > 0 else None,
            drop_last=False,
            collate_fn=trade_collate_separate,
        )

    def test_dataloader(self) -> DataLoader:
        c = self.cfg
        return DataLoader(
            self.test_dataset,
            batch_size=c.batch_size,
            shuffle=False,
            num_workers=c.num_workers,
            pin_memory=c.pin_memory,
            persistent_workers=c.persistent_workers,
            prefetch_factor=c.prefetch_factor if c.num_workers > 0 else None,
            drop_last=False,
            collate_fn=trade_collate_separate,
        )
