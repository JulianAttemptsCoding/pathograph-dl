from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np

from .trade_zarr import open_trade_zarr


SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class TradeSplit:
    t_min: int
    t_max: int


@dataclass(frozen=True)
class TradeDatasetConfig:
    base_zarr_path: str
    risk_zarr_path: str
    lookback: int = 24
    horizon: int = 1
    split: SplitName = "train"
    split_train: TradeSplit = TradeSplit(0, 815)
    split_val: TradeSplit = TradeSplit(816, 851)
    split_test: TradeSplit = TradeSplit(852, 907)
    apply_log1p: bool = True
    standardize: bool = True
    scaler_json_path: Optional[str] = None
    return_mode: Literal["separate", "concat"] = "separate"


def _month_of_year_from_t(t: int) -> int:
    # t=0 -> Jan (1)
    return (t % 12) + 1


def _sin_cos_month(m: int) -> Tuple[float, float]:
    # m in 1..12
    ang = 2.0 * np.pi * (m - 1) / 12.0
    return float(np.sin(ang)), float(np.cos(ang))


class TradeDatasetZarr:
    """Zarr-backed dataset for trade tensors.

    Returns windows ending at time t, predicting t+horizon.

    Notes:
      - Missingness is represented via mask arrays (uint8). Values for unobserved cells should be treated as 0.
      - is_estimated indicates CIF-derived import estimates (and derived risk estimates).
    """

    def __init__(self, cfg: TradeDatasetConfig):
        self.cfg = cfg
        self.h = open_trade_zarr(cfg.base_zarr_path, cfg.risk_zarr_path)

        # choose split
        split_map = {
            "train": cfg.split_train,
            "val": cfg.split_val,
            "test": cfg.split_test,
        }
        sp = split_map[cfg.split]

        # We need t such that [t-lookback+1 .. t] is valid AND t+horizon is valid within split bounds.
        self.t_start = max(sp.t_min + (cfg.lookback - 1), 0)
        self.t_end = min(sp.t_max - cfg.horizon, self.h.T - 1 - cfg.horizon)
        if self.t_end < self.t_start:
            raise ValueError(f"Split window too small for lookback/horizon. start={self.t_start}, end={self.t_end}")

        self._scaler = None
        if cfg.standardize:
            if not cfg.scaler_json_path:
                raise ValueError("standardize=True requires scaler_json_path")
            with open(cfg.scaler_json_path, "r", encoding="utf-8") as f:
                self._scaler = json.load(f)

    def __len__(self) -> int:
        return int(self.t_end - self.t_start + 1)

    def _apply_transforms(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # x is float32; mean/std are 1D per-feature
        if self.cfg.apply_log1p:
            x = np.log1p(np.maximum(x, 0.0))
        if self.cfg.standardize:
            # broadcast mean/std to x's last dim
            x = (x - mean) / (std + 1e-8)
        return x

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        t = self.t_start + idx
        L = self.cfg.lookback
        H = self.cfg.horizon

        t0 = t - (L - 1)
        t1 = t + 1  # slice end exclusive
        t_y = t + H

        # Load base window
        base_trade = self.h.base_trade[t0:t1, :, :, :]            # (L,N,N,2)
        base_mask = self.h.base_mask[t0:t1, :, :, :].astype(np.uint8)
        base_est = self.h.base_is_estimated[t0:t1, :, :, :].astype(np.uint8)

        # Load risk window
        risk_trade = self.h.risk_trade[t0:t1, :, :, :, :]         # (L,N,N,K,2)
        risk_mask = self.h.risk_mask[t0:t1, :, :, :, :].astype(np.uint8)
        risk_est = self.h.risk_is_estimated[t0:t1, :, :, :, :].astype(np.uint8)

        # Target time feature (month-of-year) for t_y
        m = _month_of_year_from_t(t_y)
        sin_m, cos_m = _sin_cos_month(m)
        time_feat = np.array([sin_m, cos_m], dtype=np.float32)

        # Apply transforms using scaler if requested
        if self.cfg.standardize:
            sc = self._scaler
            base_mean = np.array(sc["base"]["mean"], dtype=np.float32)  # (2,)
            base_std = np.array(sc["base"]["std"], dtype=np.float32)    # (2,)
            risk_mean = np.array(sc["risk"]["mean"], dtype=np.float32)  # (K*2,)
            risk_std = np.array(sc["risk"]["std"], dtype=np.float32)

            base_trade = self._apply_transforms(base_trade, base_mean, base_std)
            risk_flat = risk_trade.reshape((L, self.h.N, self.h.N, self.h.K * 2))
            risk_flat = self._apply_transforms(risk_flat, risk_mean, risk_std)
            risk_trade = risk_flat.reshape((L, self.h.N, self.h.N, self.h.K, 2))
        else:
            if self.cfg.apply_log1p:
                base_trade = np.log1p(np.maximum(base_trade, 0.0))
                risk_trade = np.log1p(np.maximum(risk_trade, 0.0))

        if self.cfg.return_mode == "separate":
            return {
                "t": np.int32(t),
                "t_y": np.int32(t_y),
                "time_feat": time_feat,                 # (2,)
                "base_trade": base_trade.astype(np.float32),
                "base_mask": base_mask,
                "base_is_estimated": base_est,
                "risk_trade": risk_trade.astype(np.float32),
                "risk_mask": risk_mask,
                "risk_is_estimated": risk_est,
            }

        # concat mode: flatten risk (K,2)->(K*2) and concatenate features along last dim
        L_, N = L, self.h.N
        risk_val = risk_trade.reshape((L_, N, N, self.h.K * 2))
        risk_m = risk_mask.reshape((L_, N, N, self.h.K * 2))
        risk_e = risk_est.reshape((L_, N, N, self.h.K * 2))

        base_val = base_trade
        base_m = base_mask
        base_e = base_est

        feat = np.concatenate([
            base_val,
            risk_val,
            base_m.astype(np.float32),
            risk_m.astype(np.float32),
            base_e.astype(np.float32),
            risk_e.astype(np.float32),
        ], axis=-1).astype(np.float32)

        return {
            "t": np.int32(t),
            "t_y": np.int32(t_y),
            "time_feat": time_feat,
            "edge_feat": feat,   # (L,N,N, 2 + 16 + 2 + 16 + 2 + 16 = 54)
        }
