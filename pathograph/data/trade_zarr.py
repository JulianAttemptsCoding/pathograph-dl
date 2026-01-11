from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import zarr


@dataclass(frozen=True)
class TradeZarrSpec:
    base_zarr_path: str
    risk_zarr_path: str


@dataclass
class TradeZarrHandles:
    base: zarr.Group
    risk: zarr.Group

    # Base arrays
    base_trade: zarr.Array
    base_mask: zarr.Array
    base_is_estimated: zarr.Array
    base_time_index: zarr.Array

    # Risk arrays
    risk_trade: zarr.Array
    risk_mask: zarr.Array
    risk_is_estimated: zarr.Array
    risk_time_index: zarr.Array

    # Metadata
    T: int
    N: int
    K: int
    channels: List[str]
    groups: List[str]


def open_trade_zarr(base_zarr_path: str, risk_zarr_path: str) -> TradeZarrHandles:
    base = zarr.open(base_zarr_path, mode="r")
    risk = zarr.open(risk_zarr_path, mode="r")

    # Verified keys from discovery
    base_trade = base["trade"]
    base_mask = base["mask"]
    base_is_estimated = base["is_estimated"]
    base_time_index = base["time_index"]

    risk_trade = risk["trade_risk"]
    risk_mask = risk["observed_mask"]
    risk_is_estimated = risk["is_estimated"]
    risk_time_index = risk["time_index"]

    # Validate alignment
    t1 = base_time_index[:]
    t2 = risk_time_index[:]
    if t1.shape != t2.shape or not np.array_equal(t1, t2):
        raise ValueError("Base and risk time_index are not identical.")

    T = int(base_trade.shape[0])
    N = int(base_trade.shape[1])
    # risk has (T,N,N,K,C)
    K = int(risk_trade.shape[3])

    channels = list(risk.attrs.get("channels", ["exports_fob_usd", "imports_fob_best_usd"]))
    groups = list(risk.attrs.get("groups", []))

    # Basic shape sanity
    if base_trade.shape != (T, N, N, 2):
        raise ValueError(f"Unexpected base_trade shape: {base_trade.shape}")
    if risk_trade.shape[:3] != (T, N, N) or risk_trade.shape[-1] != 2:
        raise ValueError(f"Unexpected risk_trade shape: {risk_trade.shape}")

    return TradeZarrHandles(
        base=base,
        risk=risk,
        base_trade=base_trade,
        base_mask=base_mask,
        base_is_estimated=base_is_estimated,
        base_time_index=base_time_index,
        risk_trade=risk_trade,
        risk_mask=risk_mask,
        risk_is_estimated=risk_is_estimated,
        risk_time_index=risk_time_index,
        T=T,
        N=N,
        K=K,
        channels=channels,
        groups=groups,
    )
