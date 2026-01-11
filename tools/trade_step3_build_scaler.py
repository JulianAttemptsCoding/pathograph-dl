from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from pathograph.data.trade_zarr import open_trade_zarr


@dataclass(frozen=True)
class Split:
    t_min: int
    t_max: int


def _masked_welford_update(sum_x, sum_x2, count, x: np.ndarray, m: np.ndarray):
    # x: (...,F), m: same but m in {0,1}
    x = x.reshape((-1, x.shape[-1]))
    m = m.reshape((-1, m.shape[-1])).astype(np.float64)
    x = x.astype(np.float64)

    sum_x += (x * m).sum(axis=0)
    sum_x2 += ((x * x) * m).sum(axis=0)
    count += m.sum(axis=0)
    return sum_x, sum_x2, count


def compute_scaler(base_zarr: str, risk_zarr: str, out_json: str, train: Split, chunk_t: int = 8):
    h = open_trade_zarr(base_zarr, risk_zarr)

    # Accumulators
    sum_base = np.zeros((2,), dtype=np.float64)
    sum2_base = np.zeros((2,), dtype=np.float64)
    cnt_base = np.zeros((2,), dtype=np.float64)

    sum_risk = np.zeros((h.K * 2,), dtype=np.float64)
    sum2_risk = np.zeros((h.K * 2,), dtype=np.float64)
    cnt_risk = np.zeros((h.K * 2,), dtype=np.float64)

    t_min = max(train.t_min, 0)
    t_max = min(train.t_max, h.T - 1)

    for t0 in range(t_min, t_max + 1, chunk_t):
        t1 = min(t0 + chunk_t, t_max + 1)

        base = h.base_trade[t0:t1, :, :, :]
        bm = h.base_mask[t0:t1, :, :, :]

        risk = h.risk_trade[t0:t1, :, :, :, :]
        rm = h.risk_mask[t0:t1, :, :, :, :]

        # log1p transform (recommended) before computing stats
        base = np.log1p(np.maximum(base, 0.0))
        risk = np.log1p(np.maximum(risk, 0.0))

        risk = risk.reshape((risk.shape[0], risk.shape[1], risk.shape[2], h.K * 2))
        rm = rm.reshape((rm.shape[0], rm.shape[1], rm.shape[2], h.K * 2))

        sum_base, sum2_base, cnt_base = _masked_welford_update(sum_base, sum2_base, cnt_base, base, bm)
        sum_risk, sum2_risk, cnt_risk = _masked_welford_update(sum_risk, sum2_risk, cnt_risk, risk, rm)

    mean_base = sum_base / np.maximum(cnt_base, 1.0)
    var_base = (sum2_base / np.maximum(cnt_base, 1.0)) - (mean_base * mean_base)
    std_base = np.sqrt(np.maximum(var_base, 1e-12))

    mean_risk = sum_risk / np.maximum(cnt_risk, 1.0)
    var_risk = (sum2_risk / np.maximum(cnt_risk, 1.0)) - (mean_risk * mean_risk)
    std_risk = np.sqrt(np.maximum(var_risk, 1e-12))

    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    payload = {
        "version": "1.0",
        "note": "Stats computed on log1p(values) using observed-only masks and train-only time indices.",
        "train": {"t_min": int(t_min), "t_max": int(t_max)},
        "base": {"mean": mean_base.tolist(), "std": std_base.tolist(), "count": cnt_base.tolist()},
        "risk": {"mean": mean_risk.tolist(), "std": std_risk.tolist(), "count": cnt_risk.tolist()},
        "K": int(h.K),
        "channels": h.channels,
        "groups": h.groups,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("WROTE", out_json)


if __name__ == "__main__":
    base = "data/processed/trade/imf_imts_step1/trade_fob_tensor.zarr"
    risk = "data/processed/trade/faostat_step2/trade_risk_tensor.zarr"
    out = "data/processed/trade/trade_step3_scaler.json"

    # Default split (train: 1950-01..2017-12)
    train = Split(0, 815)
    compute_scaler(base, risk, out, train, chunk_t=8)
