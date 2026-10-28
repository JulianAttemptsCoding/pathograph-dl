"""Score the persistence baseline on the Step-A test split.

Uses the same sample selection as TradeDataset (windows ending at t, target t+1, lookback 24, test
split 852..907, keep t only if the target month has at least one observed status) and the same masked
per-pathogen metrics as the model. Persistence predicts the last observed status in the window.

Also reports how many test targets are *new onsets* (observed 1 where persistence predicted 0): those
are the only cases where a forecaster can beat persistence.

    python tools/persistence_test_eval.py --pathogen_zarr data/processed/pathogen/status_tensor.zarr
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent))
from pathograph.metrics.masked_classification import macro_nanmean, per_pathogen_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pathogen_zarr", default="data/processed/pathogen/status_tensor.zarr")
    ap.add_argument("--split", nargs=2, type=int, default=[852, 907])
    ap.add_argument("--lookback", type=int, default=24)
    args = ap.parse_args()

    z = zarr.open(args.pathogen_zarr, mode="r")
    status = np.asarray(z["status"][:]).astype(np.float32)        # (T, N, P)
    mask = np.asarray(z["status_mask"][:]).astype(np.float32)
    names = [str(x) for x in z["pathogen_names"][:]]
    T = status.shape[0]

    t_min, t_max = args.split
    t_start, t_end = t_min + args.lookback - 1, min(t_max - 1, T - 2)
    ts = [t for t in range(t_start, t_end + 1) if np.count_nonzero(mask[t + 1]) >= 1]

    preds, ys, ms = [], [], []
    for t in ts:
        win_s, win_m = status[t - args.lookback + 1 : t + 1], mask[t - args.lookback + 1 : t + 1]
        # last observed value per (node, pathogen); 0 if never observed in the window
        last = np.zeros(status.shape[1:], dtype=np.float32)
        for k in range(args.lookback):
            last = np.where(win_m[k] > 0.5, win_s[k], last)
        preds.append(last)
        ys.append(status[t + 1])
        ms.append(mask[t + 1])
    preds, ys, ms = (torch.tensor(np.stack(a)) for a in (preds, ys, ms))

    m = per_pathogen_metrics(preds, ys, ms)
    auroc, n_auroc = macro_nanmean(m["auroc"])
    auprc, n_auprc = macro_nanmean(m["auprc"])
    obs = ms > 0.5
    brier = float(((preds - ys)[obs] ** 2).mean())
    onsets = int(((ys > 0.5) & (preds < 0.5) & obs).sum())
    positives = int(((ys > 0.5) & obs).sum())

    out = {
        "test_target_months": [ts[0] + 1, ts[-1] + 1],
        "n_windows": len(ts),
        "persistence_auroc_macro": auroc,
        "persistence_auprc_macro": auprc,
        "persistence_brier": brier,
        "pathogens_scored": n_auroc,
        "observed_positives": positives,
        "new_onsets": onsets,
        "per_pathogen": {
            n: {"auroc": float(a), "auprc": float(p), "pos": int(po), "valid": int(v)}
            for n, a, p, po, v in zip(names, m["auroc"], m["auprc"], m["pos"], m["valid"])
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
