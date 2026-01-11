from __future__ import annotations

from typing import Dict, List

import numpy as np


def _as_torch(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(x)


def trade_collate_separate(batch: List[Dict[str, np.ndarray]]) -> Dict[str, "object"]:
    """Collate for TradeDatasetZarr(return_mode='separate').

    Contract:
      - Converts numpy arrays to torch tensors.
      - Ensures masked-out entries contribute 0 to the model by multiplying values by observed_mask.
    """
    import torch

    if not batch:
        return {}

    # Scalars / small vectors
    t = torch.tensor([int(b["t"]) for b in batch], dtype=torch.int32)
    t_y = torch.tensor([int(b["t_y"]) for b in batch], dtype=torch.int32)
    time_feat = torch.stack([_as_torch(b["time_feat"]).to(torch.float32) for b in batch], dim=0)  # (B,2)

    # Main tensors
    base_trade = torch.stack([_as_torch(b["base_trade"]).to(torch.float32) for b in batch], dim=0)  # (B,L,N,N,2)
    base_mask = torch.stack([_as_torch(b["base_mask"]).to(torch.uint8) for b in batch], dim=0)
    base_est = torch.stack([_as_torch(b["base_is_estimated"]).to(torch.uint8) for b in batch], dim=0)

    risk_trade = torch.stack([_as_torch(b["risk_trade"]).to(torch.float32) for b in batch], dim=0)  # (B,L,N,N,K,2)
    risk_mask = torch.stack([_as_torch(b["risk_mask"]).to(torch.uint8) for b in batch], dim=0)
    risk_est = torch.stack([_as_torch(b["risk_is_estimated"]).to(torch.uint8) for b in batch], dim=0)

    # Zero-out masked positions (broadcast-safe). Keep masks separate for the model.
    base_trade = base_trade * base_mask.to(torch.float32)
    risk_trade = risk_trade * risk_mask.to(torch.float32)

    return {
        "t": t,
        "t_y": t_y,
        "time_feat": time_feat,
        "base_trade": base_trade,
        "base_mask": base_mask,
        "base_is_estimated": base_est,
        "risk_trade": risk_trade,
        "risk_mask": risk_mask,
        "risk_is_estimated": risk_est,
    }
