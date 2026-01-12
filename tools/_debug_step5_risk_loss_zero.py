import sys
sys.path.insert(0, '.')

import torch

from pathograph.data.trade_dataset import TradeDatasetConfig, TradeDatasetZarr
from pathograph.data.trade_collate import trade_collate_separate
from pathograph.train.trade_losses import masked_mse

cfg = TradeDatasetConfig(
    base_zarr_path='data/processed/trade/imf_imts_step1/trade_fob_tensor.zarr',
    risk_zarr_path='data/processed/trade/faostat_step2/trade_risk_tensor.zarr',
    lookback=24,
    horizon=1,
    split='train',
    standardize=False,
    apply_log1p=False,
    return_mode='separate',
    return_targets=True,
    target_kind='both',
    include_target_masks=True,
)

ds = TradeDatasetZarr(cfg)
# Take a small batch
items = [ds[i] for i in range(2)]
batch = trade_collate_separate(items)

# Predictions: persistence (use last step)
base_x_last = batch['base_trade'][:, -1]  # (B,N,N,2)
risk_x_last = batch['risk_trade'][:, -1]  # (B,N,N,K,2)

# Simple persistence: pred = last_observation
base_pred = base_x_last
risk_pred = risk_x_last

# Pull targets + masks
base_y = batch['y_base']
base_mk = batch['y_base_mask']
risk_y = batch['y_risk']
risk_mk = batch['y_risk_mask']

# Diagnostics
print('base_y shape', tuple(base_y.shape), 'risk_y shape', tuple(risk_y.shape))
print('base_mask sum', int(base_mk.sum().item()), 'risk_mask sum', int(risk_mk.sum().item()))
print('base_y abs sum', float(base_y.abs().sum().item()), 'risk_y abs sum', float(risk_y.abs().sum().item()))
print('base_pred abs sum', float(base_pred.abs().sum().item()), 'risk_pred abs sum', float(risk_pred.abs().sum().item()))

# Residual stats restricted to observed entries
base_res = (base_pred - base_y).abs()
risk_res = (risk_pred - risk_y).abs()

base_obs = base_res[base_mk.to(torch.bool)]
risk_obs = risk_res[risk_mk.to(torch.bool)]

print('base_obs count', int(base_obs.numel()), 'risk_obs count', int(risk_obs.numel()))
print('base_obs mean', float(base_obs.mean().item()) if base_obs.numel() else None)
print('risk_obs mean', float(risk_obs.mean().item()) if risk_obs.numel() else None)

base_loss = masked_mse(base_pred, base_y, base_mk)
risk_loss = masked_mse(risk_pred, risk_y, risk_mk)
print('base_loss', float(base_loss.item()), 'risk_loss', float(risk_loss.item()))
