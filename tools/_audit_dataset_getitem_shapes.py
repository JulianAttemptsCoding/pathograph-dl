import json
from pathlib import Path
import numpy as np
import sys

# Ensure repo root import
sys.path.insert(0, '.')

from pathograph.data.trade_dataset import TradeDatasetConfig, TradeDatasetZarr

OUT = Path('docs/reports/_dataset_getitem_shapes.json')

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
item = ds[0]
shapes = {}
for k, v in item.items():
    if hasattr(v, 'shape'):
        shapes[k] = {'shape': list(v.shape), 'dtype': str(v.dtype)}
    else:
        shapes[k] = {'type': type(v).__name__, 'value': str(v)}

OUT.write_text(json.dumps({'len': len(ds), 't_start': int(ds.t_start), 't_end': int(ds.t_end), 'getitem0': shapes}, indent=2), encoding='utf-8')
print('WROTE', OUT)
