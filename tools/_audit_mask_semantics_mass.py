import json
from pathlib import Path
import numpy as np
import zarr

OUT = Path('docs/reports/_mask_semantics_audit.json')

p1 = 'data/processed/trade/imf_imts_step1/trade_fob_tensor.zarr'
p2 = 'data/processed/trade/faostat_step2/trade_risk_tensor.zarr'

z1 = zarr.open(p1, mode='r')
z2 = zarr.open(p2, mode='r')

# Sample a set of times including late period where data exists
sample_t = [0, 100, 300, 600, 815, 900]

report = {'step1': {}, 'step2': {}}

def summarize_step1(t):
    trade = z1['trade'][t]          # (N,N,2)
    mask = z1['mask'][t].astype(np.int64)  # (N,N,2)
    out = {}
    codes = sorted(list(set(mask.reshape(-1).tolist())))
    out['unique_codes'] = codes
    # trade mass by code
    mass_by = {}
    count_by = {}
    for c in codes:
        m = (mask == c)
        count_by[str(c)] = int(m.sum())
        mass_by[str(c)] = float(trade[m].sum())
    out['count_by_code'] = count_by
    out['mass_by_code'] = mass_by
    return out


def summarize_step2(t):
    trade = z2['trade_risk'][t]                 # (N,N,K,2)
    mask = z2['observed_mask'][t].astype(np.int64)  # (N,N,K,2)
    out = {}
    codes = sorted(list(set(mask.reshape(-1).tolist())))
    out['unique_codes'] = codes
    mass_by = {}
    count_by = {}
    for c in codes:
        m = (mask == c)
        count_by[str(c)] = int(m.sum())
        mass_by[str(c)] = float(trade[m].sum())
    out['count_by_code'] = count_by
    out['mass_by_code'] = mass_by
    return out

for t in sample_t:
    report['step1'][str(t)] = summarize_step1(t)
    report['step2'][str(t)] = summarize_step2(t)

OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
print('WROTE', OUT)
