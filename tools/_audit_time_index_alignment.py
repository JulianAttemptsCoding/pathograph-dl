import json
from pathlib import Path
import numpy as np
import zarr

OUT = Path('docs/reports/_time_index_audit.json')

p1 = 'data/processed/trade/imf_imts_step1/trade_fob_tensor.zarr'
p2 = 'data/processed/trade/faostat_step2/trade_risk_tensor.zarr'

z1 = zarr.open(p1, mode='r')
z2 = zarr.open(p2, mode='r')

ti1 = np.array(z1['time_index'][:], dtype=np.int64)
ti2 = np.array(z2['time_index'][:], dtype=np.int64)

same = bool(ti1.shape == ti2.shape and np.all(ti1 == ti2))

summary = {
  'step1_path': p1,
  'step2_path': p2,
  'T_step1': int(ti1.shape[0]),
  'T_step2': int(ti2.shape[0]),
  'equal_arrays': same,
  'step1_head': ti1[:10].tolist(),
  'step1_tail': ti1[-10:].tolist(),
  'step2_head': ti2[:10].tolist(),
  'step2_tail': ti2[-10:].tolist(),
}

# Epoch mapping sanity check: month_index = 12*(YYYY-1950)+(MM-1)
# We cannot infer YYYY/MM without a provided mapping, but we can check monotonic + step size.
diffs = np.diff(ti1)
summary['time_index_monotonic_non_decreasing'] = bool(np.all(diffs >= 0))
summary['time_index_step_values_sample'] = sorted(list(set(diffs[:200].tolist())))[:20]
summary['time_index_all_steps_are_1'] = bool(np.all(diffs == 1))

OUT.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('WROTE', OUT)
