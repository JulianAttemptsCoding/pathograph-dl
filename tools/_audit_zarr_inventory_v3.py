import json
from pathlib import Path
import zarr

OUT = Path('docs/reports/_zarr_inventory.json')

PATHS = {
  'step1': 'data/processed/trade/imf_imts_step1/trade_fob_tensor.zarr',
  'step2': 'data/processed/trade/faostat_step2/trade_risk_tensor.zarr',
}

def inv(path: str):
    root = zarr.open(path, mode='r')
    items = []
    for name, obj in root.members():
        # obj is Array or Group
        if hasattr(obj, 'shape'):
            items.append({
                'name': name,
                'shape': list(obj.shape),
                'dtype': str(obj.dtype),
                'ndim': int(len(obj.shape)),
            })
        else:
            items.append({'name': name, 'type': type(obj).__name__})
    return {'path': path, 'type': type(root).__name__, 'members': sorted(items, key=lambda x: x['name'])}

report = {'step1': inv(PATHS['step1']), 'step2': inv(PATHS['step2'])}
OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
print('WROTE', OUT)
