import zarr
import json
from pathlib import Path

ZARR_PATH = Path('data/processed/pathogen/status_tensor.zarr')
assert ZARR_PATH.exists(), f'Missing {ZARR_PATH}'

z = zarr.open(str(ZARR_PATH), mode='r')
status = z['status'][:]          # (T, N, P)
mask = z['status_mask'][:]       # (T, N, P)

splits = {
    'train': (0, 815),
    'val':   (816, 851),
    'test':  (852, 907),
}

P = status.shape[2]

results = {}

for name, (start, end) in splits.items():
    positives = ((status[start:end+1] > 0) & (mask[start:end+1] > 0)).sum(axis=(0,1))
    observed = (mask[start:end+1] > 0).sum(axis=(0,1))
    results[name] = []
    print(f"\n{name.upper()} split (t={start}–{end})")
    for p in range(P):
        prev = float(positives[p]) / observed[p] if observed[p] > 0 else 0.0
        results[name].append({
            'pathogen': p,
            'positives': int(positives[p]),
            'observed': int(observed[p]),
            'prevalence': prev
        })
        print(f"  Pathogen {p}: {int(positives[p])}/{int(observed[p])} = {prev:.6f}")

out = Path('docs/pathogen_split_prevalence.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2))
print(f"\n[OK] Wrote prevalence audit to {out}")
