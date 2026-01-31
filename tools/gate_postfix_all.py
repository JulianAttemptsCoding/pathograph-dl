from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, check=False)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/stmm_stepA.yaml')
    ap.add_argument('--run_persistence', action='store_true')
    ap.add_argument('--run_dir', default=None)
    args = ap.parse_args()

    # 1) Pathogen tensor invariants (inline gate logic)
    gate_py = Path('tools/_tmp_gate_pathogen.py')
    gate_py.parent.mkdir(parents=True, exist_ok=True)
    gate_py.write_text(
        """
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import zarr

ZARR_PATH = Path('data/processed/pathogen/status_tensor.zarr')

def main() -> int:
    if not ZARR_PATH.exists():
        raise FileNotFoundError(f'Missing: {ZARR_PATH}')

    g = zarr.open_group(str(ZARR_PATH), mode='r')
    for k in ['status','status_mask','time_index','pathogen_names']:
        if k not in g:
            raise KeyError(f"Missing array '{k}' in {ZARR_PATH}. Found: {list(g.array_keys())}")

    status = g['status'][:]          # (T,N,P)
    mask   = g['status_mask'][:]     # (T,N,P)

    T,N,P = status.shape
    if (T,N) != (908,194):
        raise ValueError(f'Unexpected shape: {(T,N,P)} expected (908,194,P)')

    if not np.isin(status,[0,1]).all():
        raise ValueError('status must be binary')
    if not np.isin(mask,[0,1]).all():
        raise ValueError('status_mask must be binary')

    viol = int(((status==1) & (mask==0)).sum())
    if viol != 0:
        raise ValueError(f'Invariant violated: status==1 implies mask==1, viol={viol}')

    obs = (mask==1)
    ones  = int(((status==1) & obs).sum())
    zeros = int(((status==0) & obs).sum())
    if ones == 0 or zeros == 0:
        raise ValueError(f'Degenerate labels under mask: ones={ones} zeros={zeros}')

    # Split totals
    splits = {'train':(0,815), 'val':(816,851), 'test':(852,907)}
    totals = {}
    for s,(a,b) in splits.items():
        s_status = status[a:b+1]
        s_mask   = mask[a:b+1]
        pos = int(((s_status==1) & (s_mask==1)).sum())
        totals[s]=pos
    if totals['val'] == 0:
        raise ValueError('Val split has ZERO positives total (across all pathogens).')
    if totals['test'] == 0:
        raise ValueError('Test split has ZERO positives total (across all pathogens).')

    out = Path('docs/_logs/pathogen_postfix_gate_summary.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'positives_total': totals}, indent=2))
    print('[OK] pathogen gate passed:', totals)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
""".lstrip(),
        encoding='utf-8'
    )

    try:
        run([sys.executable, str(gate_py)])
    finally:
        # best-effort cleanup
        try:
            gate_py.unlink(missing_ok=True)
        except Exception:
            pass

    # 2) Pytest
    run([sys.executable, '-m', 'pytest', '-q'])

    # 3) Optional persistence baseline runner
    if args.run_persistence:
        p = Path('tools/eval_persistence_baseline.py')
        if not p.exists():
            raise FileNotFoundError('tools/eval_persistence_baseline.py not found, but --run_persistence was set')
        cmd = [sys.executable, str(p), '--config', args.config]
        if args.run_dir:
            cmd += ['--run_dir', args.run_dir]
        run(cmd)

    print('[OK] ALL POSTFIX GATES PASSED')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
