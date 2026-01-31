from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
import zarr


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _month_index_from_date(dt: pd.Timestamp) -> int:
    # month_index where 1950-01 -> 0
    return (int(dt.year) - 1950) * 12 + (int(dt.month) - 1)


def _load_node_index(candidates: List[Path]) -> pd.DataFrame:
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            if 'node_id' not in df.columns or 'iso3' not in df.columns:
                raise ValueError(f"node_index at {p} missing required columns node_id, iso3; got {list(df.columns)}")
            if len(df) != 194:
                raise ValueError(f"node_index at {p} expected 194 rows, got {len(df)}")
            if not df['node_id'].is_unique or not df['iso3'].is_unique:
                raise ValueError("node_id/iso3 must be unique")
            return df
    raise FileNotFoundError(f"node_index.csv not found in candidates: {[str(x) for x in candidates]}")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, s: str) -> None:
    _ensure_parent(path)
    path.write_text(s, encoding='utf-8')


def _write_json(path: Path, obj: dict) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def _sample_trace(df_all: pd.DataFrame, k: int) -> pd.DataFrame:
    if len(df_all) <= k:
        return df_all.copy()
    return df_all.sample(n=k, random_state=1337).copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Path to config/pathogen_step1.yaml')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite output zarr if it exists')
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))

    input_dir = Path(cfg['input_dir'])
    input_glob = cfg.get('input_glob', '*_curated_long.csv')
    time_index_path = Path(cfg['time_index_master'])
    out_zarr = Path(cfg['output_zarr'])
    manifest_out = Path(cfg['manifest_out'])
    qc_report_out = Path(cfg['qc_report_out'])
    trace_out = Path(cfg['trace_samples_out'])

    node_candidates = [Path(p) for p in cfg['node_index_candidates']]
    pathogens: List[str] = list(cfg['pathogens'])
    monotone = bool(cfg.get('monotone_forward_fill', True))
    trace_k = int(cfg.get('trace_samples_max', 50))

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")
    files = sorted(input_dir.glob(input_glob))
    if len(files) == 0:
        raise FileNotFoundError(f"No input files found in {input_dir} matching {input_glob}")

    nodes = _load_node_index(node_candidates)
    iso3_to_node = {r.iso3.strip(): int(r.node_id) for r in nodes.itertuples(index=False)}

    if not time_index_path.exists():
        raise FileNotFoundError(f"time_index_master.npy not found: {time_index_path}")
    time_index = np.load(time_index_path)
    if time_index.ndim != 1:
        raise ValueError('time_index_master.npy must be 1D')
    T = int(time_index.shape[0])
    if T != 908:
        raise ValueError(f"Expected T=908, got {T}")

    P = len(pathogens)
    pathogen_to_idx = {p: i for i, p in enumerate(pathogens)}

    # In-memory arrays (small enough; avoids complicated chunked writes)
    evidence = np.zeros((T, 194, P), dtype=np.uint8)
    evidence_mask = np.zeros((T, 194, P), dtype=np.uint8)

    parsed_rows = []
    per_file_stats = []

    required_cols = {'iso3', 'date', 'pathogen', 'value'}

    for fp in files:
        df = pd.read_csv(fp)
        missing = sorted(list(required_cols - set(df.columns)))
        if missing:
            raise ValueError(f"{fp.name} missing required columns: {missing}")

        df['iso3'] = df['iso3'].astype(str).str.strip()
        df['pathogen'] = df['pathogen'].astype(str).str.strip()
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')

        bad_dates = int(df['date_parsed'].isna().sum())
        if bad_dates > 0:
            raise ValueError(f"{fp.name} has {bad_dates} unparseable dates")

        # Map and filter
        df = df[df['iso3'].isin(iso3_to_node.keys())].copy()
        if len(df) == 0:
            per_file_stats.append({'file': str(fp), 'rows_kept': 0, 'note': 'no iso3 matched node universe'})
            continue

        # Month indices
        df['month_index'] = df['date_parsed'].apply(_month_index_from_date).astype(int)
        df = df[(df['month_index'] >= 0) & (df['month_index'] < T)].copy()

        # Pathogen filter
        df = df[df['pathogen'].isin(pathogen_to_idx.keys())].copy()

        if len(df) == 0:
            per_file_stats.append({'file': str(fp), 'rows_kept': 0, 'note': 'no rows in time range / pathogen list'})
            continue

        # Write evidence
        for r in df.itertuples(index=False):
            n = iso3_to_node[r.iso3]
            p = pathogen_to_idx[r.pathogen]
            t = int(r.month_index)
            v = 1 if (r.value is not None and float(r.value) > 0) else 0
            if v > evidence[t, n, p]:
                evidence[t, n, p] = v
            evidence_mask[t, n, p] = 1

        per_file_stats.append({
            'file': str(fp),
            'rows_in_file': int(len(pd.read_csv(fp))),
            'rows_kept': int(len(df)),
            'date_min': str(df['date_parsed'].min().date()),
            'date_max': str(df['date_parsed'].max().date()),
            'unique_iso3': int(df['iso3'].nunique()),
            'unique_pathogen': int(df['pathogen'].nunique())
        })

        parsed_rows.append(df[['iso3', 'date', 'pathogen', 'value', 'month_index']].copy())

    # Combine for trace + QC
    df_all = pd.concat(parsed_rows, ignore_index=True) if parsed_rows else pd.DataFrame(columns=['iso3','date','pathogen','value','month_index'])

    # Monotone status (cumulative max over time) + mask
    # Event-based labels: new detection only
    # status[t] = 1 iff evidence[t]==1 and evidence[t-1]==0
    status = np.zeros_like(evidence, dtype=np.uint8)
    status_mask = evidence_mask.copy()
    
    status[0] = evidence[0]
    for t in range(1, evidence.shape[0]):
        # Cast to signed int to handle negative diffs (1 -> 0 transition) safely
        current = evidence[t].astype(np.int16)
        prev = evidence[t-1].astype(np.int16)
        diff = current - prev
        status[t] = np.clip(diff, 0, 1).astype(np.uint8)

    # Output handling
    if out_zarr.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {out_zarr}. Re-run with --overwrite to replace.")
        # Remove old store (Zarr directory)
        import shutil
        shutil.rmtree(out_zarr)

    _ensure_parent(out_zarr)

    # Write Zarr (v3 compatible)
    g = zarr.open_group(str(out_zarr), mode='w')

    g.create_array('status', shape=status.shape, chunks=(1, 194, P), dtype='u1', fill_value=0)
    g.create_array('status_mask', shape=status_mask.shape, chunks=(1, 194, P), dtype='u1', fill_value=0)
    g.create_array('evidence', shape=evidence.shape, chunks=(1, 194, P), dtype='u1', fill_value=0)
    g.create_array('evidence_mask', shape=evidence_mask.shape, chunks=(1, 194, P), dtype='u1', fill_value=0)
    g.create_array('time_index', shape=(T,), chunks=(T,), dtype='i4')

    # Fixed-length unicode names
    maxlen = max(len(p) for p in pathogens)
    g.create_array('pathogen_names', shape=(P,), chunks=(P,), dtype=f'U{maxlen}')

    g['status'][:] = status
    g['status_mask'][:] = status_mask
    g['evidence'][:] = evidence
    g['evidence_mask'][:] = evidence_mask
    g['time_index'][:] = time_index.astype(np.int32)
    g['pathogen_names'][:] = np.array(pathogens, dtype=f'U{maxlen}')

    # Trace samples
    if len(df_all) > 0:
        df_trace = _sample_trace(df_all, trace_k)
        df_trace['node_id'] = df_trace['iso3'].map(iso3_to_node).astype(int)
        df_trace['pathogen_idx'] = df_trace['pathogen'].map(pathogen_to_idx).astype(int)
        _ensure_parent(trace_out)
        df_trace.to_csv(trace_out, index=False)
    else:
        _write_text(trace_out, 'iso3,date,pathogen,value,month_index,node_id,pathogen_idx\n')

    # QC stats
    events_per_pathogen = {p: int(evidence[:, :, i].sum()) for i, p in enumerate(pathogens)}
    first_month = {}
    for i, p in enumerate(pathogens):
        hits = np.where(evidence[:, :, i].sum(axis=1) > 0)[0]
        first_month[p] = int(hits.min()) if len(hits) else None

    # Write QC report
    lines = []
    lines.append(f"# Pathogen Step 1 QC Report\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    lines.append(f"\n## Inputs\n")
    lines.append(f"- node_index: {str([str(x) for x in node_candidates])}\n")
    lines.append(f"- time_index_master: {str(time_index_path)} (T={T})\n")
    lines.append(f"- pathogen inputs: {str(input_dir)} / {input_glob} (files={len(files)})\n")
    lines.append(f"\n## Outputs\n")
    lines.append(f"- zarr: {str(out_zarr)}\n")
    lines.append(f"- trace: {str(trace_out)}\n")
    lines.append(f"\n## Tensor Shapes\n")
    lines.append(f"- status: {tuple(status.shape)} dtype=uint8\n")
    lines.append(f"- status_mask: {tuple(status_mask.shape)} dtype=uint8\n")
    lines.append(f"- evidence: {tuple(evidence.shape)} dtype=uint8\n")
    lines.append(f"- evidence_mask: {tuple(evidence_mask.shape)} dtype=uint8\n")
    lines.append(f"\n## Events per Pathogen (sum of evidence)\n")
    for p in pathogens:
        lines.append(f"- {p}: {events_per_pathogen[p]} (first_month_index={first_month[p]})\n")

    lines.append(f"\n## Per-file stats\n")
    for st in per_file_stats:
        lines.append(f"- {st}\n")

    _write_text(qc_report_out, ''.join(lines))

    # Manifest
    manifest = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'inputs': {
            'node_index_used': None,
            'time_index_master': str(time_index_path),
            'pathogen_files': [str(p) for p in files]
        },
        'output': {
            'zarr': str(out_zarr),
            'arrays': {
                'status': {'shape': list(status.shape), 'dtype': 'uint8'},
                'status_mask': {'shape': list(status_mask.shape), 'dtype': 'uint8'},
                'evidence': {'shape': list(evidence.shape), 'dtype': 'uint8'},
                'evidence_mask': {'shape': list(evidence_mask.shape), 'dtype': 'uint8'},
                'time_index': {'shape': [T], 'dtype': 'int32'},
                'pathogen_names': {'shape': [P], 'dtype': f'U{maxlen}'}
            }
        },
        'qc': {
            'events_per_pathogen': events_per_pathogen,
            'per_file_stats': per_file_stats,
            'label_definition': 'event_based',
            'label_rule': 'status[t]=1 iff evidence[t]==1 and evidence[t-1]==0'
        }
    }

    # Resolve actual node_index used
    for p in node_candidates:
        if p.exists():
            manifest['inputs']['node_index_used'] = str(p)
            break

    _write_json(manifest_out, manifest)

    print(f"[OK] wrote zarr: {out_zarr}")
    print(f"[OK] wrote qc report: {qc_report_out}")
    print(f"[OK] wrote manifest: {manifest_out}")
    print(f"[OK] wrote trace: {trace_out}")
    return 0


if __name__ == '__main__':
    import sys
    raise SystemExit(main())
