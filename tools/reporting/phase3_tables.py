from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


HEADLINE = [
    "test_auroc_macro",
    "test_auprc_macro",
    "test_brier_macro",
    "test_ece_macro",
    "test_loss",
    "test_acc",
]


def load_seed_metrics(cache_dir: Path) -> pd.DataFrame:
    rows = []
    for seed_dir in sorted(cache_dir.glob("s*/")):
        p = seed_dir / "metrics.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        # lightning csv logger format: columns include "metric" and "value" OR wide columns
        if "metric" in df.columns and "value" in df.columns:
            d = {r["metric"]: r["value"] for _, r in df.iterrows()}
        else:
            # assume first row contains metrics
            d = df.iloc[0].to_dict()
        d["seed"] = seed_dir.name.lstrip("s")
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("seed").sort_index()
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    num = df.apply(pd.to_numeric, errors="coerce")
    s = pd.DataFrame({
        "mean": num.mean(axis=0),
        "std": num.std(axis=0, ddof=1),
        "min": num.min(axis=0),
        "max": num.max(axis=0),
    })
    s.index.name = "metric"
    return s


def per_pathogen_stats(df: pd.DataFrame) -> pd.DataFrame:
    # Heuristic: any metric column containing '_p' followed by digit or pathogen name
    cols = [c for c in df.columns if ("_p" in c and any(ch.isdigit() for ch in c.split("_p")[-1])) or ("per_pathogen" in c)]
    if not cols:
        return pd.DataFrame()
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    s = pd.DataFrame({
        "mean": num.mean(axis=0),
        "std": num.std(axis=0, ddof=1),
        "min": num.min(axis=0),
        "max": num.max(axis=0),
    })
    s.index.name = "metric"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="reports/phase3/_local_cache")
    ap.add_argument("--out_dir", default="reports/phase3/bundle_v1/tables")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_seed_metrics(cache_dir)
    if df.empty:
        raise SystemExit("NO_SEED_METRICS_FOUND: run phase3_collect first")

    # subset headline if present, but keep full wide too
    headline_cols = [c for c in HEADLINE if c in df.columns]
    df_wide = df.copy()
    df_head = df[headline_cols] if headline_cols else df

    df_wide.to_csv(out_dir / "phase3_seed_metrics_wide_local.csv")
    df_head.to_csv(out_dir / "phase3_seed_metrics_headline_local.csv")

    summary = summarize(df_head)
    summary.to_csv(out_dir / "phase3_summary_stats_headline_local.csv")

    pp = per_pathogen_stats(df_wide)
    if not pp.empty:
        pp.to_csv(out_dir / "phase3_per_pathogen_stats_local.csv")

    # machine-readable list of what was produced
    manifest = {
        "headline_cols": headline_cols,
        "wide_cols": list(df_wide.columns),
        "seed_count": int(df.shape[0]),
        "has_per_pathogen": bool(not pp.empty),
    }
    (out_dir / "tables_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("WROTE tables to", out_dir)


if __name__ == "__main__":
    main()
