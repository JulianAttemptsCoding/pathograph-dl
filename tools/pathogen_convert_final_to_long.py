from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def month_index_to_date_str(mi: int) -> str:
    """
    Canonical convention: month_index 0 == 1950-01.
    We emit YYYY-MM-01 as a valid ISO date string.
    """
    y = 1950 + (mi // 12)
    m = 1 + (mi % 12)
    return f"{y:04d}-{m:02d}-01"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Directory containing *_final.csv source files")
    ap.add_argument("--out_dir", required=True, help="Directory to write canonical long-format CSVs")
    ap.add_argument(
        "--node_index",
        default="data/processed/trade/imf_imts_step1/node_index.csv",
        help="Canonical node index CSV (must contain iso3 column)",
    )
    ap.add_argument(
        "--time_index_master",
        default="data/processed/meta/time_index_master.npy",
        help="Master time index (month_index) exported from trade gate",
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    node_index_path = Path(args.node_index)
    if not node_index_path.exists():
        raise FileNotFoundError(f"node_index not found: {node_index_path}")

    node_df = pd.read_csv(node_index_path)
    if "iso3" not in node_df.columns:
        raise ValueError(f"node_index.csv must contain column 'iso3'. Found: {list(node_df.columns)}")
    iso_set = set(node_df["iso3"].astype(str).str.strip())

    tmin = 0
    tmax = 10**9
    time_index_path = Path(args.time_index_master)
    if time_index_path.exists():
        t = np.load(time_index_path)
        tmin = int(np.min(t))
        tmax = int(np.max(t))

    src_files = sorted(in_dir.glob("*.csv"))
    if not src_files:
        raise FileNotFoundError(f"No CSVs found in: {in_dir}")

    written = 0
    for src in src_files:
        pathogen = re.sub(r"_final$", "", src.stem, flags=re.IGNORECASE)

        df = pd.read_csv(src)

        # Detect ISO3 column
        iso3_col = None
        for cand in ["Country_Code", "iso3", "ISO3", "country_code", "COUNTRY.ID"]:
            if cand in df.columns:
                iso3_col = cand
                break
        if iso3_col is None:
            raise ValueError(f"{src.name}: could not find ISO3 column. Columns: {list(df.columns)}")

        # D1..D5 columns are month_index events in your current “final” format
        d_cols = [c for c in df.columns if re.fullmatch(r"D\d+", str(c).strip())]
        if not d_cols:
            raise ValueError(f"{src.name}: expected D1.. columns. Columns: {list(df.columns)}")

        # Sort D columns numerically (D1, D2, ...)
        d_cols = sorted(d_cols, key=lambda x: int(str(x)[1:]))

        rows = []
        for _, r in df.iterrows():
            iso3 = str(r[iso3_col]).strip()
            if not iso3 or iso3.lower() == "nan":
                continue
            if iso3 not in iso_set:
                # Not in canonical 194-node universe
                continue

            for c in d_cols:
                v = r.get(c)
                if pd.isna(v):
                    continue
                try:
                    mi_f = float(v)
                    mi = int(round(mi_f))
                except Exception:
                    continue

                # Filter to master time range (typically 1950-01..2025-08)
                if mi < tmin or mi > tmax:
                    continue

                date_str = month_index_to_date_str(mi)
                rows.append((iso3, date_str, pathogen, 1))

        out_df = pd.DataFrame(rows, columns=["iso3", "date", "pathogen", "value"]).drop_duplicates()
        out_path = out_dir / f"{pathogen}_curated_long.csv"
        out_df.to_csv(out_path, index=False)
        print(f"[OK] wrote {out_path} rows={len(out_df)}")
        written += 1

    if written == 0:
        raise RuntimeError("No output files were written. Check inputs.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
