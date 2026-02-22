from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tools.reporting.gcloud_util import cp, ls, ls_any


@dataclass
class SeedSpec:
    seed: str
    run_prefix: str         # e.g. gs://.../phase3/adaptive_s1340
    eval_prefix: str        # e.g. gs://.../phase3/adaptive_s1340/eval_vertex_cal


def _first_existing(candidates: list[str]) -> Optional[str]:
    for u in candidates:
        if ls_any(u):
            return u
    return None


def discover_seeds(phase3_prefix: str) -> list[SeedSpec]:
    # Conservative: assume fixed known naming; allow mixed eval prefix variants.
    seeds = ["1340", "1341", "1342", "1343", "1344"]
    out: list[SeedSpec] = []
    for s in seeds:
        run = f"{phase3_prefix}/adaptive_s{s}"
        eval_candidates = [
            f"{run}/eval_vertex_cal_v2",
            f"{run}/eval_vertex_cal",
            f"{run}/eval_vertex",
        ]
        ev = _first_existing(eval_candidates)
        if ev is None:
            # still record; later stages will mark missing
            ev = eval_candidates[0]
        out.append(SeedSpec(seed=s, run_prefix=run, eval_prefix=ev))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase3_prefix", required=True)
    ap.add_argument("--out_dir", default="reports/phase3/_local_cache")
    ap.add_argument("--include_train_logs", action="store_true", help="Download training metric CSVs if present")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = discover_seeds(args.phase3_prefix)
    (out_dir / "seed_specs.json").write_text(
        json.dumps([s.__dict__ for s in seeds], indent=2), encoding="utf-8"
    )

    missing = []

    for spec in seeds:
        seed_dir = out_dir / f"s{spec.seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        metrics_uri = f"{spec.eval_prefix}/eval_logs/version_0/metrics.csv"
        bins_uri = f"{spec.eval_prefix}/eval_logs/version_0/calibration_bins_test.json"

        # download metrics.csv
        if ls_any(metrics_uri):
            cp(metrics_uri, str(seed_dir / "metrics.csv"))
        else:
            missing.append({"seed": spec.seed, "missing": metrics_uri})

        # download calibration bins if present
        if ls_any(bins_uri):
            cp(bins_uri, str(seed_dir / "calibration_bins_test.json"))

        if args.include_train_logs:
            train_candidates = [
                f"{spec.run_prefix}/lightning_logs/version_0/metrics.csv",
                f"{spec.run_prefix}/train_logs/version_0/metrics.csv",
            ]
            train_uri = _first_existing(train_candidates)
            if train_uri:
                cp(train_uri, str(seed_dir / "train_metrics.csv"))

    (out_dir / "missing.json").write_text(json.dumps(missing, indent=2), encoding="utf-8")
    print("WROTE", out_dir / "seed_specs.json")
    print("WROTE", out_dir / "missing.json")
    if missing:
        print("MISSING_COUNT", len(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
