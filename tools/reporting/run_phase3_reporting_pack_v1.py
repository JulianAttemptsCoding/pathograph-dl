from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from tools.reporting.gcloud_util import rsync


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase3_prefix", default="gs://pathograph-057a2273fe-data/runs/stepA/phase3")
    ap.add_argument("--reports_prefix", default="gs://pathograph-057a2273fe-data/runs/stepA/phase3/_reports")
    ap.add_argument("--bundle_dir", default="reports/phase3/bundle_v1")
    ap.add_argument("--cache_dir", default="reports/phase3/_local_cache")
    ap.add_argument("--no_upload", action="store_true")
    ap.add_argument("--include_train_logs", action="store_true", help="Download train logs for learning curves")
    args = ap.parse_args()

    # local dirs
    bundle = Path(args.bundle_dir)
    cache = Path(args.cache_dir)
    (bundle / "tables").mkdir(parents=True, exist_ok=True)
    (bundle / "figures").mkdir(parents=True, exist_ok=True)

    # Step 1: collect
    print("STEP1 collect")
    from tools.reporting.phase3_collect import main as collect_main
    collect_main_args = ["--phase3_prefix", args.phase3_prefix, "--out_dir", str(cache)]
    if args.include_train_logs:
        collect_main_args.append("--include_train_logs")
    import sys
    _old = sys.argv
    sys.argv = ["phase3_collect.py"] + collect_main_args
    try:
        collect_main()
    finally:
        sys.argv = _old

    # Step 2: tables
    print("STEP2 tables")
    from tools.reporting.phase3_tables import main as tables_main
    sys.argv = ["phase3_tables.py", "--cache_dir", str(cache), "--out_dir", str(bundle / "tables")]
    try:
        tables_main()
    finally:
        sys.argv = _old

    # Step 3: plots
    print("STEP3 plots")
    from tools.reporting.phase3_plots import main as plots_main
    sys.argv = ["phase3_plots.py", "--cache_dir", str(cache), "--out_dir", str(bundle / "figures")]
    try:
        plots_main()
    finally:
        sys.argv = _old

    # Step 4: render REPORT.md (lightweight)
    print("STEP4 report")
    report = bundle / "REPORT.md"
    report.write_text(
        "# Phase 3 Reporting Pack v1\n\n"
        "This bundle contains seed-level tables + plots generated from Phase3 eval outputs.\n\n"
        "## Tables\n- tables/phase3_seed_metrics_wide_local.csv\n- tables/phase3_summary_stats_headline_local.csv\n\n"
        "## Figures\n- figures/seed_box_key_metrics.png\n- figures/seed_test_auroc_macro.png (if present)\n- figures/reliability_s1340.png (if bins present)\n- figures/learning_losses_s1340.png (if train logs present)\n\n"
        "## Notes\n- ROC/PR curves require per-example predictions (not included in v1 unless available).\n"
        "- Training curves are best-effort; requires training logs in GCS.\n",
        encoding="utf-8",
    )

    # Step 5: upload
    print("STEP5 upload")
    if not args.no_upload:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = args.reports_prefix.rstrip("/") + f"/bundle_v1_{ts}"
        rsync(str(bundle).replace("\\", "/"), dst)
        print("UPLOADED_TO", dst)
    else:
        print("NO_UPLOAD")

if __name__ == "__main__":
    raise SystemExit(main())
