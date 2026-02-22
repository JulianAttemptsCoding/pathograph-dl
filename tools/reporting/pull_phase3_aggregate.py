"""Pull Phase3 aggregate CSVs from GCS and extract headline stats."""
import argparse
import csv
import subprocess
from pathlib import Path


def sh(cmd: list[str]) -> str:
    import platform
    if platform.system() == "Windows" and cmd[0] == "gcloud":
        cmd[0] = "gcloud.cmd"
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def gcs_exists(uri: str) -> bool:
    try:
        sh(["gcloud", "storage", "ls", uri])
        return True
    except Exception:
        return False


def download(uri: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    sh(["gcloud", "storage", "cp", uri, str(out)])


def read_summary_csv(p: Path) -> dict[str, dict[str, float]]:
    """Read a summary stats CSV: rows=metrics, cols=mean/std/min/max."""
    out: dict[str, dict[str, float]] = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            # Detect the index/metric column
            metric = (
                row.get("", "")
                or row.get("Unnamed: 0", "")
                or row.get("metric", "")
            )
            if not metric:
                continue

            def fnum(k: str):
                v = row.get(k)
                if v is None or v == "":
                    return None
                try:
                    return float(v)
                except Exception:
                    return None

            out[metric] = {
                "mean": fnum("mean"),
                "std": fnum("std"),
                "min": fnum("min"),
                "max": fnum("max"),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase3_aggregate_prefix", required=True,
                    help="GCS prefix, e.g. gs://bucket/runs/stepA/phase3/_aggregate")
    ap.add_argument("--out_dir", default="reports/phase3/_local_cache")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer calibrated final summary if present
    candidates = [
        f"{args.phase3_aggregate_prefix}/phase3_cal_final_summary_stats.csv",
        f"{args.phase3_aggregate_prefix}/phase3_summary_stats.csv",
    ]

    chosen = None
    for c in candidates:
        if gcs_exists(c):
            chosen = c
            break

    if not chosen:
        print("NO_SUMMARY_CSV_FOUND")
        print("Tried:")
        for c in candidates:
            print("  ", c)
        return 2

    local = out_dir / Path(chosen).name
    download(chosen, local)

    stats = read_summary_csv(local)

    keys = [
        "test_auroc_macro",
        "test_auprc_macro",
        "test_brier_macro",
        "test_ece_macro",
    ]

    print("SUMMARY_CSV:", chosen)
    for k in keys:
        if k in stats:
            s = stats[k]
            print(f"{k}: mean={s['mean']} std={s['std']} min={s['min']} max={s['max']}")
        else:
            print(f"{k}: NOT_FOUND")

    kv = {k: stats.get(k) for k in keys}
    (out_dir / "headline_stats.txt").write_text(str(kv), encoding="utf-8")
    print("WROTE:", out_dir / "headline_stats.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
