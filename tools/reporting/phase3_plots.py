from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


KEYS = ["test_auroc_macro", "test_auprc_macro", "test_brier_macro", "test_ece_macro"]


def _read_metrics(seed_dir: Path) -> dict:
    p = seed_dir / "metrics.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if "metric" in df.columns and "value" in df.columns:
        return {r["metric"]: r["value"] for _, r in df.iterrows()}
    return df.iloc[0].to_dict()


def plot_seed_bars(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    for k in [c for c in KEYS if c in df.columns]:
        vals = pd.to_numeric(df[k], errors="coerce")
        fig = plt.figure()
        plt.title(k)
        plt.plot(vals.index.astype(str), vals.values, marker="o")
        plt.xlabel("seed")
        plt.ylabel(k)
        fig.tight_layout()
        fig.savefig(out.parent / f"seed_{k}.png", dpi=200)
        plt.close(fig)


def plot_seed_box(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in KEYS if c in df.columns]
    if not cols:
        return
    data = [pd.to_numeric(df[c], errors="coerce").dropna().values for c in cols]
    fig = plt.figure()
    plt.title("Phase3 seed distribution")
    plt.boxplot(data, labels=cols)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out.parent / "seed_box_key_metrics.png", dpi=200)
    plt.close(fig)


def plot_calibration(seed_dir: Path, out_dir: Path) -> bool:
    p = seed_dir / "calibration_bins_test.json"
    if not p.exists():
        return False
    j = json.loads(p.read_text(encoding="utf-8"))
    # Best-effort: support a few plausible schemas
    # Expect arrays: bin_centers/confidence, accuracy, counts
    conf = j.get("confidence") or j.get("bin_confidence") or j.get("mean_confidence")
    acc = j.get("accuracy") or j.get("bin_accuracy") or j.get("empirical_accuracy")
    cnt = j.get("count") or j.get("bin_count") or j.get("counts")
    if conf is None or acc is None:
        return False

    conf = np.array(conf, dtype=float)
    acc = np.array(acc, dtype=float)

    fig = plt.figure()
    plt.title(f"Reliability (seed {seed_dir.name})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(conf, acc, marker="o")
    plt.xlabel("confidence")
    plt.ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(out_dir / f"reliability_{seed_dir.name}.png", dpi=200)
    plt.close(fig)
    return True


def plot_learning_curves(seed_dir: Path, out_dir: Path) -> bool:
    p = seed_dir / "train_metrics.csv"
    if not p.exists():
        return False
    df = pd.read_csv(p)
    # Heuristic: lightning metrics.csv often has columns: epoch, step, train_loss, val_loss, val_auroc_macro...
    # If it's long-form, attempt pivot.
    if "metric" in df.columns and "value" in df.columns:
        if "epoch" in df.columns:
            piv = df.pivot_table(index="epoch", columns="metric", values="value", aggfunc="last")
        else:
            return False
        dfw = piv.reset_index()
    else:
        dfw = df.copy()

    x = dfw["epoch"] if "epoch" in dfw.columns else None
    if x is None:
        return False

    candidates = [c for c in dfw.columns if any(k in c.lower() for k in ["loss", "auroc", "auprc", "acc"]) and c != "epoch"]
    if not candidates:
        return False

    # Make 2 plots max to keep it light: losses + key val metrics
    loss_cols = [c for c in candidates if "loss" in c.lower()][:4]
    metric_cols = [c for c in candidates if any(k in c.lower() for k in ["auroc", "auprc", "acc"])][:4]

    if loss_cols:
        fig = plt.figure()
        plt.title(f"Learning curve losses (seed {seed_dir.name})")
        for c in loss_cols:
            plt.plot(x, pd.to_numeric(dfw[c], errors="coerce"), label=c)
        plt.xlabel("epoch")
        plt.ylabel("value")
        plt.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"learning_losses_{seed_dir.name}.png", dpi=200)
        plt.close(fig)

    if metric_cols:
        fig = plt.figure()
        plt.title(f"Learning curve metrics (seed {seed_dir.name})")
        for c in metric_cols:
            plt.plot(x, pd.to_numeric(dfw[c], errors="coerce"), label=c)
        plt.xlabel("epoch")
        plt.ylabel("value")
        plt.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"learning_metrics_{seed_dir.name}.png", dpi=200)
        plt.close(fig)

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="reports/phase3/_local_cache")
    ap.add_argument("--out_dir", default="reports/phase3/bundle_v1/figures")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed_dir in sorted(cache.glob("s*/")):
        d = _read_metrics(seed_dir)
        if d:
            d["seed"] = seed_dir.name.lstrip("s")
            rows.append(d)

    if not rows:
        raise SystemExit("NO_METRICS_DOWNLOADED")

    df = pd.DataFrame(rows).set_index("seed").sort_index()

    plot_seed_bars(df, out_dir / "_x")
    plot_seed_box(df, out_dir / "_y")

    # calibration and learning curves per seed
    calib_any = False
    learn_any = False
    for seed_dir in sorted(cache.glob("s*/")):
        calib_any |= plot_calibration(seed_dir, out_dir)
        learn_any |= plot_learning_curves(seed_dir, out_dir)

    (out_dir / "plots_manifest.json").write_text(
        json.dumps({"has_calibration": calib_any, "has_learning_curves": learn_any, "seed_count": int(df.shape[0])}, indent=2),
        encoding="utf-8",
    )
    print("WROTE plots to", out_dir)


if __name__ == "__main__":
    main()
