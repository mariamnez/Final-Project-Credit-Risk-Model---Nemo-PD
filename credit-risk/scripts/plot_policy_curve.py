"""
Plot policy curve (recall / precision / F1 / KS vs threshold) with approve rate on a second axis.
- Reads a CSV produced by policy_curve.py (flexible to column names).
- Pulls chosen threshold from policy_choice.json (if available).
- Re-computes KS from counts if needed.
- Saves a PNG and a small snapshot CSV at the plotted threshold.

Usage (from project root):
  python scripts/plot_policy_curve.py
  python scripts/plot_policy_curve.py --curve reports/policy_curve.csv --choice reports/policy_choice.json --out reports/policy_curve.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _read_curve(curve_path: Path) -> pd.DataFrame:
    df = pd.read_csv(curve_path)

    # Normalize column names
    rename = {
        "ks_at_p": "ks",
        "KS": "ks",
        "approve_rate_at_p": "approve_rate",
        "approval_rate": "approve_rate",
        "Threshold": "threshold",
        "Recall": "recall",
        "Precision": "precision",
        "F1": "f1",
    }
    df = df.rename(columns=rename)

    for col in ["threshold", "precision", "recall", "f1", "ks", "approve_rate", "tp", "fp", "tn", "fn"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    needs_ks = ("ks" not in df.columns) or (df["ks"].nunique(dropna=True) <= 1)
    have_counts = all(c in df.columns for c in ["tp", "fp", "tn", "fn"])
    if needs_ks and have_counts:
        tpr = df["tp"] / (df["tp"] + df["fn"]).replace(0, np.nan)
        fpr = df["fp"] / (df["fp"] + df["tn"]).replace(0, np.nan)
        df["ks"] = (tpr - fpr).clip(lower=0, upper=1).fillna(0)

    if "threshold" in df.columns:
        df = df.dropna(subset=["threshold"]).sort_values("threshold")
        df = df.reset_index(drop=True)
    else:
        raise ValueError("Curve CSV must contain a 'threshold' column.")

    return df


def _read_choice(choice_path: Path) -> float | None:
    if not choice_path.exists():
        return None
    try:
        choice = json.loads(choice_path.read_text())
        t = choice.get("threshold", choice.get("t", None))
        return float(t) if t is not None else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", type=Path, default=Path("reports/policy_curve.csv"))
    ap.add_argument("--choice", type=Path, default=Path("reports/policy_choice.json"))
    ap.add_argument("--out", type=Path, default=Path("reports/policy_curve.png"))
    ap.add_argument("--snap", type=Path, default=Path("reports/policy_snapshot_at_t.csv"))
    ap.add_argument("--title", type=str, default="Policy curve")
    args = ap.parse_args()

    curve_path: Path = args.curve
    choice_path: Path = args.choice
    out_path: Path = args.out
    snap_path: Path = args.snap

    df = _read_curve(curve_path)
    t_choice = _read_choice(choice_path)

    y_left_series = {}
    for col in ["recall", "precision", "f1", "ks"]:
        if col in df.columns:
            y_left_series[col] = df[col]

    has_approve = "approve_rate" in df.columns

    # threshold to highlight
    if t_choice is None:
        if "ks" in df.columns and df["ks"].notna().any():
            t_choice = float(df.loc[df["ks"].idxmax(), "threshold"])
        elif "f1" in df.columns and df["f1"].notna().any():
            t_choice = float(df.loc[df["f1"].idxmax(), "threshold"])
        else:
            t_choice = float(df["threshold"].median())

    i = (df["threshold"] - t_choice).abs().idxmin()
    pt = df.loc[i, :]

    # Plot
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # Left axis
    colors = {
        "recall": "tab:blue",
        "precision": "tab:orange",
        "f1": "tab:green",
        "ks": "crimson",
    }
    labels = {
        "recall": "Recall",
        "precision": "Precision",
        "f1": "F1",
        "ks": "KS",
    }
    for k, s in y_left_series.items():
        ax.plot(df["threshold"], s, label=labels.get(k, k), color=colors.get(k, None), lw=2)

    ax.set_xlabel("Threshold (PD cut-off)")
    ax.set_ylabel("Recall / Precision / F1 / KS")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)

    # Right axis:
    if has_approve:
        ax2 = ax.twinx()
        ax2.plot(df["threshold"], df["approve_rate"], color="tab:blue", ls="--", lw=2, label="Approve rate")
        ax2.set_ylabel("Approve rate")
        ax2.set_ylim(0, 1.05)
    else:
        ax2 = None

    ax.axvline(t_choice, color="k", ls=":", lw=1.5, label=f"t = {t_choice:.4f}")

    # Legend
    handles, labels_ = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels_ += l2
    ax.legend(handles, labels_, loc="upper right", ncol=2, frameon=False)

    y_for_marker = None
    for c in ["f1", "precision", "recall", "ks"]:
        if c in df.columns and pd.notna(pt.get(c)):
            y_for_marker = float(pt[c])
            marker_color = colors.get(c, "k")
            ax.scatter([pt["threshold"]], [y_for_marker], s=50, color=marker_color, zorder=5)
            break

    # Title
    plt.title(args.title, pad=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    cols = [c for c in ["threshold", "approve_rate", "precision", "recall", "f1", "ks", "tp", "fp", "tn", "fn"] if c in df.columns]
    snap = df.loc[[i], cols].copy()
    snap.rename(columns={
        "threshold": "t",
        "approve_rate": "approve_rate",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "ks": "ks"
    }, inplace=True)
    snap["t"] = float(pt["threshold"])
    snap.round(6).to_csv(snap_path, index=False)

    print(f"Saved plot -> {out_path}")
    print(f"Saved snapshot -> {snap_path}")
    print(snap.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

