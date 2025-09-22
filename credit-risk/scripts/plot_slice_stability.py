# scripts/plot_slice_stability.py
# Generates: reports/slice_stability_heatmap.png

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

root = Path(".")
rep = root / "reports"

# --- Load slice-level metrics (already created earlier) ---
# Expected columns in each: key, n, bad_rate, auc, pr_auc, ks
df_state   = pd.read_csv(rep / "slice_state_top20_metrics.csv")
df_channel = pd.read_csv(rep / "slice_channel_metrics.csv")
df_purpose = pd.read_csv(rep / "slice_purpose_metrics.csv")

metrics = ["ks", "auc", "bad_rate"]

def prep_matrix(df, key_col, top=None):
    """Return (norm_values, labels, original_values) for heatmap."""
    if top:
        df = df.sort_values("n", ascending=False).head(top)
    # sort by KS (stronger segments near top)
    df = df.sort_values("ks", ascending=False).reset_index(drop=True)

    original = df[metrics].copy()
    norm = original.copy()

    # min-max scale KS and AUC to [0,1]
    for m in ["ks", "auc"]:
        lo, hi = original[m].min(), original[m].max()
        norm[m] = 0 if hi == lo else (original[m] - lo) / (hi - lo)

    # invert bad_rate so low risk => greener (closer to 1)
    lo, hi = original["bad_rate"].min(), original["bad_rate"].max()
    br_scaled = 0 if hi == lo else (original["bad_rate"] - lo) / (hi - lo)
    norm["bad_rate"] = 1 - br_scaled

    labels = df[key_col].astype(str).tolist()
    return norm.values, labels, original

# --- Build the figure with 3 stacked heatmaps ---
fig, axes = plt.subplots(3, 1, figsize=(10, 14), constrained_layout=True)

panels = [
    (df_state,   "state",   "Stability by State (top 15 by volume)", 15),
    (df_channel, "channel", "Stability by Channel",                  None),
    (df_purpose, "purpose", "Stability by Purpose",                  None),
]

last_im = None
for ax, (df, key, title, top) in zip(axes, panels):
    data, ylabels, original = prep_matrix(df, key, top)

    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    last_im = im

    ax.set_xticks(range(3))
    ax.set_xticklabels(["KS", "AUC", "Bad rate (low=good)"])
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_title(title)

    # numeric overlays (KS/AUC in 0–1; bad rate as %)
    for i in range(len(ylabels)):
        ax.text(0, i, f"{original.iloc[i]['ks']:.3f}",  ha="center", va="center", fontsize=8)
        ax.text(1, i, f"{original.iloc[i]['auc']:.3f}", ha="center", va="center", fontsize=8)
        ax.text(2, i, f"{original.iloc[i]['bad_rate']:.2%}", ha="center", va="center", fontsize=8)

cbar = fig.colorbar(last_im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
cbar.set_label("Greener = better (bad rate inverted)")

fig.suptitle("Slice Stability — model quality and risk by segment", y=0.995, fontsize=14)

out = rep / "slice_stability_heatmap.png"
plt.savefig(out, dpi=220, bbox_inches="tight")
print(f"Wrote {out}")
