from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Config ----------
OUT_PNG_DIR = Path("reports/eda_png")
OUT_CSV_DIR = Path("reports/eda_canva")

SLICE_FILES = {
    "channel": Path("reports/slice_channel_metrics.csv"),   # cols: channel,n,bad_rate,...
    "purpose": Path("reports/slice_purpose_metrics.csv"),   # cols: purpose,n,bad_rate,...
    "state20": Path("reports/slice_state_top20_metrics.csv"), # cols: state,n,bad_rate,...
    "vintage": Path("reports/slice_vintage_metrics.csv"),   # cols: vintage_q,bad_rate,pr_auc,ks
}

ABT_CANDIDATES = [
    Path("data/processed/abt_train.parquet"),
    Path("data/processed/abt_valid.parquet"),
    Path("data/processed/abt_test.parquet"),
    Path("data/processed/abt_recent.parquet"),
]

# Edges for binned features
BINS = {
    "fico": [500,580,620,660,700,740,780,850],
    "dti" : [0,20,30,36,43,50,65,1000],
    "ltv" : [0,60,70,80,90,97,100,500],
}

# ---------- Utilities ----------
def ensure_dirs(*dirs):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def try_load_abt():
    for p in ABT_CANDIDATES:
        if p.exists():
            df = pd.read_parquet(p)
            print(f"[load] ABT -> {p} | rows={len(df):,}")
            return df, p
    return None, None

def find_target_col(df):
    for c in ["default_within_24m","y_true","bad","target"]:
        if c in df.columns:
            return c
    raise ValueError("Target/label not found in ABT (looked for default_within_24m/y_true/bad/target)")

def export_csv(df, name, out_dir, downloads=False):
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    print("[csv ]", path)
    if downloads:
        d = Path.home() / "Downloads" / f"{name}.csv"
        df.to_csv(d, index=False)
        print("       copied ->", d)

def save_png(fig, name, out_dir, downloads=False, dpi=240):
    path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("[png ]", path)
    if downloads:
        d = Path.home() / "Downloads" / f"{name}.png"
        fig.savefig(d, dpi=dpi, bbox_inches="tight")
        print("       copied ->", d)

def bar_pair(ax1, ax2, x_labels, counts, rates, title):
    xs = np.arange(len(x_labels))
    ax1.bar(xs, counts, alpha=0.85)
    ax1.set_title(title, fontsize=11)
    ax1.set_ylabel("Count")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(x_labels, rotation=0)
    ax2.plot(xs, np.array(rates)*100.0, marker="o")
    ax2.set_ylabel("Bad rate (%)")
    ax2.grid(alpha=0.25, axis="y")

def plot_bin_feature(df, ycol, fcol, edges, out_png_dir, out_csv_dir, downloads):
    if fcol not in df.columns:
        print(f"[skip] {fcol} not in ABT")
        return
    s = pd.cut(df[fcol], bins=edges, include_lowest=True)
    g = df.groupby(s, observed=True).agg(Count=(ycol,"size"), Bad_rate=(ycol,"mean")).reset_index()
    g[f"{fcol}_bin"] = g.iloc[:,0].astype(str)
    g = g[[f"{fcol}_bin","Count","Bad_rate"]]

    # CSV for Canva
    export_csv(g.rename(columns={f"{fcol}_bin":fcol.capitalize()+"_bin"}), f"eda_{fcol}_bins", out_csv_dir, downloads)

    # PNG (Count + Bad rate)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4), gridspec_kw={"width_ratios":[3,2]})
    bar_pair(ax1, ax2, g[f"{fcol}_bin"], g["Count"], g["Bad_rate"], f"{fcol.upper()} bins: volume & risk")
    save_png(fig, f"eda_{fcol}_bins", out_png_dir, downloads)

def make_missingness_plot(df, out_png_dir, out_csv_dir, downloads):
    miss = (df.isna().mean()*100).sort_values(ascending=False)
    miss = miss[miss>0].head(12).round(2).reset_index()
    miss.columns = ["feature","missing_pct"]
    if miss.empty:
        print("[skip] no missing values")
        return
    export_csv(miss, "eda_missing_top", out_csv_dir, downloads)

    fig, ax = plt.subplots(figsize=(7,4))
    ax.barh(miss["feature"][::-1], miss["missing_pct"][::-1])
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missingness (top features)")
    save_png(fig, "eda_missing_top", out_png_dir, downloads)

def plot_slice_bars(name, df, cat_col, out_png_dir, out_csv_dir, downloads, title):
    # Expect columns: cat_col (str), n (Count), bad_rate
    cols_lower = {c.lower(): c for c in df.columns}
    ncol = cols_lower.get("n","n")
    brcol = cols_lower.get("bad_rate","bad_rate")
    # Clean + order by Count
    d = df[[cat_col, ncol, brcol]].copy()
    d.columns = [cat_col, "Count", "Bad_rate"]
    d = d.sort_values("Count", ascending=False)

    export_csv(d, f"eda_{name}", out_csv_dir, downloads)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4), gridspec_kw={"width_ratios":[3,2]})
    bar_pair(ax1, ax2, d[cat_col], d["Count"], d["Bad_rate"], title)
    save_png(fig, f"eda_{name}", out_png_dir, downloads)

def plot_vintage_line(df, out_png_dir, out_csv_dir, downloads):
    # Expect columns: vintage_q, bad_rate (and optionally n)
    d = df.copy()
    if "vintage_q" in d.columns:
        xcol = "vintage_q"
    elif "vintage" in d.columns:
        xcol = "vintage"
    else:
        print("[skip] vintage column not found")
        return
    d = d.sort_values(xcol)
    # Export CSV
    cols = [c for c in [xcol, "bad_rate", "n"] if c in d.columns]
    export_csv(d[cols].rename(columns={xcol:"Vintage", "bad_rate":"Bad_rate","n":"Count"}),
               "eda_vintage_trend", out_csv_dir, downloads)

    # Plot
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(d[xcol], d["bad_rate"]*100.0, marker="o")
    ax.set_title("Vintage trend: bad rate (%)")
    ax.set_ylabel("Bad rate (%)")
    ax.grid(alpha=0.3)
    ax.set_xticks(range(0,len(d[xcol]), max(1,len(d)//12)))
    ax.set_xticklabels(d[xcol].iloc[ax.get_xticks()], rotation=45, ha="right")
    save_png(fig, "eda_vintage_trend", out_png_dir, downloads)

def main(to_downloads=False):
    ensure_dirs(OUT_PNG_DIR, OUT_CSV_DIR)

    # 1) Slice files
    if SLICE_FILES["channel"].exists():
        df = pd.read_csv(SLICE_FILES["channel"])
        # normalize column name
        cat = "channel" if "channel" in df.columns else "Channel"
        plot_slice_bars("channel", df, cat, OUT_PNG_DIR, OUT_CSV_DIR, to_downloads, "Channel: volume & bad rate")

    if SLICE_FILES["purpose"].exists():
        df = pd.read_csv(SLICE_FILES["purpose"])
        cat = "purpose" if "purpose" in df.columns else "Purpose"
        plot_slice_bars("purpose", df, cat, OUT_PNG_DIR, OUT_CSV_DIR, to_downloads, "Purpose: volume & bad rate")

    if SLICE_FILES["state20"].exists():
        df = pd.read_csv(SLICE_FILES["state20"])
        cat = "state" if "state" in df.columns else "State"
        plot_slice_bars("state_top20", df, cat, OUT_PNG_DIR, OUT_CSV_DIR, to_downloads, "Top 20 states: volume & bad rate")

    if SLICE_FILES["vintage"].exists():
        df = pd.read_csv(SLICE_FILES["vintage"])
        plot_vintage_line(df, OUT_PNG_DIR, OUT_CSV_DIR, to_downloads)

    # 2) if abt is not done, make FICO/DTI/LTV + missingness
    abt, path = try_load_abt()
    if abt is not None:
        # Sample for speed if extremely large
        if len(abt) > 2_000_000:
            abt = abt.sample(2_000_000, random_state=13)
        y = find_target_col(abt)

        for f in ["fico","dti","ltv"]:
            if f in abt.columns:
                plot_bin_feature(abt, y, f, BINS[f], OUT_PNG_DIR, OUT_CSV_DIR, to_downloads)

        make_missingness_plot(abt, OUT_PNG_DIR, OUT_CSV_DIR, to_downloads)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--to-downloads", action="store_true", help="Also copy PNG/CSV to your Downloads folder")
    args = ap.parse_args()
    main(to_downloads=args.to_downloads)
