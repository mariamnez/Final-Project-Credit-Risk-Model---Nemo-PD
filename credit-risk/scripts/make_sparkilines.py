import argparse, pathlib
import pandas as pd
import matplotlib.pyplot as plt

def sparkline(csv, value_col, date_col, out, color="#2F80ED", smooth=3, target=None, ymin=None, ymax=None):
    df = pd.read_csv(csv)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    v = df[value_col].rolling(smooth, min_periods=1).mean() if smooth and smooth > 1 else df[value_col]

    fig, ax = plt.subplots(figsize=(2.4, 0.7), dpi=300)  
    ax.plot(df[date_col], v, linewidth=2, color=color)
    ax.scatter([df[date_col].iloc[-1]], [v.iloc[-1]], s=10, color=color)  

    if target is not None:
        ax.axhline(target, lw=1, ls="--", color=color, alpha=0.4)

    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("none"); fig.patch.set_alpha(0)
    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)

    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Make a sparkline PNG from a 2-column CSV (date, value).")
    p.add_argument("--csv", required=True, help="Path to CSV")
    p.add_argument("--col", required=True, help="Value column name")
    p.add_argument("--date", default="date", help="Date column name")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--color", default="#2F80ED")
    p.add_argument("--smooth", type=int, default=3)
    p.add_argument("--target", type=float)
    p.add_argument("--ymin", type=float); p.add_argument("--ymax", type=float)
    args = p.parse_args()

    sparkline(
        csv=args.csv, value_col=args.col, date_col=args.date, out=args.out,
        color=args.color, smooth=args.smooth, target=args.target,
        ymin=args.ymin, ymax=args.ymax
    )
