# scripts/make_sparklines.py
import argparse, pandas as pd, matplotlib.pyplot as plt, pathlib as p

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True, help="2-col CSV with columns: date,value")
parser.add_argument("--col", default="value")
parser.add_argument("--out", required=True)
parser.add_argument("--color", default="#2F80ED")
parser.add_argument("--width", type=float, default=500)
parser.add_argument("--height", type=float, default=120)
args = parser.parse_args()

df = pd.read_csv(args.csv, parse_dates=['date'])
df = df.sort_values('date')

plt.figure(figsize=(args.width/96, args.height/96), dpi=96)
ax = plt.gca()
ax.plot(df['date'], df[args.col], linewidth=2.2, color=args.color)

# Minimal sparkline styling
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.margins(x=0.02, y=0.15)
plt.tight_layout(pad=0)
p.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(args.out, transparent=True)
print("Saved", args.out)
