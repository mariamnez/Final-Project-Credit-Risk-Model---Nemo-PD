# scripts/peek_reports.py
import os
import pandas as pd

PATHS = [
    "reports/policy_curve.csv",
    "reports/backtest_recent_policy.csv",
    "reports/slice_vintage_metrics.csv",
    "reports/slice_state_top20_metrics.csv",
    "reports/slice_channel_metrics.csv",
    "reports/slice_purpose_metrics.csv",
]

def show(path: str):
    print(f"\n=== {path} ===")
    if not os.path.exists(path):
        print("MISSING")
        return

    df = pd.read_csv(path)

    with pd.option_context("display.max_columns", None,
                           "display.width", 160,
                           "display.max_rows", 20):
        print(df.head(10).to_string(index=False))

    nums = df.select_dtypes(include="number")
    if not nums.empty:
        stats = nums.describe().T.loc[:, ["mean", "50%", "min", "max", "std"]]
        print("\nQuick stats:\n", stats.round(5))

def main():
    for p in PATHS:
        show(p)

if __name__ == "__main__":
    main()
