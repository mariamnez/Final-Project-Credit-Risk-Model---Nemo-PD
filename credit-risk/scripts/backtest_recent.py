# scripts/backtest_recent.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

def find_prob_col(df: pd.DataFrame) -> str:
    order = ["pd_cal","pd_platt","p_cal","prob_cal",
             "pd_raw","p_raw","prob","p","pd","y_pred"]
    for n in order:
        if n in df.columns:
            return n
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]) and 0.0 <= df[c].min() <= 1.0 and df[c].max() <= 1.0:
            return c
    raise ValueError("Could not find a probability column in recent predictions.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_calibrated", action="store_true")
    args = ap.parse_args()

    preds_file = PROCESSED / "recent_predictions.parquet"
    if args.use_calibrated:
        cand = PROCESSED / "recent_predictions_calibrated.parquet"
        if cand.exists():
            preds_file = cand

    df = pd.read_parquet(preds_file)
    # we don’t have labels for the recent cohort
    pcol = find_prob_col(df)
    p = df[pcol].clip(0,1)

    choice = json.loads((REPORTS / "policy_choice.json").read_text())
    t = float(choice.get("threshold", float("nan")))
    if not np.isfinite(t):
        raise ValueError("Invalid threshold in reports/policy_choice.json")

    approve = p <= t  # lower PD = approve
    out = (df.assign(approve=approve)
             .assign(vintage_q = pd.to_datetime(df["vintage_q"]) if "vintage_q" in df.columns else pd.NaT)
             .groupby("vintage_q", dropna=False)
             .apply(lambda g: pd.Series({
                 "n": len(g),
                 "approve_rate": float(g["approve"].mean()),
                 # we don’t have default labels here; keep overall portfolio bad rate as placeholder if present
                 "approved_bad_rate": np.nan,
                 "overall_bad_rate": float(df.get("overall_bad_rate", pd.Series(dtype=float)).mean()) if "overall_bad_rate" in df else np.nan
             }))
             .reset_index())

    REPORTS.mkdir(parents=True, exist_ok=True)
    out_csv = REPORTS / "backtest_recent_policy.csv"
    out.to_csv(out_csv, index=False)

    print(f"Using preds file: {preds_file}")
    print(f"Prob column: {pcol}")
    print(f"Threshold t = {t:.6f}")
    print(f"Approve rate overall = {approve.mean():.4f}")
    print(f"Wrote: {out_csv}")

if __name__ == "__main__":
    main()
