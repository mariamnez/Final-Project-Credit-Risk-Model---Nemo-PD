# scripts/set_policy_threshold.py
import argparse, json
from pathlib import Path
import pandas as pd

def find_prob_col(cols):
    for c in ["pd_cal", "p_cal", "pd", "p", "y_pred"]:
        if c in cols: return c
    raise ValueError("No probability column found (looked for pd_cal, p_cal, pd, p, y_pred).")

def main():
    ap = argparse.ArgumentParser(description="Set policy threshold by target approval rate.")
    ap.add_argument("--approve", type=float, default=0.22,
                    help="Target approval rate (fraction, e.g. 0.22 for 22%).")
    ap.add_argument("--preds", default="data/processed/recent_predictions_calibrated.parquet",
                    help="Parquet with recent calibrated PDs.")
    ap.add_argument("--out", default="reports/policy_choice.json",
                    help="Where to write the policy choice JSON.")
    args = ap.parse_args()

    root = Path(".")
    preds = root / args.preds
    out   = root / args.out

    df = pd.read_parquet(preds)
    pcol = find_prob_col(df.columns)

    # threshold = PD quantile at the target approval rate
    t = float(df[pcol].quantile(args.approve))

    approve_rate = float((df[pcol] <= t).mean())

    choice = {
        "threshold": round(t, 6),
        "metric": "ks_constrained",   
        "prob": "y_pred"              
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(choice, indent=2))

    print(f"Set threshold -> {t:.6f}")
    print(f"Prob column   -> {pcol}")
    print(f"Approve_rate  -> {approve_rate:.4f}")
    print(f"Wrote         -> {out}")

if __name__ == "__main__":
    main()
