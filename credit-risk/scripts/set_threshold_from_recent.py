# scripts/set_threshold_from_recent.py
import argparse, json
from pathlib import Path
import pandas as pd

def main(target=0.20):
    root = Path(__file__).resolve().parents[1]
    preds_cal = root / "data" / "processed" / "recent_predictions_calibrated.parquet"
    preds_raw = root / "data" / "processed" / "recent_predictions.parquet"
    preds_path = preds_cal if preds_cal.exists() else preds_raw
    df = pd.read_parquet(preds_path)

    # find prob column
    for c in ["pd_cal","pd_platt","p_cal","prob_cal","pd","p","prob","y_pred"]:
        if c in df.columns:
            pcol = c
            break
    else:
        raise SystemExit("No probability column found in recent predictions.")

    # threshold that approves target fraction (approve if PD <= threshold)
    t = float(df[pcol].quantile(target))

    # compute actual approve rate (can differ slightly due to ties)
    approve_rate = float((df[pcol] <= t).mean())

    choice = {
        "threshold": t,
        "policy": "balanced",
        "target_approve_rate": target,
        "actual_approve_rate_recent": approve_rate,
        "prob_column": pcol,
        "source": "recent_calibrated_quantile" if preds_cal.exists() else "recent_raw_quantile",
        "file": str(preds_path)
    }

    out = root / "reports" / "policy_choice.json"
    out.write_text(json.dumps(choice, indent=2))
    print(f"Wrote -> {out}")
    print(f"prob_col={pcol} | threshold={t:.6f} | approve_rate={approve_rate:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.20, help="target approval rate (0-1)")
    args = ap.parse_args()
    main(target=args.target)
