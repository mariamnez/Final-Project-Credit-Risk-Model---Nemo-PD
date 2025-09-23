# scripts/compute_rmse.py
import argparse, math, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=r"data/processed/test_predictions.parquet",
                    help="Path to a parquet with labels + probabilities")
    ap.add_argument("--label-col", default="y_true")
    ap.add_argument("--prob-col", default=None,
                    help="If not set, auto-detect one of: pd_cal,p_cal,y_pred,pd,p,prob")
    args = ap.parse_args()

    # load parquet
    df = pd.read_parquet(args.preds)

    # pick probability column
    cand = ["pd_cal","p_cal","y_pred","pd","p","prob"]
    pcol = args.prob_col or next((c for c in cand if c in df.columns), None)
    if pcol is None:
        raise SystemExit(f"Could not find prob column; tried {cand}. Use --prob-col.")
    if args.label_col not in df.columns:
        raise SystemExit(f"Label column '{args.label_col}' not found in {args.preds}")

    y = df[args.label_col].astype(int).to_numpy()
    p = df[pcol].astype(float).to_numpy()

    brier = ((y - p) ** 2).mean()
    rmse  = math.sqrt(brier)

    print(f"file      : {args.preds}")
    print(f"label_col : {args.label_col}")
    print(f"prob_col  : {pcol}")
    print(f"Brier     : {brier:.10f}")
    print(f"RMSE      : {rmse:.6f}")

if __name__ == "__main__":
    main()
