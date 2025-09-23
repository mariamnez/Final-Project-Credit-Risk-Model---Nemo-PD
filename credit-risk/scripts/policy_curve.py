# scripts/policy_curve.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED_DEFAULTS = [
    ROOT / "data" / "processed" / "test_predictions_calibrated.parquet",
    ROOT / "data" / "processed" / "test_predictions.parquet",
]
OUT_CURVE = ROOT / "reports" / "policy_curve.csv"
OUT_CHOICE = ROOT / "reports" / "policy_choice.json"

CAL_COLS = ["pd_cal", "pd_platt", "p_cal", "prob_cal"]
RAW_COLS = ["pd", "p", "prob", "y_pred"]
Y_COLS   = ["y_true", "y", "label"]

def parse_grid(spec: str | None) -> np.ndarray:
    if not spec:
        return np.arange(0.01, 0.251, 0.005)
    if ":" in spec:
        a, b, s = map(float, spec.split(":"))
        return np.arange(a, b + 1e-12, s)
    return np.array([float(x) for x in spec.split(",")], dtype=float)

def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def ks_stat(y_true: np.ndarray, score: np.ndarray) -> float:
    from scipy.stats import ks_2samp  
    return float(ks_2samp(score[y_true == 1], score[y_true == 0]).statistic)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices={"f1","ks"}, default="f1",
                    help="Primary selection metric (f1 or ks).")
    ap.add_argument("--grid", default=None,
                    help='Threshold grid. Formats: "start:stop:step" or "t1,t2,..."')
    ap.add_argument("--use-calibrated", "--calibrated", action="store_true",
                    help="Force using calibrated PD column if present.")
    ap.add_argument("--preds", default=None,
                    help="Optional path to a predictions parquet.")
    args = ap.parse_args()

    pred_path = Path(args.preds) if args.preds else next((p for p in PRED_DEFAULTS if p.exists()), None)
    if pred_path is None:
        raise FileNotFoundError(
            f"Could not find predictions file. Tried:\n  " + "\n  ".join(map(str, PRED_DEFAULTS))
        )

    df = pd.read_parquet(pred_path)
    ycol = find_col(df, Y_COLS)
    if not ycol:
        raise ValueError(f"Could not find label column in {pred_path}. Checked {Y_COLS}")

    pcol = None
    if args.use_calibrated:
        pcol = find_col(df, CAL_COLS) or find_col(df, RAW_COLS)
    else:
        pcol = find_col(df, CAL_COLS) or find_col(df, RAW_COLS)
    if not pcol:
        raise ValueError(f"Could not find probability column. "
                         f"Checked calibrated {CAL_COLS} then raw {RAW_COLS}")

    y = df[ycol].astype(int).values
    p = df[pcol].astype(float).values
    grid = parse_grid(args.grid)

    rows = []
    for t in grid:
        approve = (p <= t)
        approve_rate = float(approve.mean())

        yhat_bad = (p >= t).astype(int)

        tp = int(((y == 1) & (yhat_bad == 1)).sum())
        fp = int(((y == 0) & (yhat_bad == 1)).sum())
        tn = int(((y == 0) & (yhat_bad == 0)).sum())
        fn = int(((y == 1) & (yhat_bad == 0)).sum())

        precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        recall    = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        ks = ks_stat(y, p)

        approved_bad_rate = float(p[approve].mean()) if approve.any() else np.nan

        rows.append({
            "threshold": float(t),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ks": ks,
            "approve_rate": approve_rate,
            "approved_bad_rate": approved_bad_rate,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })

    curve = pd.DataFrame(rows)
    OUT_CURVE.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(OUT_CURVE, index=False)

    # choose best threshold by metric
    if args.metric == "f1":
        best_idx = int(curve["f1"].idxmax())
    else:
        best_idx = int(curve["ks"].idxmax())
    choice = {
        "threshold": float(curve.loc[best_idx, "threshold"]),
        "metric": args.metric,
        "prob_col": pcol,
        "file": str(pred_path),
    }
    OUT_CHOICE.write_text(json.dumps(choice, indent=2))
    print(f"Wrote curve -> {OUT_CURVE}")
    print(f"Choice     -> {OUT_CHOICE}  (t={choice['threshold']:.6f}, metric={args.metric}, prob={pcol})")

if __name__ == "__main__":
    main()

