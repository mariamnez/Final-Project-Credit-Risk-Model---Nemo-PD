# scripts/backtest_recent_policy.py
import argparse, json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ABT_RECENT = ROOT / "data" / "processed" / "abt_recent.parquet"
RAW_PREDS   = ROOT / "data" / "processed" / "recent_predictions.parquet"
CAL_PREDS   = ROOT / "data" / "processed" / "recent_predictions_calibrated.parquet"
POLICY_JSON = ROOT / "reports" / "policy_choice.json"
OUT_CSV     = ROOT / "reports" / "backtest_recent_policy.csv"

def load_policy():
    if not POLICY_JSON.exists():
        sys.exit(f"Missing policy choice: {POLICY_JSON}. Run policy_curve/set_policy_threshold first.")
    pol = json.loads(POLICY_JSON.read_text())
    t = float(pol.get("threshold"))
    prob = pol.get("prob", None) or pol.get("prob_col", None) or "pd_cal"
    return t, prob

def pick_prob_col(df, preferred):
    candidates = [preferred, "pd_cal", "p_cal", "y_pred", "pd_raw", "p"]
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(f"No probability column found in predictions. Searched {candidates}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_calibrated", action="store_true",
                    help="Use calibrated predictions file")
    ap.add_argument("--out", type=str, default=str(OUT_CSV),
                    help="Output CSV path")
    args = ap.parse_args()

    preds_path = CAL_PREDS if args.use_calibrated else RAW_PREDS
    if not preds_path.exists():
        sys.exit(f"Missing predictions file: {preds_path}. Run score_recent.py and calibrate_recent.py.")

    if not ABT_RECENT.exists():
        sys.exit(f"Missing recent ABT: {ABT_RECENT}. Run the EDA/build pipeline to create it.")

    # Load artifacts
    t, pref_prob = load_policy()
    df_abt   = pd.read_parquet(ABT_RECENT)
    df_pred  = pd.read_parquet(preds_path)

    # Align by row order
    df = pd.concat([df_abt.reset_index(drop=True), df_pred.reset_index(drop=True)], axis=1)

    # Pick probability column
    pcol = pick_prob_col(df, pref_prob)
    df["approved"] = (df[pcol] <= t)

    cols = df.columns
    has_label = "default_within_24m" in cols
    has_vint  = "vintage_q" in cols

    # Aggregate
    if has_vint:
        g = df.groupby("vintage_q", as_index=False)
    else:
        g = [(None, df)]

    rows = []
    def summarize(d):
        n = len(d)
        approve_rate = float(d["approved"].mean())
        row = {
            "n": n,
            "approve_rate": approve_rate,
        }
        if has_label:
            overall_bad_rate  = float(d["default_within_24m"].mean())
            approved_bad_rate = float(d.loc[d["approved"], "default_within_24m"].mean()) if d["approved"].any() else float("nan")
            row.update({
                "approved_bad_rate": approved_bad_rate,
                "overall_bad_rate": overall_bad_rate,
            })
        return row

    if has_vint:
        for vint, d in g:
            row = {"vintage_q": vint}
            row.update(summarize(d))
            rows.append(row)
    else:
        row = summarize(df)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"Wrote -> {args.out}")

    if len(out)==1:
        ar = out.iloc[0]["approve_rate"]
        print(f"approve_rate={ar:.4f}"
              + ("" if not has_label else
                 f", approved_bad_rate={out.iloc[0]['approved_bad_rate']:.4%}, overall_bad_rate={out.iloc[0]['overall_bad_rate']:.4%}"))
    else:
        print(out.head())

if __name__ == "__main__":
    main()
