# scripts/slice_approved_share_recent.py
import json
import pathlib
import pandas as pd

ROOT = pathlib.Path(".")
POLICY = ROOT / "reports" / "policy_choice.json"
PRED = ROOT / "data" / "processed" / "recent_predictions_calibrated.parquet"
ABT_RECENT = ROOT / "data" / "processed" / "abt_recent.parquet"
OUT = ROOT / "reports" / "slice_approved_share_recent.csv"

SLICE_CANDIDATES = ["channel", "purpose", "state", "msa", "vintage_q"]

def load_policy():
    d = json.loads(POLICY.read_text())
    t = float(d["threshold"])
    prob = d.get("prob") or d.get("prob_col") or "pd_cal"
    return t, prob

def ensure_slice_columns(df_pred: pd.DataFrame) -> pd.DataFrame:
    """If no slice columns exist in df_pred, try to augment from ABT_RECENT."""
    have = [c for c in SLICE_CANDIDATES if c in df_pred.columns]
    if have:
        return df_pred 

    if not ABT_RECENT.exists():
        print(f"[warn] No slice cols in predictions and {ABT_RECENT} not found. "
              "Will fall back to global stats.")
        return df_pred

    df_abt = pd.read_parquet(ABT_RECENT, columns=["loan_id", *SLICE_CANDIDATES] if "loan_id" in df_pred.columns or "loan_id" in df_abt.columns else SLICE_CANDIDATES)

    if "loan_id" in df_pred.columns and "loan_id" in df_abt.columns:
        df_merged = df_pred.merge(df_abt[["loan_id", *[c for c in SLICE_CANDIDATES if c in df_abt.columns]]],
                                  on="loan_id", how="left", validate="one_to_one")
        return df_merged

    if len(df_pred) == len(df_abt):
        df_pred = df_pred.reset_index(drop=True)
        df_abt = df_abt.reset_index(drop=True)
        for c in SLICE_CANDIDATES:
            if c in df_abt.columns and c not in df_pred.columns:
                df_pred[c] = df_abt[c]
        return df_pred

    print("[warn] Could not augment slices (no loan_id match and length mismatch). "
          "Will fall back to global stats.")
    return df_pred

def main():
    assert PRED.exists(), f"Missing predictions file: {PRED}"
    t, prob_col = load_policy()

    df = pd.read_parquet(PRED)
    if prob_col not in df.columns:
        prob_col = next((c for c in ["pd_cal", "p_cal", "pd", "p", "y_pred"] if c in df.columns), None)
        if not prob_col:
            raise ValueError("No probability column found in predictions file.")

    df = ensure_slice_columns(df)
    slices = [c for c in SLICE_CANDIDATES if c in df.columns]

    df = df.assign(approved=(df[prob_col] <= t))

    if not slices:
        out = pd.DataFrame([{
            "slice": "GLOBAL",
            "approve_rate": float(df["approved"].mean()),
            "n": int(len(df))
        }])
        out.to_csv(OUT, index=False)
        print(f"[info] No slice columns available; wrote global stats -> {OUT}")
        print(out)
        return

    out = (df.groupby(slices, dropna=False)["approved"]
             .agg(["mean", "count"])
             .rename(columns={"mean": "approve_rate", "count": "n"})
             .reset_index())

    out.to_csv(OUT, index=False)
    print(f"Wrote -> {OUT}")
    print(out.head(10))

if __name__ == "__main__":
    main()
