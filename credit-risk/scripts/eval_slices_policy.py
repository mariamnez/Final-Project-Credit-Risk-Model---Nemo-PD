# scripts/eval_slices_policy.py
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(".").resolve()
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

CAL_FILE = DATA / "test_predictions_calibrated.parquet"
RAW_FILE = DATA / "test_predictions.parquet"
POLICY_JSON = REPORTS / "policy_choice.json"

TEST_ABT = DATA / "abt_test.parquet"   # to bring grouping columns


def _load_predictions():
    src = CAL_FILE if CAL_FILE.exists() else RAW_FILE
    if not src.exists():
        raise FileNotFoundError(f"Missing predictions: {src}")
    dfp = pd.read_parquet(src)
    score_col = "y_pred_cal" if "y_pred_cal" in dfp.columns else "y_pred"
    y = dfp["y_true"].astype(int).to_numpy()
    p = dfp[score_col].astype(float).to_numpy()
    pid = dfp["loan_id"] if "loan_id" in dfp.columns else pd.Series(range(len(dfp)))
    return dfp.assign(score=p, loan_id=pid), score_col, src


def _load_policy():
    if not POLICY_JSON.exists():
        raise FileNotFoundError(POLICY_JSON)
    pol = json.loads(POLICY_JSON.read_text())
    return float(pol["threshold"]), pol.get("metric", "f1"), pol.get("score_col", "y_pred")


def _join_test_features(dfp: pd.DataFrame) -> pd.DataFrame:
    if not TEST_ABT.exists():
        raise FileNotFoundError(TEST_ABT)
    dft = pd.read_parquet(TEST_ABT)
    keep_cols = [c for c in ["loan_id", "vintage_q", "channel", "purpose", "state", "msa"] if c in dft.columns]
    dft = dft[keep_cols].copy()
    df = dfp.merge(dft, on="loan_id", how="left")
    # ensure vintage is datetime
    if "vintage_q" in df.columns:
        df["vintage_q"] = pd.to_datetime(df["vintage_q"])
    return df


def _policy_metrics(df: pd.DataFrame, threshold: float, group: str) -> pd.DataFrame:
    # approved = predicted good = score < t
    df = df.copy()
    df["approved"] = (df["score"] < threshold)
    # y_true == 1 => default/bad
    g = df.groupby(group, dropna=False)
    out = g.apply(lambda d: pd.Series({
        "n": int(len(d)),
        "approve_rate": float(d["approved"].mean()),
        "approved": int(d["approved"].sum()),
        "approved_bad_rate": float(( (d["approved"]) & (d["y_true"] == 1) ).sum() / max(1, d["approved"].sum())),
        "overall_bad_rate": float(d["y_true"].mean())
    })).reset_index()
    return out


def main():
    dfp, score_col, src = _load_predictions()
    thr, metric, sc_used = _load_policy()
    print(f"Using source={src.name} score_col={score_col} threshold={thr:.6f} by {metric}")

    df = _join_test_features(dfp)

    # ---- vintage (always if present)
    if "vintage_q" in df.columns:
        vint = _policy_metrics(df, thr, "vintage_q").sort_values("vintage_q")
        vint.to_csv(REPORTS / "policy_vintage_metrics.csv", index=False)
        print("Wrote:", REPORTS / "policy_vintage_metrics.csv")

    # ---- channel
    if "channel" in df.columns:
        ch = _policy_metrics(df, thr, "channel").sort_values("approve_rate")
        ch.to_csv(REPORTS / "policy_channel_metrics.csv", index=False)
        print("Wrote:", REPORTS / "policy_channel_metrics.csv")

    # ---- purpose
    if "purpose" in df.columns:
        pu = _policy_metrics(df, thr, "purpose").sort_values("approve_rate")
        pu.to_csv(REPORTS / "policy_purpose_metrics.csv", index=False)
        print("Wrote:", REPORTS / "policy_purpose_metrics.csv")

    # ---- state top 20 by volume
    if "state" in df.columns:
        vol = df["state"].value_counts().head(20).index.tolist()
        st = _policy_metrics(df[df["state"].isin(vol)], thr, "state").sort_values("approve_rate")
        st.to_csv(REPORTS / "policy_state_top20_metrics.csv", index=False)
        print("Wrote:", REPORTS / "policy_state_top20_metrics.csv")


if __name__ == "__main__":
    main()
