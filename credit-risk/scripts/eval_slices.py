# scripts/eval_slices.py
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# ---- inputs (robust fallbacks) ----
PRED_PATHS = [
    DATA / "test_predictions_calibrated.parquet",
    DATA / "test_predictions.parquet",
]
ABT_PATHS = [
    DATA / "abt.parquet",          # full ABT (if present)
    DATA / "abt_test.parquet",     # test-only ABT (if present)
]

SLICE_CANDIDATES = ["vintage_q", "state", "channel", "purpose"]
ID_CANDIDATES = ["loan_id", "unique_loan_id", "loanid", "id"]

def ks_stat(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # KS = max(TPR - FPR)
    # guard against degenerate cases
    y = np.asarray(y_true)
    s = np.asarray(y_score)
    if y.min() == y.max():  # only one class present
        return np.nan
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(tpr - fpr))

def first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None

def resolve_id(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    for c in ID_CANDIDATES:
        if c in df.columns:
            return c, df
    # synthesize a row id if nothing exists
    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    return "row_id", df

def safe_auc(y, p) -> float:
    y = np.asarray(y)
    if y.min() == y.max():
        return np.nan
    return float(roc_auc_score(y, p))

def safe_pr_auc(y, p) -> float:
    y = np.asarray(y)
    if y.min() == y.max():
        return np.nan
    return float(average_precision_score(y, p))

def compute_slice_metrics(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by, observed=True)
    out = g.apply(
        lambda t: pd.Series(
            {
                "n": len(t),
                "bad_rate": float(t["y_true"].mean()),
                "auc": safe_auc(t["y_true"].values, t["y_pred"].values),
                "pr_auc": safe_pr_auc(t["y_true"].values, t["y_pred"].values),
                "ks": ks_stat(t["y_true"].values, t["y_pred"].values),
            }
        )
    )
    out.index.name = by
    return out.reset_index()

def main():
    pred_path = first_existing(*PRED_PATHS)
    if pred_path is None:
        raise FileNotFoundError("No predictions found. Expected one of: "
                                + ", ".join(str(p) for p in PRED_PATHS))

    abt_path = first_existing(*ABT_PATHS)
    if abt_path is None:
        raise FileNotFoundError("No ABT found. Expected one of: "
                                + ", ".join(str(p) for p in ABT_PATHS))

    preds = pd.read_parquet(pred_path)
    abt = pd.read_parquet(abt_path)

    # Identify id columns and slice columns that exist
    id_col_preds, preds = resolve_id(preds)
    id_col_abt, abt = resolve_id(abt)
    # If ids differ, align names for merge
    if id_col_preds != id_col_abt:
        preds = preds.rename(columns={id_col_preds: id_col_abt})
        id_col = id_col_abt
    else:
        id_col = id_col_preds

    have_slices = [c for c in SLICE_CANDIDATES if c in abt.columns]
    if not have_slices:
        raise ValueError("No slice columns found in ABT. "
                         f"Looked for: {SLICE_CANDIDATES}")

    # keep only needed meta cols
    keep_cols = [id_col] + have_slices
    meta = abt.loc[:, keep_cols].copy()

    # predictions columns (robust): prefer calibrated col names
    prob_cols_pref = ["y_pred_cal", "pd_cal", "pd_platt", "p_cal", "prob_cal",
                      "y_pred", "pd", "p", "prob"]
    pcol = next((c for c in prob_cols_pref if c in preds.columns), None)
    if pcol is None:
        raise KeyError(f"No probability column found in predictions. "
                       f"Tried: {prob_cols_pref}")
    ycol = next((c for c in ["y_true", "label", "default_within_24m"] if c in preds.columns), None)
    if ycol is None:
        raise KeyError("No label column found in predictions. Expected one of "
                       "['y_true', 'label', 'default_within_24m'].")

    df = preds[[id_col, ycol, pcol]].copy()
    df = df.rename(columns={ycol: "y_true", pcol: "y_pred"})

    # Join meta for slicing
    df = df.merge(meta, on=id_col, how="left")

    # Compute and write each slice table that exists
    if "vintage_q" in have_slices:
        vint = compute_slice_metrics(df, "vintage_q")
        vint.to_csv(REPORTS / "slice_vintage_metrics.csv", index=False)

    if "state" in have_slices:
        st = compute_slice_metrics(df, "state").sort_values("n", ascending=False)
        st.head(20).to_csv(REPORTS / "slice_state_top20_metrics.csv", index=False)

    if "channel" in have_slices:
        ch = compute_slice_metrics(df, "channel")
        ch.to_csv(REPORTS / "slice_channel_metrics.csv", index=False)

    if "purpose" in have_slices:
        pu = compute_slice_metrics(df, "purpose")
        pu.to_csv(REPORTS / "slice_purpose_metrics.csv", index=False)

    # small manifest for debugging
    manifest = {
        "preds": str(pred_path),
        "abt": str(abt_path),
        "id_col": id_col,
        "used_slices": have_slices,
        "n_rows_joined": int(len(df)),
    }
    (REPORTS / "slice_eval_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Wrote slice CSVs into", REPORTS)
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
