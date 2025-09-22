# scripts/score_recent.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

MODEL_TXT = REPORTS / "lgbm_model.txt"
MEDIANS_JSON = REPORTS / "feature_medians.json"
RECENT_ABT = DATA / "abt_recent.parquet"
OUT_PATH = DATA / "recent_predictions.parquet"

def load_medians():
    if MEDIANS_JSON.exists():
        return json.loads(MEDIANS_JSON.read_text())
    return {}

def main():
    assert MODEL_TXT.exists(), f"Missing model file: {MODEL_TXT}"
    assert RECENT_ABT.exists(), f"Missing recent ABT: {RECENT_ABT}"
    medians = load_medians()

    df = pd.read_parquet(RECENT_ABT)
    print(f"Loaded recent ABT: {len(df):,} rows")

    booster = lgb.Booster(model_file=str(MODEL_TXT))
    feat = booster.feature_name()

    # Keep *exactly* model features, in order
    X = df.reindex(columns=feat)

    # Coerce all to numeric; non-numeric becomes NaN
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Fill missing by medians when available, otherwise 0
    for c in X.columns:
        if c in medians and medians[c] is not None:
            X[c] = X[c].fillna(medians[c])
        else:
            X[c] = X[c].fillna(0.0)

    # Predict raw PD as 1D numpy
    preds = booster.predict(X.values)

    out = pd.DataFrame({
        "pd_raw": preds.astype(float)
    })
    # Optional: keep a key to join back (if present)
    for k in ("loan_id", "orig_date", "msa"):
        if k in df.columns and k not in out.columns:
            out[k] = df[k].values

    out.to_parquet(OUT_PATH, index=False)
    print(f"Wrote -> {OUT_PATH} | rows={len(out):,}")

if __name__ == "__main__":
    main()
