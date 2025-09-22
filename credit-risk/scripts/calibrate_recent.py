# scripts/calibrate_recent.py
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parents[1]
DATA    = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

RAW_PRED = DATA / "recent_predictions.parquet"
SCALER   = REPORTS / "platt_scaler.json"
OUT      = DATA / "recent_predictions_calibrated.parquet"

def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0/(1.0 + np.exp(-z))

def main():
    assert RAW_PRED.exists(), f"Missing raw preds: {RAW_PRED}"
    assert SCALER.exists(),   f"Missing platt scaler: {SCALER} (run scripts\\calibrate_platt.py)"

    params = json.loads(SCALER.read_text())
    a = float(params["a"]); b = float(params["b"])

    df = pd.read_parquet(RAW_PRED)
    pcol = next(c for c in df.columns if c.lower() in {"pd_raw","p_raw","pd","p","prob"})
    p = df[pcol].clip(1e-6, 1-1e-6).to_numpy()
    x = np.log(p/(1-p))

    df["pd_cal"] = sigmoid(a + b*x)
    df.to_parquet(OUT, index=False)
    print(f"Wrote -> {OUT} | rows={len(df):,}")

if __name__ == "__main__":
    main()
