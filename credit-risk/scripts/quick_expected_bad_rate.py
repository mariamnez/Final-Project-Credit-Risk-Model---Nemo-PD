# scripts/quick_expected_bad_rate.py
import json
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
preds = root / "data/processed/recent_predictions_calibrated.parquet"
pol   = json.loads((root / "reports/policy_choice.json").read_text())
t     = float(pol["threshold"])
df    = pd.read_parquet(preds)
pcol  = next(c for c in ["pd_cal","p_cal","y_pred","pd_raw","p"] if c in df)
approved = df[df[pcol] <= t]
print(f"approve_rate = {len(approved)/len(df):.4f}")
print(f"expected_bad_rate_on_approved = {approved[pcol].mean():.4%}")
