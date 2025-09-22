# scripts/check_threshold.py
from pathlib import Path
import json
import pandas as pd

root = Path(__file__).resolve().parents[1]

# 1) load chosen threshold
choice = json.loads((root / "reports" / "policy_choice.json").read_text())
t = float(choice["threshold"])

# 2) pick recent predictions file (calibrated if present)
preds_cal = root / "data" / "processed" / "recent_predictions_calibrated.parquet"
preds_raw = root / "data" / "processed" / "recent_predictions.parquet"
preds_path = preds_cal if preds_cal.exists() else preds_raw

df = pd.read_parquet(preds_path)

# 3) find probability column
for c in ["pd_cal", "pd_platt", "p_cal", "prob_cal", "pd", "p", "prob", "y_pred"]:
    if c in df.columns:
        pcol = c
        break
else:
    raise SystemExit("No probability/prob column found in recent predictions.")

approve_rate = float((df[pcol] <= t).mean())

print(f"file       : {preds_path}")
print(f"threshold  : {t:.6f}")
print(f"prob_col   : {pcol}")
print(f"approve_rate = {approve_rate:.4f}")
