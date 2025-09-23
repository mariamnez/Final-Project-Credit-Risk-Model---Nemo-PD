# scripts/peek_pd_recent.py
from pathlib import Path
import pandas as pd

RECENT = Path(r"data/processed/recent_predictions_calibrated.parquet")

df = pd.read_parquet(RECENT)

candidates = ["pd_cal", "p_cal", "y_pred", "pd", "p"]
prob_col = next(c for c in candidates if c in df.columns)

q = df[prob_col].quantile([0, .001, .01, .05, .10, .20, .50, .80, .90, .95, .99, 1.0])

print("file     :", RECENT)
print("prob_col :", prob_col)
print("\nquantiles:")
print(q.to_string())

for t in [0.002, 0.005, 0.010, 0.020, 0.030, 0.050, 0.100]:
    print(f"share <= {t:0.3f} :", float((df[prob_col] <= t).mean()))
