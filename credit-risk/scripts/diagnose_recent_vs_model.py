# scripts/diagnose_recent_vs_model.py
from pathlib import Path
import json
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

MODEL_TXT = REPORTS / "lgbm_model.txt"
RECENT_ABT = DATA / "abt_recent.parquet"

def main():
    booster = lgb.Booster(model_file=str(MODEL_TXT))
    feat = booster.feature_name()
    df = pd.read_parquet(RECENT_ABT)

    model_set = set(feat)
    abt_set = set(df.columns)

    print(f"Model features: {len(feat)}   Recent columns: {df.shape[1]}")
    missing = [c for c in feat if c not in abt_set]
    extra = [c for c in df.columns if c not in model_set]

    print("\nMissing in ABT (should be 0):", missing[:20], "..." if len(missing)>20 else "")
    print("\nExtra columns in ABT:", extra[:20], "..." if len(extra)>20 else "")

    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if obj_cols:
        print("\nObject-typed cols (first 10 shown):", obj_cols[:10])
        for c in obj_cols[:5]:
            print(f"  {c}: sample ->", df[c].dropna().astype(str).head(3).tolist())

if __name__ == "__main__":
    main()
