import pandas as pd
import numpy as np
from pathlib import Path
import json

CURVE = Path("reports/policy_curve.csv")
CHOICE = Path("reports/policy_choice.json")
OUT_WIDE = Path("reports/policy_curve_canva_wide.csv")
OUT_LONG = Path("reports/policy_curve_canva_long.csv")
OUT_APPROVE = Path("reports/policy_curve_canva_approve.csv")

def read_curve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "ks_at_p": "ks",
        "approve_rate_at_p": "approve_rate",
        "Threshold": "threshold",
        "Recall": "recall",
        "Precision": "precision",
        "F1": "f1",
    })
    for c in ["threshold","precision","recall","f1","ks","approve_rate","tp","fp","tn","fn"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["threshold"]).sort_values("threshold").reset_index(drop=True)

    if ("ks" not in df.columns) or (df["ks"].nunique(dropna=True) <= 1):
        if all(c in df.columns for c in ["tp","fp","tn","fn"]):
            tpr = df["tp"] / (df["tp"] + df["fn"]).replace(0, np.nan)
            fpr = df["fp"] / (df["fp"] + df["tn"]).replace(0, np.nan)
            df["ks"] = (tpr - fpr).clip(0,1).fillna(0)

    return df

def choice_t(path: Path) -> float|None:
    if not path.exists():
        return None
    try:
        j = json.loads(path.read_text())
        return float(j.get("threshold", j.get("t")))
    except Exception:
        return None

def main():
    df = read_curve(CURVE)
    t = choice_t(CHOICE)
    if t is None and "ks" in df.columns and df["ks"].notna().any():
        t = float(df.loc[df["ks"].idxmax(),"threshold"])

    keep = ["threshold","recall","precision","f1","ks","approve_rate"]
    keep = [k for k in keep if k in df.columns]
    df = df[keep].copy()

    if t is not None:
        span = 0.06  
        df = df[(df["threshold"] >= t - span/2) & (df["threshold"] <= t + span/2)]

    # Wide format
    df_wide = df.copy().round(6)
    df_wide.to_csv(OUT_WIDE, index=False)

    # Long format
    value_cols = [c for c in ["recall","precision","f1","ks"] if c in df.columns]
    df_long = df.melt(id_vars=["threshold"], value_vars=value_cols,
                      var_name="metric", value_name="value").dropna()
    df_long = df_long.sort_values(["metric","threshold"]).round(6)
    df_long.to_csv(OUT_LONG, index=False)

    if "approve_rate" in df.columns:
        df[["threshold","approve_rate"]].round(6).to_csv(OUT_APPROVE, index=False)

    print(f"WIDE  -> {OUT_WIDE}")
    print(f"LONG  -> {OUT_LONG}")
    if "approve_rate" in df.columns:
        print(f"APPROVE -> {OUT_APPROVE}")

if __name__ == "__main__":
    main()
