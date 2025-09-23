# scripts/build_macro_features.py
import pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(".")
ABT_IN  = ROOT/"data/processed/abt.parquet"
ABT_OUT = ROOT/"data/processed/abt_plus.parquet"

EXT = ROOT/"data/external"
UNEMP = EXT/"unemployment_state_monthly.csv"
HPI   = EXT/"fhfa_hpi_state_monthly.csv"
PMMS  = EXT/"pmms_30y_rate_monthly.csv"

def safe_read_csv(path, **kw):
    if path.exists():
        df = pd.read_csv(path, **kw)
        for c in df.columns:  
            df.rename(columns={c: c.strip()}, inplace=True)
        return df
    print(f"[warn] missing {path}; continuing without it")
    return pd.DataFrame()

def prep_month(s):
    s = pd.to_datetime(s, errors="coerce")
    return s.dt.to_period("M").astype(str)

def main():
    df = pd.read_parquet(ABT_IN)
    if "orig_date" in df and df["orig_date"].notna().any():
        df["orig_month"] = prep_month(df["orig_date"])
    else:
        df["orig_month"] = prep_month(pd.to_datetime(df["first_payment_date"], errors="coerce") - pd.offsets.MonthBegin(1))

    # unemployment
    un = safe_read_csv(UNEMP)
    if not un.empty:
        un["date"] = un["date"].astype(str)
        df = df.merge(un.rename(columns={"date":"orig_month"}), how="left",
                      left_on=["state","orig_month"], right_on=["state","orig_month"])

    hpi = safe_read_csv(HPI)
    if not hpi.empty:
        hpi["date"] = pd.to_datetime(hpi["date"])
        hpi = hpi.sort_values(["state","date"])
        hpi["hpi_12m_chg"] = hpi.groupby("state")["hpi_index"].pct_change(12)
        hpi["orig_month"] = hpi["date"].dt.to_period("M").astype(str)
        hpi = hpi.drop(columns=["date"])
        df = df.merge(hpi, how="left", on=["state","orig_month"])

    # PMMS (market rate) and rate spread
    pm = safe_read_csv(PMMS)
    if not pm.empty:
        pm["orig_month"] = pm["date"].astype(str)
        pm = pm[["orig_month","pmms_30y"]]
        df = df.merge(pm, how="left", on="orig_month")
        if "orig_rate" in df.columns:
            df["rate_spread"] = df["orig_rate"] - df["pmms_30y"]

    added = [c for c in ["unemp_rate","hpi_index","hpi_12m_chg","pmms_30y","rate_spread"] if c in df.columns]
    print("Added cols:", added)

    df.to_parquet(ABT_OUT, index=False)
    print(f"Wrote -> {ABT_OUT}  | rows={len(df):,}")

if __name__ == "__main__":
    main()
