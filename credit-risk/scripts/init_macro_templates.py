# scripts/init_macro_templates.py
from pathlib import Path
import pandas as pd

ROOT = Path(".")
ABT = ROOT / "data/processed/abt.parquet"
OUT = ROOT / "data/external"
OUT.mkdir(parents=True, exist_ok=True)

DATE_CANDIDATES = [
    "first_payment_date","first_payment_dt","first_pay_date",
    "orig_date","min_fp","max_fp","min_last","max_last"
]
STATE_CANDIDATES = ["state","addr_state","property_state","st"]

def pick_first(df, cols):
    for c in cols:
        if c in df.columns:
            return c
    return None

def derive_months(df: pd.DataFrame) -> list[str]:
    for c in DATE_CANDIDATES:
        if c in df.columns:
            dt = pd.to_datetime(df[c], errors="coerce")
            if dt.notna().any():
                m = dt.dt.to_period("M").astype(str)
                vals = sorted({x for x in m if x != "NaT"})
                if vals:
                    return vals

    for c in ["vintage_q","orig_vintage","vintage"]:
        if c in df.columns:
            try:
                v = pd.PeriodIndex(df[c].astype(str), freq="Q")
                months = v.asfreq("M", "start_time").strftime("%Y-%m")
                vals = sorted({x for x in months if x != "NaT"})
                if vals:
                    return vals
            except Exception:
                pass

    return [p.strftime("%Y-%m") for p in pd.period_range("2017-01", "2025-03", freq="M")]

def main():
    assert ABT.exists(), f"ABT not found: {ABT}"
    df = pd.read_parquet(ABT)

    months = derive_months(df)
    state_col = pick_first(df, STATE_CANDIDATES)
    if state_col is None or df[state_col].dropna().empty:
        raise RuntimeError("No state column found (state/addr_state/property_state/st).")

    states = sorted(df[state_col].dropna().astype(str).unique().tolist())

    # Unemployment (state monthly)
    pd.DataFrame([(m, s, "") for m in months for s in states],
                 columns=["date","state","unemp_rate"]).to_csv(OUT/"unemployment_state_monthly.csv", index=False)

    # FHFA HPI (state monthly)
    pd.DataFrame([(m, s, "") for m in months for s in states],
                 columns=["date","state","hpi_index"]).to_csv(OUT/"fhfa_hpi_state_monthly.csv", index=False)

    # PMMS 30Y (national monthly)
    pd.DataFrame([(m, "") for m in months],
                 columns=["date","pmms_30y"]).to_csv(OUT/"pmms_30y_rate_monthly.csv", index=False)

    print("Templates written to:", OUT.resolve())
    print("  unemployment_state_monthly.csv")
    print("  fhfa_hpi_state_monthly.csv")
    print("  pmms_30y_rate_monthly.csv")
    print(f"Months: {months[0]} .. {months[-1]}  |  #months={len(months)}  #states={len(states)}")

if __name__ == "__main__":
    main()
