# scripts/build_kpi_history.py
import pandas as pd
import pathlib as p
import re

root = p.Path(__file__).resolve().parents[1]  # project root
src  = root / 'reports' / 'slice_vintage_metrics.csv'

def to_date(s: str):
    s = str(s).strip()
    d = pd.to_datetime(s, errors='coerce')
    if pd.notna(d):
        return d
    m = re.match(r'(\d{4})-?Q([1-4])', s, re.I)           # 2021Q3 or 2021-Q3
    if m:
        year = int(m.group(1)); q = int(m.group(2))
        month = {1:1, 2:4, 3:7, 4:10}[q]
        return pd.Timestamp(year, month, 1)
    m = re.match(r'(\d{4})[-/](\d{1,2})$', s)              # 2021-07 or 2021/07
    if m:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
    return pd.NaT

def main():
    if not src.exists():
        raise FileNotFoundError(f"Cannot find {src}")

    df = pd.read_csv(src)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    for c in ('vintage','vintage_q','present','month','period','date'):
        if c in df.columns:
            datecol = c
            break
    else:
        datecol = df.columns[0]

    df['date'] = df[datecol].map(to_date)
    df = df.dropna(subset=['date']).sort_values('date')

    outdir = root / 'reports' / 'kpi_history'
    outdir.mkdir(parents=True, exist_ok=True)

    def write_two_col(src_col, out_name):
        s = df[['date', src_col]].dropna().copy()
        s.columns = ['date', 'value']
        dst = outdir / f'{out_name}.csv'
        s.to_csv(dst, index=False)
        print(f'Wrote: {dst}')

    candidates = [
        ('auc_roc', 'auc'),
        ('auc', 'auc'),
        ('ks', 'ks'),
        ('approve_rate', 'approve_rate'),
        ('bad_rate', 'bad_rate'),
        ('brier', 'brier'),
        ('ece', 'calibration'),
        ('calibration', 'calibration'),
    ]

    found_any = False
    for src_col, out_name in candidates:
        if src_col in df.columns:
            write_two_col(src_col, out_name)
            found_any = True

    if not found_any:
        print("No KPI columns found. Open reports/slice_vintage_metrics.csv and check column names.")

if __name__ == "__main__":
    main()
