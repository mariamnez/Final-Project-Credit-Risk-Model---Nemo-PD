# src/features/build_features.py
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  
DATA = ROOT / "data" / "processed"
RPTS = ROOT / "reports"
DATA.mkdir(parents=True, exist_ok=True)
RPTS.mkdir(parents=True, exist_ok=True)

ABT = DATA / "abt.parquet"                   
OUTS = {
    "train": DATA / "abt_train.parquet",
    "valid": DATA / "abt_valid.parquet",
    "test":  DATA / "abt_test.parquet",
    "recent":DATA / "abt_recent.parquet",
}
MEDIANS_JSON = RPTS / "feature_medians.json"

# helpers
def _clip(series, low=None, high=None):
    s = series.copy()
    if low  is not None: s = s.clip(lower=low)
    if high is not None: s = s.clip(upper=high)
    return s

def _safe(series, default=np.nan):
    try: return series.astype(float)
    except Exception: return pd.Series(default, index=series.index)

def target_mean_encode(train, col, y, smoothing=100.0):
    """
    Mean target encoding with smoothing on TRAIN only.
    Returns mapping (dict) and a transform function for any df.
    """
    g = train.groupby(col)[y].agg(["mean","count"])
    global_mean = train[y].mean()
    smoothed = (g["mean"]*g["count"] + global_mean*smoothing) / (g["count"] + smoothing)
    mapping = smoothed.to_dict()
    def transform(df):
        out = df[col].map(mapping).astype(float)
        return out.fillna(global_mean)
    return mapping, transform

df = pd.read_parquet(ABT)
y_col = next((c for c in ["default_within_24m","y","label"] if c in df.columns), None)
vintage_col = next((c for c in ["vintage_q","orig_vintage_q","orig_yrq"] if c in df.columns), None)

assert y_col is not None, "Label column not found (expected default_within_24m / y)."
assert vintage_col is not None, "vintage (quarter) column not found."

# minimal cleaning / guards
for c in ["fico","dti","ltv","cltv","orig_rate"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
if "dti" in df:  df["dti"]  = _clip(df["dti"],  0, 65)
if "ltv" in df:  df["ltv"]  = _clip(df["ltv"],  0, 120)
if "cltv" in df: df["cltv"] = _clip(df["cltv"], 0, 130)
if "fico" in df: df["fico"] = _clip(df["fico"], 300, 850)

# engineered features (all from ABT)
if "orig_rate" in df and vintage_col in df:
    med_by_vint = df.groupby(vintage_col)["orig_rate"].transform("median")
    df["rate_spread_vintage"] = (df["orig_rate"] - med_by_vint).astype(float)

if "fico" in df:
    df["fico_gap"] = (850.0 - df["fico"]).astype(float)

if "ltv" in df:
    df["ltv_80"] = np.maximum(df["ltv"] - 80.0, 0.0)
    df["ltv_90"] = np.maximum(df["ltv"] - 90.0, 0.0)

if "dti" in df:
    df["dti_43"] = np.maximum(df["dti"] - 43.0, 0.0)

if set(["fico","ltv"]).issubset(df.columns):
    df["int_fico_gap_x_ltv"] = (850.0 - df["fico"]) * df["ltv"]
if set(["dti","ltv"]).issubset(df.columns):
    df["int_dti_x_ltv"] = df["dti"] * df["ltv"]

# (train≤2019, valid=2020, test=2021–2022, recent≥2023)
# Assumes vintage like '2019Q4' or datetime-like month
def vint_to_year(v):
    if pd.isna(v): return np.nan
    s = str(v)
    for key in ["Q","q","-","/"]:
        if key in s:
            try: return int(s[:4])
            except: break
    try:
        return pd.to_datetime(v).year
    except:
        return np.nan

years = df[vintage_col].map(vint_to_year)
mask_train = years <= 2019
mask_valid = years == 2020
mask_test  = (years >= 2021) & (years <= 2022)
mask_recent= years >= 2023

splits = {
    "train": df[mask_train].copy(),
    "valid": df[mask_valid].copy(),
    "test":  df[mask_test].copy(),
    "recent":df[mask_recent].copy()
}

for col in ["state","channel","purpose"]:
    if col in df.columns:
        mapping, xf = target_mean_encode(splits["train"], col, y_col, smoothing=100.0)
        for k in splits:
            splits[k][f"{col}_te"] = xf(splits[k])


drop_cols = {
    y_col, "loan_id", "loanid", "first_payment_date", "first_90dpd",
    "orig_date", "orig_dt", vintage_col
}
for k in splits:
    s = splits[k]
    # keep numeric
    keep = [c for c in s.columns
            if (c not in drop_cols)
            and (np.issubdtype(s[c].dtype, np.number))]
    splits[k] = s[keep + [y_col, vintage_col]].copy()

medians = splits["train"].drop(columns=[y_col, vintage_col], errors="ignore").median(numeric_only=True).to_dict()
MEDIANS_JSON.write_text(json.dumps(medians, indent=2))

# write splits
for name, dfp in splits.items():
    dfp.to_parquet(OUTS[name], index=False)

print(f"train: {len(splits['train']):,} | valid: {len(splits['valid']):,} | test: {len(splits['test']):,} | recent: {len(splits['recent']):,}")
print(f"Saved medians -> {MEDIANS_JSON}")
for k,p in OUTS.items(): print(f"Wrote {k} -> {p}")
