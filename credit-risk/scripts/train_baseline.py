# scripts/train_baseline.py
"""
Baseline training script for Credit Risk PD@24m

- Loads slices directly from data/processed/abt.parquet using DuckDB (RAM-safe)
- Splits by vintage (train 2017–2019, valid 2020, test 2021–2022 by default)
- Models:
    1) Logistic Regression (numeric only, with preprocessing pipeline)
    2) LightGBM (numeric + categorical, native categorical handling)
- Outputs to:
    models/
      - logit_numeric.joblib
      - lgbm_all.joblib
      - lgbm_feature_meta.json  (column order & categorical list)
    reports/
      - metrics.json
      - calibration_valid_lgbm.png
      - policy_curve_valid_lgbm.csv
      - feature_importance_lgbm.csv
      - split_sizes.json

You can tweak splits or downsampling via CLI flags; run with -h to see options.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import math
import warnings

import duckdb
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb
import matplotlib.pyplot as plt


# ----------------------- defaults & schema -----------------------

NUMERIC = [
    "fico",
    "orig_upb",
    "orig_rate",
    "ltv",
    "cltv",
    "dti",
    "loan_term",
    "num_units",
]

CATEGORICAL = [
    "channel",
    "purpose",
    "occupancy",
    "product",
    "property_type",
    "state",
    "first_time_buyer",
]

TARGET = "default_within_24m"

# Default time windows (by vintage_q)
DEFAULT_SPLIT = dict(
    train=("2017-01-01", "2019-12-31"),
    valid=("2020-01-01", "2020-12-31"),
    test=("2021-01-01", "2022-12-31"),
)


# ----------------------- utilities -----------------------

def ks_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov–Smirnov score for binary classifier."""
    order = np.argsort(y_prob)
    y = y_true[order]
    if y.sum() == 0 or y.sum() == len(y):
        return 0.0
    cum_pos = np.cumsum(y) / y.sum()
    cum_neg = np.cumsum(1 - y) / (len(y) - y.sum())
    return float(np.max(np.abs(cum_pos - cum_neg)))


def policy_curve(y_true: np.ndarray, y_prob: np.ndarray, points: int = 101) -> pd.DataFrame:
    """Returns threshold grid with approval rate, bad-rate among approved, and default capture."""
    thr = np.quantile(y_prob, np.linspace(0, 1, points))
    out = []
    total_defaults = y_true.sum()
    for t in thr:
        approve = y_prob < t  # approve if predicted PD < threshold
        approval_rate = approve.mean()
        bad_in_approved = np.nan
        captured_default_share = np.nan
        if approve.any():
            bad_in_approved = float(y_true[approve].mean())
        if total_defaults > 0:
            captured_default_share = float(y_true[~approve].sum() / total_defaults)
        out.append((float(t), float(approval_rate), bad_in_approved, captured_default_share))
    return pd.DataFrame(out, columns=["threshold", "approval_rate", "bad_rate_in_approved", "captured_defaults_share"])


def ensure_columns(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}")


def load_split(
    abt_path: Path,
    start: str,
    end: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    target: str = TARGET,
    downsample_neg_ratio: float | None = None,
    seed: int = 42,
):
    """
    Loads a split from ABT using DuckDB. Optionally downsample negatives to ratio*N_pos.
    Returns (X_num, X_cat, y, df) where df is the full slice (for summaries).
    """
    con = duckdb.connect()
    cols = numeric_cols + categorical_cols + [target, "vintage_q"]
    col_list = ", ".join(cols)
    sql = f"""
        SELECT {col_list}
        FROM read_parquet('{abt_path.as_posix()}')
        WHERE vintage_q BETWEEN DATE '{start}' AND DATE '{end}'
          AND {target} IS NOT NULL
    """
    df = con.execute(sql).df()

    # normalize types
    if df.empty:
        raise ValueError(f"No rows in split {start}..{end}")
    df[target] = df[target].astype("bool").astype("int8")
    for c in categorical_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    # Optional simple downsampling to keep training light
    if downsample_neg_ratio and downsample_neg_ratio > 0:
        pos = df[df[target] == 1]
        neg = df[df[target] == 0]
        take = min(len(neg), int(downsample_neg_ratio * max(1, len(pos))))
        if take < len(neg):  # only sample if we need to
            neg = neg.sample(take, random_state=seed)
        df = pd.concat([pos, neg], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # split features/target
    ensure_columns(df, numeric_cols + categorical_cols + [target], "load_split")
    X_num = df[numeric_cols].copy()
    X_cat = df[categorical_cols].copy()
    y = df[target].to_numpy()
    return X_num, X_cat, y, df


# ----------------------- main -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--abt", default="data/processed/abt.parquet", help="Path to combined ABT parquet")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--reports_dir", default="reports")
    parser.add_argument("--train_start", default=DEFAULT_SPLIT["train"][0])
    parser.add_argument("--train_end", default=DEFAULT_SPLIT["train"][1])
    parser.add_argument("--valid_start", default=DEFAULT_SPLIT["valid"][0])
    parser.add_argument("--valid_end", default=DEFAULT_SPLIT["valid"][1])
    parser.add_argument("--test_start", default=DEFAULT_SPLIT["test"][0])
    parser.add_argument("--test_end", default=DEFAULT_SPLIT["test"][1])
    parser.add_argument("--neg_ratio", type=float, default=10.0,
                        help="Downsample negatives to N*positives for TRAIN and VALID. Use 0 to disable.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)

    ABT = Path(args.abt)
    MDIR = Path(args.models_dir); MDIR.mkdir(parents=True, exist_ok=True)
    RDIR = Path(args.reports_dir); RDIR.mkdir(parents=True, exist_ok=True)

    print("Loading splits from:", ABT)

    # Train/Valid with optional downsampling; keep Test full
    Xtr_num, Xtr_cat, ytr, dtr = load_split(
        ABT, args.train_start, args.train_end, NUMERIC, CATEGORICAL,
        downsample_neg_ratio=(None if args.neg_ratio <= 0 else args.neg_ratio), seed=args.seed
    )
    Xva_num, Xva_cat, yva, dva = load_split(
        ABT, args.valid_start, args.valid_end, NUMERIC, CATEGORICAL,
        downsample_neg_ratio=(None if args.neg_ratio <= 0 else args.neg_ratio), seed=args.seed
    )
    Xte_num, Xte_cat, yte, dte = load_split(
        ABT, args.test_start, args.test_end, NUMERIC, CATEGORICAL,
        downsample_neg_ratio=None, seed=args.seed
    )

    # Save split sizes
    split_sizes = {
        "train": int(len(dtr)),
        "valid": int(len(dva)),
        "test": int(len(dte)),
        "pos_train": int(ytr.sum()), "pos_valid": int(yva.sum()), "pos_test": int(yte.sum()),
    }
    (RDIR / "split_sizes.json").write_text(json.dumps(split_sizes, indent=2))
    print("Split sizes:", split_sizes)

    # ---------------- Logistic Regression (numeric only) ----------------
    print("\nTraining Logistic Regression (numeric only)…")
    logit_pipe = Pipeline(steps=[
        ("pre", ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=[
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler(with_mean=True, with_std=True))
                ]), NUMERIC),
                # We ignore categoricals here; this is a numeric-only baseline
            ],
            remainder="drop"
        )),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=-1, solver="lbfgs", random_state=args.seed))
    ])
    logit_pipe.fit(Xtr_num, ytr)
    p_va_log = logit_pipe.predict_proba(Xva_num)[:, 1]
    p_te_log = logit_pipe.predict_proba(Xte_num)[:, 1]

    # ---------------- LightGBM (numeric + categorical) ----------------
    print("Training LightGBM (numeric + categorical)…")
    # Prepare frames with correct dtypes (categoricals as category)
    Xtr_all = pd.concat([Xtr_num.copy(), Xtr_cat.copy()], axis=1)
    Xva_all = pd.concat([Xva_num.copy(), Xva_cat.copy()], axis=1)
    Xte_all = pd.concat([Xte_num.copy(), Xte_cat.copy()], axis=1)
    for c in CATEGORICAL:
        if c in Xtr_all.columns:
            Xtr_all[c] = Xtr_all[c].astype("category")
            Xva_all[c] = Xva_all[c].astype("category")
            Xte_all[c] = Xte_all[c].astype("category")

    # Handle missing numerics (median) for LightGBM too
    for c in NUMERIC:
        med = Xtr_all[c].median()
        Xtr_all[c] = Xtr_all[c].fillna(med)
        Xva_all[c] = Xva_all[c].fillna(med)
        Xte_all[c] = Xte_all[c].fillna(med)

    # LightGBM class weight via scale_pos_weight
    pos = float(ytr.sum()); neg = float(len(ytr) - ytr.sum())
    spw = (neg / pos) if pos > 0 else 1.0

    lgbm = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.seed,
        n_jobs=-1,
        scale_pos_weight=spw,
    )
    cat_idx = [Xtr_all.columns.get_loc(c) for c in CATEGORICAL if c in Xtr_all.columns]
    lgbm.fit(
        Xtr_all, ytr,
        eval_set=[(Xva_all, yva)],
        eval_metric="auc",
        categorical_feature=cat_idx,
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    p_va_lgb = lgbm.predict_proba(Xva_all)[:, 1]
    p_te_lgb = lgbm.predict_proba(Xte_all)[:, 1]

    # ---------------- Evaluation ----------------
    def eval_block(name: str, y_true: np.ndarray, p: np.ndarray) -> dict:
        safe_brier = float(brier_score_loss(y_true, p)) if len(np.unique(y_true)) > 1 else float("nan")
        safe_logloss = float(log_loss(y_true, p, labels=[0,1])) if len(np.unique(y_true)) > 1 else float("nan")
        return dict(
            model=name,
            auc=float(roc_auc_score(y_true, p)),
            pr_auc=float(average_precision_score(y_true, p)),
            ks=float(ks_score(y_true, p)),
            brier=safe_brier,
            logloss=safe_logloss,
        )

    metrics = {
        "valid": [
            eval_block("logit_numeric", yva, p_va_log),
            eval_block("lgbm_all",     yva, p_va_lgb),
        ],
        "test": [
            eval_block("logit_numeric", yte, p_te_log),
            eval_block("lgbm_all",     yte, p_te_lgb),
        ]
    }
    (Path(args.reports_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nMetrics:\n", json.dumps(metrics, indent=2))

    # ---------------- Calibration & Policy (valid) ----------------
    plt.figure()
    CalibrationDisplay.from_predictions(yva, p_va_lgb, n_bins=20)
    plt.title("Calibration — LightGBM (valid)")
    plt.tight_layout()
    plt.savefig(Path(args.reports_dir) / "calibration_valid_lgbm.png", dpi=140)
    plt.close()

    pc = policy_curve(yva, p_va_lgb)
    pc.to_csv(Path(args.reports_dir) / "policy_curve_valid_lgbm.csv", index=False)

    # ---------------- Feature importance (gain) ----------------
    fi = pd.Series(lgbm.booster_.feature_importance(importance_type="gain"),
                   index=Xtr_all.columns).sort_values(ascending=False)
    fi.to_csv(Path(args.reports_dir) / "feature_importance_lgbm.csv")

    # ---------------- Persist models ----------------
    dump(logit_pipe, Path(args.models_dir) / "logit_numeric.joblib")
    dump(lgbm,      Path(args.models_dir) / "lgbm_all.joblib")
    meta = dict(feature_order=list(Xtr_all.columns), categorical=CATEGORICAL)
    (Path(args.models_dir) / "lgbm_feature_meta.json").write_text(json.dumps(meta, indent=2))

    print("\nArtifacts written:")
    print(f"  Models : {Path(args.models_dir).resolve()}")
    print(f"  Reports: {Path(args.reports_dir).resolve()}")


if __name__ == "__main__":
    main()
