# scripts/train_lgbm_tuned.py
import json, math, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import ParameterSampler

ROOT = Path(".")
DATA = ROOT / "data" / "processed"
RPTS = ROOT / "reports"
RPTS.mkdir(parents=True, exist_ok=True)

TRAIN = DATA / "abt_train.parquet"
VALID = DATA / "abt_valid.parquet"
TEST  = DATA / "abt_test.parquet"

LABEL = "default_within_24m"

def ks_stat(y_true, p):
    df = pd.DataFrame({"y": y_true, "p": p})
    df = df.sort_values("p")
    c1 = (df["y"] == 1).cumsum() / (df["y"] == 1).sum()
    c0 = (df["y"] == 0).cumsum() / (df["y"] == 0).sum()
    return float((c1 - c0).abs().max())

def columns_to_use(df):
    drop = {
        LABEL, "loan_id", "first_90dpd", "first_payment_date",
        "orig_date", "vintage_q", "msa"
    }
    use = [c for c in df.columns if c not in drop]
    return use

def build_monotone_constraints(cols):
    monotone = {  
        "fico": -1,
        "dti": +1,
        "ltv": +1,
        "cltv": +1,
        "orig_rate": +1
    }
    out = []
    for c in cols:
        out.append(monotone.get(c, 0))
    return out

def fit_and_eval(params, Xtr, ytr, Xva, yva, monotone=None):
    lgb_train = lgb.Dataset(Xtr, label=ytr, free_raw_data=True)
    lgb_valid = lgb.Dataset(Xva, label=yva, reference=lgb_train, free_raw_data=True)

    p = params.copy()
    p.update(dict(
        objective="binary",
        metric="binary_logloss",
        boosting_type="gbdt",
        verbose=-1
    ))
    if monotone is not None:
        p["monotone_constraints"] = monotone

    booster = lgb.train(
        p,
        lgb_train,
        num_boost_round=4000,
        valid_sets=[lgb_valid],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
    )

    pv = booster.predict(Xva, num_iteration=booster.best_iteration)
    metrics = {
        "auc_roc": float(roc_auc_score(yva, pv)),
        "auc_pr":  float(average_precision_score(yva, pv)),
        "ks":      ks_stat(yva, pv),
        "brier":   float(brier_score_loss(yva, pv)),
        "best_iter": int(booster.best_iteration)
    }
    return booster, metrics

def main():
    assert TRAIN.exists() and VALID.exists() and TEST.exists(), "Missing split files."

    dtr = pd.read_parquet(TRAIN)
    dva = pd.read_parquet(VALID)
    dte = pd.read_parquet(TEST)

    feats = columns_to_use(dtr)
    Xtr, ytr = dtr[feats], dtr[LABEL].astype(int).values
    Xva, yva = dva[feats], dva[LABEL].astype(int).values
    Xte, yte = dte[feats], dte[LABEL].astype(int).values

    # class imbalance
    pos_rate = ytr.mean() if ytr.mean() > 0 else 0.01
    spw = (1.0 - pos_rate) / pos_rate  
    print(f"Train pos_rate={pos_rate:.5f}  →  scale_pos_weight≈{spw:.1f}")

    # base param grid
    grid = {
        "learning_rate": [0.03, 0.05, 0.07, 0.10],
        "num_leaves": [31, 63, 95, 127],
        "min_data_in_leaf": [1000, 3000, 6000, 10000, 15000],
        "feature_fraction": [0.7, 0.8, 0.9, 1.0],
        "bagging_fraction": [0.7, 0.8, 0.9, 1.0],
        "bagging_freq": [0, 1, 5],
        "lambda_l1": [0.0, 0.1, 0.5, 1.0],
        "lambda_l2": [0.0, 0.1, 0.5, 1.0],
        "max_depth": [-1, 8, 10, 12],
        "scale_pos_weight": [spw*0.7, spw, spw*1.3],
    }
    sampler = list(ParameterSampler(grid, n_iter=30, random_state=42))

    mono = build_monotone_constraints(feats)
    use_mono = any(m != 0 for m in mono)
    print(f"Monotone constraints active: {use_mono}")

    best = None
    best_booster = None
    for i, params in enumerate(sampler, 1):
        booster, m = fit_and_eval(params, Xtr, ytr, Xva, yva, monotone=mono if use_mono else None)
        tag = f"[{i:02d}/{len(sampler)}] KS={m['ks']:.4f} AUC={m['auc_roc']:.4f} iter={m['best_iter']}"
        if best is None or m["ks"] > best["ks"]:
            best, best_booster = m, booster
            print("★", tag, "← new best")
        else:
            print("  ", tag)

    # evaluate on TEST
    pt = best_booster.predict(Xte, num_iteration=best_booster.best_iteration)
    test_metrics = {
        "auc_roc": float(roc_auc_score(yte, pt)),
        "auc_pr":  float(average_precision_score(yte, pt)),
        "ks":      ks_stat(yte, pt),
        "brier":   float(brier_score_loss(yte, pt)),
        "n_test": int(len(yte)),
        "pos_rate_test": float(yte.mean())
    }
    print("Test metrics (tuned):", json.dumps(test_metrics, indent=2))

    # save artifacts
    model_txt = RPTS / "lgbm_model_tuned.txt"
    best_booster.save_model(str(model_txt))
    (RPTS / "metrics_tuned.json").write_text(json.dumps(test_metrics, indent=2))

    # feature importance
    fi = pd.DataFrame({
        "feature": feats,
        "gain": best_booster.feature_importance(importance_type="gain"),
        "split": best_booster.feature_importance(importance_type="split"),
        "monotone": build_monotone_constraints(feats)
    }).sort_values("gain", ascending=False)
    fi.to_csv(RPTS / "feature_importance_tuned.csv", index=False)

    out = pd.DataFrame({"y_true": yte, "y_pred": pt})
    out.to_parquet(DATA / "test_predictions_tuned.parquet", index=False)

    print(f"Wrote model -> {model_txt}")
    print("Wrote TEST preds -> data/processed/test_predictions_tuned.parquet")
    print("Wrote metrics -> reports/metrics_tuned.json")
    print("Wrote importance -> reports/feature_importance_tuned.csv")

if __name__ == "__main__":
    main()
