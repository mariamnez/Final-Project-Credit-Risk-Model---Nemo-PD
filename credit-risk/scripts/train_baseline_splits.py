# scripts/train_baseline_splits.py
# Train LightGBM on prepared splits and evaluate on TEST.
# Outputs:
#   - reports/lgbm_model.txt
#   - reports/metrics.json
#   - reports/feature_importance.csv
#   - data/processed/test_predictions.parquet
#   - reports/roc_curve.png, reports/pr_curve.png, reports/calibration_curve.png

from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)

# Paths & constants
ROOT = Path(".")
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

TRAIN_PQ = DATA / "abt_train.parquet"
VALID_PQ = DATA / "abt_valid.parquet"
TEST_PQ  = DATA / "abt_test.parquet"

MODEL_TXT = REPORTS / "lgbm_model.txt"
METRICS_JSON = REPORTS / "metrics.json"
FEATIMP_CSV = REPORTS / "feature_importance.csv"
TEST_PREDS_PQ = DATA / "processed" / "test_predictions.parquet"  

ROC_PNG = REPORTS / "roc_curve.png"
PR_PNG = REPORTS / "pr_curve.png"
CAL_PNG = REPORTS / "calibration_curve.png"

LABEL_COL = "default_within_24m"  
ID_COL = "loan_id"                
VINTAGE_COL = "vintage_q"         

# Utilities
def ks_stat(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """KS statistic via ROC curve (max TPR - FPR)."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))

def select_features(df: pd.DataFrame, label_col: str) -> list:
    """Select model features by excluding obvious non-features."""
    drop_like = {
        label_col,
        "first_90dpd",
        "first_payment_date",
        "orig_date",
        "orig_dt",
        "report_dt",
        "as_of_date",
        "snapshot_date",
        "issue_dt",
        "disbursement_date",
        ID_COL,
    }
    feats = [c for c in df.columns if c not in drop_like and df[c].dtype != "datetime64[ns]"]
    return feats

def cat_columns(df: pd.DataFrame, feats: list) -> list:
    """Categorical columns as object or category among features."""
    return [c for c in feats if str(df[c].dtype) in ("object", "category")]

def ensure_dirs():
    REPORTS.mkdir(parents=True, exist_ok=True)
    (DATA / "processed").mkdir(parents=True, exist_ok=True)

def plot_curves(y, p):
    import matplotlib.pyplot as plt

    # ROC
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], "--", lw=1.5)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve")
    plt.tight_layout()
    plt.savefig(ROC_PNG, dpi=150)
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y, p)
    plt.figure(figsize=(8, 6))
    plt.plot(rec, prec, lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curve")
    plt.tight_layout()
    plt.savefig(PR_PNG, dpi=150)
    plt.close()

    # Calibration (reliability by quantile bins)
    dfc = pd.DataFrame({"y": y, "p": p})
    dfc["bin"] = pd.qcut(dfc["p"], q=10, duplicates="drop")
    calib = dfc.groupby("bin").agg(pred=("p", "mean"), obs=("y", "mean")).reset_index(drop=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(calib["pred"], calib["obs"])
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed default rate")
    plt.title("Calibration (quantile bins)")
    plt.tight_layout()
    plt.savefig(CAL_PNG, dpi=150)
    plt.close()

# Main
def main():
    ensure_dirs()

    # Load splits
    df_tr = pd.read_parquet(TRAIN_PQ)
    df_va = pd.read_parquet(VALID_PQ)
    df_te = pd.read_parquet(TEST_PQ)

    # Features & label
    feats = select_features(df_tr, LABEL_COL)
    cats = cat_columns(df_tr, feats)

    # Coerce categoricals for LightGBM
    for c in cats:
        for df in (df_tr, df_va, df_te):
            df[c] = df[c].astype("category")

    print(f"Using {len(feats)} features")

    Xtr, ytr = df_tr[feats], df_tr[LABEL_COL].astype(int).to_numpy()
    Xva, yva = df_va[feats], df_va[LABEL_COL].astype(int).to_numpy()
    Xte, yte = df_te[feats], df_te[LABEL_COL].astype(int).to_numpy()

    dtrain = lgb.Dataset(
        Xtr, label=ytr, categorical_feature=cats, free_raw_data=False
    )
    dvalid = lgb.Dataset(
        Xva, label=yva, categorical_feature=cats, reference=dtrain, free_raw_data=False
    )

    # LightGBM params
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 64,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbose": -1,
        "seed": 42,
        "num_threads": 0,  
    }

    num_boost_round = 3000
    stopping_rounds = 200
    eval_period = 50

    # Train with LightGBM
    try:
        gbm = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds, verbose=False),
                lgb.log_evaluation(eval_period),
            ],
        )
    except TypeError:
        gbm = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            valid_names=["valid"],
            early_stopping_rounds=stopping_rounds,
            verbose_eval=eval_period,
        )

    best_iter = gbm.best_iteration if hasattr(gbm, "best_iteration") and gbm.best_iteration else gbm.current_iteration()

    # Predict on TEST
    p_test = gbm.predict(Xte, num_iteration=best_iter)

    # Metrics
    metrics = {
        "auc_roc": float(roc_auc_score(yte, p_test)),
        "auc_pr": float(average_precision_score(yte, p_test)),
        "ks": float(ks_stat(yte, p_test)),
        "brier": float(brier_score_loss(yte, p_test)),
        "n_test": int(len(yte)),
        "pos_rate_test": float(float(np.mean(yte))),
    }
    print("Test metrics:", json.dumps(metrics, indent=2))

    MODEL_TXT.parent.mkdir(parents=True, exist_ok=True)
    gbm.save_model(str(MODEL_TXT), num_iteration=best_iter)

    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    fi = pd.DataFrame(
        {
            "feature": gbm.feature_name(),
            "gain": gbm.feature_importance(importance_type="gain"),
            "split": gbm.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)
    fi.to_csv(FEATIMP_CSV, index=False)

    # Save test predictions parquet
    out_cols = {}
    if ID_COL in df_te.columns:
        out_cols[ID_COL] = df_te[ID_COL].values
    if VINTAGE_COL in df_te.columns:
        out_cols[VINTAGE_COL] = df_te[VINTAGE_COL].values

    out_cols["y_true"] = yte
    out_cols["y_pred"] = p_test
    pd.DataFrame(out_cols).to_parquet(TEST_PREDS_PQ, index=False)

    # Plots
    plot_curves(yte, p_test)

if __name__ == "__main__":
    main()
