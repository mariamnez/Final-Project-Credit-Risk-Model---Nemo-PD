# Final-Project-Credit-Risk-Model---Nemo-PD
# Credit-Risk PD Model (Freddie Mac)

End-to-end probability-of-default (PD) project using Freddie Mac Single-Family Loan-Level data.  
Pipeline covers data prep, model training (LightGBM), calibration (Platt), policy/threshold selection, back-testing on a recent cohort, monitoring assets, and a Streamlit demo for single-app scoring.

---

## TL;DR – Current test metrics (this run)

- **AUC (test):** ~0.759  
- **KS (at chosen threshold):** ~0.415  
- **Brier (calibrated):** ~0.0110  
- **Recent cohort bad rate:** ~1.99%  
- **Chosen PD threshold (policy):** ≈ 0.010–0.014 (drives ~20% approvals; see `reports/policy_curve.csv`)

---

## Repo Layout

### Repo Layout

```text
credit-risk/
├── data/
│   ├── processed/
│   │   ├── test_predictions.parquet
│   │   └── test_predictions_calibrated.parquet
│   └── external/                 # (optional macro templates if used)
├── reports/
│   ├── metrics.json
│   ├── metrics_tuned.json
│   ├── platt_scaler.json
│   ├── lgbm_model_tuned.txt
│   ├── policy_curve.csv
│   ├── policy_choice.json
│   ├── backtest_recent_policy.csv
│   └── eda_*.(png|csv)           # EDA assets for slides/dashboards
├── scripts/
│   ├── train_baseline_splits.py
│   ├── train_tuned.py
│   ├── calibrate_platt.py
│   ├── policy_curve.py
│   ├── set_policy_threshold.py
│   ├── check_threshold.py
│   ├── backtest_recent_policy.py
│   ├── compute_rmse.py
│   ├── slice_*.py                # fairness/stability slices
│   └── make_eda_assets.py
└── streamlit_app/
    └── app.py



---

## Environment

```bash
# Create/activate (Anaconda)
conda create -n nemo-pd python=3.10 -y
conda activate nemo-pd

# Install
pip install lightgbm pandas numpy scikit-learn matplotlib seaborn shap streamlit pyarrow

# 0) From repo root
cd path/to/credit-risk
conda activate nemo-pd

# 1) Train (baseline splits: train 2017–2019, valid 2020, test 2021–2022)
python scripts/train_baseline_splits.py

# 2) (Optional) hyperparameter pass
python scripts/train_tuned.py

# 3) Calibrate test predictions (Platt scaling)
python scripts/calibrate_platt.py --preds data/processed/test_predictions.parquet

# 4) Build policy curve (use calibrated PDs)
python scripts/policy_curve.py --metric ks --use-calibrated

# 5) Choose threshold (via config or target approve rate)
#    a) by target approval on recent cohort:
python scripts/set_policy_threshold.py --approve 0.22
#    b) or by best KS/F1 from policy_curve.csv (writes policy_choice.json)

# 6) Sanity check threshold on recent cohort
python scripts/check_threshold.py
python scripts/backtest_recent_policy.py --use_calibrated

# 7) Quality metrics (probability quality)
python scripts/compute_rmse.py --preds data/processed/test_predictions_calibrated.parquet


# From repo root
python -m streamlit run "streamlit_app/app.py"
# App expects:
# reports/lgbm_model_tuned.txt
# reports/platt_scaler.json
# reports/policy_choice.json
# reports/feature_medians.json

# Data source
Freddie Mac Single-Family Loan-Level Dataset (public). We train on 2017–2019, validate 2020, test 2021–2022, and keep 2023 as a “recent” holdout for policy/backtest charts.
