from pathlib import Path
import json, pandas as pd

root = Path(".")
outd = root / "data" / "processed" / "exports"
outd.mkdir(parents=True, exist_ok=True)

# 1) Policy choice (threshold + which prob column was used)
policy = json.loads((root/"reports"/"policy_choice.json").read_text())
t = float(policy["threshold"])
prob_hint = policy.get("prob", "y_pred_cal")  # 'y_pred' | 'y_pred_cal' | 'pd_cal'

# 2) TEST predictions (calibrated)
testp = root/"data"/"processed"/"test_predictions_calibrated.parquet"
df_test = pd.read_parquet(testp)
pcol_test = next(c for c in [prob_hint, "y_pred_cal", "y_pred"] if c in df_test.columns)
keep_test = [c for c in ["loan_id","vintage_q","state","channel","purpose","y_true",pcol_test] if c in df_test.columns]
df_test = df_test[keep_test].rename(columns={pcol_test:"pd"})
df_test["decision_at_t"] = (df_test["pd"] <= t).astype(int)
df_test.to_csv(outd/"test_predictions_for_tableau.csv", index=False)

# 3) RECENT predictions (calibrated)
recentp = root/"data"/"processed"/"recent_predictions_calibrated.parquet"
df_recent = pd.read_parquet(recentp)
pcol_recent = next(c for c in [prob_hint, "pd_cal", "y_pred_cal", "pd", "y_pred"] if c in df_recent.columns)
keep_recent = [c for c in ["loan_id","vintage_q","state","channel","purpose",pcol_recent] if c in df_recent.columns]
df_recent = df_recent[keep_recent].rename(columns={pcol_recent:"pd"})
df_recent["decision_at_t"] = (df_recent["pd"] <= t).astype(int)
df_recent.to_csv(outd/"recent_predictions_for_tableau.csv", index=False)

# 4) Already-built artifacts: copy to exports for one place to point Tableau
artifacts = [
    ("reports/policy_curve.csv", "policy_curve.csv"),
    ("reports/backtest_recent_policy.csv", "backtest_recent_policy.csv"),
    ("reports/feature_importance.csv", "feature_importance.csv"),
    ("reports/slice_vintage_metrics.csv", "slice_vintage_metrics.csv"),
    ("reports/slice_state_top20_metrics.csv", "slice_state_top20_metrics.csv"),
    ("reports/slice_channel_metrics.csv", "slice_channel_metrics.csv"),
    ("reports/slice_purpose_metrics.csv", "slice_purpose_metrics.csv"),
]
for src, dst in artifacts:
    p = root/src
    if p.exists():
        pd.read_csv(p).to_csv(outd/dst, index=False)

# 5) Tiny summary for the deck
summary = {
    "threshold": t,
    "metric": policy.get("metric","(see policy_choice.json)"),
    "prob_col": prob_hint,
}
(root/"reports"/"tableau_summary.json").write_text(json.dumps(summary, indent=2))
print(f"Exports written to: {outd.resolve()}")
print(summary)
