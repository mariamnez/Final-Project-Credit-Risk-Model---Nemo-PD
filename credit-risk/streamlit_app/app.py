import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import streamlit as st

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "reports" / "lgbm_model_tuned.txt"
SCALER_PATH = ROOT / "reports" / "platt_scaler.json"        # optional
POLICY_PATH = ROOT / "reports" / "policy_choice.json"
MEDIANS_PATH = ROOT / "reports" / "feature_medians.json"    # used to fill defaults

# ---------- Load artifacts ----------
@st.cache_resource(show_spinner=False)
def load_model():
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    feat_order = booster.feature_name()
    return booster, feat_order

@st.cache_resource(show_spinner=False)
def load_json(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default

booster, FEATURE_ORDER = load_model()
platt = load_json(SCALER_PATH, default=None)  # {"a":..., "b":...} or None
policy = load_json(POLICY_PATH, default={"threshold": 0.01})
medians = load_json(MEDIANS_PATH, default={})

THRESHOLD = float(policy.get("threshold", 0.01))

# ---------- Helpers ----------
def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x))

def apply_platt(p_raw: np.ndarray, params: dict | None):
    """
    Platt scaling on probabilities. We defensively handle either style:
    p_cal = sigmoid(a + b*logit(p))  (common)
    If 'use_logit' is False or missing, fall back to sigmoid(a + b*p).
    """
    if params is None:
        return p_raw
    a = float(params.get("a", 0.0))
    b = float(params.get("b", 1.0))
    mode = params.get("mode", "logit")  # "logit" (default) or "linear"
    p = np.clip(p_raw, 1e-6, 1 - 1e-6)
    if mode == "linear":
        z = a + b * p
    else:  # "logit"
        z = a + b * np.log(p / (1 - p))
    return sigmoid(z)

def set_one_hot(row_dict: dict, feature_prefix: str, choice_code: str):
    """
    If model has one-hot columns like purpose_P, purpose_C..., set chosen one to 1, others 0.
    If model has a single categorical code column (e.g., 'purpose'), we put the raw code.
    """
    # one-hots present?
    one_hot_cols = [c for c in FEATURE_ORDER if c.startswith(feature_prefix + "_")]
    if one_hot_cols:
        for c in one_hot_cols:
            row_dict[c] = 1.0 if c.endswith("_" + choice_code) else 0.0
    elif feature_prefix in FEATURE_ORDER:
        row_dict[feature_prefix] = choice_code  # LightGBM can accept string cats if trained that way

def compute_engineered(row_dict: dict):
    # Compute common engineered fields if the model expects them
    if "fico" in FEATURE_ORDER:
        fico = float(row_dict["fico"])
    elif "credit_score" in FEATURE_ORDER:
        fico = float(row_dict["credit_score"])
        row_dict["fico"] = fico
    else:
        fico = float(row_dict.get("fico", medians.get("fico", 740)))

    # gap and interaction often used in your model
    if "fico_gap" in FEATURE_ORDER:
        row_dict["fico_gap"] = 850.0 - fico
    if "int_fico_gap_x_ltv" in FEATURE_ORDER:
        ltv = float(row_dict.get("ltv", row_dict.get("cltv", medians.get("ltv", 80))))
        row_dict["int_fico_gap_x_ltv"] = (850.0 - fico) * ltv

    # If model expects cltv and we only gathered ltv, mirror it
    if "cltv" in FEATURE_ORDER and "cltv" not in row_dict:
        row_dict["cltv"] = float(row_dict.get("ltv", medians.get("cltv", medians.get("ltv", 80.0))))

def build_feature_row(user_inputs: dict) -> pd.DataFrame:
    # start from medians for robustness
    row = {k: medians.get(k, 0.0) for k in FEATURE_ORDER}

    # numeric direct mappings if present in the model
    for key, val in user_inputs.items():
        if key in FEATURE_ORDER:
            row[key] = val

    # Support common numeric names used in your project
    if "fico" in user_inputs and "fico" in FEATURE_ORDER:
        row["fico"] = user_inputs["fico"]
    if "dti" in user_inputs and "dti" in FEATURE_ORDER:
        row["dti"] = user_inputs["dti"]
    if "ltv" in user_inputs and "ltv" in FEATURE_ORDER:
        row["ltv"] = user_inputs["ltv"]
    if "orig_rate" in FEATURE_ORDER and "orig_rate" in user_inputs:
        row["orig_rate"] = user_inputs["orig_rate"]

    # Set one-hots / cat codes if those columns exist
    set_one_hot(row, "purpose", user_inputs["purpose_code"])
    set_one_hot(row, "channel", user_inputs["channel_code"])
    set_one_hot(row, "occupancy", user_inputs["occ_code"])
    # Property type & state if one-hotted in the model
    set_one_hot(row, "property_type", user_inputs["prop_code"])
    set_one_hot(row, "state", user_inputs["state_code"])

    # Add engineered columns if required
    compute_engineered(row)

    # Ensure column order matches the model
    X = pd.DataFrame([[row.get(f, 0.0) for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
    return X

def score_pd(X: pd.DataFrame) -> float:
    # LightGBM returns probability for class=1 (default)
    p_raw = booster.predict(X, raw_score=False).astype(float)[0]
    p_cal = float(apply_platt(np.array([p_raw]), platt)[0])
    return p_cal

# ---------- UI ----------
st.set_page_config(page_title="Credit Risk Demo", page_icon="✅", layout="centered")

st.title("Credit Eligibility Demo")
st.caption("Educational demo. Not a lending offer. Predictions come from the trained PD model with Platt calibration and a policy threshold.")

with st.form("inputs"):
    col1, col2 = st.columns(2)
    with col1:
        fico = st.slider("FICO score", 580, 850, 740, step=1)
        ltv = st.slider("LTV (%)", 30, 97, 80, step=1)
        dti = st.slider("Debt-to-Income (%)", 10, 60, 36, step=1)
        rate = st.slider("Interest rate (%)", 2.0, 12.0, 6.25, step=0.05)
    with col2:
        state = st.selectbox("Property state", ["California","Texas","Florida","New York","Illinois","Ohio","Georgia","Pennsylvania","New Jersey","Arizona","Virginia","North Carolina","Washington","Colorado","Tennessee","Michigan","Minnesota","Indiana","Massachusetts","Maryland","Other"], index=0)
        purpose = st.selectbox("Purpose", ["Purchase","No cash-out refinance","Cash-out refinance"], index=0)
        channel = st.selectbox("Channel", ["Retail","Correspondent","Broker"], index=0)
        occ = st.selectbox("Occupancy", ["Primary residence","Second home","Investor"], index=0)
        prop = st.selectbox("Property type", ["Single Family","Condo/Coop","Planned Unit Dev","Manufactured"], index=0)

    submitted = st.form_submit_button("Check my eligibility")

# map UI choices to the short codes often used in your features
state_code_map = {
    "California":"CA","Texas":"TX","Florida":"FL","New York":"NY","Illinois":"IL","Ohio":"OH","Georgia":"GA","Pennsylvania":"PA",
    "New Jersey":"NJ","Arizona":"AZ","Virginia":"VA","North Carolina":"NC","Washington":"WA","Colorado":"CO","Tennessee":"TN",
    "Michigan":"MI","Minnesota":"MN","Indiana":"IN","Massachusetts":"MA","Maryland":"MD","Other":"OT"
}
purpose_code_map = {"Purchase":"P","No cash-out refinance":"N","Cash-out refinance":"C"}
channel_code_map = {"Retail":"R","Correspondent":"C","Broker":"B"}
occ_code_map = {"Primary residence":"P","Second home":"S","Investor":"I"}
prop_code_map = {"Single Family":"SF","Condo/Coop":"CO","Planned Unit Dev":"PU","Manufactured":"MH"}

if submitted:
    user = {
        "fico": float(fico),
        "ltv": float(ltv),
        "dti": float(dti),
        "orig_rate": float(rate),
        "state_code": state_code_map[state],
        "purpose_code": purpose_code_map[purpose],
        "channel_code": channel_code_map[channel],
        "occ_code": occ_code_map[occ],
        "prop_code": prop_code_map[prop],
    }
    X = build_feature_row(user)
    pd_hat = score_pd(X)  # calibrated PD
    approve = pd_hat <= THRESHOLD

    st.subheader("Result")
    colA, colB = st.columns([1,1])
    with colA:
        st.metric("Predicted PD (12/24 mo)", f"{pd_hat*100:.2f}%")
        st.metric("Policy threshold t", f"{THRESHOLD*100:.2f}%")
    with colB:
        st.markdown("### Decision")
        if approve:
            st.success("✅ **Approved** (PD ≤ threshold)")
        else:
            st.error("❌ **Declined** (PD > threshold)")

    # Explain top drivers with SHAP (optional, single row)
    try:
        explainer = shap.TreeExplainer(booster)
        shap_vals = explainer.shap_values(X)[1] if isinstance(explainer.shap_values(X), list) else explainer.shap_values(X)
        shap_df = (pd.DataFrame({"feature": FEATURE_ORDER, "shap": shap_vals[0], "value": X.iloc[0].values})
                   .reindex(FEATURE_ORDER).sort_values("shap", key=np.abs, ascending=False).head(8))
        st.markdown("### Top contributing features")
        st.dataframe(shap_df, use_container_width=True)
    except Exception as e:
        st.info("Feature attributions unavailable on this environment.")
