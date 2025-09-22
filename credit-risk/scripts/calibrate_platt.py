# scripts/calibrate_platt.py
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parents[1]
DATA    = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# Most common locations/names produced earlier
CANDIDATES = [
    DATA / "valid_predictions.parquet",
    DATA / "test_predictions.parquet",
]

LABEL_PRIOR = {"y", "label", "default_within_24m", "target"}
PROB_PRIOR  = {"p", "pd", "pd_raw", "prob", "prob1", "p_raw", "prediction", "pred"}

def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0/(1.0 + np.exp(-z))

def fit_platt(x_logit, y, max_iter=200, tol=1e-6):
    """Logistic regression with intercept via Newton–Raphson on y ~ sigmoid(a + b*x)."""
    n  = x_logit.shape[0]
    X  = np.c_[np.ones(n), x_logit]    # [1, x]
    th = np.array([0.0, 1.0], dtype=float)

    for _ in range(max_iter):
        z = X @ th
        p = sigmoid(z)
        grad = X.T @ (y - p)
        w = p * (1 - p)
        H = -(X.T * w) @ X  # negative Hessian
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            H = H - 1e-4*np.eye(2)
            step = np.linalg.solve(H, grad)
        th_new = th - step
        if np.linalg.norm(th_new - th) < tol:
            th = th_new
            break
        th = th_new
    return float(th[0]), float(th[1])   # a, b

def pick_label(df: pd.DataFrame) -> str | None:
    cols = list(df.columns)
    # 1) by name
    for c in cols:
        if c.lower() in LABEL_PRIOR:
            return c
    # 2) by binary-ness
    for c in cols:
        s = df[c].dropna()
        if s.nunique() <= 2 and set(map(float, s.unique())) <= {0.0, 1.0}:
            return c
    return None

def pick_prob(df: pd.DataFrame, ycol: str | None) -> str | None:
    cols = list(df.columns)
    # 1) by name
    for c in cols:
        if c == ycol: 
            continue
        if c.lower() in PROB_PRIOR:
            return c
    # 2) by value range (mostly in [0,1], non-binary, floaty)
    cands = []
    for c in cols:
        if c == ycol:
            continue
        s = df[c].dropna()
        if s.size == 0:
            continue
        if s.dtype.kind not in "fc":
            continue
        frac01 = (s.between(0, 1)).mean()
        if frac01 > 0.98 and s.nunique() > 10:  # looks like a probability
            cands.append(c)
    if not cands:
        return None
    # prefer names that hint probability
    cands.sort(key=lambda c: (0 if any(k in c.lower() for k in ["pred","prob","pd","p_"]) else 1, -df[c].nunique()))
    return cands[0]

def find_preds() -> pd.DataFrame:
    tried = []
    for path in CANDIDATES:
        if not path.exists():
            tried.append(str(path))
            continue
        df = pd.read_parquet(path)
        ycol = pick_label(df)
        pcol = pick_prob(df, ycol)
        if ycol and pcol:
            print(f"Using file: {path}")
            print(f"  label column = {ycol}")
            print(f"  prob  column = {pcol}")
            return df[[pcol, ycol]].rename(columns={pcol: "p_raw", ycol: "y"})
        tried.append(f"{path} (cols={list(df.columns)})")
    raise FileNotFoundError(
        "Could not auto-detect prediction & label columns.\n"
        "Tried:\n  - " + "\n  - ".join(tried) +
        "\nIf your file/columns are named differently, please share one line of df.head()."
    )

def main():
    df = find_preds().dropna()
    y = df["y"].astype(int).to_numpy()
    p = df["p_raw"].clip(1e-6, 1-1e-6).to_numpy()
    x = np.log(p / (1 - p))  # logit

    a, b = fit_platt(x, y)
    (REPORTS / "platt_scaler.json").write_text(json.dumps({"a": a, "b": b}, indent=2))
    print(f"Wrote scaler -> {REPORTS/'platt_scaler.json'}  (a={a:.6f}, b={b:.6f})")

    # Optional sanity metrics; skip if sklearn not present
    try:
        from sklearn.metrics import brier_score_loss, roc_auc_score
        before = {
            "brier": float(brier_score_loss(y, p)),
            "auc":   float(roc_auc_score(y, p)),
        }
        p_cal = sigmoid(a + b * x)
        after = {
            "brier": float(brier_score_loss(y, p_cal)),
            "auc":   float(roc_auc_score(y, p_cal)),
        }
        (REPORTS/"calibration_metrics.json").write_text(json.dumps({"before": before, "after": after, "n": int(len(y))}, indent=2))
        print("Calibration metrics:", json.dumps({"before": before, "after": after}, indent=2))
    except Exception as e:
        print(f"(Skipped optional metrics: {e})")

if __name__ == "__main__":
    main()


