# scripts/audit_abt.py
"""
Audit the combined ABT to verify shape, label, missingness, outliers, categories,
and potential leakage fields. Writes a human-readable JSON report and prints a
console summary.

Usage:
  python scripts/audit_abt.py --abt data/processed/abt.parquet --out reports
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json

import duckdb
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abt", default="data/processed/abt.parquet",
                    help="Path to combined ABT parquet")
    ap.add_argument("--out", default="reports", help="Directory for audit outputs")
    args = ap.parse_args()

    ABT = Path(args.abt)
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    if not ABT.exists():
        raise FileNotFoundError(f"ABT file not found: {ABT}")

    # Feature expectations (used for rules/tops; only applied if present)
    NUMERIC = ["fico", "ltv", "cltv", "dti", "orig_rate", "orig_upb", "loan_term", "num_units"]
    CATEG   = ["channel", "purpose", "occupancy", "product", "property_type",
               "state", "first_time_buyer", "msa"]
    TARGET  = "default_within_24m"

    # Broad plausibility caps (we CAP outliers later, not drop)
    RULES = {
        "fico":      (300, 850),
        "ltv":       (0, 150),
        "cltv":      (0, 200),
        "dti":       (0, 80),
        "orig_rate": (0, 15),
        "orig_upb":  (1, 2_000_000),
        "loan_term": (60, 480),
        "num_units": (1, 4),
    }

    con = duckdb.connect()

    # ---------- basic shape ----------
    rows = con.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(ABT)]).fetchone()[0]
    uniq = con.execute("SELECT COUNT(DISTINCT loan_id) FROM read_parquet(?)", [str(ABT)]).fetchone()[0]

    # ---------- date/label summary ----------
    hdr = con.execute(f"""
        SELECT
          MIN(first_payment_date) AS min_fp,
          MAX(first_payment_date) AS max_fp,
          MIN(vintage_q)          AS min_vint,
          MAX(vintage_q)          AS max_vint,
          MIN(last_perf_month)    AS min_last,
          MAX(last_perf_month)    AS max_last,
          AVG(CASE WHEN {TARGET} THEN 1 ELSE 0 END)::DOUBLE AS bad_rate,
          SUM(CASE WHEN {TARGET} IS NULL THEN 1 ELSE 0 END) AS label_nulls
        FROM read_parquet(?)
    """, [str(ABT)]).df()

    # ---------- column list ----------
    cols = con.execute("SELECT * FROM read_parquet(?) LIMIT 0", [str(ABT)]).df().columns.tolist()

    # ---------- missingness (loop per column for compatibility) ----------
    miss_rows = []
    for c in cols:
        # COUNT(*) - COUNT(col) counts NULLs; works for numeric and categorical
        nulls = con.execute(f"SELECT COUNT(*) - COUNT({c}) FROM read_parquet(?)", [str(ABT)]).fetchone()[0]
        miss_rows.append((c, int(nulls)))
    missing = pd.DataFrame(miss_rows, columns=["col", "nulls"])
    missing["miss_rate"] = missing["nulls"] / (rows if rows else 1)
    missing = missing.sort_values("miss_rate", ascending=False)

    # ---------- outliers vs rules ----------
    out_rows = []
    for c, (lo, hi) in RULES.items():
        if c in cols:
            cnt = con.execute(
                f"SELECT SUM(({c} < {lo}) OR ({c} > {hi}) OR {c} IS NULL) FROM read_parquet(?)",
                [str(ABT)]
            ).fetchone()[0]
            out_rows.append((c, int(cnt)))
    outliers = pd.DataFrame(out_rows, columns=["feature", "count_flagged"]).sort_values(
        "count_flagged", ascending=False
    )

    # ---------- categorical top counts ----------
    tops = {}
    for c in ["channel", "purpose", "occupancy", "product", "property_type", "state", "first_time_buyer"]:
        if c in cols:
            tops[c] = con.execute(
                f"SELECT {c} AS val, COUNT(*) AS n FROM read_parquet(?) GROUP BY 1 ORDER BY n DESC LIMIT 10",
                [str(ABT)]
            ).df().to_dict(orient="records")

    # ---------- leakage probe ----------
    suspects = [k for k in cols if any(tag in k.lower() for tag in
                                       ["delinq", "dq_", "foreclos", "reo", "modif", "prepay", "current_", "status"])]

    # ---------- pack report ----------
    report = {
        "rows": rows,
        "unique_loan_id": uniq,
        "duplicates": rows - uniq,
        "date_summary": hdr.to_dict(orient="records")[0] if not hdr.empty else {},
        "missing_top": missing.head(20).to_dict(orient="records"),
        "outlier_flags": outliers.to_dict(orient="records"),
        "categorical_tops": tops,
        "suspect_columns": suspects,
        "columns": cols,
    }

    # write JSON (default=str serializes timestamps safely)
    (OUT / "audit_abt.json").write_text(json.dumps(report, indent=2, default=str))

    # ---------- console summary ----------
    print(json.dumps({k: report[k] for k in ["rows", "unique_loan_id", "duplicates"]}, indent=2))
    print("\nDates/label:\n", hdr.to_string(index=False))
    print("\nTop missing:\n", missing.head(10).to_string(index=False))
    print("\nOutlier flags:\n", outliers.to_string(index=False))
    print("\nSuspect columns:", suspects)
    print("\nWrote full report ->", (OUT / "audit_abt.json").resolve())


if __name__ == "__main__":
    main()
