# scripts/build_abt_min.py
from __future__ import annotations
from pathlib import Path
import argparse
import os
import duckdb


def list_files(dir_: Path):
    files = sorted([p for p in dir_.rglob("*.txt")] + [p for p in dir_.rglob("*.csv")])
    return [str(p) for p in files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="repo root (default: .)")
    ap.add_argument("--years", default="", help="comma list, e.g., 2019,2020")
    ap.add_argument("--out", default="data/processed/abt.parquet")
    args = ap.parse_args()

    ROOT = Path(args.root).resolve()
    ACQ = ROOT / "data" / "raw" / "acq"
    PERF = ROOT / "data" / "raw" / "perf"
    OUT = ROOT / args.out
    OUT.parent.mkdir(parents=True, exist_ok=True)
    TMP = ROOT / "data" / "interim"
    TMP.mkdir(parents=True, exist_ok=True)

    acq_files = list_files(ACQ)
    perf_files = list_files(PERF)

    if args.years:
        keep = {y.strip() for y in args.years.split(",")}
        acq_files = [p for p in acq_files if any(y in p for y in keep)]
        perf_files = [p for p in perf_files if any(y in p for y in keep)]

    if not acq_files or not perf_files:
        raise SystemExit(f"No files found. acq={len(acq_files)} perf={len(perf_files)}")

    print(f"ROOT: {ROOT}")
    print(f"ACQ:  {ACQ} | files: {len(acq_files)}")
    print(f"PERF: {PERF} | files: {len(perf_files)}")
    print(f"OUT:  {OUT}")

    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA threads={os.cpu_count()}")
    con.execute(f"PRAGMA temp_directory='{TMP.as_posix()}'")
    con.execute("PRAGMA enable_progress_bar=true")

    def load_headerless(table: str, files: list[str]):
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {table} AS
            SELECT *
            FROM read_csv_auto(?, union_by_name=true, header=false, sample_size=-1)
            LIMIT 0
            """,
            [files[0]],
        )
        for i, f in enumerate(files, 1):
            con.execute(
                f"""
                INSERT INTO {table}
                SELECT *
                FROM read_csv_auto(?, union_by_name=true, header=false, sample_size=-1)
                """,
                [f],
            )
            print(f"[{table}] {i}/{len(files)} loaded: {Path(f).name}")

    print("Reading acquisition…")
    load_headerless("acq_raw", acq_files)

    print("Reading performance…")
    load_headerless("perf_raw", perf_files)

    # Map by index
    # ACQ indices (0-based) per Freddie "Standard"
    con.execute(
        """
        CREATE OR REPLACE TABLE acq AS
        SELECT
          t.column19                                              AS loan_id,
          TRY_CAST(t.column00 AS INTEGER)                         AS fico,
          date_trunc('month',
            COALESCE(
              try_strptime(t.column01::VARCHAR,'%Y-%m-%d'),
              try_strptime(t.column01::VARCHAR,'%Y%m')
            )
          )::DATE                                                 AS first_payment_date,
          NULL::DATE                                              AS orig_date,
          TRY_CAST(t.column10 AS DOUBLE)                          AS orig_upb,
          TRY_CAST(t.column12 AS DOUBLE)                          AS orig_rate,
          TRY_CAST(t.column11 AS DOUBLE)                          AS ltv,
          TRY_CAST(t.column08 AS DOUBLE)                          AS cltv,
          TRY_CAST(t.column09 AS DOUBLE)                          AS dti,
          t.column13                                              AS channel,
          t.column20                                              AS purpose,
          t.column07                                              AS occupancy,
          t.column15                                              AS product,
          t.column17                                              AS property_type,
          t.column16                                              AS state,
          t.column04                                              AS msa,
          TRY_CAST(t.column21 AS INTEGER)                         AS loan_term,
          TRY_CAST(t.column06 AS INTEGER)                         AS num_units,
          t.column23                                              AS seller_name,
          t.column24                                              AS servicer_name,
          t.column02                                              AS first_time_buyer
        FROM acq_raw AS t;
        """
    )

    # PERF indices
    con.execute(
        """
        CREATE OR REPLACE TABLE perf AS
        SELECT
          t.column00                                              AS loan_id,
          date_trunc('month',
            COALESCE(
              try_strptime(t.column01::VARCHAR,'%Y-%m-%d'),
              try_strptime(t.column01::VARCHAR,'%Y%m')
            )
          )::DATE                                                 AS report_month,
          t.column03                                             AS dq_status
        FROM perf_raw AS t;
        """
    )

    con.execute("ALTER TABLE perf ADD COLUMN dq_num INTEGER;")
    con.execute("UPDATE perf SET dq_num = TRY_CAST(dq_status AS INTEGER);")
    con.execute("UPDATE perf SET dq_num = -1 WHERE dq_num IS NULL;")

    # First 90+ and censoring
    con.execute(
        """
        CREATE OR REPLACE TABLE dq3 AS
        SELECT loan_id, MIN(report_month) AS first_90dpd
        FROM perf
        WHERE dq_num >= 3
        GROUP BY loan_id;
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE lastrep AS
        SELECT loan_id, MAX(report_month) AS last_perf_month
        FROM perf
        GROUP BY loan_id;
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE abt AS
        SELECT
          a.*,
          date_trunc('quarter', a.first_payment_date) AS vintage_q,
          a.first_payment_date + INTERVAL 24 MONTH    AS window_end,
          lr.last_perf_month,
          d.first_90dpd,
          CASE WHEN d.first_90dpd IS NOT NULL
                 AND d.first_90dpd <= (a.first_payment_date + INTERVAL 24 MONTH)
               THEN TRUE ELSE FALSE END              AS default_within_24m
        FROM acq a
        LEFT JOIN lastrep lr USING (loan_id)
        LEFT JOIN dq3 d USING (loan_id)
        WHERE lr.last_perf_month IS NOT NULL
          AND lr.last_perf_month >= (a.first_payment_date + INTERVAL 24 MONTH);
        """
    )

    con.execute(f"COPY (SELECT * FROM abt) TO '{OUT.as_posix()}' (FORMAT PARQUET);")
    con.execute(
        f"COPY (SELECT * FROM abt USING SAMPLE 5000 ROWS) "
        f"TO '{OUT.with_name('abt_sample.parquet').as_posix()}' (FORMAT PARQUET);"
    )

    rows = con.execute("SELECT COUNT(*) FROM abt").fetchone()[0]
    bad = con.execute(
        "SELECT AVG(CASE WHEN default_within_24m THEN 1 ELSE 0 END) FROM abt"
    ).fetchone()[0]
    bad_str = "n/a" if bad is None else f"{bad:.4f}"

    print(f"\nABT saved: {OUT} | rows={rows:,} | bad_rate={bad_str}")
    if rows == 0:
        print("NOTE: No loans have a full 24-month window for this year (expected for recent vintages).")


if __name__ == "__main__":
    main()
