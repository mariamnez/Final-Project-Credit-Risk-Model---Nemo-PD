# scripts/inspect_raw_min.py
from pathlib import Path
import duckdb
import sys

# --- paths ---
ROOT = Path(__file__).resolve().parents[1]
ACQ  = ROOT / "data" / "raw" / "acq"
PERF = ROOT / "data" / "raw" / "perf"

def head_line(p: Path) -> str:
    """Return the first line (for delimiter/header sniffing)."""
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        return f.readline().strip()

def guess_delim(line: str) -> str:
    """Very simple delimiter guess for display purposes only."""
    counts = { "|": line.count("|"), ",": line.count(","), "\t": line.count("\t") }
    return max(counts, key=counts.get) if counts else "|"

def union_columns(files, header: bool):
    """Return column names DuckDB infers across ALL files (no data read)."""
    if not files:
        return []
    con = duckdb.connect(":memory:")
    header_sql = "true" if header else "false"
    # read_csv_auto will sniff types & delimiter; we only vary header=True/False
    df0 = con.execute(
        f"SELECT * FROM read_csv_auto(?, union_by_name=true, header={header_sql}, sample_size=20000) LIMIT 0",
        [list(map(str, files))]
    ).df()
    return list(df0.columns)

def looks_like_named_cols(cols) -> bool:
    """Heuristic: if all columns look like 'column0', it's headerless."""
    if not cols:
        return False
    return any(not c.lower().startswith("column") for c in cols)

def main():
    acq_files = sorted(list(ACQ.rglob("*.txt")) + list(ACQ.rglob("*.csv")))
    perf_files = sorted(list(PERF.rglob("*.txt")) + list(PERF.rglob("*.csv")))

    print(f"repo root: {ROOT}")
    print(f"acq dir:   {ACQ}  | files: {len(acq_files)}")
    print(f"perf dir:  {PERF} | files: {len(perf_files)}")
    if not acq_files or not perf_files:
        print(">> Missing files. Make sure UNZIPPED .txt are under data/raw/acq and data/raw/perf")
        sys.exit(1)

    # sample lines & delimiter guess
    acq_head = head_line(acq_files[0])
    perf_head = head_line(perf_files[0])
    print("\n-- sample first lines --")
    print("acq :", acq_files[0].name, "|", acq_head[:200])
    print("perf:", perf_files[0].name, "|", perf_head[:200])

    print("\n-- delimiter guess --")
    print("acq :", guess_delim(acq_head))
    print("perf:", guess_delim(perf_head))

    # show unioned columns with header=True and header=False
    acq_cols_h  = union_columns(acq_files, header=True)
    perf_cols_h = union_columns(perf_files, header=True)
    acq_cols_nh  = union_columns(acq_files, header=False)
    perf_cols_nh = union_columns(perf_files, header=False)

    print("\n-- unioned columns (header=True) --")
    print("acq :", acq_cols_h)
    print("perf:", perf_cols_h)

    print("\n-- unioned columns (header=False) --")
    print("acq :", acq_cols_nh[:10], ("... total " + str(len(acq_cols_nh))) if len(acq_cols_nh) > 10 else "")
    print("perf:", perf_cols_nh[:10], ("... total " + str(len(perf_cols_nh))) if len(perf_cols_nh) > 10 else "")

    # simple recommendation
    acq_header = looks_like_named_cols(acq_cols_h)
    perf_header = looks_like_named_cols(perf_cols_h)
    print("\n-- recommendation --")
    if acq_header and perf_header:
        print("Use header=True in the builder.")
    else:
        print("Use header=False (files are likely headerless / pipe-delimited).")

if __name__ == "__main__":
    main()

