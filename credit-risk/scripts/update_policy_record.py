# scripts/update_policy_record.py
import argparse, json, pathlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approve-rate", type=float)
    ap.add_argument("--expected-bad", type=float)
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--prob-col", type=str)
    ap.add_argument("--file", type=str, default="reports/policy_choice.json")
    args = ap.parse_args()

    p = pathlib.Path(args.file)
    d = json.loads(p.read_text()) if p.exists() else {}
    if args.approve_rate is not None:
        d["approve_rate_recent"] = args.approve_rate
    if args.expected_bad is not None:
        d["expected_bad_rate_on_approved"] = args.expected_bad
    if args.threshold is not None:
        d["threshold"] = args.threshold
    if args.prob_col is not None:
        d["prob"] = args.prob_col

    p.write_text(json.dumps(d, indent=2))
    print("Updated ->", p)

if __name__ == "__main__":
    main()
