"""Audit a log of chatbot answers (the batch version of the Monitor tab in the app).

The input is a CSV or JSONL file with an `answer` column (other columns are kept). The output adds the
probe reading of the weakest claim in each answer and a `flagged` column for answers the model does not believe.

Usage:
    python scripts/audit_log.py my_log.csv --out audited.csv --tau 0.5
"""
import argparse
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probity import DEFAULT_TAU, Polygraph  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="CSV or JSONL file with an 'answer' column")
    ap.add_argument("--out", default="audited.csv")
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU, help="answers below this probability are flagged")
    ap.add_argument("--probe", default=str(ROOT / "results" / "app_probes.npz"))
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    path = pathlib.Path(args.log)
    df = pd.read_json(path, lines=True) if path.suffix == ".jsonl" else pd.read_csv(path)
    if "answer" not in df.columns:
        sys.exit("The log needs an 'answer' column.")
    if args.max_rows:
        df = df.head(args.max_rows)

    poly = Polygraph.from_pretrained(args.model, probe=args.probe, layer=args.layer)
    results = [poly.check(str(a)) for a in df.answer]
    df["p_true"] = [r["p_true"] for r in results]
    df["verdict"] = [r["verdict"] for r in results]
    df["weakest_claim"] = [r["weakest_claim"] for r in results]
    df["trusted_reading"] = [r["trusted"] for r in results]
    df["flagged"] = df.p_true.notna() & (df.p_true < args.tau)
    df.to_csv(args.out, index=False)
    scored = df.p_true.notna().sum()
    print(f"{len(df)} answers, {scored} with a factual claim, {int(df.flagged.sum())} flagged (tau = {args.tau:.2f}). Saved {args.out}")


if __name__ == "__main__":
    main()
