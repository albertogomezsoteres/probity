"""Check one or more claims with the probe from the command line.

Usage (needs a GPU and access to google/gemma-2-2b-it on Hugging Face):
    python scripts/check_claims.py "Anyang-si is located in Iraq." "The Spanish word 'uno' means 'one'."
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probity import Polygraph  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("texts", nargs="+", help="sentences or full replies to check")
    ap.add_argument("--probe", default=str(ROOT / "results" / "app_probes.npz"))
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    args = ap.parse_args()

    poly = Polygraph.from_pretrained(args.model, probe=args.probe, layer=args.layer)
    for text in args.texts:
        r = poly.check(text)
        p = "  n/a" if r["p_true"] is None else f"{r['p_true']:.2f}"
        print(f"{p}  {r['verdict']:<17} {text}")


if __name__ == "__main__":
    main()
