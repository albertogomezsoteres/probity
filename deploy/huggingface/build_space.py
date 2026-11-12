"""Assemble the Hugging Face Space from the repository: the app entry point, the package and the results it reads.

Usage::

    python deploy/huggingface/build_space.py build/space

The output folder is uploaded as is by the deploy workflow (.github/workflows/deploy-space.yml).
"""

from __future__ import annotations

import argparse
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent

# Files from results/ that the app reads.
RESULTS = [
    "app_probes.npz",  # calibrated probes at every layer and the familiarity test
    "summary01b.json",  # the layer chosen on validation
    "summary05.json",  # guard threshold and guard results
    "statements.csv",  # labels of the test statements (calibration error in the Evidence tab)
    "claims.csv",  # the sample log of the Monitor tab and the geography demo candidates
    "demo_cases_highstakes.json",  # medicine and finance demo candidates
    "demo_cases.json",  # verified demo cases (optional, written by `probity app --find-demos`)
    "demo_cases_app_highstakes.json",
]
FIGURES = ["fig7_guard_results.png", "fig3_shortcut.png", "fig2_negation.png", "fig6_deletion_test.png"]


def build(out: pathlib.Path) -> list[str]:
    """Write the Space into ``out`` (replacing it) and return the files written."""
    if out.exists():
        shutil.rmtree(out)
    (out / "results").mkdir(parents=True)
    (out / "figures").mkdir()
    (out / "configs").mkdir()
    for name in ("app.py", "README.md", "requirements.txt"):
        shutil.copy2(HERE / name, out / name)
    shutil.copytree(ROOT / "src" / "probity", out / "probity", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ROOT / "configs" / "default.yaml", out / "configs" / "default.yaml")
    for name in RESULTS:
        if (ROOT / "results" / name).exists():
            shutil.copy2(ROOT / "results" / name, out / "results" / name)
    for name in FIGURES:
        shutil.copy2(ROOT / "figures" / name, out / "figures" / name)
    shutil.copy2(ROOT / "LICENSE", out / "LICENSE")
    return sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?", default=str(ROOT / "build" / "space"))
    files = build(pathlib.Path(ap.parse_args().out))
    print(f"{len(files)} files written")


if __name__ == "__main__":
    main()
