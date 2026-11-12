"""The reports are rebuilt from results/ alone, without a model."""

import json

from probity.reports import figures, robustness


def test_robustness_checks_reproduce_the_saved_file(repo_root, capsys):
    out = robustness.compute(repo_root / "results")
    saved = json.loads((repo_root / "results" / "robustness_checks.json").read_text())
    assert json.loads(json.dumps(out)) == saved
    robustness.print_report(out)
    assert "E. Guard" in capsys.readouterr().out


def test_every_figure_is_generated(repo_root, tmp_path):
    names = figures.generate(repo_root / "results", tmp_path, repo_root / "assets" / "fonts")
    committed = sorted(p.stem for p in (repo_root / "figures").glob("*.png"))
    assert names == committed
    assert all((tmp_path / f"{n}.svg").exists() for n in names)
