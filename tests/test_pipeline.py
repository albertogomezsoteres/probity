"""End-to-end test of the training stage on synthetic activations (no model, no GPU)."""

import json

import numpy as np
import pandas as pd

from probity import data as data_mod
from probity.config import load_config
from probity.pipeline import compare_with_published, stage_train
from probity.tracking import start_run


def _frame_and_acts(n=500, d=12, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {"statement": [f"Fact number {i} is stated here." for i in range(n)], "label": y, "dataset": "cities"}
    )
    df["dataset"] = np.where(np.arange(n) % 2 == 0, "cities", "sp_en_trans")
    df = data_mod.add_fact_splits(df, load_config().split)
    acts = {}
    for L in range(4):
        X = rng.normal(size=(n, d)).astype(np.float32)
        X[:, 0] += (L == 2) * 3 * (2 * y - 1)
        acts[L] = X
    return df, acts


def test_stage_train_writes_probes_and_metrics(tmp_path):
    cfg = load_config(
        overrides=[
            f"run.output_dir={tmp_path}",
            "evaluation.ask_model=false",
            "familiarity.n_components=4",
            "evaluation.n_bootstrap=50",
        ]
    )
    df, acts = _frame_and_acts()
    with start_run(cfg, "train") as run:
        metrics = stage_train(cfg, df, acts, run=run)
    assert metrics["layer_selected_on_validation"] == 2
    assert metrics["test_polygraph_auroc"] > 0.95
    lo, hi = metrics["test_polygraph_ci"]
    assert lo <= metrics["test_polygraph_auroc"] <= hi
    assert {"app_probes.npz", "metrics.json", "layer_selection.csv", "results_test_per_dataset.csv"} <= {
        p.name for p in run.dir.iterdir()
    }
    assert json.loads((run.dir / "metrics.json").read_text())["app"]["layer"] == 2


def test_comparison_with_published_results(repo_root):
    published = json.loads((repo_root / "results" / "summary01b.json").read_text())
    same = {k: published[k] for k in ("layer_selected_on_validation", "test_polygraph_auroc", "test_polygraph_acc")}
    table = compare_with_published(same, repo_root / "results")
    assert table.match.all() and len(table) == 3
    worse = dict(same, test_polygraph_auroc=published["test_polygraph_auroc"] - 0.01)
    assert not compare_with_published(worse, repo_root / "results").match.all()
