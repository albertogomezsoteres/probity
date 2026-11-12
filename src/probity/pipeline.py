"""The reproducible training pipeline: data -> hidden states -> probes -> evaluation.

Each stage can run on its own (``probity data``, ``probity extract``, ``probity train``) or all together
(``probity run``). Every stage is recorded in ``<output_dir>/runs/`` (see :mod:`probity.tracking`), and the
final step compares the reproduced numbers with the published ones in ``results/``.

Stage          GPU   Output
-------------  ----  ------------------------------------------------------------------
data           no    ``data/statements.csv`` (+ fact-grouped splits), checksums verified
extract        yes   ``activations/<key>.npz``: last-token hidden states at every layer
train          no*   ``app_probes.npz``, ``metrics.json``, layer and per-dataset tables
                     (*the ask-the-model baseline needs the model; skip it with
                     ``--set evaluation.ask_model=false``)
"""

from __future__ import annotations

import json
import math
import pathlib
import shutil
from typing import Any

import numpy as np
import pandas as pd

from . import data as data_mod
from .activations import HiddenStateReader, cache_key, load_activations, load_model, save_activations
from .config import Config
from .evaluation import accuracy_at_zero, auroc, bootstrap_auroc_ci, expected_calibration_error, per_dataset_auroc
from .tracking import Run, get_logger, set_seed, start_run, write_json
from .training import fit_probe, fit_probe_bank, select_layer

log = get_logger("pipeline")

STATEMENT_COLUMNS = ["statement", "label", "dataset", "topic", "negated", "split"]


def _data_dir(cfg: Config) -> pathlib.Path:
    d = cfg.output_dir / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def statements_path(cfg: Config) -> pathlib.Path:
    return _data_dir(cfg) / "statements.csv"


# ---------- stage 1: data ----------
def stage_data(cfg: Config, run: Run | None = None) -> pd.DataFrame:
    """Download and verify the datasets, then build the statements with both splits."""
    df = data_mod.prepare_statements(cfg)
    path = statements_path(cfg)
    df[STATEMENT_COLUMNS].to_csv(path, index=False)
    df.to_csv(path.with_name("statements_with_splits.csv"), index=False)
    digest = data_mod.sha256(path)
    same = digest == cfg.data.statements_sha256
    (log.info if same else log.warning)(
        "statements.csv %s the published file (sha256 %s)", "matches" if same else "DIFFERS from", digest[:12]
    )
    counts = df.split3.value_counts().to_dict()
    log.info("%d statements, %d facts; train/val/test = %s", len(df), df.fact.nunique(), counts)
    if run is not None:
        run.log_metrics(
            {
                "n_statements": len(df),
                "n_facts": int(df.fact.nunique()),
                "split3_counts": counts,
                "statements_sha256": digest,
                "matches_published_statements": same,
            }
        )
        run.log_artifact(path)
    return df


def load_statements_with_splits(cfg: Config) -> pd.DataFrame:
    """The table written by :func:`stage_data` (built on the fly if missing)."""
    p = statements_path(cfg).with_name("statements_with_splits.csv")
    if not p.exists():
        return stage_data(cfg)
    return pd.read_csv(p)


# ---------- stage 2: hidden states ----------
def activations_path(cfg: Config, statements: list[str]) -> pathlib.Path:
    d = cfg.output_dir / "activations"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"acts_{cfg.model.id.split('/')[-1]}_{cache_key(statements, cfg.model.id + cfg.model.fact_prompt)}.npz"


def stage_extract(
    cfg: Config, df: pd.DataFrame, reader: HiddenStateReader | None = None, run: Run | None = None
) -> dict[int, np.ndarray]:
    """Read the last-token hidden states of every statement at every layer (cached on disk)."""
    statements = df.statement.tolist()
    path = activations_path(cfg, statements)
    if path.exists():
        log.info("using cached activations %s", path.name)
        return load_activations(path)
    if reader is None:
        reader = HiddenStateReader(*load_model(cfg.model), cfg.model)
    layers = list(range(reader.n_layers + 1))
    acts = reader.read(reader.texts(statements), layers)
    save_activations(path, acts)
    log.info("saved %d layers x %d statements to %s", len(layers), len(statements), path)
    if run is not None:
        run.log_metrics({"n_layers": reader.n_layers, "d_model": reader.d_model, "activations": str(path)})
        run.log_artifact(path)
    return acts


# ---------- stage 3: probes and evaluation ----------
def stage_train(
    cfg: Config,
    df: pd.DataFrame,
    acts: dict[int, np.ndarray],
    reader: HiddenStateReader | None = None,
    run: Run | None = None,
) -> dict[str, Any]:
    """Choose the layer on validation facts, report the test set once, and fit the calibrated probe bank."""
    set_seed(cfg.run.seed)
    y = df.label.to_numpy().astype(int)
    train, val, test = data_mod.split_masks(df)
    groups = df.fact.to_numpy()

    # A. layer selection on validation facts (notebook 01b)
    if cfg.probe.layer is None:
        layer, val_auc = select_layer(acts, y, train, val, cfg.probe)
    else:
        layer, val_auc = cfg.probe.layer, {}
    log.info("layer %d selected on validation", layer)

    # B. refit on train + validation, read the test set once
    p_final = fit_probe(acts[layer][train | val], y[train | val], cfg.probe)
    s_te = p_final.score(acts[layer][test])
    test_auc = auroc(y[test], s_te)
    test_ci = bootstrap_auroc_ci(y[test], s_te, groups[test], cfg.evaluation.n_bootstrap, cfg.run.seed)
    metrics: dict[str, Any] = {
        "model": cfg.model.id,
        "layer_selected_on_validation": int(layer),
        "test_polygraph_auroc": round(test_auc, 3),
        "test_polygraph_ci": [round(x, 3) for x in test_ci],
        "test_polygraph_acc": round(accuracy_at_zero(y[test], s_te), 3),
        "n_test": int(test.sum()),
    }
    columns = {"polygraph_auroc": s_te}

    # C. baseline: ask the model directly
    if cfg.evaluation.ask_model:
        if reader is None:
            reader = HiddenStateReader(*load_model(cfg.model), cfg.model)
        ask = reader.ask_scores(df.statement[test].tolist())
        ask_ci = bootstrap_auroc_ci(y[test], ask, groups[test], cfg.evaluation.n_bootstrap, cfg.run.seed)
        metrics.update(
            test_ask_model_auroc=round(auroc(y[test], ask), 3), test_ask_model_ci=[round(x, 3) for x in ask_ci]
        )
        columns["ask_model_auroc"] = ask
    per_ds = per_dataset_auroc(df.dataset[test].to_numpy(), y[test], columns)

    # D. calibrated probes at every layer and the familiarity test, for the app (notebook 04)
    probe_layers = [L for L in sorted(acts) if L >= 1]
    bank = fit_probe_bank(acts, y, train, val, test, layer, probe_layers, cfg.probe, cfg.familiarity)
    lo, hi = cfg.evaluation.unsure_low, cfg.evaluation.unsure_high
    p_test = bank.TEST_P
    metrics["app"] = {
        "layer": int(layer),
        "test_auroc": round(float(bank.AUC_TEST[layer]), 3),
        "test_accuracy": round(float(((p_test > 0.5) == y[test]).mean()), 3),
        "ece": round(expected_calibration_error(p_test, y[test]), 3),
        "unsure_share": round(float(((p_test > lo) & (p_test < hi)).mean()), 3),
    }
    log.info("test AUROC %.3f [%.3f, %.3f] at layer %d", test_auc, *test_ci, layer)

    if run is not None:
        bank.save(run.dir / "app_probes.npz")
        per_ds.to_csv(run.dir / "results_test_per_dataset.csv", index=False)
        pd.DataFrame({"layer": list(val_auc), "val_auroc": list(val_auc.values())}).to_csv(
            run.dir / "layer_selection.csv", index=False
        )
        run.log_metrics(metrics)
        for name in ("app_probes.npz", "results_test_per_dataset.csv", "layer_selection.csv"):
            run.log_artifact(run.dir / name)
    return metrics


# ---------- reproducibility check ----------
def compare_with_published(metrics: dict[str, Any], results_dir: pathlib.Path) -> pd.DataFrame:
    """Side-by-side table of reproduced vs published headline numbers (``results/summary01b.json``)."""
    published = json.loads((results_dir / "summary01b.json").read_text())
    keys = [
        "layer_selected_on_validation",
        "test_polygraph_auroc",
        "test_polygraph_acc",
        "test_ask_model_auroc",
        "n_test",
    ]
    rows = []
    for k in keys:
        if k in metrics and k in published:
            a, b = metrics[k], published[k]
            match = (a == b) if isinstance(a, int) else math.isclose(float(a), float(b), abs_tol=0.0015)
            rows.append({"metric": k, "published": b, "reproduced": a, "match": match})
    return pd.DataFrame(rows)


def publish(run_dir: pathlib.Path, results_dir: pathlib.Path) -> None:
    """Copy the probes of a run into ``results/`` (they are what the app and the package load)."""
    shutil.copy2(run_dir / "app_probes.npz", results_dir / "app_probes.npz")
    log.info("published %s -> %s", run_dir / "app_probes.npz", results_dir / "app_probes.npz")


def run_all(cfg: Config, publish_results: bool = False) -> dict[str, Any]:
    """Run every stage in one go and check the result against the published numbers."""
    with start_run(cfg, "data") as run:
        df = stage_data(cfg, run)
    reader = HiddenStateReader(*load_model(cfg.model), cfg.model)
    with start_run(cfg, "extract") as run:
        acts = stage_extract(cfg, df, reader, run)
    with start_run(cfg, "train") as run:
        metrics = stage_train(cfg, df, acts, reader, run)
        table = compare_with_published(metrics, cfg.results_dir)
        table.to_csv(run.dir / "comparison_with_published.csv", index=False)
        log.info("reproduced vs published:\n%s", table.to_string(index=False))
        write_json(run.dir / "metrics.json", metrics)
        if publish_results:
            publish(run.dir, cfg.results_dir)
    return metrics
