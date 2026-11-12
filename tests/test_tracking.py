"""Tests for seeding and run records."""

import json

import numpy as np
import pytest
import yaml

from probity.config import load_config
from probity.tracking import environment_info, git_info, set_seed, start_run


def test_set_seed_makes_numpy_reproducible():
    set_seed(7)
    a = np.random.rand(3)  # noqa: NPY002
    set_seed(7)
    assert np.array_equal(a, np.random.rand(3))  # noqa: NPY002


def test_environment_info_lists_versions():
    info = environment_info()
    assert info["python"] and "numpy" in info["packages"]


def test_git_info_outside_a_repository(tmp_path):
    assert git_info(tmp_path)["commit"] is None


def test_run_writes_config_metadata_and_metrics(tmp_path):
    cfg = load_config(overrides=[f"run.output_dir={tmp_path}"])
    with start_run(cfg, "unit") as run:
        run.log_metrics({"auroc": np.float64(0.98), "layer": np.int64(14)})
        run.log_artifact(tmp_path / "x.npz")
    meta = json.loads((run.dir / "run.json").read_text())
    assert meta["status"] == "completed" and meta["stage"] == "unit" and meta["seed"] == 0
    assert meta["data"]["commit"] == cfg.data.commit
    assert json.loads((run.dir / "metrics.json").read_text()) == {"auroc": 0.98, "layer": 14}
    assert yaml.safe_load((run.dir / "config.yaml").read_text())["run"]["output_dir"] == str(tmp_path)
    assert (run.dir / "run.log").read_text()


def test_failed_run_is_recorded(tmp_path):
    cfg = load_config(overrides=[f"run.output_dir={tmp_path}"])
    with pytest.raises(RuntimeError), start_run(cfg, "unit") as run:
        raise RuntimeError("boom")
    assert json.loads((run.dir / "run.json").read_text())["status"] == "failed"
