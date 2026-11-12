"""Tests for the typed configuration."""

import pytest

from probity.config import Config, DatasetSpec, load_config, save_config


def test_default_yaml_matches_the_code_defaults(in_repo):
    assert load_config("configs/default.yaml") == Config()


def test_defaults_are_the_published_settings():
    cfg = Config()
    assert cfg.model.id == "google/gemma-2-2b-it"
    assert (cfg.split.train_pct, cfg.split.val_pct) == (60, 20)
    assert cfg.probe.C == 0.1 and cfg.guard.tau == 0.5
    assert len(cfg.data.datasets) == 9
    assert all(isinstance(d, DatasetSpec) and len(d.sha256) == 64 for d in cfg.data.datasets.values())


def test_overrides_are_parsed_as_yaml_values():
    cfg = load_config(overrides=["probe.C=0.05", "run.seed=3", "probe.layer=14", "evaluation.ask_model=false"])
    assert cfg.probe.C == 0.05 and cfg.run.seed == 3 and cfg.probe.layer == 14 and cfg.evaluation.ask_model is False


@pytest.mark.parametrize("bad", ["probe.Cc=1", "nosuch.key=1", "probe=1"])
def test_unknown_keys_fail_loudly(bad):
    with pytest.raises((KeyError, ValueError, TypeError)):
        load_config(overrides=[bad])


def test_unknown_key_in_a_file_fails(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("probe:\n  lr: 0.1\n")
    with pytest.raises(KeyError, match="probe"):
        load_config(p)


def test_save_and_load_round_trip(tmp_path):
    cfg = load_config(overrides=["guard.tau=0.4"])
    save_config(cfg, tmp_path / "c.yaml")
    assert load_config(tmp_path / "c.yaml") == cfg
