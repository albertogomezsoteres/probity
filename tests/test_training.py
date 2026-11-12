"""Tests for probe training, calibration, the familiarity test and the probe bank."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from probity.config import FamiliarityConfig, ProbeConfig
from probity.probe import Probe
from probity.training import Familiarity, ProbeBank, fit_platt, fit_probe, fit_probe_bank, select_layer, sigmoid


def _synthetic(n=600, d=16, seed=0, signal_layers=(2,)):
    """Activations at layers 0..3; only `signal_layers` carry the label along dimension 0."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    acts = {}
    for L in range(4):
        X = rng.normal(size=(n, d)).astype(np.float32)
        if L in signal_layers:
            X[:, 0] += 3 * (2 * y - 1)
        acts[L] = X
    split = rng.choice(["train", "val", "test"], size=n, p=[0.6, 0.2, 0.2])
    return acts, y, split == "train", split == "val", split == "test"


def test_folded_probe_equals_scaler_plus_logistic_regression():
    acts, y, tr, _, _ = _synthetic()
    X = acts[2]
    cfg = ProbeConfig()
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(C=cfg.C, max_iter=cfg.max_iter, tol=cfg.tol).fit(sc.transform(X[tr]), y[tr])
    p = fit_probe(X[tr], y[tr], cfg)
    np.testing.assert_allclose(p.score(X), clf.decision_function(sc.transform(X)), rtol=1e-6, atol=1e-6)


def test_layer_selection_finds_the_informative_layer():
    acts, y, tr, va, _ = _synthetic()
    layer, val_auc = select_layer(acts, y, tr, va, ProbeConfig())
    assert layer == 2 and val_auc[2] > 0.95 > max(val_auc[L] for L in (0, 1, 3))


def test_platt_scaling_is_monotonic_and_calibrated_direction():
    rng = np.random.default_rng(0)
    s = rng.normal(size=500)
    y = (s + rng.normal(scale=0.5, size=500) > 0).astype(int)
    a, c = fit_platt(s, y)
    assert a > 0
    p = sigmoid(a * np.sort(s) + c)
    assert np.all(np.diff(p) >= 0) and 0.3 < p[len(p) // 2] < 0.7


def test_familiarity_flags_far_away_points():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 8))
    fam = Familiarity.fit(X, FamiliarityConfig(n_components=4))
    assert fam.distance(X[:5] * 0).max() < fam.distance(X[:5] + 50).min()


@pytest.fixture
def bank():
    acts, y, tr, va, te = _synthetic()
    return fit_probe_bank(acts, y, tr, va, te, 2, [1, 2, 3], ProbeConfig(), FamiliarityConfig(n_components=4)), y, te


def test_probe_bank_matches_the_published_file_format(bank, repo_root):
    b, _, _ = bank
    published = np.load(repo_root / "results" / "app_probes.npz")
    assert set(b.__dict__) == set(published.files)
    assert b.W.shape[0] == 4 and np.all(b.W[0] == 0) and np.isnan(b.AUC_TEST[0])
    assert b.AUC_TEST[2] > 0.95 and np.all(np.diff(b.TEST_DIST) >= 0)


def test_probe_bank_round_trip_and_runtime_probe(bank, tmp_path):
    b, y, te = bank
    path = tmp_path / "app_probes.npz"
    b.save(path)
    loaded = ProbeBank.load(path)
    np.testing.assert_array_equal(loaded.W, b.W)
    probe = Probe.from_npz(str(path), layer=2)
    assert probe.ood_threshold == pytest.approx(np.percentile(b.TEST_DIST, 99))
    acts, *_ = _synthetic()
    p = probe.p_true(acts[2][te])
    assert ((p > 0.5) == y[te]).mean() > 0.95


def test_published_probes_load(repo_root):
    probe = Probe.from_npz(str(repo_root / "results" / "app_probes.npz"), layer=14)
    assert probe.w.shape == (2304,) and probe.platt_a > 0 and probe.ood_threshold is not None


def test_activations_round_trip(tmp_path):
    from probity.activations import load_activations, save_activations

    acts, *_ = _synthetic(n=20)
    save_activations(tmp_path / "a.npz", acts)
    loaded = load_activations(tmp_path / "a.npz")
    assert sorted(loaded) == sorted(acts) and all(np.array_equal(loaded[L], acts[L]) for L in acts)
