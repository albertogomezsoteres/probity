"""Tests for the evaluation metrics."""

import math

import numpy as np

from probity.evaluation import (
    accuracy_at_zero,
    auroc,
    bootstrap_auroc_ci,
    expected_calibration_error,
    per_dataset_auroc,
)


def test_auroc_is_nan_with_one_class():
    assert math.isnan(auroc([1, 1, 1], [0.1, 0.2, 0.3]))
    assert auroc([0, 1], [0.1, 0.9]) == 1.0


def test_accuracy_uses_the_sign_of_the_score():
    assert accuracy_at_zero(np.array([0, 1, 1]), np.array([-1.0, 2.0, -0.5])) == 2 / 3


def test_bootstrap_interval_contains_the_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    s = y + rng.normal(scale=0.8, size=400)
    groups = np.repeat(np.arange(200), 2)
    lo, hi = bootstrap_auroc_ci(y, s, groups, n=200, seed=0)
    assert lo < auroc(y, s) < hi
    assert (lo, hi) == bootstrap_auroc_ci(y, s, groups, n=200, seed=0)


def test_ece_is_zero_for_perfect_calibration():
    p = np.repeat([0.05, 0.95], 100)
    y = np.r_[np.zeros(95), np.ones(5), np.ones(95), np.zeros(5)]
    assert expected_calibration_error(p, y) < 1e-9


def test_per_dataset_skips_single_class_datasets():
    ds = np.array(["a"] * 6 + ["b"] * 6)
    y = np.array([0, 1] * 3 + [1] * 6)
    t = per_dataset_auroc(ds, y, {"polygraph_auroc": y.astype(float)})
    assert t.dataset.tolist() == ["a"] and t.polygraph_auroc.iloc[0] == 1.0
