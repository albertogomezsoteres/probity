"""Metrics with the evaluation protocol of the project (see ``experiments/protocol.md``).

AUROC is the main metric because it does not depend on a threshold. Confidence intervals resample whole facts
(or subjects), never single sentences, because statements about the same fact are not independent.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


def auroc(y: Sequence[int] | np.ndarray, s: Sequence[float] | np.ndarray) -> float:
    """Area under the ROC curve; NaN when only one class is present."""
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def accuracy_at_zero(y: np.ndarray, s: np.ndarray) -> float:
    """Accuracy of the sign of a probe score (positive means true)."""
    return float(accuracy_score(y, (np.asarray(s) > 0).astype(int)))


def bootstrap_auroc_ci(
    y: np.ndarray, s: np.ndarray, groups: Sequence[str] | np.ndarray, n: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95% interval for AUROC, resampling whole groups (all statements about one fact move together)."""
    rng = np.random.default_rng(seed)
    y, s, groups = np.asarray(y), np.asarray(s), np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    vals = []
    for _ in range(n):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        if 0 < y[idx].sum() < len(idx):
            vals.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width probability bins."""
    p, y = np.asarray(p), np.asarray(y)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    return float(
        sum(
            abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean()
            for b in range(n_bins)
            if (bins == b).any()
        )
    )


def per_dataset_auroc(
    datasets: Sequence[str] | np.ndarray, y: np.ndarray, columns: dict[str, np.ndarray], min_n: int = 5
) -> pd.DataFrame:
    """AUROC of every score column, dataset by dataset (datasets with one class or fewer than ``min_n`` are skipped)."""
    datasets = np.asarray(datasets)
    rows = []
    for d in pd.unique(datasets):
        m = datasets == d
        if m.sum() < min_n or len(np.unique(y[m])) < 2:
            continue
        rows.append({"dataset": d, "n": int(m.sum()), **{k: auroc(y[m], v[m]) for k, v in columns.items()}})
    return pd.DataFrame(rows).round(3)
