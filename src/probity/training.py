"""Train the truth probes, calibrate them and fit the familiarity test.

The probe is a logistic regression on standardized hidden states. Because standardization is linear, the fitted
scaler and classifier fold into a single direction ``w`` and bias ``b``: ``score(x) = x @ w + b``. That is the
form stored in ``app_probes.npz`` and used at run time.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .config import FamiliarityConfig, ProbeConfig
from .tracking import get_logger

log = get_logger("training")


@dataclass(frozen=True)
class LinearProbe:
    """A truth direction ``w`` and bias ``b``. Positive scores mean "true"."""

    w: np.ndarray
    b: float

    def score(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b


def fit_probe(X: np.ndarray, y: np.ndarray, cfg: ProbeConfig) -> LinearProbe:
    """Standardize, fit an L2 logistic regression and fold both into one linear probe."""
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=cfg.C, max_iter=cfg.max_iter, tol=cfg.tol).fit(sc.transform(X), y)
    w = clf.coef_[0] / sc.scale_
    b = float(clf.intercept_[0] - (sc.mean_ / sc.scale_) @ clf.coef_[0])
    return LinearProbe(w, b)


def select_layer(
    acts: dict[int, np.ndarray], y: np.ndarray, train: np.ndarray, val: np.ndarray, cfg: ProbeConfig
) -> tuple[int, dict[int, float]]:
    """Fit one probe per layer on ``train`` and pick the layer with the best validation AUROC."""
    val_auc = {}
    for L in sorted(acts):
        p = fit_probe(acts[L][train], y[train], cfg)
        val_auc[L] = float(roc_auc_score(y[val], p.score(acts[L][val])))
    best = max(val_auc, key=lambda L: val_auc[L])
    return best, val_auc


def fit_platt(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Platt scaling: ``p = sigmoid(a * score + c)``, fitted on held-out (validation) scores."""
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(scores[:, None], y)
    return float(platt.coef_[0, 0]), float(platt.intercept_[0])


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function (overflow for very negative inputs correctly gives 0)."""
    with np.errstate(over="ignore"):
        return 1 / (1 + np.exp(-z))


@dataclass
class Familiarity:
    """Mahalanobis distance in the top principal components of the training activations."""

    mu: np.ndarray
    comp: np.ndarray
    var: np.ndarray

    @classmethod
    def fit(cls, X_train: np.ndarray, cfg: FamiliarityConfig) -> Familiarity:
        mu = X_train.mean(0)
        _, S, Vt = np.linalg.svd(X_train - mu, full_matrices=False)
        k = cfg.n_components
        return cls(mu, Vt[:k], (S[:k] ** 2) / (len(X_train) - 1))

    def distance(self, X: np.ndarray) -> np.ndarray:
        return (((X - self.mu) @ self.comp.T) ** 2 / self.var).sum(1)


@dataclass
class ProbeBank:
    """Calibrated probes at every layer plus the familiarity test at the app layer (``app_probes.npz``)."""

    W: np.ndarray  # [n_layers + 1, d]; row 0 (embeddings) is unused
    B: np.ndarray
    PA: np.ndarray
    PC: np.ndarray
    AUC_VAL: np.ndarray
    AUC_TEST: np.ndarray
    MU: np.ndarray
    COMP: np.ndarray
    VAR: np.ndarray
    TEST_DIST: np.ndarray  # sorted distances of the test statements
    TEST_P: np.ndarray  # calibrated test probabilities at the app layer

    def save(self, path: str | pathlib.Path) -> None:
        np.savez(path, **self.__dict__)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> ProbeBank:
        z = np.load(path)
        return cls(**{k: z[k] for k in cls.__dataclass_fields__})


def fit_probe_bank(
    acts: dict[int, np.ndarray],
    y: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    app_layer: int,
    layers: Sequence[int],
    probe_cfg: ProbeConfig,
    fam_cfg: FamiliarityConfig,
) -> ProbeBank:
    """Fit, calibrate and score one probe per layer, as in notebook 04.

    Probes are trained on ``train``, calibrated on ``val`` and scored once on ``test``. Activations are rounded
    to ``probe_cfg.bank_activation_dtype`` first, as they were stored when the published probes were fitted.
    """
    n_rows = max(layers) + 1
    d = acts[layers[0]].shape[1]
    W = np.zeros((n_rows, d), np.float32)
    B, PA, PC = np.zeros(n_rows), np.ones(n_rows), np.zeros(n_rows)
    AUC_VAL, AUC_TEST = np.full(n_rows, np.nan), np.full(n_rows, np.nan)
    dtype = np.dtype(probe_cfg.bank_activation_dtype)
    for L in layers:
        XL = acts[L].astype(dtype).astype(np.float32)
        p = fit_probe(XL[train], y[train], probe_cfg)
        W[L], B[L] = p.w, p.b  # stored (and used at run time) in float32
        s_va, s_te = XL[val] @ W[L] + B[L], XL[test] @ W[L] + B[L]
        PA[L], PC[L] = fit_platt(s_va, y[val])
        AUC_VAL[L], AUC_TEST[L] = roc_auc_score(y[val], s_va), roc_auc_score(y[test], s_te)
        log.debug("layer %d: val AUROC %.3f, test AUROC %.3f", L, AUC_VAL[L], AUC_TEST[L])
    XA = acts[app_layer].astype(dtype).astype(np.float32)
    fam = Familiarity.fit(XA[train], fam_cfg)
    test_dist = np.sort(fam.distance(XA[test]))
    test_p = sigmoid(PA[app_layer] * (XA[test] @ W[app_layer] + B[app_layer]) + PC[app_layer])
    return ProbeBank(W, B, PA, PC, AUC_VAL, AUC_TEST, fam.mu, fam.comp, fam.var, test_dist, test_p)
