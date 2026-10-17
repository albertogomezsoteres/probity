"""The linear truth probe: logistic regression on one layer of the residual stream.

The weights are trained in notebook 01b (train facts only) and saved by the app notebook to `results/app_probes.npz`.
The score is calibrated with Platt scaling, so `p_true` can be read as a probability. A Mahalanobis distance
in a 64-dimensional PCA space tells us if a claim looks like the statements the probe was trained on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UNSURE = (0.30, 0.70)  # readings between these two values are shown as "undecided"


@dataclass
class Probe:
    """Linear probe at one layer with Platt calibration and a familiarity (out-of-distribution) test."""
    layer: int
    w: np.ndarray
    b: float
    platt_a: float
    platt_c: float
    mu: np.ndarray | None = None
    comp: np.ndarray | None = None
    var: np.ndarray | None = None
    ood_threshold: float | None = None

    @classmethod
    def from_npz(cls, path: str, layer: int, ood_percentile: float = 99) -> "Probe":
        """Load the probe for `layer` from an .npz file written by the app notebook."""
        z = np.load(path)
        thr = float(np.percentile(z["TEST_DIST"], ood_percentile)) if "TEST_DIST" in z else None
        return cls(layer, z["W"][layer].astype(np.float32), float(z["B"][layer]), float(z["PA"][layer]), float(z["PC"][layer]),
                   z["MU"] if "MU" in z else None, z["COMP"] if "COMP" in z else None, z["VAR"] if "VAR" in z else None, thr)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Raw probe score (a dot product with the truth direction). Positive means true."""
        return X @ self.w + self.b

    def p_true(self, X: np.ndarray) -> np.ndarray:
        """Calibrated probability that the model treats each claim as true."""
        return 1 / (1 + np.exp(-(self.platt_a * self.score(X) + self.platt_c)))

    def familiar(self, X: np.ndarray) -> np.ndarray:
        """False for claims that are far from the training statements (the reading is less reliable)."""
        if self.mu is None or self.ood_threshold is None:
            return np.ones(len(X), bool)
        d = (((X - self.mu) @ self.comp.T) ** 2 / self.var).sum(1)
        return d <= self.ood_threshold


def verdict_word(p: float | None) -> str:
    """Turn a probability into the label shown in the app."""
    if p is None:
        return "no factual claim"
    return "believed false" if p <= UNSURE[0] else "undecided" if p < UNSURE[1] else "believed true"
