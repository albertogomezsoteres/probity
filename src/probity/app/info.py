"""Everything the app shows that does not need the model: probe statistics, guard settings and figures."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..probe import UNSURE

# Figures shown in the "Evidence and limits" tab (files in figures/, made by `probity figures`).
EVIDENCE_FIGURES = [
    ("fig7_guard_results", "Polygraph Guard on held-out subjects, before and after."),
    (
        "fig3_shortcut",
        "Inside the conversation, layer 20 tracked the instruction to lie rather than the truth of the claim.",
    ),
    ("fig2_negation", "A probe trained without negations reads negated statements backwards."),
    ("fig6_deletion_test", "Each attribution method checked with a deletion test against random words."),
]

GUARD_DEFAULT = {"tau": 0.30, "n_samples": 4}


@dataclass
class AppInfo:
    """Probe bank, guard settings and published statistics, loaded from ``results/``."""

    layer: int  # the layer the needle reads (chosen on validation)
    W: np.ndarray
    B: np.ndarray
    PA: np.ndarray
    PC: np.ndarray
    AUC_TEST: np.ndarray
    MU: np.ndarray
    COMP: np.ndarray
    VAR: np.ndarray
    TEST_DIST: np.ndarray
    TEST_P: np.ndarray
    y_test: np.ndarray | None = None
    guard_summary: dict[str, Any] | None = None
    figures: list[tuple[pathlib.Path, str]] = field(default_factory=list)
    ood_percentile: float = 99.0

    @property
    def n_layers(self) -> int:
        return int(self.W.shape[0] - 1)

    @property
    def all_layers(self) -> list[int]:
        return list(range(1, self.n_layers + 1))

    @property
    def tau(self) -> float:
        return float(self.guard_summary["tau"] if self.guard_summary else GUARD_DEFAULT["tau"])

    @property
    def n_guard(self) -> int:
        return int(self.guard_summary["n_samples"] if self.guard_summary else GUARD_DEFAULT["n_samples"])

    @property
    def ood_threshold(self) -> float:
        return float(np.percentile(self.TEST_DIST, self.ood_percentile))

    def calibrated(self, score: Any, layer: int | None = None) -> Any:
        """Calibrated probability that the model treats a claim as true, from a raw probe score."""
        L = self.layer if layer is None else layer
        with np.errstate(over="ignore"):
            return 1 / (1 + np.exp(-(self.PA[L] * score + self.PC[L])))

    @property
    def app_stats(self) -> dict[str, Any]:
        """Held-out statistics of the needle (as printed by notebook 04)."""
        stats: dict[str, Any] = {
            "layer": self.layer,
            "test_auroc": float(self.AUC_TEST[self.layer]),
            "n_test": len(self.TEST_P),
            "unsure_share": float(((UNSURE[0] < self.TEST_P) & (UNSURE[1] > self.TEST_P)).mean()),
            "ece": float("nan"),
            "test_accuracy": float("nan"),
        }
        if self.y_test is not None:
            from ..evaluation import expected_calibration_error

            stats["ece"] = expected_calibration_error(self.TEST_P, self.y_test)
            stats["test_accuracy"] = float(((self.TEST_P > 0.5) == self.y_test).mean())
        return stats

    @classmethod
    def from_results(
        cls,
        results_dir: str | pathlib.Path = "results",
        probes: str | pathlib.Path | None = None,
        figures_dir: str | pathlib.Path = "figures",
        ood_percentile: float = 99.0,
    ) -> AppInfo:
        """Load the probe bank and the published summaries."""
        res = pathlib.Path(results_dir)
        z = np.load(probes or res / "app_probes.npz")
        layer = int(json.loads((res / "summary01b.json").read_text())["layer_selected_on_validation"])
        guard = res / "summary05.json"
        y_test = _test_labels(res)
        figs = [
            (pathlib.Path(figures_dir) / f"{n}.png", c)
            for n, c in EVIDENCE_FIGURES
            if (pathlib.Path(figures_dir) / f"{n}.png").exists()
        ]
        return cls(
            layer=layer,
            **{k: z[k] for k in ("W", "B", "PA", "PC", "AUC_TEST", "MU", "COMP", "VAR", "TEST_DIST", "TEST_P")},
            y_test=y_test if y_test is not None and len(y_test) == len(z["TEST_P"]) else None,
            guard_summary=json.loads(guard.read_text()) if guard.exists() else None,
            figures=figs,
            ood_percentile=ood_percentile,
        )


def _test_labels(res: pathlib.Path) -> np.ndarray | None:
    """Labels of the fact-grouped test statements, in the order used for ``TEST_P``."""
    path = res / "statements.csv"
    if not path.exists():
        return None
    import pandas as pd

    from ..config import SplitConfig
    from ..data import add_fact_splits

    df = add_fact_splits(pd.read_csv(path), SplitConfig())
    return df.label.to_numpy()[(df.split3 == "test").to_numpy()].astype(int)
