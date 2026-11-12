"""Typed configuration for the Probity pipeline.

Every setting that affects a result lives here, with the values used for the published results as defaults.
`configs/default.yaml` mirrors these defaults (a test keeps the two in sync), and any value can be overridden
from the command line, for example ``probity train --set probe.C=0.05 --set run.seed=1``.
"""

from __future__ import annotations

import dataclasses
import pathlib
from dataclasses import dataclass, field
from typing import Any

import yaml

DEFAULT_CONFIG = pathlib.Path("configs") / "default.yaml"


@dataclass(frozen=True)
class DatasetSpec:
    """One Geometry of Truth dataset: its topic, whether it is negated, and the SHA-256 of the CSV file."""

    topic: str
    negated: bool
    sha256: str


def _default_datasets() -> dict[str, DatasetSpec]:
    return {
        "cities": DatasetSpec("geography", False, "f2560a0c1758c69935b50125111b574b71896bb75930dd87b1d8c973c1fa086e"),
        "neg_cities": DatasetSpec(
            "geography", True, "66c5c84de98ca15570138c205ab0e3907b57750083fac72a4eacafc3d9ebcea7"
        ),
        "sp_en_trans": DatasetSpec(
            "translation", False, "9afa97c1b8af539a05f47655a80a5c7832c6fdb914885fd24cb4bcccbe2795b5"
        ),
        "neg_sp_en_trans": DatasetSpec(
            "translation", True, "bec33e37047135b15cdd18c0cb0c66b65fb435ace08910cbcc220ce1ef610c96"
        ),
        "larger_than": DatasetSpec(
            "numbers", False, "406d6872e97910f3af5a031600d4932b5a567b8666e7545c8b446057471ad723"
        ),
        "smaller_than": DatasetSpec(
            "numbers", False, "ee6b7cf668d7e0fb3f49a84ca8d510cb7e484ae029ef86ed238362e26ef61a44"
        ),
        "companies_true_false": DatasetSpec(
            "business", False, "75a31139def9838dcaa4e2f1e5f3680f44f09f3c94553f9ad3b3a5a476124c8c"
        ),
        "counterfact_true_false": DatasetSpec(
            "knowledge", False, "5af51456d6dbd3a53d8e2ee6e4aadaca7eb02ff9136ad825866458e7283b5273"
        ),
        "common_claim_true_false": DatasetSpec(
            "claims", False, "1889b4793c2ebd13c767eea8b65ebd4a5c2da700567677d512097b495a262576"
        ),
    }


@dataclass
class RunConfig:
    """Where outputs go and how the run is seeded."""

    seed: int = 0
    output_dir: str = "outputs"  # activations, run logs and artifacts (not committed)
    results_dir: str = "results"  # the published results used by the README, the figures and the app
    log_level: str = "INFO"


@dataclass
class ModelConfig:
    """The language model whose hidden states are read."""

    id: str = "google/gemma-2-2b-it"
    dtype: str = "bfloat16"
    device: str = "cuda"
    attn_implementation: str = "eager"
    batch_size: int = 32
    # Statements are read as the model's own answer to this request (the "assistant" format of notebook 01).
    fact_prompt: str = "Tell me one fact."


@dataclass
class DataConfig:
    """The labelled statements: The Geometry of Truth (Marks and Tegmark, 2023), pinned to one commit."""

    repo: str = "saprmarks/geometry-of-truth"
    commit: str = "5d1c630c44f7e50bda7ad86d601ccadf9abc5ddb"
    cache_dir: str = "data/raw"
    max_per_dataset: int = 1000  # balanced cap per dataset (half true, half false)
    test_fraction: float = 0.2  # the stratified split of notebook 01
    statements_sha256: str = "3f34d6f3a29b54c0cade156541e02fc41fcb0925501964e07df2b6cdf9df6267"
    datasets: dict[str, DatasetSpec] = field(default_factory=_default_datasets)


@dataclass
class SplitConfig:
    """Fact-grouped train / validation / test split and cross-fitting folds (notebook 01b)."""

    train_pct: int = 60
    val_pct: int = 20  # the test set is the remaining 20%
    n_folds: int = 5


@dataclass
class ProbeConfig:
    """Logistic-regression probe on standardized activations."""

    C: float = 0.1
    max_iter: int = 2000
    tol: float = 1e-3
    layer: int | None = None  # None: choose the layer on validation facts
    # The app probes were fitted on activations stored in float16 (notebook 04); kept for exact reproduction.
    bank_activation_dtype: str = "float16"


@dataclass
class FamiliarityConfig:
    """Out-of-distribution flag: Mahalanobis distance in the top principal components."""

    n_components: int = 64
    percentile: float = 99.0


@dataclass
class EvaluationConfig:
    """How results are measured."""

    n_bootstrap: int = 1000
    ask_model: bool = True  # compare with simply asking the model "true or false?"
    unsure_low: float = 0.30
    unsure_high: float = 0.70


@dataclass
class GuardConfig:
    """Polygraph Guard (notebook 05): threshold and resampling settings."""

    tau: float = 0.50  # chosen on development subjects
    n_samples: int = 4
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 160


@dataclass
class AppConfig:
    """The interactive app."""

    probes: str = "results/app_probes.npz"
    lime_samples: int = 160
    ig_steps: int = 32
    monitor_max_rows: int = 500


@dataclass
class TrackingConfig:
    """Optional experiment tracking. Every run is always recorded locally in `<output_dir>/runs/`."""

    wandb: bool = False
    wandb_project: str = "probity"


@dataclass
class Config:
    """The full configuration of a run."""

    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    familiarity: FamiliarityConfig = field(default_factory=FamiliarityConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    guard: GuardConfig = field(default_factory=GuardConfig)
    app: AppConfig = field(default_factory=AppConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def to_dict(self) -> dict[str, Any]:
        """Plain dictionary (for YAML files and run metadata)."""
        return dataclasses.asdict(self)

    @property
    def output_dir(self) -> pathlib.Path:
        return pathlib.Path(self.run.output_dir)

    @property
    def results_dir(self) -> pathlib.Path:
        return pathlib.Path(self.run.results_dir)


def _build(cls: Any, values: dict[str, Any], where: str) -> Any:
    """Create dataclass `cls` from a dictionary, rejecting unknown keys so typos fail loudly."""
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(names)
    if unknown:
        raise KeyError(f"Unknown config key(s) in {where or 'root'}: {', '.join(sorted(unknown))}")
    kwargs: dict[str, Any] = {}
    for key, value in values.items():
        default = getattr(cls(), key) if cls is not DatasetSpec else None
        path = f"{where}.{key}" if where else key
        if dataclasses.is_dataclass(default):
            if not isinstance(value, dict):
                raise TypeError(f"Config section {path} must be a mapping, got {value!r}")
            kwargs[key] = _build(type(default), value, path)
        elif key == "datasets" and isinstance(value, dict):
            kwargs[key] = {name: _build(DatasetSpec, spec, f"{path}.{name}") for name, spec in value.items()}
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _set(d: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            raise KeyError(f"Unknown config section: {dotted}")
        d = d[k]
    if keys[-1] not in d:
        raise KeyError(f"Unknown config key: {dotted}")
    d[keys[-1]] = value


def load_config(path: str | pathlib.Path | None = None, overrides: list[str] | tuple[str, ...] = ()) -> Config:
    """Load a YAML config on top of the defaults and apply ``section.key=value`` overrides.

    With ``path=None``, ``configs/default.yaml`` is used when it exists, otherwise the built-in defaults.
    """
    data = Config().to_dict()
    p = pathlib.Path(path) if path is not None else DEFAULT_CONFIG
    if path is not None or p.exists():
        loaded = yaml.safe_load(p.read_text()) or {}
        for section, values in loaded.items():
            if section not in data:
                raise KeyError(f"Unknown config section: {section}")
            if isinstance(values, dict) and section != "data":
                data[section].update(values)
            elif isinstance(values, dict):
                datasets = values.pop("datasets", None)
                data[section].update(values)
                if datasets is not None:
                    data[section]["datasets"] = datasets
            else:
                raise TypeError(f"Section {section} must be a mapping")
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Overrides look like section.key=value, got: {item}")
        key, raw = item.split("=", 1)
        _set(data, key.strip(), yaml.safe_load(raw))
    return _build(Config, data, "")


def save_config(cfg: Config, path: str | pathlib.Path) -> None:
    """Write the resolved configuration to YAML."""
    pathlib.Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
