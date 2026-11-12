"""Reproducibility utilities: seeding, logging and a record of every run.

Each pipeline stage runs inside :func:`start_run`, which creates ``<output_dir>/runs/<run_id>/`` with

* ``config.yaml``: the fully resolved configuration,
* ``run.json``: run id, stage, status, timings, git commit, data version, software and hardware versions,
* ``metrics.json``: the numbers the stage produced,
* ``run.log``: the log of the stage.

Weights & Biases is used in addition when ``tracking.wandb`` is enabled and the package is installed.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import importlib.metadata
import json
import logging
import os
import pathlib
import platform
import random
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import Config, save_config

LOGGER_NAME = "probity"
_PACKAGES = ("probity", "numpy", "pandas", "scikit-learn", "torch", "transformers", "accelerate", "gradio", "shap")


def get_logger(name: str | None = None) -> logging.Logger:
    """Logger under the ``probity`` namespace."""
    return logging.getLogger(LOGGER_NAME if not name else f"{LOGGER_NAME}.{name}")


def setup_logging(level: str | int = "INFO") -> logging.Logger:
    """Send ``probity`` logs to the console with timestamps (idempotent)."""
    logger = get_logger()
    logger.setLevel(level)
    if not any(getattr(h, "_probity", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
        handler._probity = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch, and ask PyTorch for deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - also seed the legacy global RNG used by third-party code
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def git_info(cwd: str | pathlib.Path = ".") -> dict[str, Any]:
    """Current commit, branch and whether the working tree has uncommitted changes."""

    def git(*args: str) -> str:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "branch": None, "dirty": None}


def environment_info() -> dict[str, Any]:
    """Software and hardware versions that can change a result."""
    versions: dict[str, str | None] = {}
    for pkg in _PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "cuda": None,
        "gpu": None,
    }
    try:
        import torch

        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pathlib.Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o).__name__}")


def write_json(path: str | pathlib.Path, obj: Any) -> None:
    """Write JSON with NumPy types converted."""
    pathlib.Path(path).write_text(json.dumps(obj, indent=2, default=_json_default) + "\n")


@dataclass
class Run:
    """A pipeline run: collects metrics and artifacts and writes them to its directory."""

    id: str
    stage: str
    dir: pathlib.Path
    config: Config
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    _wandb: Any = None

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Record metrics (merged into ``metrics.json`` and sent to W&B when enabled)."""
        self.metrics.update(metrics)
        write_json(self.dir / "metrics.json", self.metrics)
        if self._wandb is not None:
            flat = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            if flat:
                self._wandb.log(flat)

    def log_artifact(self, path: str | pathlib.Path) -> None:
        """Remember a file produced by this run."""
        self.artifacts.append(str(path))


@contextlib.contextmanager
def start_run(cfg: Config, stage: str) -> Iterator[Run]:
    """Create a run directory, log to it, and write ``run.json`` when the stage ends (also on failure)."""
    setup_logging(cfg.run.log_level)
    now = dt.datetime.now(dt.timezone.utc)
    run_id = f"{now:%Y%m%d-%H%M%S}-{stage}-{uuid.uuid4().hex[:6]}"
    run_dir = cfg.output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    logger = get_logger()
    file_handler = logging.FileHandler(run_dir / "run.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    logger.addHandler(file_handler)

    run = Run(run_id, stage, run_dir, cfg)
    if cfg.tracking.wandb:
        try:
            import wandb

            run._wandb = wandb.init(
                project=cfg.tracking.wandb_project, name=run_id, job_type=stage, config=cfg.to_dict()
            )
        except ImportError:
            logger.warning("tracking.wandb is enabled but wandb is not installed; recording locally only")

    meta: dict[str, Any] = {
        "run_id": run_id,
        "stage": stage,
        "started_at": now.isoformat(timespec="seconds"),
        "seed": cfg.run.seed,
        "model": cfg.model.id,
        "data": {"repo": cfg.data.repo, "commit": cfg.data.commit},
        "git": git_info(),
        "environment": environment_info(),
    }
    logger.info("run %s started (%s)", run_id, run_dir)
    t0 = time.perf_counter()
    status = "failed"
    try:
        yield run
        status = "completed"
    finally:
        meta.update(
            status=status,
            finished_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            duration_s=round(time.perf_counter() - t0, 2),
            artifacts=run.artifacts,
        )
        write_json(run_dir / "run.json", meta)
        if not (run_dir / "metrics.json").exists():
            write_json(run_dir / "metrics.json", run.metrics)
        logger.info("run %s %s in %.1fs", run_id, status, meta["duration_s"])
        logger.removeHandler(file_handler)
        file_handler.close()
        if run._wandb is not None:
            run._wandb.finish(exit_code=0 if status == "completed" else 1)
