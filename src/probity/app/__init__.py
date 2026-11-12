"""The interactive Probity app (Gradio): guarded chat, monitor, explanations, brain scan, evidence and docs.

Launch it with ``probity app`` on a machine with a GPU, or deploy it to Hugging Face Spaces with the files in
``deploy/huggingface``. The pieces are:

* :mod:`~probity.app.info`: probe bank and published statistics (no model),
* :mod:`~probity.app.engine`: everything that runs the model,
* :mod:`~probity.app.views` and :mod:`~probity.app.theme`: the HTML and the design system,
* :mod:`~probity.app.server`: the Gradio layout and event handlers,
* :mod:`~probity.app.demos`: one-click demo cases.
"""

from __future__ import annotations

import pathlib
from typing import Any

from ..config import Config


def create_app(
    cfg: Config, figures_dir: str | pathlib.Path = "figures", find_demos: bool = False
) -> tuple[Any, dict[str, Any]]:
    """Load the probes and the model and build the app. Returns ``(blocks, launch_kwargs)``."""
    from .demos import DemoCases, find_demo_cases
    from .engine import Engine
    from .info import AppInfo
    from .server import build_demo

    info = AppInfo.from_results(cfg.results_dir, cfg.app.probes, figures_dir, cfg.familiarity.percentile)
    engine = Engine.load(cfg, info)
    demos = find_demo_cases(engine, cfg.results_dir) if find_demos else DemoCases.load(cfg.results_dir)
    return build_demo(engine, demos, cfg.results_dir)


__all__ = ["create_app"]
