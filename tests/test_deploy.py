"""The Hugging Face Space bundle contains everything the app reads."""

import importlib.util
import pathlib

from probity.app.demos import DemoCases
from probity.app.info import AppInfo


def _build_space():
    path = pathlib.Path(__file__).resolve().parents[1] / "deploy" / "huggingface" / "build_space.py"
    spec = importlib.util.spec_from_file_location("build_space", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_space_bundle_is_self_contained(tmp_path):
    files = _build_space().build(tmp_path / "space")
    space = tmp_path / "space"
    for f in ("app.py", "README.md", "requirements.txt", "configs/default.yaml", "probity/app/static/probity.css"):
        assert f in files
    assert not any("__pycache__" in f for f in files)
    info = AppInfo.from_results(space / "results", figures_dir=space / "figures")
    assert info.layer == 14 and len(info.figures) == 4 and info.y_test is not None
    assert DemoCases.load(space / "results").geography
    assert "sdk: gradio" in (space / "README.md").read_text()
