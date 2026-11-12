"""Shared test fixtures."""

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root() -> pathlib.Path:
    """The repository root (configs/, results/ and figures/ live here)."""
    return ROOT


@pytest.fixture
def in_repo(monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Run the test from the repository root, like the CLI."""
    monkeypatch.chdir(ROOT)
    return ROOT
