"""Tests for the command-line interface."""

import json

import pytest

from probity.cli import build_parser, main


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for cmd in ("data", "extract", "train", "run", "check", "audit", "info"):
        assert cmd in out


def test_overrides_are_collected():
    args = build_parser().parse_args(["train", "--set", "probe.C=0.05", "-s", "run.seed=2", "--publish"])
    assert args.set == ["probe.C=0.05", "run.seed=2"] and args.publish


def test_info_prints_the_resolved_config(capsys, in_repo):
    assert main(["info", "--set", "guard.tau=0.4"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["config"]["guard"]["tau"] == 0.4 and "python" in info["environment"]
