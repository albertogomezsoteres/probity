"""Command-line interface: ``probity <command>``. Run ``probity --help`` for the list of commands."""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Callable, Sequence

from .config import Config, load_config
from .polygraph import Polygraph
from .tracking import get_logger, setup_logging, start_run

log = get_logger("cli")


def _config(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config, args.set or ())
    setup_logging(cfg.run.log_level)
    return cfg


# ---------- pipeline ----------
def cmd_data(args: argparse.Namespace) -> int:
    from .pipeline import stage_data

    cfg = _config(args)
    with start_run(cfg, "data") as run:
        stage_data(cfg, run)
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    from .pipeline import load_statements_with_splits, stage_extract

    cfg = _config(args)
    df = load_statements_with_splits(cfg)
    with start_run(cfg, "extract") as run:
        stage_extract(cfg, df, run=run)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .pipeline import (
        compare_with_published,
        load_statements_with_splits,
        publish,
        stage_extract,
        stage_train,
    )

    cfg = _config(args)
    df = load_statements_with_splits(cfg)
    acts = stage_extract(cfg, df)
    with start_run(cfg, "train") as run:
        metrics = stage_train(cfg, df, acts, run=run)
        print(compare_with_published(metrics, cfg.results_dir).to_string(index=False))
        if args.publish:
            publish(run.dir, cfg.results_dir)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run_all

    run_all(_config(args), publish_results=args.publish)
    return 0


# ---------- using the probe ----------
def _polygraph(args: argparse.Namespace) -> tuple[Config, Polygraph]:
    cfg = _config(args)
    layer = args.layer if args.layer is not None else _published_layer(cfg)
    probe = args.probe or cfg.app.probes
    return cfg, Polygraph.from_pretrained(
        cfg.model.id, probe=probe, layer=layer, device=cfg.model.device, tau=cfg.guard.tau
    )


def _published_layer(cfg: Config) -> int:
    import json

    p = cfg.results_dir / "summary01b.json"
    return int(json.loads(p.read_text())["layer_selected_on_validation"]) if p.exists() else 14


def cmd_check(args: argparse.Namespace) -> int:
    _, poly = _polygraph(args)
    for text in args.texts:
        r = poly.check(text)
        p = "  n/a" if r["p_true"] is None else f"{r['p_true']:.2f}"
        print(f"{p}  {r['verdict']:<17} {text}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    import pandas as pd

    cfg, poly = _polygraph(args)
    tau = cfg.guard.tau if args.tau is None else args.tau
    path = pathlib.Path(args.log)
    df = pd.read_json(path, lines=True) if path.suffix in (".jsonl", ".json") else pd.read_csv(path)
    if "answer" not in df.columns:
        log.error("The log needs an 'answer' column.")
        return 2
    if args.max_rows:
        df = df.head(args.max_rows)
    results = [poly.check(str(a)) for a in df.answer]
    df["p_true"] = [r["p_true"] for r in results]
    df["verdict"] = [r["verdict"] for r in results]
    df["weakest_claim"] = [r["weakest_claim"] for r in results]
    df["trusted_reading"] = [r["trusted"] for r in results]
    df["flagged"] = df.p_true.notna() & (df.p_true < tau)
    df.to_csv(args.out, index=False)
    scored = int(df.p_true.notna().sum())
    print(
        f"{len(df)} answers, {scored} with a factual claim, {int(df.flagged.sum())} flagged (tau = {tau:.2f}). Saved {args.out}"
    )
    return 0


# ---------- reports (no GPU) ----------
def cmd_figures(args: argparse.Namespace) -> int:
    from .reports.figures import generate

    cfg = _config(args)
    names = generate(cfg.results_dir, args.out, args.fonts)
    print(f"{len(names)} figures in {args.out}/")
    return 0


def cmd_robustness(args: argparse.Namespace) -> int:
    from .reports.robustness import run

    cfg = _config(args)
    run(cfg.results_dir, write=not args.no_write)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    import json

    from .tracking import environment_info, git_info

    cfg = _config(args)
    print(json.dumps({"git": git_info(), "environment": environment_info(), "config": cfg.to_dict()}, indent=2))
    return 0


# ---------- parser ----------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="probity",
        description="Probity: a white-box polygraph and guardrail for open language models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Pipeline: data -> extract -> train (or `run` for all three). Docs: README.md",
    )
    from . import __version__

    parser.add_argument("--version", action="version", version=f"probity {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", default=None, help="YAML config (default: configs/default.yaml)")
    common.add_argument(
        "-s", "--set", action="append", metavar="KEY=VALUE", help="override a config value, e.g. probe.C=0.05"
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    def add(name: str, help_: str, func: Callable[[argparse.Namespace], int]) -> argparse.ArgumentParser:
        p = sub.add_parser(name, parents=[common], help=help_, description=help_)
        p.set_defaults(func=func)
        return p

    add("data", "Download, verify and split the labelled statements (no GPU).", cmd_data)
    add("extract", "Read the hidden states of every statement at every layer (GPU).", cmd_extract)
    p = add("train", "Choose the layer, evaluate on held-out facts and fit the app probes.", cmd_train)
    p.add_argument("--publish", action="store_true", help="copy the new app_probes.npz into results/")
    p = add("run", "Run the full pipeline (data, extract, train) and compare with the published results.", cmd_run)
    p.add_argument("--publish", action="store_true", help="copy the new app_probes.npz into results/")

    for name, help_, func in [
        ("check", "Check one or more sentences or replies with the probe (GPU).", cmd_check),
        ("audit", "Audit a CSV/JSONL log of answers and flag the ones the model does not believe (GPU).", cmd_audit),
    ]:
        p = add(name, help_, func)
        p.add_argument("--probe", default=None, help="probe file (default: app.probes in the config)")
        p.add_argument("--layer", type=int, default=None, help="probe layer (default: the published layer)")
        if name == "check":
            p.add_argument("texts", nargs="+", help="sentences or full replies to check")
        else:
            p.add_argument("log", help="CSV or JSONL file with an 'answer' column")
            p.add_argument("--out", default="audited.csv")
            p.add_argument("--tau", type=float, default=None, help="flag answers below this probability")
            p.add_argument("--max-rows", type=int, default=None)

    p = add("figures", "Rebuild the README and poster figures from results/ (no GPU).", cmd_figures)
    p.add_argument("--out", default="figures", help="output directory")
    p.add_argument("--fonts", default="assets/fonts", help="directory with the IBM Plex fonts")
    p = add("robustness", "Re-run the robustness checks on results/ (no GPU).", cmd_robustness)
    p.add_argument("--no-write", action="store_true", help="print the report without updating robustness_checks.json")
    add("info", "Print the resolved config and the software, hardware and git versions.", cmd_info)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
