"""The labelled statements: download, checksum, balance and split.

The data are the curated true/false datasets of *The Geometry of Truth* (Marks and Tegmark, 2023), pinned to
one commit and verified with SHA-256 checksums, so the statements can never change silently.

Two splits are used, exactly as in the notebooks:

* ``split`` (notebook 01): a stratified 80/20 split inside every dataset and label, kept for reference.
* ``split3`` and ``fold`` (notebook 01b): train / validation / test and cross-fitting folds drawn over **facts**,
  so "X is in Y" and "X is not in Y" always land in the same split. Every reported number uses this split.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
import shutil
import urllib.request

import numpy as np
import pandas as pd

from .config import Config, DataConfig, SplitConfig
from .tracking import get_logger

log = get_logger("data")

RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/datasets/{name}.csv"
SPLITS = ("train", "val", "test")


def sha256(path: str | pathlib.Path) -> str:
    """SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_datasets(cfg: DataConfig, force: bool = False) -> pathlib.Path:
    """Download the dataset CSVs at the pinned commit into ``cache_dir`` and verify their checksums.

    Files already present with the right checksum are not downloaded again.
    """
    out = pathlib.Path(cfg.cache_dir) / cfg.commit[:7]
    out.mkdir(parents=True, exist_ok=True)
    for name, spec in cfg.datasets.items():
        path = out / f"{name}.csv"
        if path.exists() and not force and sha256(path) == spec.sha256:
            continue
        url = RAW_URL.format(repo=cfg.repo, commit=cfg.commit, name=name)
        log.info("downloading %s", url)
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        got = sha256(tmp)
        if got != spec.sha256:
            tmp.unlink()
            raise ValueError(f"Checksum mismatch for {name}.csv: expected {spec.sha256}, got {got}")
        tmp.replace(path)
    return out


def load_statements(cfg: DataConfig, raw_dir: str | pathlib.Path, seed: int = 0) -> pd.DataFrame:
    """Balanced statements with their dataset, topic and the stratified split of notebook 01.

    Each dataset is capped at ``max_per_dataset`` statements, half true and half false, sampled with ``seed``.
    """
    raw_dir = pathlib.Path(raw_dir)
    frames = []
    for name, spec in cfg.datasets.items():
        d = pd.read_csv(raw_dir / f"{name}.csv")[["statement", "label"]]
        d = d.dropna().drop_duplicates("statement")
        d["label"] = d["label"].astype(int)
        n_per_class = min(cfg.max_per_dataset // 2, d["label"].value_counts().min())
        d = pd.concat([d[d.label == c].sample(n_per_class, random_state=seed) for c in (0, 1)])
        d["dataset"], d["topic"], d["negated"] = name, spec.topic, spec.negated
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    # stratified train/test split inside every dataset x label cell
    rng = np.random.default_rng(seed)
    df["split"] = "train"
    for _, cell in df.groupby(["dataset", "label"]).groups.items():
        rows = np.array(list(cell))
        rng.shuffle(rows)
        df.loc[rows[: round(len(rows) * cfg.test_fraction)], "split"] = "test"
    return df.reset_index(drop=True)


_COMPANY_VERBS = r"(?:engages|is|operates|has|was|provides|develops|manufactures|offers|specializes|focuses)"


def fact_id(statement: str, dataset: str) -> str:
    """Key shared by every statement about the same fact (city, Spanish word, number pair, company or subject)."""
    s, ds = statement, dataset
    if ds in ("cities", "neg_cities"):
        m = re.match(r"The city of (.+?) is", s)
        return "city:" + (m.group(1) if m else s)
    if ds in ("sp_en_trans", "neg_sp_en_trans"):
        m = re.match(r"The Spanish word '(.+?)'", s)
        return "sp:" + (m.group(1) if m else s)
    if ds in ("larger_than", "smaller_than"):
        m = re.match(r"(.+?) is (?:larger|smaller) than (.+?)\.", s)
        return "num:" + "|".join(sorted(m.groups())) if m else "num:" + s
    if ds == "counterfact_true_false":
        return "cf:" + " ".join(s.split()[:6])
    if ds == "companies_true_false":  # group every claim about the same company
        m = re.match(rf"(.+?) {_COMPANY_VERBS}\b", s)
        return "co:" + (m.group(1) if m else s)
    # Caveat: other datasets are grouped by sentence, so related claims can still fall in different splits.
    return ds + ":" + s


def add_fact_splits(df: pd.DataFrame, cfg: SplitConfig, seed: int = 0) -> pd.DataFrame:
    """Add ``fact``, ``split3`` (train / val / test) and ``fold``, all drawn over facts with an md5 hash."""
    df = df.copy()
    df["fact"] = [fact_id(s, d) for s, d in zip(df.statement, df.dataset)]
    h = np.array([int(hashlib.md5((f + str(seed)).encode()).hexdigest(), 16) % 100 for f in df.fact])
    df["split3"] = np.where(h < cfg.train_pct, "train", np.where(h < cfg.train_pct + cfg.val_pct, "val", "test"))
    df["fold"] = h % cfg.n_folds
    check_no_leakage(df)
    return df


def check_no_leakage(df: pd.DataFrame) -> None:
    """Fail if a fact appears in more than one split or fold."""
    if df.groupby("fact").split3.nunique().max() > 1:
        raise AssertionError("A fact appears in more than one split")
    if df.groupby("fact").fold.nunique().max() > 1:
        raise AssertionError("A fact appears in more than one fold")


def prepare_statements(cfg: Config) -> pd.DataFrame:
    """Download (if needed), verify and build the statements table with both splits."""
    raw = download_datasets(cfg.data)
    df = load_statements(cfg.data, raw, seed=cfg.run.seed)
    return add_fact_splits(df, cfg.split, seed=cfg.run.seed)


def split_masks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boolean masks for the fact-grouped train, validation and test sets."""
    return tuple((df.split3 == s).to_numpy() for s in SPLITS)  # type: ignore[return-value]
