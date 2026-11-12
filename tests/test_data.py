"""Tests for the data stage: checksums, balancing and fact-grouped splits."""

import hashlib
import io

import pandas as pd
import pytest

from probity import data as data_mod
from probity.config import DataConfig, DatasetSpec, SplitConfig, load_config


@pytest.mark.parametrize(
    ("statement", "dataset", "expected"),
    [
        ("The city of Krasnodar is in Russia.", "cities", "city:Krasnodar"),
        ("The city of Krasnodar is not in Russia.", "neg_cities", "city:Krasnodar"),
        ("The Spanish word 'uno' means 'one'.", "sp_en_trans", "sp:uno"),
        ("The Spanish word 'uno' does not mean 'one'.", "neg_sp_en_trans", "sp:uno"),
        ("Fifty-two is larger than fifty-one.", "larger_than", "num:Fifty-two|fifty-one"),
        ("Fifty-one is smaller than fifty-two.", "smaller_than", "num:Fifty-one|fifty-two"),
        ("Thermo Fisher engages in the provision of chips.", "companies_true_false", "co:Thermo Fisher"),
        (
            "The mother tongue of Danielle Darrieux is French.",
            "counterfact_true_false",
            "cf:The mother tongue of Danielle Darrieux",
        ),
        ("Spiders can walk on water.", "common_claim_true_false", "common_claim_true_false:Spiders can walk on water."),
    ],
)
def test_fact_id_groups_statements_about_the_same_fact(statement, dataset, expected):
    assert data_mod.fact_id(statement, dataset) == expected


def _toy(n=400):
    rows = []
    for i in range(n):
        rows.append({"statement": f"The city of C{i} is in X.", "label": i % 2, "dataset": "cities"})
        rows.append({"statement": f"The city of C{i} is not in X.", "label": 1 - i % 2, "dataset": "neg_cities"})
    return pd.DataFrame(rows)


def test_fact_splits_keep_a_fact_and_its_negation_together():
    df = data_mod.add_fact_splits(_toy(), SplitConfig())
    assert df.groupby("fact").split3.nunique().max() == 1
    assert df.groupby("fact").fold.nunique().max() == 1
    shares = df.split3.value_counts(normalize=True)
    assert 0.5 < shares["train"] < 0.7 and 0.1 < shares["val"] < 0.3 and 0.1 < shares["test"] < 0.3


def test_fact_splits_are_deterministic():
    a = data_mod.add_fact_splits(_toy(), SplitConfig(), seed=0)
    b = data_mod.add_fact_splits(_toy(), SplitConfig(), seed=0)
    c = data_mod.add_fact_splits(_toy(), SplitConfig(), seed=1)
    assert a.equals(b) and not a.split3.equals(c.split3)


def test_leakage_check_fails_on_a_leaking_split():
    df = data_mod.add_fact_splits(_toy(), SplitConfig())
    df.loc[df.index[0], "split3"] = "test" if df.split3.iloc[0] != "test" else "train"
    with pytest.raises(AssertionError):
        data_mod.check_no_leakage(df)


def _csv(n_true, n_false):
    rows = [f"True fact {i}.,1" for i in range(n_true)] + [f"False fact {i}.,0" for i in range(n_false)]
    return ("statement,label\n" + "\n".join(rows) + "\n").encode()


def test_load_statements_balances_and_splits(tmp_path):
    (tmp_path / "a.csv").write_bytes(_csv(30, 50))
    cfg = DataConfig(max_per_dataset=40, datasets={"a": DatasetSpec("t", False, "x")})
    df = data_mod.load_statements(cfg, tmp_path)
    assert df.label.value_counts().to_dict() == {0: 20, 1: 20}
    assert (df.split == "test").sum() == 8


def test_download_verifies_checksums(tmp_path, monkeypatch):
    body = _csv(3, 3)
    good = hashlib.sha256(body).hexdigest()
    monkeypatch.setattr(data_mod.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(body))
    cfg = DataConfig(cache_dir=str(tmp_path), datasets={"a": DatasetSpec("t", False, good)})
    out = data_mod.download_datasets(cfg)
    assert data_mod.sha256(out / "a.csv") == good

    bad = DataConfig(cache_dir=str(tmp_path / "b"), datasets={"a": DatasetSpec("t", False, "0" * 64)})
    with pytest.raises(ValueError, match="Checksum mismatch"):
        data_mod.download_datasets(bad)


@pytest.mark.network
def test_data_stage_reproduces_the_published_statements(tmp_path, repo_root):
    cfg = load_config(overrides=[f"data.cache_dir={tmp_path}"])
    df = data_mod.prepare_statements(cfg)
    published = pd.read_csv(repo_root / "results" / "statements.csv")
    cols = list(published.columns)
    assert df[cols].astype(str).equals(published.astype(str))
    out = tmp_path / "statements.csv"
    df[cols].to_csv(out, index=False)
    assert data_mod.sha256(out) == cfg.data.statements_sha256
