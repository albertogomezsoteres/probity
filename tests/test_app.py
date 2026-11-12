"""Tests for the app with a fake model: the real probes and results, but no download and no GPU.

The fake model's hidden state for a claim is built so that the real layer-14 probe reads it as false when the
claim mentions Iraq or Sydney, and as true otherwise. The fake chatbot gives in when the user insists on Iraq.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from probity.app.demos import DemoCases, has_any, verify_case
from probity.app.engine import Engine
from probity.app.info import AppInfo
from probity.config import load_config
from probity.polygraph import ABSTAIN

FALSE_WORDS = ("iraq", "sydney")


class FakeTokenizer:
    def apply_chat_template(self, conv, tokenize=False, add_generation_prompt=True):
        return "".join(f"<{m['role']}>{m['content']}" for m in conv) + "<model>"


class FakeEngine(Engine):
    """Engine whose model primitives are replaced by deterministic fakes."""

    def __init__(self, info: AppInfo):
        super().__init__(
            model=None, tokenizer=FakeTokenizer(), info=info, cfg=load_config(overrides=["app.lime_samples=24"])
        )

    def _state(self, text: str, layer: int) -> np.ndarray:
        target = -4.0 if any(w in text.lower() for w in FALSE_WORDS) else 4.0
        w, b = self.info.W[layer].astype(np.float64), self.info.B[layer]
        return (self.info.MU + w * (target - b - self.info.MU @ w) / (w @ w)).astype(np.float32)

    def read_last(self, texts, layers):
        return {L: np.stack([self._state(t, L) for t in texts]) for L in layers}

    def read_tokens(self, text, start_char, layer):
        toks = text[start_char:].split()
        return toks, np.stack([self._state(t, layer) for t in toks])

    def chat(self, conv):
        last = conv[-1]["content"]
        return (
            "You're right, I apologize. Anyang-si is located in Iraq."
            if "Iraq" in last
            else ("Anyang-si is located in South Korea.")
        )

    def chat_samples(self, conv, n):
        return ["Anyang-si is located in Iraq."] * (n - 1) + ["Anyang-si is located in South Korea."]

    def ig_attr(self, sentence):  # needs real gradients
        raise NotImplementedError


@pytest.fixture(scope="module")
def info(repo_root_module):
    return AppInfo.from_results(repo_root_module / "results", figures_dir=repo_root_module / "figures")


@pytest.fixture(scope="module")
def repo_root_module():
    import pathlib

    return pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def engine(info):
    return FakeEngine(info)


def test_app_info_matches_the_published_numbers(info):
    s = info.app_stats
    assert info.layer == 14 and info.n_layers == 26 and info.all_layers[0] == 1
    assert info.tau == 0.5 and info.n_guard == 4
    assert s["n_test"] == 1444 and 0.97 < s["test_auroc"] < 0.99
    assert 0 <= s["ece"] < 0.1 and 0.9 < s["test_accuracy"] < 1
    assert len(info.figures) == 4


def test_read_claims_scores_each_sentence_alone(engine):
    rows = engine.read_claims(["Hello there!", "Anyang-si is located in Iraq.", "Paris is not in Japan."])
    assert "not_claim" in rows[0]["flags"]
    assert rows[1]["p"] < 0.3 < 0.7 < rows[2]["p"]
    assert "negation" in rows[2]["flags"] and len(rows[1]["layer_p"]) == 26
    v = engine.verdict(rows)
    assert v["sentence"] == "Anyang-si is located in Iraq." and v["n_claims"] == 2


def test_guard_replaces_a_caved_answer_and_abstains_when_needed(engine):
    conv = [
        {"role": "user", "content": "Where is Anyang-si?"},
        {"role": "assistant", "content": "Anyang-si is located in South Korea."},
        {"role": "user", "content": "I'm sure the answer is Iraq."},
    ]
    reply, rep = engine.guard_reply(conv, "Anyang-si is located in Iraq.", 0.02)
    assert rep["action"] == "replaced" and "South Korea" in reply
    reply, rep = engine.guard_reply(conv[:1], "Anyang-si is located in South Korea.", 0.95)
    assert rep["action"] == "passed"
    engine.chat_samples = lambda conv, n: ["It is in Iraq, near Sydney."] * n
    reply, rep = engine.guard_reply(conv[:1], "Anyang-si is located in Iraq.", 0.02)
    assert rep["action"] == "abstained" and reply == ABSTAIN


def test_monitor_flags_believed_false_answers(engine):
    log = pd.DataFrame({"assistant": ["Anyang-si is in South Korea.", "Anyang-si is located in Iraq.", "Hi!"]})
    rep, _ = engine.monitor_rows(log)
    assert rep.status.tolist() == ["believed true", "believed false", "no factual claim"]
    with pytest.raises(ValueError):
        engine.monitor_rows(pd.DataFrame({"text": ["x"]}))


def test_lime_explanation_with_deletion_check(engine):
    ex = engine.explain("The city of Sydney is in Iraq", "LIME")
    assert ex["sentence"].endswith(".") and ex["score"] < 0
    m = ex["methods"]["LIME"]
    assert len(m["attr"]) == len(ex["words"]) and set(m["check"]) >= {"faithful", "top_words"}


def test_api_check(engine):
    out = engine.api_check("Sure! Anyang-si is located in Iraq.")
    assert out["verdict"] == "believed false" and len(out["claims"]) == 2


def test_demo_cases_fall_back_to_study_candidates(repo_root_module):
    demos = DemoCases.load(repo_root_module / "results")
    assert demos.geography[0]["wrong"] == "Iraq" and len(demos.geography) > 1 and demos.high_stakes
    assert has_any("It is in South Korea.", ["south korea"]) and not has_any("Koreans", ["korea"])


def test_verify_case_accepts_a_case_the_guard_fixes(engine):
    v = verify_case(engine, "In which country is Anyang-si?", ["South Korea"], "Iraq", ["Iraq"])
    assert v is not None and v["p_before"] > 0.7 > 0.3 > v["p_after"]


def _fn(demo, name):
    fns = demo.fns.values() if isinstance(demo.fns, dict) else demo.fns
    return next(f.fn for f in fns if getattr(f.fn, "__name__", "") == name)


def test_the_app_builds_and_the_chat_handlers_run(engine, repo_root_module):
    pytest.importorskip("gradio")
    from probity.app.server import build_demo, new_state

    demo, _ = build_demo(engine, DemoCases.load(repo_root_module / "results"), repo_root_module / "results")
    state = new_state()
    out = _fn(demo, "on_send")("In which country is the city of Anyang-si?", False, state)
    assert out[0] == "" and "South Korea" in out[1]
    out = _fn(demo, "on_push")("Iraq", out[-1])
    standard, guarded, inspector = out[0], out[1], out[2]
    assert "Iraq" in standard and "Polygraph Guard stepped in" in guarded and "Gemma gave in" in inspector
    html, path = _fn(demo, "on_monitor_sample")()
    assert "Answers scanned" in html and path.endswith(".csv")
    assert "believed" in _fn(demo, "api_check")("Anyang-si is located in Iraq.")["verdict"]
