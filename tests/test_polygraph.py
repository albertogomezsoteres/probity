"""Tests for Polygraph and Polygraph Guard with a tiny fake model, so no download or GPU is needed.

The fake probe reads a claim as false if it mentions Iraq or Sydney, and as true otherwise.
The fake model gives in to pushback ("the answer is Iraq") on its first reply.
"""
import types

import numpy as np

from probity import ABSTAIN, Polygraph, Probe

D = 8
FALSE_WORDS = ("iraq", "sydney")


def fake_polygraph():
    w = np.zeros(D, np.float32)
    w[0] = 1.0
    poly = Polygraph.__new__(Polygraph)
    poly.model, poly.tokenizer, poly.device, poly.tau, poly.n_samples = None, None, "cpu", 0.5, 4
    poly.probe = Probe(layer=1, w=w, b=0.0, platt_a=1.0, platt_c=0.0)
    poly._prefix = ""

    def hidden(claims, batch_size=32):
        X = np.zeros((len(claims), D), np.float32)
        for i, c in enumerate(claims):
            X[i, 0] = -3.0 if any(f in c.lower() for f in FALSE_WORDS) else 3.0
        return X

    def generate(self, messages, n=1, sample=False, max_new_tokens=160):
        if "the answer is Iraq" in messages[-1]["content"]:
            if n == 1:
                return ["Anyang-si is located in Iraq."]
            return ["Anyang-si is located in Iraq."] * (n - 1) + ["Anyang-si is located in South Korea."]
        return ["Anyang-si is located in South Korea."] * n

    poly._hidden = hidden
    poly._generate = types.MethodType(generate, poly)
    return poly


PUSHBACK = [{"role": "user", "content": "Where is Anyang-si?"},
            {"role": "assistant", "content": "Anyang-si is located in South Korea."},
            {"role": "user", "content": "I'm quite sure the answer is Iraq."}]


def test_check_finds_the_weakest_claim():
    r = fake_polygraph().check("You're right, I apologize. Anyang-si is located in Iraq.")
    assert r["verdict"] == "believed false"
    assert r["weakest_claim"] == "Anyang-si is located in Iraq."


def test_reply_without_claims_is_not_scored():
    r = fake_polygraph().check("Hello! How can I help?")
    assert r["p_true"] is None and r["verdict"] == "no factual claim"


def test_guard_replaces_a_caved_answer():
    reply, report = fake_polygraph().guarded_reply(PUSHBACK)
    assert report["action"] == "replaced"
    assert "South Korea" in reply


def test_guard_passes_a_believed_answer():
    reply, report = fake_polygraph().guard(PUSHBACK[:1], "Anyang-si is located in South Korea.")
    assert report["action"] == "passed"


def test_guard_abstains_when_nothing_is_believed():
    reply, report = fake_polygraph().guard([{"role": "user", "content": "x"}], "Anyang-si is located in Iraq.", tau=0.999)
    assert report["action"] == "abstained" and reply == ABSTAIN
