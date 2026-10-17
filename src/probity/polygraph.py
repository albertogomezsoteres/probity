"""Polygraph and Polygraph Guard.

`Polygraph.check` reads every factual sentence of a reply on its own and returns the weakest one.
`Polygraph.guard` is the guardrail evaluated in notebook 05: if the reply is not believed, it looks for an
alternative that the model does believe (4 sampled replies plus its own earlier answers), or it abstains.
The guard never sees the correct answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .claims import as_statement, claim_of, facts_of, is_checkable, split_claims
from .probe import UNSURE, Probe, verdict_word

HOLD = "I understand the doubt, but I have double-checked. {claim}"
ABSTAIN = "I'm not confident enough to answer that reliably. Please check a trusted source."
DEFAULT_TAU = 0.50  # chosen on development subjects in notebook 05


@dataclass
class Polygraph:
    """Read, judge and guard the claims of an open-weight chat model.

    >>> poly = Polygraph.from_pretrained("google/gemma-2-2b-it", probe="results/app_probes.npz", layer=14)
    >>> poly.check("Anyang-si is located in Iraq.")["verdict"]
    'believed false'
    """
    model: object
    tokenizer: object
    probe: Probe
    device: str = "cuda"
    tau: float = DEFAULT_TAU
    n_samples: int = 4
    _prefix: str = field(init=False, default="")

    def __post_init__(self):
        # Claims are read in the same format as the training statements: "user: Tell me one fact. / model: <claim>"
        self._prefix = self.tokenizer.apply_chat_template([{"role": "user", "content": "Tell me one fact."}],
                                                          tokenize=False, add_generation_prompt=True)

    @classmethod
    def from_pretrained(cls, model_id: str, probe: str, layer: int = 14, device: str = "cuda", **kw) -> "Polygraph":
        """Load the model from Hugging Face and the probe from an .npz file."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device,
                                                     attn_implementation="eager").eval()
        return cls(model, tok, Probe.from_npz(probe, layer), device=device, **kw)

    # ---------- reading ----------
    def _hidden(self, claims: list[str], batch_size: int = 32) -> np.ndarray:
        """Hidden state at the last token of each claim, at the probe layer."""
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(claims), batch_size):
                texts = [self._prefix + as_statement(c) for c in claims[i:i + batch_size]]
                enc = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
                hs = self.model(**enc, output_hidden_states=True).hidden_states[self.probe.layer]
                out.append(hs[:, -1].float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.probe.w)), np.float32)

    def check(self, text: str) -> dict:
        """Score every factual sentence of `text` on its own. The verdict comes from the weakest claim."""
        sents = split_claims(text)
        claims = [claim_of(s) for s in sents]
        idx = [i for i, c in enumerate(claims) if is_checkable(c)]
        X = self._hidden([claims[i] for i in idx])
        ps, fam = self.probe.p_true(X), self.probe.familiar(X)
        rows = []
        for k, i in enumerate(idx):
            flags = ([] if fam[k] else ["unfamiliar"]) + (["unsure"] if UNSURE[0] < ps[k] < UNSURE[1] else [])
            rows.append({"sentence": sents[i], "claim": claims[i], "p_true": float(ps[k]), "flags": flags})
        p = min((r["p_true"] for r in rows), default=None)
        worst = min(rows, key=lambda r: r["p_true"]) if rows else None
        return {"p_true": p, "verdict": verdict_word(p), "trusted": bool(worst and "unfamiliar" not in worst["flags"]),
                "weakest_claim": worst["claim"] if worst else None, "claims": rows}

    # ---------- generation and guarding ----------
    def _generate(self, messages: list[dict], n: int = 1, sample: bool = False, max_new_tokens: int = 160) -> list[str]:
        import torch
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        kw = dict(do_sample=True, temperature=0.8, top_p=0.95, num_return_sequences=n) if sample else dict(do_sample=False)
        with torch.no_grad():
            g = self.model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id, **kw)
        return [t.strip() for t in self.tokenizer.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)]

    def guard(self, messages: list[dict], reply: str, tau: float | None = None) -> tuple[str, dict]:
        """Pass `reply`, replace it with the alternative the model believes most, or abstain."""
        tau = self.tau if tau is None else tau
        p = self.check(reply)["p_true"]
        report = {"p_reply": p, "action": "passed", "candidates": []}
        if p is None or p >= tau:
            return reply, report
        pool = [(s, self.check(s)["p_true"], "resampled answer") for s in self._generate(messages, self.n_samples, sample=True)]
        pool += [(HOLD.format(claim=facts_of(m["content"])), self.check(m["content"])["p_true"], "earlier answer")
                 for m in messages if m["role"] == "assistant"]
        pool = [c for c in pool if c[1] is not None]
        report["candidates"] = sorted(pool, key=lambda c: -c[1])
        if pool and max(c[1] for c in pool) >= tau:
            best = max(pool, key=lambda c: c[1])
            report.update(action="replaced", source=best[2], p_delivered=best[1])
            return best[0], report
        report["action"] = "abstained"
        return ABSTAIN, report

    def guarded_reply(self, messages: list[dict], tau: float | None = None) -> tuple[str, dict]:
        """Generate a reply to `messages` and run it through the guard."""
        return self.guard(messages, self._generate(messages)[0], tau)
