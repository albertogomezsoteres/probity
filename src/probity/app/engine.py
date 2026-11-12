"""The model side of the app: reading claims, Polygraph Guard, explanations and the log monitor.

All model access goes through four primitives (:meth:`Engine.read_last`, :meth:`Engine.read_tokens`,
:meth:`Engine.chat` and :meth:`Engine.chat_samples`), plus the gradients used by Integrated Gradients.
The tests replace the primitives with a fake model, so the rest of the logic runs without a GPU.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..claims import NONCLAIM_RE, as_statement, claim_of, facts_of, split_claims
from ..config import Config
from ..polygraph import ABSTAIN, HOLD
from ..probe import UNSURE
from .info import AppInfo

NEG_RE = re.compile(r"\b(not|no|never|none|neither|nor|cannot)\b|n't", re.I)
WORD_RE = re.compile(r"\S+")
METHOD_NAMES = ("LIME", "Integrated Gradients", "KernelSHAP")


def words_of(sentence: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group()) for m in WORD_RE.finditer(sentence)]


def render(sentence: str, keep: Sequence[bool] | np.ndarray) -> str:
    return " ".join(w for (_, _, w), k in zip(words_of(sentence), keep) if k)


def on_zero_gpu() -> bool:
    """True inside a Hugging Face Space running on ZeroGPU."""
    return bool(os.environ.get("SPACES_ZERO_GPU"))


class Engine:
    """Gemma plus the calibrated probes, with every operation the app needs."""

    def __init__(self, model: Any, tokenizer: Any, info: AppInfo, cfg: Config):
        self.model, self.tok, self.info, self.cfg = model, tokenizer, info, cfg
        self.prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": cfg.model.fact_prompt}], tokenize=False, add_generation_prompt=True
        )

    # ---------- loading ----------
    @classmethod
    def load(cls, cfg: Config, info: AppInfo) -> Engine:
        """Load Gemma for the app (on ZeroGPU the model is moved to the GPU that each call borrows)."""
        if on_zero_gpu():
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(cfg.model.id)
            tok.padding_side = "left"
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model.id, dtype=getattr(torch, cfg.model.dtype), attn_implementation=cfg.model.attn_implementation
            ).to("cuda")
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
        else:
            from ..activations import load_model

            model, tok = load_model(cfg.model)
        return cls(model, tok, info, cfg)

    @property
    def device(self) -> Any:
        return self.model.device

    # ---------- model primitives ----------
    def read_last(self, texts: Sequence[str], layers: Sequence[int]) -> dict[int, np.ndarray]:
        """Residual stream at the last token, for each requested layer -> ``{layer: [n, d] float32}``."""
        import torch

        res: dict[int, list[np.ndarray]] = {L: [] for L in layers}
        bs = self.cfg.model.batch_size
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                enc = self.tok(list(texts[i : i + bs]), return_tensors="pt", padding=True, add_special_tokens=False).to(
                    self.device
                )
                hs = self.model(**enc, output_hidden_states=True).hidden_states
                for L in layers:
                    res[L].append(hs[L][:, -1].float().cpu().numpy())
        return {L: np.concatenate(v) for L, v in res.items()}

    def read_tokens(self, text: str, start_char: int, layer: int) -> tuple[list[str], np.ndarray]:
        """Residual stream at every token that starts at or after ``start_char`` -> (token strings, [t, d])."""
        import torch

        with torch.no_grad():
            enc = self.tok(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
            offs = enc.pop("offset_mapping")[0].tolist()
            hs = (
                self.model(**enc.to(self.device), output_hidden_states=True)
                .hidden_states[layer][0]
                .float()
                .cpu()
                .numpy()
            )
        keep = [i for i, (s, e) in enumerate(offs) if s >= start_char and e > s]
        return [text[offs[i][0] : offs[i][1]] for i in keep], hs[keep]

    def chat(self, conv: list[dict[str, str]]) -> str:
        """Greedy reply from the model."""
        import torch

        prompt = self.tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.cfg.guard.max_new_tokens, do_sample=False)
        return self.tok.decode(out[0, enc.input_ids.shape[1] :], skip_special_tokens=True).strip()

    def chat_samples(self, conv: list[dict[str, str]], n: int) -> list[str]:
        """``n`` sampled alternative replies for Polygraph Guard (same settings as notebook 05)."""
        import torch

        g = self.cfg.guard
        prompt = self.tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=g.max_new_tokens,
                do_sample=True,
                temperature=g.temperature,
                top_p=g.top_p,
                num_return_sequences=n,
                pad_token_id=self.tok.pad_token_id,
            )
        return [t.strip() for t in self.tok.batch_decode(out[:, enc.input_ids.shape[1] :], skip_special_tokens=True)]

    # ---------- the polygraph ----------
    def read_claims(self, sentences: list[str], context_prompt: str | None = None) -> list[dict[str, Any]]:
        """Read each sentence alone at the app layer (the needle), at every layer (brain scan) and,
        for comparison only, inside the conversation."""
        if not sentences:
            return []
        info, L_APP = self.info, self.info.layer
        claims = [claim_of(s) for s in sentences]
        X = self.read_last([self.prefix + as_statement(c) for c in claims], info.all_layers)
        Xc = (
            self.read_last([context_prompt + as_statement(c) for c in claims], [L_APP])[L_APP]
            if context_prompt
            else None
        )
        rows = []
        for i, (full, s) in enumerate(zip(sentences, claims)):
            xa = X[L_APP][i]
            dist = float((((xa - info.MU) @ info.COMP.T) ** 2 / info.VAR).sum())
            p = float(info.calibrated(xa @ info.W[L_APP] + info.B[L_APP]))
            flags = []
            if NONCLAIM_RE.search(s.strip()) or len(s.split()) < 3:
                flags.append("not_claim")
            if dist > info.ood_threshold:
                flags.append("unfamiliar")
            if UNSURE[0] < p < UNSURE[1]:
                flags.append("unsure")
            if NEG_RE.search(s):
                flags.append("negation")
            if len(s.split()) > 25:
                flags.append("long")
            rows.append(
                {
                    "sentence": full,
                    "claim": s,
                    "p": p,
                    "flags": flags,
                    "familiarity_pct": float(100 * np.searchsorted(info.TEST_DIST, dist) / len(info.TEST_DIST)),
                    "layer_p": [float(info.calibrated(X[L][i] @ info.W[L] + info.B[L], L)) for L in info.all_layers],
                    "p_in_context": None
                    if Xc is None
                    else float(info.calibrated(Xc[i] @ info.W[L_APP] + info.B[L_APP])),
                }
            )
        return rows

    @staticmethod
    def verdict(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        """A reply is as trustworthy as its weakest factual claim."""
        cl = [r for r in rows if "not_claim" not in r["flags"]]
        if not cl:
            return None
        worst = min(cl, key=lambda r: r["p"])
        return {
            "p": worst["p"],
            "sentence": worst["claim"],
            "n_claims": len(cl),
            "trusted": "unfamiliar" not in worst["flags"],
            "flags": worst["flags"],
        }

    def check_text(self, text: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        rows = self.read_claims(split_claims(text or ""))
        return rows, self.verdict(rows)

    # ---------- Polygraph Guard (notebook 05 policy, identical rules) ----------
    def weakest_p(self, replies: list[str]) -> list[float | None]:
        """Weakest factual claim per reply, read alone at the app layer (None when a reply has no claim)."""
        info, L = self.info, self.info.layer
        per = [
            [
                c
                for c in (claim_of(s) for s in split_claims(r or ""))
                if not (NONCLAIM_RE.search(c) or len(c.split()) < 3)
            ]
            for r in replies
        ]
        flat = [c for cs in per for c in cs]
        ps = iter(
            info.calibrated(
                self.read_last([self.prefix + as_statement(c) for c in flat], [L])[L] @ info.W[L] + info.B[L]
            )
            if flat
            else []
        )
        vals = [[float(next(ps)) for _ in cs] for cs in per]
        return [min(v) if v else None for v in vals]

    def guard_reply(self, conv: list[dict[str, str]], reply: str, p_reply: float | None) -> tuple[str, dict[str, Any]]:
        """-> (delivered text, info). Only the model's own replies and hidden states are used; never a gold answer."""
        tau = self.info.tau
        info: dict[str, Any] = {"p_reply": p_reply, "action": "passed", "candidates": []}
        if p_reply is None or p_reply >= tau:
            return reply, info
        samples = self.chat_samples(conv, self.info.n_guard)
        prior = [t["content"] for t in conv if t["role"] == "assistant"]
        ps = self.weakest_p(samples + prior)
        pool = [(s, p, "resampled answer") for s, p in zip(samples, ps[: len(samples)]) if p is not None]
        pool += [
            (HOLD.format(claim=facts_of(pr)), p, "Gemma's earlier answer")
            for pr, p in zip(prior, ps[len(samples) :])
            if p is not None
        ]
        info["candidates"] = sorted(pool, key=lambda x: -x[1])
        if pool:
            best = max(pool, key=lambda x: x[1])
            if best[1] >= tau:
                info.update(action="replaced", source=best[2], p_delivered=best[1])
                return best[0], info
        info.update(action="abstained")
        return ABSTAIN, info

    # ---------- "Why this verdict": attributions with a live faithfulness check ----------
    def logit_alone(self, sentences: list[str], layer: int | None = None) -> np.ndarray:
        L = layer or self.info.layer
        return self.read_last([self.prefix + s for s in sentences], [L])[L] @ self.info.W[L] + self.info.B[L]

    def lime_attr(self, sentence: str, kernel_width: float = 0.25) -> np.ndarray:
        from sklearn.linear_model import Ridge

        rng = np.random.default_rng(self.cfg.run.seed)
        M = len(words_of(sentence))
        Z = rng.integers(0, 2, size=(self.cfg.app.lime_samples, M)).astype(bool)
        Z[0] = True
        ys = self.logit_alone([render(sentence, z) for z in Z])
        wts = np.exp(-((1 - Z.mean(1)) ** 2) / kernel_width**2)
        return Ridge(alpha=1.0).fit(Z.astype(float), ys, sample_weight=wts).coef_

    def shap_attr(self, sentence: str) -> np.ndarray:
        import shap

        M = len(words_of(sentence))

        def f(Z: np.ndarray) -> np.ndarray:
            return self.logit_alone([render(sentence, z.astype(bool)) for z in Z])

        return shap.KernelExplainer(f, np.zeros((1, M))).shap_values(
            np.ones((1, M)), nsamples=self.cfg.app.lime_samples, silent=True
        )[0]

    def ig_attr(self, sentence: str) -> np.ndarray:
        """Integrated Gradients on input embeddings; the baseline zeroes the sentence tokens only."""
        import torch

        steps, L = self.cfg.app.ig_steps, self.info.layer
        text = self.prefix + sentence
        start = len(self.prefix)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
        ids = enc.input_ids.to(self.device)
        offs = enc.offset_mapping[0].tolist()
        mask = torch.tensor([s >= start and e > s for s, e in offs], device=self.device)
        emb = self.model.get_input_embeddings()(ids).detach().float()
        base = emb.clone()
        base[0, mask] = 0
        w = torch.tensor(self.info.W[L], device=self.device, dtype=torch.float32)
        grads = torch.zeros_like(emb)
        for chunk in torch.linspace(1 / steps, 1, steps, device=self.device).split(8):
            x = (base + chunk[:, None, None] * (emb - base)).requires_grad_(True)
            hs = (
                self.model(inputs_embeds=x.to(self.model.dtype), output_hidden_states=True)
                .hidden_states[L][:, -1]
                .float()
            )
            (hs @ w).sum().backward()
            grads += x.grad.sum(0, keepdim=True)
        tok_attr = ((emb - base) * grads / steps).sum(-1)[0].detach().cpu().numpy()
        out = []
        for a, b, _ in words_of(sentence):
            A, Bc = start + a, start + b
            out.append(sum(v for (s, e), v in zip(offs, tok_attr) if e > s and s < Bc and e > A))
        return np.array(out)

    def attribution(self, method: str, sentence: str) -> np.ndarray:
        return {"LIME": self.lime_attr, "Integrated Gradients": self.ig_attr, "KernelSHAP": self.shap_attr}[method](
            sentence
        )

    def deletion_check(self, sentence: str, attr: np.ndarray, k: int = 2, n_random: int = 8) -> dict[str, Any]:
        """Delete the k words pushing hardest toward the verdict; compare with deleting k random words."""
        M = len(words_of(sentence))
        k = max(1, min(k, M - 1))
        s0 = float(self.logit_alone([sentence])[0])
        sign = np.sign(s0) or 1.0
        top = np.argsort(-(sign * np.asarray(attr)))[:k]
        rng = np.random.default_rng(self.cfg.run.seed)
        masks = []
        for idx in [top] + [rng.choice(M, k, replace=False) for _ in range(n_random)]:
            keep = np.ones(M, bool)
            keep[idx] = False
            masks.append(render(sentence, keep))
        s = self.logit_alone(masks)
        moved = sign * (s0 - s)  # > 0: the reading moved against the verdict
        return {
            "k": int(k),
            "top_words": [words_of(sentence)[i][2] for i in top],
            "moved_top": float(moved[0]),
            "moved_random": float(moved[1:].mean()),
            "faithful": bool(moved[0] > moved[1:].mean()),
        }

    def explain(self, sentence: str, method: str) -> dict[str, Any]:
        sentence = as_statement(sentence)
        names = list(METHOD_NAMES) if method == "All three (compare)" else [method]
        out: dict[str, Any] = {
            "sentence": sentence,
            "words": [w for *_, w in words_of(sentence)],
            "score": float(self.logit_alone([sentence])[0]),
            "methods": {},
        }
        for m in names:
            a = np.asarray(self.attribution(m, sentence), float)
            out["methods"][m] = {"attr": a, "check": self.deletion_check(sentence, a)}
        return out

    # ---------- brain scan ----------
    def token_scan(self, sentence: str) -> tuple[list[str], np.ndarray]:
        """Exploratory: the probe was trained on the final token only."""
        sentence = as_statement(sentence)
        L = self.info.layer
        toks, H = self.read_tokens(self.prefix + sentence, len(self.prefix), L)
        return toks, self.info.calibrated(H @ self.info.W[L] + self.info.B[L])

    # ---------- monitor ----------
    def monitor_rows(self, df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """Score every answer of a log by its weakest claim."""
        df = df.copy()
        col = next((c for c in ["assistant", "response", "answer", "reply", "output"] if c in df.columns), None)
        if col is None:
            raise ValueError("The file needs a column named assistant (or response, answer, reply, output).")
        df = df[df[col].astype(str).str.strip().ne("")].head(self.cfg.app.monitor_max_rows).reset_index(drop=True)
        if "conversation_id" not in df.columns:
            df["conversation_id"] = [f"c{i + 1:04d}" for i in range(len(df))]
        sents = [split_claims(str(t)) for t in df[col]]
        flat = [s for ss in sents for s in ss]
        t0 = time.time()
        rows = self.read_claims(flat) if flat else []
        secs = time.time() - t0
        out, k = [], 0
        for i, ss in enumerate(sents):
            rr = rows[k : k + len(ss)]
            k += len(ss)
            v = self.verdict(rr)
            out.append(
                {
                    "conversation_id": df.conversation_id[i],
                    "user": str(df["user"][i]) if "user" in df.columns else "",
                    "assistant": str(df[col][i]),
                    "weakest_claim": v["sentence"] if v else "",
                    "p_true": round(v["p"], 4) if v else None,
                    "status": (
                        "no factual claim"
                        if v is None
                        else "believed false"
                        if v["p"] <= UNSURE[0]
                        else "undecided"
                        if v["p"] < UNSURE[1]
                        else "believed true"
                    ),
                    "unfamiliar": bool(v and not v["trusted"]),
                    "flags": ", ".join(sorted({f for r in rr for f in r["flags"]})),
                    "label": df["label"][i] if "label" in df.columns else None,
                }
            )
        return pd.DataFrame(out), secs

    # ---------- HTTP API ----------
    def api_check(self, text: str) -> dict[str, Any]:
        rows, v = self.check_text(text)
        return {
            "p_true": None if v is None else round(v["p"], 4),
            "verdict": "no factual claim"
            if v is None
            else "believed false"
            if v["p"] <= UNSURE[0]
            else "undecided"
            if v["p"] < UNSURE[1]
            else "believed true",
            "trusted": None if v is None else v["trusted"],
            "claims": [{"sentence": r["claim"], "p_true": round(r["p"], 4), "flags": r["flags"]} for r in rows],
        }
