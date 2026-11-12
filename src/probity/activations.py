"""Load the language model and read its hidden states.

A statement is read as the model's own answer: the chat template of ``user: <fact_prompt>`` followed by the
statement, and the residual stream is taken at the **last token** (the final period) of every requested layer.
Layer 0 is the embedding output and layer ``n`` the output of block ``n``.
"""

from __future__ import annotations

import hashlib
import pathlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from .config import ModelConfig
from .tracking import get_logger

log = get_logger("activations")


def load_model(cfg: ModelConfig) -> tuple[Any, Any]:
    """Load the tokenizer and the model in eval mode, with gradients disabled.

    The tokenizer pads on the left, so position -1 is always the last real token.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.id)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.id,
        dtype=getattr(torch, cfg.dtype),
        device_map=cfg.device,
        attn_implementation=cfg.attn_implementation,
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log.info("loaded %s (%d layers) on %s", cfg.id, model.config.num_hidden_layers, cfg.device)
    return model, tok


def chat_prefix(tokenizer: Any, fact_prompt: str) -> str:
    """The text that precedes every statement, so the statement reads as the model's own reply."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": fact_prompt}], tokenize=False, add_generation_prompt=True
    )


class HiddenStateReader:
    """Batched reader of last-token hidden states."""

    def __init__(self, model: Any, tokenizer: Any, cfg: ModelConfig):
        self.model, self.tok, self.cfg = model, tokenizer, cfg
        self.prefix = chat_prefix(tokenizer, cfg.fact_prompt)
        self.n_layers: int = model.config.num_hidden_layers
        self.d_model: int = model.config.hidden_size

    def texts(self, statements: Sequence[str]) -> list[str]:
        """Statements in the assistant format used for training and reading."""
        return [self.prefix + s for s in statements]

    def read(self, texts: Sequence[str], layers: Sequence[int], progress: bool = True) -> dict[int, np.ndarray]:
        """``{layer: array [n_texts, d_model]}`` of float32 hidden states at the last token."""
        import torch
        from tqdm.auto import tqdm

        out: dict[int, list[np.ndarray]] = {L: [] for L in layers}
        bs = self.cfg.batch_size
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), bs), desc="hidden states", disable=not progress):
                enc = self.tok(list(texts[i : i + bs]), return_tensors="pt", padding=True, add_special_tokens=False).to(
                    self.model.device
                )
                hs = self.model(**enc, output_hidden_states=True).hidden_states
                for L in layers:
                    out[L].append(hs[L][:, -1].float().cpu().numpy())
        return {L: np.concatenate(v) for L, v in out.items()}

    def ask_scores(self, statements: Sequence[str]) -> np.ndarray:
        """Baseline: ``logit(True) - logit(False)`` when the model is asked whether a statement is true."""
        import torch

        t_id = self.tok.encode("True", add_special_tokens=False)[0]
        f_id = self.tok.encode("False", add_special_tokens=False)[0]
        prompts = [
            self.tok.apply_chat_template(
                [{"role": "user", "content": ASK_TEMPLATE + s}], tokenize=False, add_generation_prompt=True
            )
            for s in statements
        ]
        out = []
        bs = self.cfg.batch_size
        with torch.no_grad():
            for i in range(0, len(prompts), bs):
                enc = self.tok(prompts[i : i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(
                    self.model.device
                )
                lg = self.model(**enc).logits[:, -1].float()
                out.append((lg[:, t_id] - lg[:, f_id]).cpu().numpy())
        return np.concatenate(out)


ASK_TEMPLATE = "Is the following statement true or false? Answer with exactly one word: True or False.\n\nStatement: "


def cache_key(texts: Sequence[str], model_id: str) -> str:
    """Short hash identifying a set of texts read by one model."""
    return hashlib.md5(("\n".join(texts) + model_id).encode()).hexdigest()[:10]


def save_activations(path: str | pathlib.Path, acts: dict[int, np.ndarray]) -> None:
    """Store activations as one stacked array ``[layers, n, d]`` plus the layer list."""
    layers = sorted(acts)
    np.savez(path, layers=np.array(layers), acts=np.stack([acts[L] for L in layers]))


def load_activations(path: str | pathlib.Path) -> dict[int, np.ndarray]:
    """Inverse of :func:`save_activations`."""
    with np.load(path) as z:
        acts, layers = z["acts"], z["layers"]  # read the stacked array once
    return {int(L): acts[i] for i, L in enumerate(layers)}
