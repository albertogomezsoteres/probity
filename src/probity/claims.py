"""Turn a chat reply into short factual claims that the probe can read.

The probe was trained on single statements, so a reply is split into sentences and each sentence is cleaned:
markdown is removed and polite openers such as "I'm sorry, but" are dropped. Sentences that are not factual
claims (greetings, questions, very short replies) are skipped.
"""
from __future__ import annotations

import re

# Sentences that start like this are conversation, not facts.
NONCLAIM_RE = re.compile(
    r"^(i\b|i'm|i’m|sure\b|okay|ok\b|great|hello|hi\b|hey\b|here\b|let me|let's|as an ai|sorry|you're|you are|"
    r"thank|please|feel free|absolutely|of course|certainly|good question)|\?$", re.I)

# Polite openers that are removed before reading the claim ("I'm sorry, but X" becomes "X").
LEAD_RE = re.compile(
    r"^(?:(?:i'm sorry|i am sorry|sorry|you're right|you are right|you're correct|you are correct|i apologize|"
    r"my apologies|apologies|actually|yes|no|well|indeed|of course|certainly|okay|ok|sure|you're absolutely right|"
    r"great question|good question|thank you|thanks)\b[^,.!]*[,.!]\s*(?:but\s+|however,?\s+)?)+", re.I)


def split_claims(text: str) -> list[str]:
    """Split a reply into sentences, removing markdown and list markers."""
    t = re.sub(r"[*_#`>]+", "", text or "")
    out: list[str] = []
    for line in t.splitlines():
        line = re.sub(r"^\s*([-•]|\d+[.)])\s+", "", line).strip()
        if line:
            out += [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", line) if s.strip()]
    return out


def claim_of(sentence: str) -> str:
    """Drop leading apologies or agreement ("I'm sorry, but X" -> "X")."""
    c = LEAD_RE.sub("", sentence.strip()).strip()
    return c[:1].upper() + c[1:] if len(c.split()) >= 3 else sentence.strip()


def is_checkable(claim: str) -> bool:
    """True if the sentence looks like a factual claim (not a greeting, question or a two-word reply)."""
    return not (NONCLAIM_RE.search(claim.strip()) or len(claim.split()) < 3)


def as_statement(s: str) -> str:
    """Make sure the claim ends with punctuation, like the training statements."""
    s = s.strip()
    return s if s[-1:] in ".!?" else s + "."


def facts_of(reply: str) -> str:
    """Keep only the factual sentences of a reply (used when the guard repeats an earlier answer)."""
    return " ".join(as_statement(c) for c in (claim_of(s) for s in split_claims(reply)) if is_checkable(c))
