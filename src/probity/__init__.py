"""Probity: an AI polygraph for open language models.

A linear probe reads the hidden states of Gemma 2 2B-it and estimates whether the model believes each claim it makes.
Polygraph Guard uses that reading to stop answers the model itself does not believe.
"""
from .claims import claim_of, split_claims
from .polygraph import ABSTAIN, DEFAULT_TAU, Polygraph
from .probe import Probe

__all__ = ["Polygraph", "Probe", "split_claims", "claim_of", "ABSTAIN", "DEFAULT_TAU"]
__version__ = "1.0.0"
