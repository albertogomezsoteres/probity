"""Entry point of the Probity Hugging Face Space (built by build_space.py; see deploy/README.md)."""

import os

import spaces  # noqa: F401  (on ZeroGPU, `spaces` must be imported before torch)
from huggingface_hub import login

from probity.app import create_app
from probity.config import load_config

# Gemma is a gated model: add a read token with access to it as the Space secret HF_TOKEN.
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])

cfg = load_config("configs/default.yaml")
demo, launch_style = create_app(cfg)
demo.queue().launch(**launch_style)
