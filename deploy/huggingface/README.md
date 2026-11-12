---
title: Probity
emoji: 🔎
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.31.0
python_version: "3.11"
app_file: app.py
pinned: true
license: mit
short_description: A white-box polygraph and guardrail for open LLMs
models:
  - google/gemma-2-2b-it
datasets: []
tags:
  - interpretability
  - explainable-ai
  - guardrails
  - probing
---

# Probity

Probity reads the hidden states of Gemma 2 2B-it while it talks, flags the claims the model itself believes are
false, and stops them before they reach the user.

Source code, experiments and results: https://github.com/albertogomezsoteres/probity

This Space is deployed automatically from the `main` branch of the repository by a GitHub Actions workflow.
