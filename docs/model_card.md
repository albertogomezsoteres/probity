# Model card: Probity truth probe

## Overview

| | |
|---|---|
| **What it is** | A calibrated linear probe (logistic regression) on the residual stream of Gemma 2 2B-it, plus a familiarity (out-of-distribution) test |
| **Input** | The hidden state at the last token of a claim, read as the model's own answer to "Tell me one fact." |
| **Output** | `p_true`: the probability that the model represents the claim as true |
| **Layer** | 14 of 26, chosen on validation facts |
| **File** | [`results/app_probes.npz`](../results/app_probes.npz): one calibrated probe per layer (`W`, `B`, `PA`, `PC`), held-out AUROC per layer, and the familiarity test (`MU`, `COMP`, `VAR`, `TEST_DIST`) |
| **Base model** | [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) (not included; Gemma Terms of Use) |
| **Training** | `probity train` ([pipeline](pipeline.md)); notebooks 01b and 04 |
| **License** | MIT (the probe); Gemma Terms of Use (the base model) |

## Intended use

- Flag answers of Gemma 2 2B-it that the model itself does not believe, for example after a user pushes back or
  under a hidden instruction to lie, and block them with Polygraph Guard.
- Audit logs of short factual answers (`probity audit`).
- Research on truth representations, probing and shortcut learning.

**Out of scope:** fact checking (the probe reads the model's belief, not the truth), other models (a probe must be
retrained for every model), long answers, reasoning, opinions and code, and any high-stakes decision without a
human in the loop.

## Training data

7,696 balanced true/false statements from nine datasets of *The Geometry of Truth* (cities, negated cities,
Spanish-English translations and their negations, number comparisons, companies, counterfactual facts and common
claims). Splits are drawn over facts: 4,649 train, 1,603 validation and 1,444 test statements. See the
[data card](data_card.md).

## Evaluation

All numbers are on held-out facts, with 95% bootstrap intervals over facts or subjects.

| Metric | Value |
|---|---|
| Test AUROC (layer 14, refitted on train + validation) | **0.981** [0.975, 0.987] |
| Test accuracy | 0.928 |
| Asking Gemma "true or false?" (baseline) | 0.957 [0.946, 0.966] |
| App probe: test AUROC / accuracy / calibration error (ECE) | 0.980 / 0.923 / 0.019 |
| Negated translations, probe trained without negations | 0.011 (reads them backwards) |
| Negated translations, final probe (trained with negations) | 0.999 |
| Medicine and finance (no retraining) | 0.80, vs 0.96 for asking |
| Polygraph Guard: wrong answers after confident pushback | 29.9% → 4.2% [1.4, 7.0] |
| Polygraph Guard: valid corrections still accepted | 99.5% |

## Limitations and risks

- **Belief, not truth.** If Gemma learned a wrong fact, the probe reads it as true.
- **Shortcuts.** Read inside a conversation, a layer-20 probe tracked the instruction to lie rather than the claim
  (notebook 02b). Probity reads every claim on its own to avoid this.
- **Domain shift.** Transfer to medicine and finance is partial (AUROC 0.80).
- **Decodable is not causal.** The probe shows the information is present, not that the model uses it.
- **Adaptive attacks** against the probe were not tested.

The full discussion, including misuse risks and stakeholders, is in [`limitations.md`](limitations.md).
