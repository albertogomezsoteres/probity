# Limitations, failure modes and responsible use

## What the results do not show

- **One model.** All results are for Gemma 2 2B-it. A probe must be trained again for every model, and I did not test if the recipe transfers.
- **Short factual claims only.** Cities, translations, numbers, companies and simple world knowledge. I did not test long answers, reasoning, code, opinions or advice.
- **Few honest mistakes.** Gemma rarely made clear mistakes on these questions (4 cases in notebook 02), so I cannot say how well the probe catches hallucinations. The strong results are for cases where the model knows the answer: lies and giving in to pressure.
- **"Believed false" is not "false".** The probe reads what the model treats as true. If Gemma learned a wrong fact, the probe marks it as true. Probity is not a fact checker.
- **Decodable is not used.** A probe shows that the information is present in the hidden states. It does not show that the model uses it to choose its words. A causal test (steering) would be needed for that.

## Known failure modes

| Failure | Evidence | What I did |
|---|---|---|
| Probe trained without negations reads them backwards | AUROC 0.011 on negated translations (01b) | Train with negations |
| Reading in context shifts the scores and adds shortcuts | Layer 14 lies 0.643, layer 20 caving 0.679 (02) | Read claims alone |
| Layer 20 reads the instruction instead of the claim | Prompt AUROC 0.996, truth inside lie prompt 0.358 (02b) | Use layer 14 |
| Integrated Gradients does not pass the deletion test on lies | Deletion: -3.2 vs -1.1 for random words (03); the test removes words, which favours LIME and KernelSHAP | LIME by default, live deletion test |
| Attribution methods disagree and depend on the number of words | 86% of LIME's top words vs 38% of KernelSHAP's are in the instruction (03, robustness checks) | Shortcut claim rests on the 2x2 test, not on XAI |
| Pooled numbers hide a harder source | Read alone: cities 0.998 to 1.000, general knowledge 0.825 to 0.916 (robustness checks) | Report both |
| Replies that start with "I" are not scored | 4 of the 9 pushback failures that got through (05) | Open problem |
| Keyword grading ignores negation | "I cannot confirm that X" counts as naming X | Disclosed; affects a few cases |
| Claims that only make sense in context ("It is in Iraq") | By design, claims are read alone | Open problem |
| Guard under a lie instruction | Abstains 78.6% of the time; correct answers stay at about 15% (05) | Shown honestly in the results |
| Transfer to new domains is partial | Medicine and finance: probe AUROC 0.80 vs 0.96 for asking; guard 52.7% to 31.1% with more abstentions (06) | Per-domain probes as future work |
| Guard under pushback relies on the earlier answer | 54 of 55 fixes keep Gemma's first answer (05) | Disclosed |
| Unfamiliar inputs | Readings far from the training data | Familiarity flag in the app |

## Uncertainty

- Every main number has a 95% bootstrap interval that resamples facts or subjects.
- τ = 0.50 is at the edge of the grid, so a different grid might find a slightly better threshold.
- Generation on a GPU is not perfectly deterministic, so a rerun can change the last digit of some numbers.

## Misuse risks

- **False confidence.** A green reading can make people trust a wrong answer that the model believes. The app always shows the reading as "what the model believes", never as "true".
- **Attacking the probe.** Someone who knows the probe could look for prompts or fine-tuning that move the reading without changing the answer. I did not test adaptive attacks.
- **Hiding behind the guard.** A company could use a guard like this to claim its chatbot "does not lie". The guard only covers the cases tested here.
- **Surveillance.** Reading the internal state of a model is harmless, but the same Monitor could be used on user logs. The app does not store conversations, and the audit script only reads files the owner provides.

## Stakeholders

- **Companies that self-host small open models** (the main audience). They can read hidden states, which closed API models do not allow.
- **End users,** who see fewer wrong answers after pushback but may see "I'm not confident enough" more often.
- **Safety and audit teams,** who can use the Monitor to review logs.
- **Researchers,** who can reuse the protocol (fact-grouped splits, controls, deletion test) to check their own probes.

## Deployment notes

- Retrain and recalibrate the probe for every model and every new kind of content.
- Choose τ with your own cost of false alarms vs missed errors. The threshold is a policy decision, not a technical constant.
- Keep a human in the loop for high-stakes answers. Abstaining is a feature, not a bug.
