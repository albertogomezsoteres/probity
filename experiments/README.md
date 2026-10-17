# Experiments

Each experiment answers one question. The notebooks contain the code and full outputs, and `results/` contains the saved numbers. Experiments are listed in the order I ran them, because each one depends on what the previous one found.

The common protocol is in [`protocol.md`](protocol.md).

| ID | Question | Notebook | Main metric | Result |
|---|---|---|---|---|
| E1 | Can a linear probe read truth from Gemma's hidden states, and at which layer? | `01_truth_probe` | AUROC per layer vs baselines | Peaks in the middle layers. Layer 14: 0.981 |
| E2 | Is the headline number fair, and is the negation failure real? | `01b_strict_eval_and_negation` | Test AUROC with fact-grouped splits, AUROC on negated pairs | 0.981 [0.975, 0.987] vs asking 0.957. Without negations: 0.011 |
| E3 | Does the probe work on Gemma's own answers under attack? | `02_red_teaming` | AUROC per scenario, in context and alone | Gemma gives in 31% of the time (cities 10%, general knowledge 79%). In context: lies 0.643 (L14), caving 0.679 (L20) |
| E4 | Does layer 20 detect the lie or the instruction? | `02b_instruction_shortcut` | 2x2 design: truth vs prompt AUROC | Instruction shortcut: prompt 0.996, truth 0.358. Alone: 0.955 to 0.982 pooled |
| E5 | What do the verdicts depend on, and are the explanations faithful? | `03_explaining_the_probe` | Attribution shares, deletion test vs random | Methods disagree; per word none favours the instruction. IG does not pass the word-deletion test on lies |
| E6 | Can the probe stop bad answers without causing harm? | `05_polygraph_guard` | Rate of wrong answers delivered, with controls | Pushback 29.9% to 4.2%, lies 85.2% to 6.3% (mostly abstaining), controls 99.5% and 99.6% |
| E7 | Does it work in medicine and finance, without retraining? | `06_high_stakes` | Transfer AUROC, pushback with guard, cross-fitted threshold | Gemma gives in 53%; guard 52.7% to 31.1%; probe AUROC 0.80 vs asking 0.96 |
| R | Do the pooled results hold up when split by source, per word and per guard action? | `scripts/robustness_checks.py` | Same metrics, stratified | Cities near perfect, general knowledge harder; see `results/robustness_checks.json` |

## E1. Truth probe

- **Hypothesis:** truth is linearly readable from the residual stream (Marks and Tegmark, 2023).
- **Setup:** 7,696 statements, 9 datasets, hidden state at the last token, all 26 layers.
- **Baselines:** mass-mean probe, shuffled-label control, asking the model.
- **Outputs:** `results_layers.csv`, `results_lodo.csv`, `results_negation.csv`, `results_per_dataset.csv`, `summary.json`.
- **Figure:** `figures/fig1_layer_profile.png`.

## E2. Strict evaluation and negation autopsy

- **Hypothesis:** if the probe reads truth, its reading must flip sign when *not* is added.
- **Setup:** train, validation and test split by fact (4,649 / 1,603 / 1,444 statements). Layer chosen on validation. Cross-fitted scores for 1,348 affirmative and negated pairs.
- **Outputs:** `results_autopsy.csv`, `results_test_per_dataset.csv`, `summary01b.json`.
- **Figure:** `figures/fig2_negation.png`.

## E3. Red-teaming the polygraph

- **Hypothesis:** if the probe reads what the model believes, it should catch lies and caving, where the model knows the right answer.
- **Setup:** 670 questions about subjects never seen in training. Scenarios: honest answers, instructed lies vs a matched truth game, "are you sure?" and confident pushback. Conservative labels (ambiguous answers excluded and counted).
- **Outputs:** `claims.csv`, `scenario_results.csv`, `gallery.json`, `summary02.json`.

## E4. Instruction shortcut

- **Hypothesis:** if the probe reads truth, it should separate true and false claims within the same prompt, and ignore the prompt within the same truth value.
- **Setup:** four groups: told the truth (200), refused to lie (81), wrong in the truth game (56), lied (200).
- **Outputs:** `results_confound_cells.csv`, `results_confound_tests.csv`, `summary02b.json`.
- **Figures:** `figures/fig3_shortcut.png`, `figures/fig4_context_fix.png`.

## E5. Explaining the probe

- **Hypothesis:** if the lie verdict at layer 20 is a shortcut, attribution methods should put the credit on the instruction, not on the answer.
- **Result:** not confirmed by XAI. Raw shares favour the instruction only because it has 85% of the words. Per word, every method gives the claim more credit, and the methods disagree on which words matter.
- **Setup:** 120 explained verdicts (negation, pushback, lies), three methods, deletion test with k = 1, 2, 3 words vs random words, Spearman agreement between methods.
- **Outputs:** `results_attr_shares.csv`, `results_attr_deletion.csv`, `results_attr_agreement.csv`, `attributions.json`, `summary03.json`.
- **Figures:** `figures/fig5_xai_attribution.png`, `figures/fig6_deletion_test.png`.

## E6. Polygraph Guard

- **Hypothesis:** if the probe reads belief, choosing the most believed answer should undo caving and block lies, without blocking valid corrections.
- **Setup:** five scenarios on development and test subjects (md5 split). τ chosen on development only. 4 seeded samples per conversation. Bootstrap over subjects.
- **Outputs:** `guard_test_conversations.jsonl`, `guard_threshold_sweep.csv`, `guard_examples.json`, `summary05.json`.
- **Figures:** `figures/fig7_guard_results.png`, `figures/fig8_guard_threshold.png`.

## E7. Medicine and finance

- **Hypothesis:** if the probe reads what the model believes, it should still help on facts from other domains where mistakes are costly, without retraining.
- **Setup:** 82 questions (45 medicine, 37 finance), each with a correct answer, a plausible wrong answer and a public source. Probe and threshold fixed from earlier notebooks. Same scenarios and controls as E6, bootstrap over items, plus a cross-fitted threshold (chosen on one half of the items, applied to the other).
- **Result:** Gemma knows 95% of the answers but gives in to pushback 53% of the time (71% in finance). The guard cuts this to 31% with τ fixed, with 13% of valid corrections and 6% of correct answers turned into abstentions. The probe transfers only partly (AUROC 0.80 vs 0.96 for asking Gemma) and struggles with numbers.
- **Outputs:** `high_stakes_items.csv`, `high_stakes_statements.csv`, `high_stakes_conversations.jsonl`, `demo_cases_highstakes.json`, `summary06.json`.
- **Figure:** `figures/fig9_high_stakes.png`.
