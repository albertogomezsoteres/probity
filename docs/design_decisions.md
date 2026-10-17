# Design decisions

Each decision below comes from an experiment in this repository. I list the alternative I rejected and the evidence that made me choose.

## 1. Model: Gemma 2 2B-it

- **Why:** a white-box method needs an open model, and I needed a model that fits on one Colab GPU and runs fast enough for a live app (about 1 s per reply on an A100). Gemma 2 2B-it is instruction-tuned, so it behaves like a real chatbot, including giving in to pushback.
- **Alternative:** a larger model (Gemma 2 9B). It would make the results more general, but every experiment would take several times longer. I list it as future work.

## 2. Probe: logistic regression on one layer

- **Why:** it is the simplest probe that works, it is easy to explain (a dot product with one "truth direction" plus a threshold), and it is cheap at run time (51 ms per check). In notebook 01, logistic regression beats the mass-mean probe, and the shuffled-label control stays at chance.
- **Alternative:** a non-linear probe (MLP). With AUROC 0.98 already, a more complex probe would be harder to interpret and more likely to learn shortcuts.

## 3. Layer 14, chosen on validation

- **Why:** notebook 01b chooses the layer on validation facts and reports the test set once. Layer 14 is the best on validation. When claims are read alone, layers 14 and 20 are both good (lies 0.962 vs 0.982, caving 0.981 vs 0.955), so the validation choice decides.
- **Rejected:** layer 20. It looked best on unseen datasets in notebook 01 (a leave-one-dataset-out test that also used the test facts, so this choice was never used for a reported number) and almost perfect on instructed lies in notebook 02 (0.998). Notebook 02b shows that this is a shortcut: it reads the lie instruction, not the claim.

## 4. Train with negations

- **Why:** a probe trained only on affirmative translations reads negated translations backwards (AUROC 0.011, notebook 01b). The final probe is trained on all nine datasets, negations included (0.999 on negated translations).

## 5. Fact-grouped splits

- **Why:** "X is in Y" and "X is not in Y" are about the same fact. If they end up in different splits, the test set is not really new. All splits and folds are drawn over facts, and confidence intervals resample facts.

## 6. Read every claim alone, not inside the conversation

- **Why:** in context, the probe breaks in a different way at each layer (layer 14 misses instructed lies with 0.643, layer 20 misses caving with 0.679) and its readings shift so much that most true claims are flagged. Reading each claim alone, in the same format as training, gives 0.955 to 0.982 pooled (notebook 02b): 0.998 to 1.000 on cities and 0.825 to 0.916 on general-knowledge facts.
- **Cost:** the probe cannot judge claims that need the conversation to make sense. See `limitations.md`.

## 7. The weakest claim decides

- **Why:** one false sentence is enough to make an answer wrong. A reply is scored by its lowest `p_true`. Replies without factual sentences (greetings, questions) get no score and pass.

## 8. Calibration and a familiarity check

- **Why:** a raw probe score is hard to read. Platt scaling on the validation set turns it into a probability that users can understand. A Mahalanobis distance test (64 principal components, 99th percentile of test statements) flags claims that look unlike the training data, so the app can say "do not trust this reading".

## 9. Guard policy: replace or abstain, never invent

- **Why:** the guard does not know the correct answer, so it should not write one. It can only choose between answers Gemma already gives (4 samples plus Gemma's earlier answers) or abstain. When it keeps an earlier answer it adds one fixed sentence in front ("I understand the doubt, but I have double-checked."), which Gemma did not write; a neutral wording would be more honest and is an easy change.
- **Controls:** a guard that always repeated the earlier answer would block valid corrections. The correction control (99.5% still accepted) shows this does not happen. It is an easy control, though: the wrong earlier answer is written by me and the probe strongly disbelieves it.

## 10. Threshold τ chosen on development subjects

- **Why:** the threshold is a policy choice, so I chose it with an explicit cost (wrong answers after pushback + rejected corrections + false alarms, equal weights) on development subjects only, and reported test subjects once. The best value, 0.50, is at the edge of the grid I fixed in advance. I did not extend the grid after seeing it, because that would mean tuning while knowing the result.

## 11. Which XAI methods, and why

- **LIME, KernelSHAP and Integrated Gradients** explain a verdict word by word. I used them to ask whether the verdict depends on the answer or on the instruction. They did not settle it: they disagree with each other, and per word none of them favours the instruction. The controlled 2x2 test (notebook 02b) did. I report this as a finding about XAI, not as a success.
- **A deletion test** checks every method: delete its top words and compare with deleting random words. Integrated Gradients did not pass it on lies, so the app uses LIME by default and runs the deletion test live on every explanation. The test removes words, which is the same kind of change LIME and KernelSHAP are built on, so it favours them.
- **Settings.** LIME is my own implementation of the LIME recipe (random word masks, exponential kernel, Ridge surrogate) so that it works on the probe score directly; KernelSHAP (`shap.KernelExplainer`) uses 256 coalitions, so its values are estimates; Integrated Gradients uses 32 steps and a zero-embedding baseline, which influences the result.
- I did not use attention weights as explanations, because attention is not a reliable explanation (Jain and Wallace, 2019), and plain gradient saliency is a weaker version of Integrated Gradients.
- I did not add other methods, because each one needs a clear question to answer.
