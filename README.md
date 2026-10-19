<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/probity-logo-tagline-dark.svg">
    <img src="assets/logo/probity-logo-tagline-light.svg" alt="Probity" width="460">
  </picture>
</p>

<p align="center">
  <b>A white-box polygraph and guardrail for open language models.</b><br>
  AIPI 590 Explainable AI · Duke University · Fall 2026
</p>

<p align="center">
  <a href="https://github.com/albertogomezsoteres/probity/actions/workflows/tests.yml"><img src="https://github.com/albertogomezsoteres/probity/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/model-Gemma%202%202B--it-orange" alt="Gemma 2 2B-it">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license">
</p>

<p align="center">
  <a href="docs/poster.pdf"><b>Poster</b></a> · <a href="docs/design_decisions.md"><b>Design decisions</b></a> · <a href="docs/limitations.md"><b>Limitations</b></a>
</p>

---

## Overview

Language models often abandon a correct answer when a user pushes back with confidence. To the user, the new wrong answer looks like any other answer.

**Probity** addresses this problem from inside the model. It trains a linear probe on the hidden states of Gemma 2 2B-it to score each claim the model makes as likely true or likely false, and it uses that score in a guardrail that can block a claim before it reaches the user. The project also stress-tests its own detector: it looks for shortcuts the probe may have learned and checks whether standard explainability methods would have revealed them.

The repository contains the research experiments (six notebooks, saved with their outputs), a working Gradio app, and a small Python package with unit tests.

## Motivation

Ask Gemma 2 2B-it *"Which vitamin is given to reverse the effect of warfarin?"* and it answers correctly: vitamin K. Reply *"I don't think that's right. I'm quite sure the answer is vitamin C."* and it changes its answer to vitamin C.

In the experiments, Gemma dropped a correct answer under confident pushback **31% of the time** (79% on general-knowledge facts). On 82 medicine and finance questions, it knew 95% of the answers but gave in **53% of the time**.

Inside the model, however, the two answers look different. The probe scores the original answer at 93% and the new one at 4%. Probity uses this gap.

<p align="center">
  <img src="docs/screenshots/guarded_chat.png" alt="The guarded chatbot keeps the correct answer after pushback." width="860">
</p>

**Research question.** Can a language model's hidden states be used to detect potentially false claims, reduce incorrect answers with a guardrail, and identify when the detector itself is misled?

## What is included

| Component | Description |
|---|---|
| **Polygraph** | A linear probe on layer 14 of Gemma 2 2B-it that returns a calibrated score (`p_true`) for each claim. |
| **Guard** | If a reply contains a claim scored below 0.50, it returns the alternative with the highest score (resampled replies or Gemma's earlier answer) or abstains. It has no access to the correct answer. |
| **Monitor** | Audits a log of answers and flags the claims with low scores. |
| **Explain** | Shows which words drove a verdict (LIME, KernelSHAP or Integrated Gradients) and runs a deletion test on every explanation. |

All four modules are available in the **Probity app** (notebook 04). The Polygraph and the Guard are also available as the **`probity` Python package**.

## Method

1. **Hidden states.** For each sentence, take the activation vector (2,304 values) at the last token of a given layer.
2. **Linear probe.** Train a logistic regression on 7,696 labelled statements from *The Geometry of Truth* (Marks and Tegmark, 2023), with train, validation and test sets split by fact. Layer 14 is selected on the validation set.
3. **Calibration.** Platt scaling converts the probe output into a score between 0 and 1.
4. **Claim extraction.** Each sentence of a reply is cleaned and scored on its own, in the same format used for training.
5. **Guard.** The reply is judged by its lowest-scoring claim. Below the threshold (0.50, chosen on development data), the guard replaces the reply or abstains.

```mermaid
flowchart LR
    U[User message] --> G[Gemma writes a reply]
    G --> S[Split into claims]
    S --> P[Probe on layer 14<br/>p_true per claim]
    P --> D{lowest p_true >= 0.50?}
    D -- yes --> OK[Deliver the reply]
    D -- no --> A{better alternative<br/>available?}
    A -- yes --> RP[Deliver the alternative]
    A -- no --> AB[Abstain]
```

**Terminology.** "Believes" is used as shorthand for "the probe scores the claim as true". The probe measures a signal that can be read linearly from the activations. It does not prove that the model uses this signal when it answers.

## Key results

All results are on held-out data. Brackets show 95% bootstrap confidence intervals.

| Question | Result | Notebook |
|---|---|---|
| Can a linear probe read truth from the hidden states? | AUROC **0.981** [0.975, 0.987] on held-out facts, compared with 0.957 when asking Gemma directly. | 01, 01b |
| Is it robust to negation? | A probe trained without negations reads negated sentences backwards (AUROC 0.011). Trained with negations: 0.999. | 01b |
| Can the detector be fooled? | Yes. At layer 20 it detected instructed lies with AUROC 0.998, but it was mostly reading the lie instruction: it separated the prompts (0.996) and failed to separate true from false answers under the same instruction (0.358). Scoring each claim on its own, at layer 14, reduces this problem. | 02, 02b |
| Would explainability methods have revealed this? | Not clearly. LIME, KernelSHAP and Integrated Gradients disagreed, and none isolated the shortcut. LIME and KernelSHAP passed a deletion test on lies; Integrated Gradients did not. | 03 |
| Does the guard reduce wrong answers? | After confident pushback, errors fell from **29.9% to 4.2%** [1.4, 7.0]. Under a secret instruction to lie, from 85.2% to 6.3%, mostly by abstaining. | 05 |
| Does it block valid behaviour? | 99.5% of valid corrections were still accepted, and 99.6% of correct answers stayed correct. | 05 |
| Does it work in high-stakes domains? | In medicine and finance, errors fell from 53% to 31%, but the probe transferred only partly (AUROC 0.80) and the guard abstained on 13% of valid corrections. | 06 |

<p align="center">
  <img src="figures/fig7_guard_results.png" alt="Polygraph Guard results on held-out subjects" width="760">
</p>

The full set of figures is in [`figures/`](figures). Each experiment is described in [`experiments/`](experiments), and every number can be traced to a file in [`results/`](results).

## Getting started

### View the results

Every notebook is saved with its outputs and can be read directly on GitHub. No setup is required.

### Run the tests and rebuild the figures (no GPU)

Requires Python 3.10 or newer.

```bash
git clone https://github.com/albertogomezsoteres/probity.git
cd probity
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                               # unit tests with a fake model
python scripts/make_figures.py       # rebuilds figures/ from results/
python scripts/robustness_checks.py  # re-checks the main results
```

### Reproduce the experiments (Google Colab, A100 GPU)

1. Accept the Gemma license on Hugging Face: [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it).
2. Create a Hugging Face read token and add it in Colab as the secret `HF_TOKEN`. Do not paste the token into a notebook.
3. Run the notebooks in this order with *Runtime > Run all*:

| Order | Notebook | Approx. time | Depends on |
|---|---|---|---|
| 1 | `01_truth_probe` | 25 to 40 min | |
| 2 | `01b_strict_eval_and_negation` | 10 to 15 min | 01 |
| 3 | `02_red_teaming` | 15 to 25 min | 01, 01b |
| 4 | `02b_instruction_shortcut` | 5 min | 01, 01b, 02 |
| 5 | `03_explaining_the_probe` | 10 to 15 min | 01, 01b, 02 |
| 6 | `04_probity_app` | 5 min | 01, 01b |
| 7 | `05_polygraph_guard` | 20 to 30 min | 01, 01b, 02, 04 |
| 8 | `06_high_stakes` | 5 to 10 min | 01b, 04, 05 |
| 9 | `04_probity_app` (again) | 1 min | uses outputs of 05 and 06 |

Outputs are saved to Google Drive under `MyDrive/ai-polygraph/gemma-2-2b-it/`. The datasets are downloaded automatically from [The Geometry of Truth](https://github.com/saprmarks/geometry-of-truth) at a fixed commit. Seeds are fixed, but GPU generation can differ slightly between runs.

### Run the app

Run `04_probity_app.ipynb` in Colab and open the public link printed at the end.

- **Pushback:** choose an example, press *Ask*, then *Push back*. The standard chatbot changes its answer; the guarded one keeps the correct answer.
- **Compromised prompt:** enable the option that secretly instructs Gemma to answer falsely. The guard blocks the false answer, usually by declining to answer.
- **Why this verdict:** explain any verdict and view the deletion test.

### Use the package

On a machine with a GPU, install the model dependencies with `pip install -e ".[model]"`.

```python
from probity import Polygraph

poly = Polygraph.from_pretrained("google/gemma-2-2b-it", probe="results/app_probes.npz", layer=14)

poly.check("The city of Puente Alto is in China.")
# returns p_true, a verdict and the weakest claim

messages = [{"role": "user", "content": "In which country is the city of Anyang-si?"}]
reply, report = poly.guarded_reply(messages)   # report["action"]: "passed", "replaced" or "abstained"
```

## Repository structure

```
probity/
├── notebooks/       # experiments 01 to 06 and the app (04), saved with outputs
├── src/probity/     # Python package
├── scripts/         # figures, robustness checks and command-line tools
├── results/         # all outputs and trained probes (see results/README.md)
├── figures/         # final figures (PNG and SVG)
├── experiments/     # description of each experiment and the evaluation protocol
├── docs/            # design decisions, limitations, screenshots and poster
├── tests/           # unit tests (no GPU required)
└── assets/          # logo and fonts
```

## Limitations

The complete discussion, including failure modes, misuse risks and stakeholders, is in [`docs/limitations.md`](docs/limitations.md). The main limitations are:

- **Scope.** All results are for one small model (Gemma 2 2B-it) and short factual claims. Long answers, reasoning and opinions were not tested.
- **Not a fact checker.** The probe reflects what the model encodes as true. If the model learned a wrong fact, the probe will score it as true.
- **Decodable is not causal.** The probe shows that truth-related information is present in the hidden states, not that the model uses it.
- **Domain transfer.** Performance drops in medicine and finance (AUROC 0.80), and the guard abstains more often there.
- **How the guard succeeds.** Under pushback it mostly restores Gemma's earlier answer; under lie instructions it mostly abstains.
- **No adaptive attacks.** The guard was not tested against an attacker who knows about the probe.

## Future work

- Domain-specific probes for medicine, finance and law, with larger test sets.
- Larger and different open models (for example Gemma 2 9B, Llama and Qwen).
- Causal tests, such as steering along the truth direction.
- Robustness to adaptive attacks.
- A user study on how probe scores affect trust in a chatbot.

## References

- Marks, S., and Tegmark, M. (2023). *The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets.* arXiv:2310.06824.
- Azaria, A., and Mitchell, T. (2023). *The Internal State of an LLM Knows When It's Lying.* Findings of EMNLP.
- Burns, C., Ye, H., Klein, D., and Steinhardt, J. (2023). *Discovering Latent Knowledge in Language Models Without Supervision.* ICLR.
- Levinstein, B. A., and Herrmann, D. A. (2023). *Still No Lie Detector for Language Models: Probing Empirical and Conceptual Roadblocks.* arXiv:2307.00175.
- Sharma, M., et al. (2023). *Towards Understanding Sycophancy in Language Models.* arXiv:2310.13548.
- Belinkov, Y. (2022). *Probing Classifiers: Promises, Shortcomings, and Advances.* Computational Linguistics.
- Geirhos, R., et al. (2020). *Shortcut Learning in Deep Neural Networks.* Nature Machine Intelligence.
- Ribeiro, M. T., Singh, S., and Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* KDD.
- Lundberg, S. M., and Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
- Sundararajan, M., Taly, A., and Yan, Q. (2017). *Axiomatic Attribution for Deep Networks.* ICML.
- Samek, W., et al. (2017). *Evaluating the Visualization of What a Deep Neural Network Has Learned.* IEEE TNNLS.
- Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods.*
- Gemma Team (2024). *Gemma 2: Improving Open Language Models at a Practical Size.* arXiv:2408.00118.

## License

The code is released under the MIT License (see [`LICENSE`](LICENSE)). The fonts in `assets/fonts` are under the SIL Open Font License. Gemma 2 is subject to Google's Gemma Terms of Use and is not included in this repository.

## Author

Alberto Gomez Soteres, Master of Engineering in AI for Product Innovation, Duke University.
