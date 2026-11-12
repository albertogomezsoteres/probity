# The training and evaluation pipeline

The pipeline turns labelled statements into the calibrated probes that the app and the package use, and reports
their accuracy with the protocol in [`experiments/protocol.md`](../experiments/protocol.md). It reproduces
notebooks 01b (the headline evaluation) and 04 (the app probes) from the command line.

```
probity data  ──►  probity extract  ──►  probity train  ──►  results/app_probes.npz (with --publish)
  statements        hidden states        layer, metrics,
  + splits          (cached)             calibrated probes
```

`probity run` runs the three stages and ends with a table that compares the reproduced numbers with the published
ones in `results/summary01b.json`.

## Stages

### 1. Data (`probity data`, no GPU)

- Downloads the nine curated datasets of *The Geometry of Truth* at commit `5d1c630` and verifies each file
  against the SHA-256 checksum in [`configs/default.yaml`](../configs/default.yaml).
- Balances every dataset to at most 1,000 statements (half true, half false) with a fixed seed.
- Groups statements by fact (`probity.data.fact_id`) and draws train / validation / test (60 / 20 / 20) and five
  cross-fitting folds over facts with an md5 hash. The stage fails if a fact appears in two splits.
- Writes `outputs/data/statements.csv`, which is byte-identical to `results/statements.csv` (the checksum is
  verified and logged), and `statements_with_splits.csv`.

### 2. Hidden states (`probity extract`, GPU)

Each statement is read as Gemma's own answer to "Tell me one fact." The residual stream at the last token is
stored for every layer (0 = embeddings) in `outputs/activations/`, keyed by a hash of the statements and the model,
so a second run reuses the file.

### 3. Probes and evaluation (`probity train`)

1. One logistic-regression probe (standardized activations, L2, `C = 0.1`) per layer on the training facts; the
   layer with the best validation AUROC is selected (layer 14 for the published results).
2. The probe is refitted on train + validation and the test set is read **once**: AUROC, accuracy and a 95%
   bootstrap interval that resamples facts.
3. Baseline: asking Gemma "true or false?" on the same test statements (`evaluation.ask_model`, needs the model).
4. The probe bank for the app: one probe per layer trained on train, calibrated on validation (Platt scaling) and
   scored on test, plus the familiarity test (Mahalanobis distance in 64 principal components).

Outputs, in the run directory: `app_probes.npz`, `metrics.json`, `layer_selection.csv`,
`results_test_per_dataset.csv` and, with `probity run`, `comparison_with_published.csv`. `--publish` copies the
new probes into `results/`.

## Configuration

Every setting is in [`configs/default.yaml`](../configs/default.yaml) and typed in `probity.config`. Use another file
with `--config my.yaml`, or override single values:

```bash
probity train --set probe.C=0.05 --set run.seed=1 --set evaluation.ask_model=false
```

Unknown keys are rejected, so a typo cannot silently fall back to a default.

## Run records

Each stage writes `outputs/runs/<timestamp>-<stage>-<id>/`:

| File | Content |
|---|---|
| `config.yaml` | the fully resolved configuration |
| `run.json` | status, timings, seed, model, data commit, git commit and dirty flag, Python, package, CUDA and GPU versions, artifacts |
| `metrics.json` | the numbers the stage produced |
| `run.log` | the log of the stage |

Set `tracking.wandb: true` to also log to Weights & Biases (`pip install wandb`).

## Determinism

Seeds are set for Python, NumPy and PyTorch, and cuDNN runs in deterministic mode. The data stage is exactly
reproducible. Hidden states can differ in the last digits across GPUs and library versions, which can move a
metric in the third decimal; the comparison table uses a tolerance of 0.0015 for that reason.

## Reports without a GPU

`probity figures` and `probity robustness` rebuild `figures/` and `results/robustness_checks.json` from the files
in `results/`. Both outputs are checked against the committed files by the test suite.
