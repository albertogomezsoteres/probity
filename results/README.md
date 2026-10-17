# Results

Every file here was written by a notebook (or by `scripts/robustness_checks.py`) and is used as is by `scripts/make_figures.py`, the README and the app. Nothing was edited by hand. Some JSON summaries (for example `summary02.json`) contain `NaN` where a metric is undefined, such as an AUROC for a scenario with almost no false claims. Python's `json` module reads these files without problems; strict JSON parsers need `NaN` replaced by `null`.

| File | Notebook | Content |
|---|---|---|
| `statements.csv` | 01 | The 7,696 labelled statements with dataset, topic and the notebook 01 split |
| `polygraph_probe.npz` | 01 | Probe weights at the best layer (first pass) |
| `results_layers.csv` | 01 | AUROC and accuracy per layer, format and method (probe, mass-mean, shuffled control) |
| `results_lodo.csv` | 01 | Leave-one-dataset-out AUROC per layer |
| `results_negation.csv` | 01 | The "not" test |
| `results_per_dataset.csv` | 01 | AUROC per dataset at the best layer |
| `summary.json` | 01 | Main numbers of notebook 01 |
| `results_test_per_dataset.csv` | 01b | Test AUROC per dataset, probe vs asking the model |
| `results_autopsy.csv` | 01b | Negation autopsy per layer, probe and domain |
| `summary01b.json` | 01b | Headline test AUROC with intervals, autopsy at layer 14 |
| `claims.csv` | 02 | Every generated reply with scenario, label and readings (in context and alone) |
| `scenario_results.csv` | 02 | AUROC, detection rate and false alarms per scenario and layer |
| `gallery.json` | 02 | Examples with token-by-token readings for the app |
| `summary02.json` | 02 | Main numbers of notebook 02 |
| `results_confound_cells.csv` | 02b | The four groups of the 2x2 design with readings |
| `results_confound_tests.csv` | 02b | Truth and prompt AUROC tests with intervals |
| `summary02b.json` | 02b | Main numbers of notebook 02b, including the alone vs in-context AUROCs |
| `attributions.json` | 03 | Word attributions per example and method |
| `results_attr_shares.csv` | 03 | Share of attribution on key words |
| `results_attr_deletion.csv` | 03 | Deletion test per method and k |
| `results_attr_agreement.csv` | 03 | Spearman agreement between methods |
| `summary03.json` | 03 | Main numbers of notebook 03 |
| `app_probes.npz` | 04 | Calibrated probes at every layer and the data for the familiarity test (PCA basis, test distances) |
| `guard_test_conversations.jsonl` | 05 | Every test conversation with the unguarded and guarded answer |
| `guard_threshold_sweep.csv` | 05 | τ sweep on development subjects |
| `guard_examples.json` | 05 | Example conversations for the app |
| `summary05.json` | 05 | Guard results on test subjects, τ, latency |
| `high_stakes_items.csv` | 06 | The 82 medicine and finance questions with answers, accepted words and sources |
| `high_stakes_statements.csv` | 06 | Correct and wrong statement for each item with probe and ask-the-model readings |
| `high_stakes_conversations.jsonl` | 06 | Every verifiable conversation with the unguarded and guarded answer |
| `demo_cases_highstakes.json` | 06 | Cases where Gemma gave in and the guard restored the answer (candidates for the app) |
| `summary06.json` | 06 | Transfer, guard results by domain, cross-fitted threshold and the trade-off table |
| `robustness_checks.json` | `scripts/robustness_checks.py` | Main results split by source, attribution per word, and what the guard did in each scenario |
