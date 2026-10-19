# Evaluation protocol

These rules apply to every experiment. The goal is that no result depends on a choice made after looking at the test data.

## Data and splits

- Statements come from *The Geometry of Truth* at a fixed commit (`5d1c630`).
- Statements about the same fact share a `fact_id` and always go to the same split. Splits are about 60% train, 20% validation and 20% test, by fact.
- The questions in notebooks 02 to 05 are about cities and subjects that **never appear** in the probe's training statements.
- In notebook 05, conversations are split into development and test **by subject** with an md5 hash, so no subject is in both.

## Model choices

- The layer is chosen on validation facts. The threshold τ is chosen on development subjects. The test set is used once.
- Two choices did touch test statements, and neither is used for a reported number: the "robust" layer 20 in notebook 01 comes from a leave-one-dataset-out test over all statements, and the app's familiarity warning uses the 99th percentile of test statements.
- Probes are retrained inside each notebook from the saved statements with the same seed, so a notebook never depends on temporary files from another run.

## Labels

- A claim is **true** only if it names the correct answer and no other known answer. It is **false** only if it names the known wrong answer and not the correct one. Everything else is excluded, and the counts are printed in the notebook.
- I checked the labelled tables by eye before trusting any number.

## Statistics

- AUROC is the main metric because it does not depend on a threshold.
- 95% confidence intervals use 1,000 bootstrap resamples of facts or subjects, not of sentences. Exception: the notebook 02b intervals resample claims; `scripts/robustness_checks.py` recomputes the main AUROCs with intervals over subjects.
- For the guard, I report the rate before and after, the interval of the difference, and how often the guard abstained or stepped in.

## Controls

| Control | What it rules out |
|---|---|
| Shuffled-label probe | The probe fits noise |
| Asking the model | The probe adds nothing over the model's own answer |
| Leave one dataset out | The probe only memorized one topic |
| Negated pairs | The probe reads "sounds like a fact" instead of truth |
| 2x2 truth vs prompt design | The probe reads the instruction instead of the claim |
| Random-word deletion | An explanation looks good but is not faithful |
| Valid corrections | The guard just repeats the old answer |
| Plain questions | The guard changes correct answers (false alarms) |
