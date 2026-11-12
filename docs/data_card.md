# Data card: labelled statements

## Source

*The Geometry of Truth* (Marks and Tegmark, 2023), [saprmarks/geometry-of-truth](https://github.com/saprmarks/geometry-of-truth),
pinned to commit `5d1c630c44f7e50bda7ad86d601ccadf9abc5ddb`. The files are downloaded by `probity data` and checked
against the SHA-256 checksums in [`configs/default.yaml`](../configs/default.yaml). The source repository does not
state a license; `results/statements.csv` keeps the sampled statements for reproducibility, with this attribution.

## Composition

| Dataset | Topic | Negated | Example |
|---|---|---|---|
| `cities` | geography | no | "The city of Krasnodar is in Russia." |
| `neg_cities` | geography | yes | "The city of Krasnodar is not in Russia." |
| `sp_en_trans` | translation | no | "The Spanish word 'uno' means 'one'." |
| `neg_sp_en_trans` | translation | yes | "The Spanish word 'uno' does not mean 'one'." |
| `larger_than`, `smaller_than` | numbers | no | "Fifty-one is larger than fifty-two." |
| `companies_true_false` | business | no | "Thermo Fisher ... semiconductor products." |
| `counterfact_true_false` | world knowledge | no | "The mother tongue of Danielle Darrieux is French." |
| `common_claim_true_false` | common claims (noisier labels) | no | "Spiders can use surface tension to walk on water." |

Each dataset is capped at 1,000 statements, half true and half false, sampled with seed 0: **7,696 statements** in
total.

## Splits

- **Fact-grouped split (used for every reported number).** Statements about the same fact (the same city, Spanish
  word, number pair, company or subject) share a `fact_id` and always land in the same split: 4,649 train, 1,603
  validation and 1,444 test statements (60 / 20 / 20 by fact, md5 hash with seed 0). Five cross-fitting folds are
  drawn the same way.
- **Stratified split (`split` column, notebook 01 only).** 80 / 20 inside every dataset and label; kept for
  reference.

## Known issues

- Grouping uses surface keys, so semantically related claims with different keys (mostly in `common_claim`) can
  still fall in different splits.
- `common_claim_true_false` has noisier labels; the probe and the baseline are both weaker on it.
- English only, short single-sentence statements.

## Other data in `results/`

The red-teaming, guard and high-stakes experiments (notebooks 02 to 06) use questions generated from these
datasets and a hand-written set of 82 medicine and finance questions (`high_stakes_items.csv`). Every file is
listed in [`results/README.md`](../results/README.md).
