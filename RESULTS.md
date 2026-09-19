# Generated Results

Generated from MLflow tracking store (`mlflow.db`), experiment `orbital_threat_assessment`, dataset `full_population_932335`.

**Test-set metrics — touched exactly once per run, after hyperparameter selection on the validation set.** See src/otda/train.py for the split/search methodology and legacy/README.md for the originally published (test-set-selected, 100K-sample) numbers this supersedes.

## Experiment 1 — Orbital Only (no MOID)

| Model | Recall | Precision | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 1.000 | 0.149 | 0.260 | 0.258 | 0.997 |
| Decision Tree | 0.998 | 0.343 | 0.511 | 0.678 | 0.997 |
| Random Forest | 0.983 | 0.359 | 0.526 | 0.450 | 0.998 |
| LinearSVC | 1.000 | 0.151 | 0.262 | 0.242 | 0.997 |

## Experiment 2 — MOID Included

| Model | Recall | Precision | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 1.000 | 0.340 | 0.508 | 0.814 | 1.000 |
| Decision Tree | 0.985 | 0.944 | 0.964 | 0.999 | 1.000 |
| Random Forest | 0.990 | 0.938 | 0.963 | 0.997 | 1.000 |
| LinearSVC | 1.000 | 0.413 | 0.584 | 0.821 | 1.000 |

## Interpretation: why Exp2 is near-perfect and Exp1 isn't

Exp2's decision tree scores (0.999 PR-AUC here vs. 0.988 in the original 100K-sample run) aren't a new leakage artifact — they're the same mechanism the original notebook already identified: MOID is part of NASA's own two-condition PHA definition (`moid <= 0.05 AU` and `H <= 22`), so a tree given that feature is learning the label's own definitional boundary, not an independent signal. Training on the full 932,335-row population instead of the 100K sample gives the tree far more examples near that boundary to split on, which is why the score moved closer to 1.0 rather than staying flat.

Exp1's decision tree — no MOID, orbital mechanics only — moved from 0.665 (legacy) to 0.678 here: a modest, expected improvement from more training data and validation-scored (rather than test-set-scored) hyperparameter selection, not a discontinuity. That contrast is the sanity check: if the full-population switch were introducing leakage rather than just more data, Exp1 would have jumped the same way Exp2 did, and it didn't — its features don't encode the label's own definition, so there's no boundary for extra rows to sharpen.
