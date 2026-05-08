# docs

Writeups and reusable artifacts from the TBXT hit-identification work.

## Index

- [`HACKATHON_PLAN.md`](HACKATHON_PLAN.md) — overall execution plan for the May 9 event.
- [`folds-pacmap-kmeans6/`](folds-pacmap-kmeans6/README.md) — reusable
  6-fold chemical-space split (Morgan → PaCMAP → KMeans). Shared artifact
  consumed by all modeling experiments.
- [`classification-models-try1-rjg/`](classification-models-try1-rjg/README.md)
  — first-pass binder/non-binder classifier ensembles (CheMeleon transfer,
  XGBoost, ± validation folds), TukeyHSD comparison on the distinct
  holdout fold. rjg, 2026-05-08.
- [`regression-models-try1-rjg/`](regression-models-try1-rjg/README.md)
  — first-pass continuous pKD regression counterpart of the classifier
  experiment. Same four ensemble variants, squared-error TukeyHSD, and
  train-mean baseline. rjg, 2026-05-08.

## Adding a new experiment

Conventions:
1. Create `docs/<name>/README.md` with results + linked figures.
2. Mirror as `scripts/<name>/` (numbered scripts) and
   `data/<name>/` (output artifacts + trained model state).
3. Cross-link consumed artifacts (e.g. fold assignments) rather than
   regenerating them, so upstream improvements propagate.
