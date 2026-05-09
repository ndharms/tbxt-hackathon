# Modeling Notes

XGBoost-based ligand-based virtual screening models for TBXT hit identification.
Both approaches use Butina-clustered (Tanimoto, threshold 0.4) scaffold-aware 5-fold
cross-validation and Optuna hyperparameter optimization (TPE sampler, AUC-PR objective).

## 1. Binary classifier (`scripts/build_xgb_classifier.py`)

Classifies compounds as active (pKD >= 4.5) or inactive (pKD <= 3.5), excluding
compounds in the ambiguous zone. Positives are upweighted so total positive weight
equals total negative weight.

**Features:** MACCS keys (167) + RDKit 2D descriptors (217) = 384 per molecule.

We ran a series of ablation experiments across feature sets, negative sources, and
Optuna budgets:

| Experiment | Features | Negatives | Trials | AUC-ROC | AUC-PR | F1 |
|-----------|----------|-----------|--------|---------|--------|-----|
| MACCS+Desc+Gobbi, Enamine, 25 trials | 640 | SPR + Enamine | 25 | 0.843 | 0.386 | 0.405 |
| MACCS+Desc (no Gobbi), Enamine, 25 trials | 384 | SPR + Enamine | 25 | 0.848 | 0.394 | 0.414 |
| MACCS+Desc (no Gobbi), Enamine, 50 trials | 384 | SPR + Enamine | 50 | 0.843 | 0.400 | 0.430 |
| **MACCS+Desc, SPR-only, 50 trials** | **384** | **SPR only** | **50** | **0.791** | **0.431** | **0.492** |

### Key findings

- **Gobbi pharmacophore features add noise.** Removing them improved AUC-PR from
  0.386 to 0.394 and reduced the feature vector from 640 to 384.

- **Enamine decoys inflate AUC-ROC but hurt AUC-PR.** They are easy negatives that
  boost discrimination (0.84 AUC-ROC) but dilute the precision-recall signal. Removing
  them dropped AUC-ROC to 0.79 but improved AUC-PR from 0.40 to 0.43 and F1 from
  0.43 to 0.49.

- **SPR-only training produces a better-calibrated model.** The optimal threshold
  sits at 0.475 (near the natural 0.5) vs 0.627 with Enamine decoys. Fold variance
  also tightened (worst-fold AUC-PR 0.35 vs 0.25).

- **Additional Optuna trials provide diminishing returns.** Going from 25 to 50
  trials improved AUC-PR by ~0.01. The model plateaus around AUC-PR ~0.43 with
  these features.

The current saved model uses the SPR-only, MACCS+Desc configuration (bolded row).
At the F1-maximizing OOF threshold (0.475): precision = 0.479, recall = 0.506.

The `TBXTClassifier` API supports both ensemble (5 CV models) and single final-model
inference.

Report: `models/xgb_classifier/report.html`

## 2. Pairwise comparison classifier (`scripts/build_pairwise_classifier.py`)

Instead of predicting activity class, this model predicts whether molecule A is more
potent than molecule B by >= 1.0 pKD unit. Training pairs are balanced by subsampling
negatives to match positives; test pairs are left unbalanced for fair evaluation. Pair
predictions are aggregated into per-molecule ranking scores (mean P(mol > other)),
then evaluated via correlation with true pKD.

We ran a series of ablation experiments to determine the best feature representation:

| Experiment | Features/pair | Best AUC-PR | Mean Prec | Mean Rec | Mean F1 | Runtime |
|-----------|--------------|------------|----------|---------|--------|---------|
| MACCS+Desc, [f(A),f(B),f(A)-f(B)], 8 trials | 1152 | 0.396 | -- | -- | -- | ~36 min |
| +Gobbi, [f(A),f(B),f(A)-f(B)], 8 trials | 1920 | 0.396 | 0.47 | 0.24 | 0.32 | ~76 min |
| +Gobbi, **[f(A)-f(B)] only**, 8 trials | 640 | 0.398 | 0.38 | 0.50 | 0.43 | ~7 min |
| +Gobbi, **[f(A)-f(B)] only**, 32 trials | 640 | 0.401 | 0.39 | 0.47 | 0.43 | ~25 min |

### Key findings

- **Gobbi pharmacophore features add no value** in the pairwise setting. AUC-PR was
  identical (0.396) with and without them when using the full [f(A), f(B), f(A)-f(B)]
  representation.

- **The difference fingerprint f(A)-f(B) alone captures all the ranking signal.** It
  matches or exceeds the full concatenated representation while using 1/3 the features
  and running 10x faster. This makes sense for the pairwise task: the absolute
  representations f(A) and f(B) are redundant when the model only needs to learn
  *relative* potency.

- **The diff-only model trades precision for recall** compared to the full representation
  (precision 0.39 vs 0.47, recall 0.47 vs 0.24), with a meaningfully better F1 (0.43 vs
  0.32). It catches more true positive pairs at the cost of more false positives.

- **Additional HPO trials provide diminishing returns.** Going from 8 to 32 Optuna trials
  improved AUC-PR from 0.398 to 0.401. The model appears to plateau around AUC-PR ~0.40
  with these features and this dataset.

- **Ranking correlations are modest but real:** mean Spearman rho ~0.32, Kendall tau ~0.22,
  Pearson r ~0.25 between predicted pair scores and true pKD. The model captures
  directional potency trends but is not a precise pKD predictor.

Report: `models/pairwise_classifier/report.html`

## Performance commentary

The binary classifier reaches AUC-PR ~0.43 and the pairwise classifier ~0.40,
suggesting we are near the ceiling for hand-crafted fingerprint features (MACCS +
RDKit descriptors) on this dataset. The SPR data has inherent noise (batch effects
across 14 campaigns, within-compound pKD standard deviations documented in the SAR
diagnostics), which limits how much signal any model can extract.

A key finding is that Enamine REAL diversity decoys, while boosting AUC-ROC, actually
hurt precision-recall performance. The SPR-only binary classifier produces better
calibrated predictions with less fold variance. This makes sense: the decoys are
structurally dissimilar to the SPR compounds and provide an easy discrimination signal
that doesn't generalize to the harder active-vs-inactive boundary within the chemical
series under study.

The binary classifier is the more practical tool for screening: it can rapidly triage
large compound sets with ~48% precision and ~51% recall at its optimal threshold. The
pairwise classifier offers a complementary view: ranking compounds by relative potency
rather than applying a hard activity cutoff. For the hackathon submission, the binary
classifier is the primary scoring tool, with pairwise rankings available as a
secondary signal.

Neither model has been validated on truly external compounds. The Butina scaffold split
provides some protection against memorization, but performance on structurally novel
scaffolds from the onepot library is unknown.
