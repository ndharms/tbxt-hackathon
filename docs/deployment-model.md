# Deployment model: `xgb_maccs_fp_pocket_phys_deploy`

**Status:** frozen as the project's deployment choice for scoring onepot's
3.4B compound library.

## What it is

A 6-booster XGBoost ensemble trained with leave-one-fold-out CV on all
708 filtered-label compounds. Features are MACCS keys + pocket similarity
scores + physchem descriptors (183 columns total).

Each fold in {0, 1, 2, 3, 4, 5} is held out in turn; one booster is fit
on the other five. Every training compound appears in 5 of the 6 training
sets, and each has exactly one out-of-fold (OOF) prediction from the
booster that did not see it. For **novel** compounds at inference time,
all 6 boosters are averaged.

| Property | Value |
|---|---|
| Task | Binary: P(pKD > 5) |
| Training compounds | 708 (87 positives, 621 negatives; 12.3% prevalence) |
| Features | 183 total: MACCS (167) + pocket (8) + physchem (8) |
| Boosters | 6 (leave-one-fold-out) |
| Per-booster hyperparams | n_estimators=61, max_depth=6, lr=0.05, subsample=0.9, colsample_bytree=0.6 |
| Class imbalance handling | `scale_pos_weight = n_neg / n_pos` per training fold |
| Seed | 0 |
| Overall OOF AUROC | 0.764 |
| Overall OOF AUPRC | 0.337 (random baseline: 0.123) |

The featurization pipeline is deliberately simple: every input becomes a
183-dim float vector through a deterministic procedure. No SMILES
normalization beyond RDKit canonicalization, no data augmentation, no
calibration layer.

## Why this model

Full story in [try4 README](classification-models-try4-rjg/README.md); short
version:

1. **Try1 (top-quartile label)** was too noisy to produce a rank-able
   model. AUROC on the OOD holdout was 0.48, no better than random.
2. **Try2 (filter pKD 3-5 gray zone)** fixed the label. Holdout had only
   1 positive out of 62 so ranking metrics were unreliable, but the XGB
   family clearly ranked that one positive at #2/62 (vs CheMeleon's
   28-33/62).
3. **Try3 (holdout = fold 3 with 29 positives)** gave the first
   well-powered evaluation: xgb_no_val Morgan+physchem scored AUROC 0.786.
4. **Try4 (ablate fingerprint type x feature set, 9 variants)** showed
   that MACCS + pocket + physchem was strictly better:
   - Holdout AUROC **0.844** (vs Morgan+physchem's 0.781)
   - Holdout AUPRC **0.523** (vs 0.430)
   - Mean fractional rank of true positives **0.208** (median 0.124)
   - Top-30 precision **17/30** true binders (3.6x random)
   - Feature importance: 86% MACCS, 6% pocket, 8% physchem — the only
     feature-set variant where the non-FP columns actually contribute.

The deployment recipe differs from try4 in one respect: instead of
training on 5 folds and evaluating on fold 3 (what gave us the 0.844
holdout AUROC), we train 6 boosters with each fold held out in turn.
This makes no fold special, lets every training compound contribute to
5/6 boosters, and reduces variance on predictions for novel compounds
(the actual screening task). **The tradeoff: we can no longer cite a
0.844 "holdout AUROC" because fold 3 is now in 5/6 training sets.**
Overall OOF AUROC here (0.764) is the closest comparable number.

## How to use it

### From Python

```python
from tbxt_hackathon.deployment import DeploymentModel

model = DeploymentModel.load()  # loads all 6 boosters
probs = model.predict([
    "CCOC(=O)c1ccc(O)cc1",
    "Nc1ccc(Cl)cc1",
])
# probs: (2,) array of P(binder) in [0, 1]

# If you want the per-booster matrix to compute variance / uncertainty:
per_model = model.predict(smiles_list, return_per_model=True)
# per_model: (n_compounds, 6) matrix
```

### From the CLI

```bash
# CSV input
uv run python scripts/deployment-model/02_score_smiles.py \
    --input my_compounds.csv \
    --smiles-col smiles \
    --id-col compound_id \
    --output scored.csv

# SDF input
uv run python scripts/deployment-model/02_score_smiles.py \
    --input library.sdf \
    --sdf \
    --output library_scored.csv

# Keep only the top-N by ensemble mean
uv run python scripts/deployment-model/02_score_smiles.py \
    --input onepot_sample.csv \
    --smiles-col smiles --id-col id \
    --output onepot_scored_top1k.csv \
    --top-n 1000
```

Output columns: `compound_id`, `canonical_smiles`, `p_binder_ensemble_mean`,
`p_binder_fold_{0..5}`, `rank` (1 = best).

## Retraining

To retrain from scratch (only necessary if training data changes):

```bash
# 1. rebuild filtered labels + pocket features (reuse try4 artifacts
#    if they're still current)
uv run python scripts/classification-models-try4-rjg/01_make_filtered_labels.py
uv run python scripts/classification-models-try4-rjg/02_make_pocket_features.py

# 2. retrain the 6-booster ensemble
uv run python scripts/deployment-model/01_train_ensemble.py
```

The training script writes to `data/deployment-model/` (6 boosters +
metadata + OOF predictions). Seed is fixed at 0 so reruns with the
same input are bitwise-reproducible.

## Feature pipeline details

The `DEPLOYMENT_FEATURE_SPEC` dict in
[`src/tbxt_hackathon/deployment.py`](../src/tbxt_hackathon/deployment.py)
is the single source of truth for the featurization recipe and is
written to `data/deployment-model/feature_spec.json` at training time.
Anyone running inference outside this repo should verify they're using
the same pipeline.

Column ordering in the 183-dim vector:

| Columns | Slice | Source |
|---|---|---|
| 0 - 166 (167 cols) | MACCS keys | `rdkit.Chem.MACCSkeys.GenMACCSKeys` |
| 167 - 174 (8 cols) | Pocket scores | `tbxt_hackathon.pocket_assigner.PocketAssigner.score`; for each pocket P in {A, B, C, D}: `pocket_P_tanimoto` (max ECFP4 Tanimoto to pocket fragments), `pocket_P_substruct` (1.0 if any pocket fragment is a subgraph of the query, else 0.0) |
| 175 - 182 (8 cols) | Physchem | RDKit `Descriptors.MolWt`, `Crippen.MolLogP`, `Lipinski.{NumHDonors, NumHAcceptors, NumRotatableBonds}`, `mol.GetNumHeavyAtoms()`, `rdMolDescriptors.CalcNumRings`, `Descriptors.TPSA` |

All columns are cast to float32 before concatenation. No null handling
is done during inference — if any SMILES fails to parse it is dropped
from the batch before featurization.

## Caveats

- **Prevalence-matching.** The training set is 12.3% positives (87/708)
  after the pKD filter. Scoring compounds from a distribution with very
  different positive rate (e.g. onepot 3.4B has unknown but presumably
  much lower prevalence) will produce probabilities that aren't
  calibrated on the target population. **Use `p_binder_ensemble_mean`
  as a ranking score, not a probability.**
- **Fold 0 is a bit of an outlier.** The leave-fold-0-out booster has
  per-fold OOF AUROC 0.473, worse than random. Fold 0 (112 compounds,
  6 positives) is structurally distinct in ways the other folds don't
  sample well. The other 5 boosters compensate in the ensemble mean.
- **Ensemble probabilities are typically lower than single-model
  predictions on training compounds.** This is because 5 of 6 boosters
  have seen each training compound. For out-of-sample SMILES, all 6
  are equally naive — the mean should be taken at face value.
- **The model is a ranker, not an uncertainty estimate.** For fragment
  selection we pair this with structure-based sanity checking (docking
  against pocket A/C) downstream.

## Artifacts

```
data/deployment-model/
|-- xgb_deploy_fold_{0..5}.ubj       # the 6 boosters (tracked via git-lfs)
|-- feature_spec.json                # 183-column recipe
|-- fold_assignments_used.csv        # frozen training fold assignment
|-- training_predictions.csv         # OOF preds per training compound
`-- metrics.json                     # per-fold and overall OOF metrics
```

Per-booster file size: 91-128 KB; total ensemble footprint < 1 MB.
