# TBXT Hit Identification Hackathon

Computational workflows for identifying small-molecule binders of human TBXT (Brachyury),
a T-box transcription factor and key dependency of chordoma.

**Event:** May 9, 2026 -- Boston, Pillar VC. Hosted by muni in collaboration with Rowan and onepot.

Got 3rd place! Compound performance TBD.

This work was drafted and executed with a lot of AI assistance, and in order to get it done in the limited time window,
review of the AI outputs was kept high level. There may be methodological or epistemic errors or inconsistencies.

## Goal

Design and rank 4 non-covalent small-molecule ligands for the TBXT DNA-binding domain
(residues ~42-219), sourced from [onepot's 3.4B compound library](https://www.onepot.ai/).
Winning compounds will be synthesized by onepot and tested via SPR against full-length
TBXT (G177D) by the Chordoma Foundation.

## Approach

Ligand-based virtual screening using ~1,500 compounds with SPR binding affinity data (pKD)
from prior screening campaigns. Structure-based validation (docking against PDB 6F59 and
SGC fragment co-structures) may be added as a follow-up step.

## Data

The primary dataset is 2,143 SPR binding affinity measurements across 14 experimental
batches (Oct 2020 -- Jan 2023), merged from the
[Zenodo dataset](https://zenodo.org/) released alongside
[Newman et al. (2025) *Nature Communications*](https://doi.org/10.1038/s41467-025-56213-1).
After cleaning, this yields 1,545 unique compounds with validated SMILES and pKD values.

See `notebooks/01-data-prep.ipynb` for the full cleaning pipeline and QC visuals.


## Modeling

The deployment model is a 6-booster XGBoost ensemble trained on MACCS keys
+ pocket similarity + physchem descriptors (183 features). See
[`docs/deployment-model.md`](docs/deployment-model.md) for the full recipe,
metrics, and inference instructions.

Quick inference:

```python
from tbxt_hackathon.deployment import DeploymentModel
model = DeploymentModel.load()
probs = model.predict(["CCOC(=O)c1ccc(O)cc1", "Nc1ccc(Cl)cc1"])
```

Or from the CLI:

```bash
uv run python scripts/deployment-model/02_score_smiles.py \
    --input compounds.csv --smiles-col smiles --id-col id \
    --output scored.csv
```

### Model development history

Four iterations got us to the current deployment model. Each has its own
directory under `docs/` with a detailed README:

- [`classification-models-try1-rjg`](docs/classification-models-try1-rjg/README.md)
  — four ensembles on top-quartile pKD label; all tied with random on OOD
  holdout. Label was too noisy to learn from.
- [`classification-models-try2-rjg`](docs/classification-models-try2-rjg/README.md)
  — dropped pKD 3-5 gray zone. +0.06 to +0.12 OOF AUROC; XGB cleanly
  beat CheMeleon on ranking. Holdout too small (1 positive) for
  aggregate metrics.
- [`classification-models-try3-rjg`](docs/classification-models-try3-rjg/README.md)
  — switched holdout to fold 3 (29 positives). xgb_no_val (Morgan +
  physchem) AUROC 0.786.
- [`classification-models-try4-rjg`](docs/classification-models-try4-rjg/README.md)
  — ablated fingerprint type × feature set (9 variants).
  **maccs_fp_pocket_phys** won with holdout AUROC 0.844, AUPRC 0.523.
  This is the recipe baked into the deployment model.

Separate documentation tracks the older approaches that were superseded:

- **Binary classifier** (`scripts/build_xgb_classifier.py`) -- early XGBoost
  active/inactive classifier using MACCS + RDKit descriptors. Superseded by
  the try4 MACCS ablation.
- **Pairwise classifier** (`scripts/build_pairwise_classifier.py`) -- relative
  potency ranking using difference fingerprints. AUC-PR 0.40, Spearman rho
  ~0.32 with true pKD. Not integrated into the current deployment pipeline.

Older ablation notes live at [`docs/modeling-notes.md`](docs/modeling-notes.md).


## Setup

```bash
uv sync
uv run jupyter lab
```

## Repo structure

```
scripts/                    Standalone modeling and utility scripts
  deployment-model/          Final model: train + score CLIs
  classification-models-try{1..4}-rjg/   Iterative development
  build_xgb_classifier.py    Early binary classifier (superseded)
  build_pairwise_classifier.py  Pairwise classifier (superseded)
models/                     Early model outputs (superseded)
  xgb_classifier/            Binary classifier params + report
  pairwise_classifier/       Pairwise classifier params + report
notebooks/                  Jupyter notebooks (numbered in execution order)
docs/                       Write-ups and figures
  deployment-model.md        Deployment model recipe, metrics, usage
  classification-models-try{1..4}-rjg/  Per-try READMEs + plots
data/
  deployment-model/          Final 6-booster ensemble + metadata
  classification-models-try{1..4}-rjg/  Per-try artifacts (models, preds)
  zenodo/                    Raw merged SPR data
  processed/                 Cleaned modeling-ready CSVs
  structures/                PDB files + SGC fragments (pocket features)
src/tbxt_hackathon/         Shared Python utilities
  deployment.py              Inference pipeline for the deployment model
  fingerprints.py            Morgan / RDKit-path / MACCS builders
  xgb_baseline.py            XGBoost training helpers (class + reg)
  pocket_assigner.py         SGC fragment similarity scoring
  chemeleon_transfer.py      CheMeleon encoder wrapper (not used in deployment)
```
