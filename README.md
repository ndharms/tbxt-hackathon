# TBXT Hit Identification Hackathon

Computational workflows for identifying small-molecule binders of human TBXT (Brachyury),
a T-box transcription factor and key dependency of chordoma.

**Event:** May 9, 2026 -- Boston, Pillar VC. Hosted by muni in collaboration with Rowan and onepot.

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

Two complementary XGBoost approaches, both using Butina-clustered scaffold-aware 5-fold
CV with Optuna HPO (AUC-PR objective). See [`docs/modeling-notes.md`](docs/modeling-notes.md)
for full ablation results, findings, and performance commentary.

- **Binary classifier** (`scripts/build_xgb_classifier.py`) -- active/inactive classification
  using MACCS + RDKit descriptors (384 features). Best config: SPR-only negatives,
  AUC-PR 0.43, F1 0.49 at optimal threshold.
- **Pairwise classifier** (`scripts/build_pairwise_classifier.py`) -- relative potency
  ranking using difference fingerprints. AUC-PR 0.40, Spearman rho ~0.32 with true pKD.


## Setup

```bash
uv sync
uv run jupyter lab
```

## Repo structure

```
scripts/                    Standalone modeling and utility scripts
  build_xgb_classifier.py    Binary active/inactive classifier
  build_pairwise_classifier.py  Pairwise potency comparison classifier
models/
  xgb_classifier/           Binary classifier outputs (models, params, report)
  pairwise_classifier/       Pairwise classifier outputs (models, params, report)
notebooks/                  Jupyter notebooks (numbered in execution order)
docs/                       Write-ups and figures
data/
  zenodo/                   Raw merged SPR data
  processed/                Cleaned modeling-ready CSVs
  structures/               PDB files (for future docking)
src/tbxt_hackathon/         Shared Python utilities
```
