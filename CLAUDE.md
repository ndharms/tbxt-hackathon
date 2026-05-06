# TBXT Hit Identification Hackathon

## What this is
Computational playground for the TBXT Hit Identification Hackathon (May 9 2026, Boston).
Goal: submit 4 ranked non-covalent small-molecule SMILES that credibly bind human TBXT (Brachyury),
sourced from onepot's 3.4B compound library.

## Target
- Human TBXT (Brachyury), T-box transcription factor
- Domain of interest: DNA-binding domain (DBD), residues ~42–219
- PDB 6F59 (TBXT bound to DNA); SGC fragment co-structures available
- UniProt O15178

## Strategy
Ligand-based virtual screening as primary approach. Structure-based validation (docking)
may be added later as a separate task.

## Data
- `data/zenodo/tbxt_spr_merged.csv` — 2143 SPR binding affinity records (pKD) from 14 experimental
  batches. See `notebooks/01-data-prep.ipynb` for schema, cleaning pipeline, and QC visuals.
- `data/zenodo/uploads/` — raw password-protected Excel files from HD Biosciences (password: HDB)
- `data/Naar_SMILES.xlsx` — 135 previously screened compound SMILES
- `data/structures/` — PDB files (empty; for future docking work)

## Setup
```bash
uv sync
uv run jupyter lab
```

## Design constraints (from Chordoma Foundation)
- Non-covalent, small molecules, within onepot library
- LogP <= 6, HBD <= 6, HBA <= 12, MW <= 600
- Ideal: 10–30 heavy atoms, HBD+HBA <= 11, cLogP < 5, <5 ring systems, <=2 fused rings
- Avoid: acid halides, aldehydes, diazo, imines, >2 fused benzene rings

## Conventions
- SMILES are never silently normalized; preserve exactly as received from source
- pKD = -log10(KD_M); higher = more potent
- All data operations should include shape assertions
- Use RDKit for all cheminformatics; pandas for tabular data
