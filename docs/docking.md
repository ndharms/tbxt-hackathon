# Docking Pipeline

AutoDock Vina docking against TBXT binding pockets. Three modes:
- `dock_single()` -- detailed report with interaction analysis and 3D visualization
- `dock_batch()` -- fast scoring of many SMILES against a single pocket
- `dock_screen()` -- multi-pocket screen using PocketAssigner to route compounds to
  the right pocket (A, C, or D), then batch-docking each group

## Environment setup

Vina requires conda (pip build fails on Apple Silicon):

```bash
conda create -n tbxt-dock python=3.11
conda activate tbxt-dock
conda install -c conda-forge vina=1.2.6
pip install -e .        # project deps (meeko, rdkit, py3Dmol, etc.)
```

All subsequent commands assume `conda activate tbxt-dock`.

## Quick start

```python
from tbxt_hackathon.docking import dock_single, dock_batch, dock_screen

POCKET = "data/structures/TGT_TBXT_A_pocket.pdb"

# --- Single compound (detailed report + 3D viz) ---
report = dock_single("CC(=O)Nc1ccc(F)c(F)c1", POCKET)
print(report.summary())
report.show_3d()          # interactive 3D in Jupyter

# --- Batch scoring against one pocket ---
import pandas as pd
smiles_list = pd.read_csv("my_compounds.csv")["smiles"].tolist()
df = dock_batch(smiles_list, POCKET)

# --- Multi-pocket screen (~6K compounds) ---
df = dock_screen(smiles_list, checkpoint_dir="checkpoints/dock")
df.to_csv("docking_scores.csv", index=False)
```

## Single-compound mode: `dock_single`

Returns a `DockingReport` with full interaction analysis and 3D visualization.

```python
report = dock_single(
    smiles,                 # input SMILES
    pocket_pdb,             # PDB with protein ATOM + reference ligand HETATM
    n_conformers=3,         # independent conformer/docking runs
    exhaustiveness=32,      # Vina search depth (higher = slower, more thorough)
    n_poses=9,              # poses kept per conformer run
)
```

**Report contents:**

| Attribute            | Description                                      |
|----------------------|--------------------------------------------------|
| `report.score`       | Best (minimum) Vina score across all conformers   |
| `report.ligand_efficiency` | -score / heavy_atom_count                  |
| `report.n_heavy_atoms` | Heavy atom count                               |
| `report.mw`          | Molecular weight                                  |
| `report.poses`       | List of `PoseResult` (all conformers, all poses)  |
| `report.best_pose`   | `PoseResult` with the lowest score                |
| `report.interactions`| Shortcut to `best_pose.interactions`              |
| `report.summary()`   | Text summary with score, interactions, etc.       |
| `report.show_3d()`   | 3D view of best pose in Jupyter (py3Dmol)         |
| `report.show_3d(pose_idx=N)` | View any specific pose                   |
| `report.show_all_poses()`    | Overlay all poses (best = green, rest = gray) |

Each `PoseResult` contains:
- `score`, `inter_energy`, `intra_energy` (kcal/mol)
- `conformer_idx`, `pose_idx`
- `interactions`: list of detected protein-ligand interactions
- `pose_sdf`: SDF block of the docked ligand (for downstream use)

Interaction types detected: H-bonds, pi-stacking (face-to-face and
edge-to-face), hydrophobic contacts, salt bridges.

## Batch mode: `dock_batch`

Scores many compounds and returns a DataFrame. Uses lower exhaustiveness
and fewer poses for speed.

```python
df = dock_batch(
    smiles_list,            # list of SMILES strings
    pocket_pdb,             # same pocket PDB
    n_conformers=3,         # conformer runs per compound
    exhaustiveness=8,       # lower than single mode for speed
    n_poses=1,              # only need top pose for scoring
)
```

**Output columns:**

| Column              | Description                                          |
|---------------------|------------------------------------------------------|
| `smiles`            | Input SMILES                                          |
| `score`             | Best (min) Vina score across conformers (kcal/mol)    |
| `ligand_efficiency` | -score / heavy_atom_count                             |
| `n_heavy_atoms`     | Heavy atom count                                      |
| `mw`                | Molecular weight                                      |
| `score_std`         | Std dev of scores across conformer runs               |
| `status`            | "ok" or error message                                 |

Compounds that fail (bad SMILES, embedding failure, etc.) get `NaN` scores
and the error in `status` rather than crashing the batch.

## Multi-pocket screen: `dock_screen`

Routes each compound to its best-matching pocket via PocketAssigner, then
batch-docks each group against the correct pocket PDB. This is the
recommended entry point for screening large compound sets.

```python
df = dock_screen(
    smiles_list,                        # list of SMILES
    fragment_csv="data/structures/sgc_fragments.csv",
    n_conformers=3,                     # conformer runs per compound
    exhaustiveness=3,                   # Vina exhaustiveness
    dock_unassigned=True,               # dock unassigned compounds against pocket A
    checkpoint_dir="checkpoints/dock",  # per-pocket parquet checkpoints
)
```

**Available pockets:** A, C, D (pocket B excluded due to poor pose reliability).

**Pocket assignment:** uses PocketAssigner (fragment substructure matching +
ECFP4 Tanimoto similarity). Compounds that don't match any pocket are docked
against pocket A by default (`dock_unassigned=True, default_pocket="A"`).

**Output columns:**

| Column              | Description                                          |
|---------------------|------------------------------------------------------|
| `smiles`            | Input SMILES                                          |
| `pocket`            | Assigned pocket (A, C, or D)                          |
| `score`             | Best (min) Vina score across conformers (kcal/mol)    |
| `ligand_efficiency` | -score / heavy_atom_count                             |
| `n_heavy_atoms`     | Heavy atom count                                      |
| `mw`                | Molecular weight                                      |
| `score_std`         | Std dev of scores across conformer runs               |
| `pocket_tanimoto`   | Max Tanimoto to fragments in the assigned pocket      |
| `pocket_substruct`  | Whether the compound contains a pocket fragment       |
| `status`            | "ok" or error message                                 |

**Checkpointing:** when `checkpoint_dir` is set, each pocket group writes
a parquet file on completion (e.g. `dock_A.parquet`). If the file exists on
a subsequent run, that pocket is loaded from disk instead of re-docked. This
allows restarting interrupted runs without losing progress.

**Runtime estimate:** ~7 sec/compound at exhaustiveness=3, n_conformers=3.
For 6K compounds routed to one pocket each, expect ~12 hours.

**Two-stage workflow:**

```python
# Stage 1: fast screen
df = dock_screen(smiles_list, exhaustiveness=3, checkpoint_dir="ckpt/fast")

# Stage 2: re-dock top hits with higher accuracy
top = df.nsmallest(50, "score")
for pocket in top["pocket"].unique():
    pocket_smiles = top.loc[top["pocket"] == pocket, "smiles"].tolist()
    df_refined = dock_batch(
        pocket_smiles,
        f"data/structures/TGT_TBXT_{pocket}_pocket.pdb",
        n_conformers=3,
        exhaustiveness=32,
    )
```

## Pocket PDB format

The pocket PDB must contain:
- Protein coordinates as `ATOM` records (used as the rigid receptor)
- A reference ligand as `HETATM` records (used only to define the search
  box center and size; stripped before docking)

The search box is set to the ligand centroid with 10 A padding in each
dimension.

## How it works

1. **Receptor prep**: protein ATOM records are converted to PDBQT with AD4
   atom types (no OpenBabel or ADFR Suite needed). The reference ligand
   defines the search box and is then discarded.
2. **Ligand prep**: SMILES -> RDKit 3D conformer (ETKDGv3 + MMFF) -> Meeko
   PDBQT. Each conformer run uses a different random seed.
3. **Docking**: AutoDock Vina scores and optimizes each ligand in the
   search box. Returns poses ranked by score.
4. **Coordinate mapping**: docked coordinates are mapped back to the RDKit
   molecule using Meeko's atom index map, preserving bond connectivity.
5. **Interaction detection**: distance-based detection of H-bonds (<3.5 A),
   pi-stacking (<5.5 A), hydrophobic contacts (<4.0 A, C-C only), and
   salt bridges (<4.0 A).
6. **Visualization**: py3Dmol renders the protein (cartoon), ligand
   (ball-and-stick), interacting residues (white sticks with labels),
   and interaction lines (color-coded dashed cylinders).

## Interaction color key (3D view)

| Color  | Interaction   |
|--------|---------------|
| Gold   | H-bond        |
| Green  | Pi-stacking   |
| Gray   | Hydrophobic   |
| Red    | Salt bridge   |
