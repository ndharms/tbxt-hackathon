# RMSD Calculation Script

Calculate Root Mean Square Deviation (RMSD) between two PDB structures.

## Usage

Basic usage:
```bash
uv run python scripts/calculate_rmsd.py ref.pdb target.pdb
```

Compare only backbone atoms (recommended for overall structure comparison):
```bash
uv run python scripts/calculate_rmsd.py ref.pdb target.pdb --align-backbone
```

Compare specific chain:
```bash
uv run python scripts/calculate_rmsd.py ref.pdb target.pdb --chain A
```

Compare TBXT DNA-binding domain (residues 42-219):
```bash
uv run python scripts/calculate_rmsd.py ref.pdb target.pdb --residue-range 42 219 --align-backbone
```

## Options

- `--chain CHAIN_ID`: Specify which chain to compare (default: first chain)
- `--align-backbone`: Only align backbone atoms (N, CA, C, O) - recommended for structure comparison
- `--residue-range START END`: Only compare atoms in specified residue range

## Output

The script prints:
- RMSD value in Ångströms
- Number of atoms used for alignment
- Alignment method (all atoms vs backbone only)
- Residue range (if specified)

## Use in Python

```python
from pathlib import Path
from scripts.calculate_rmsd import calculate_rmsd

result = calculate_rmsd(
    ref_pdb=Path("data/structures/6f59.pdb"),
    target_pdb=Path("data/structures/model.pdb"),
    align_backbone=True,
    residue_range=(42, 219)  # TBXT DBD
)

print(f"RMSD: {result['rmsd']:.2f} Å")
print(f"Aligned {result['n_atoms']} atoms")
```
