"""Filter onepot nearest-neighbor compounds by physicochemical property ranges.

Reads onepot_neighbors_assigned.csv, computes RDKit descriptors, applies filters,
and writes the passing compounds to onepot_neighbors_filtered.csv.

Usage:
    uv run python scripts/filter_onepot_properties.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, FilterCatalog, rdMolDescriptors

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT = REPO_ROOT / "data" / "onepot_neighbors_assigned.csv"
OUTPUT = REPO_ROOT / "data" / "onepot_neighbors_filtered.csv"

MW_MIN = 300
MW_MAX = 600
LOGP_MAX = 6.0
HBD_MAX = 6
HBA_MAX = 12
HBD_HBA_SUM_MAX = 11
HEAVY_MIN = 10
HEAVY_MAX = 30
NUM_RINGS_MAX = 5
NUM_FUSED_RINGS_MAX = 2
FUSED_BENZENE_MAX = 2

EXCLUDE_SMARTS = {
    "acid_halide": "[CX3](=O)[F,Cl,Br,I]",
    "aldehyde": "[CX3H1](=O)",
    "diazo": "[#6]=[N+]=[N-]",
    "azide": "[NX1]=[NX2]=[NX1]",
    "imine": "[CX3]=[NX2]",
    "long_alkyl_chain": "[CH2][CH2][CH2][CH2][CH2][CH2]",
    "isocyanate": "[NX2]=C=O",
    "isothiocyanate": "[NX2]=C=S",
    "sulfonyl_halide": "[SX4](=O)(=O)[F,Cl,Br,I]",
    "epoxide": "[OX2r3]1[#6r3][#6r3]1",
    "anhydride": "[CX3](=O)[OX2][CX3](=O)",
}

_COMPILED_SMARTS = {name: Chem.MolFromSmarts(sma) for name, sma in EXCLUDE_SMARTS.items()}


def max_fused_ring_size(mol: Chem.Mol) -> int:
    ri = mol.GetRingInfo()
    if ri.NumRings() == 0:
        return 0
    bond_rings = [set(r) for r in ri.BondRings()]
    n = len(bond_rings)
    if n <= 1:
        return 1

    adjacency: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if bond_rings[i] & bond_rings[j]:
                adjacency[i].add(j)
                adjacency[j].add(i)

    visited = [False] * n
    max_component = 0
    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        size = 0
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            size += 1
            stack.extend(adjacency[node] - {node for node in range(n) if visited[node]})
        max_component = max(max_component, size)
    return max_component


_PAINS_CATALOG = FilterCatalog.FilterCatalog(
    FilterCatalog.FilterCatalogParams(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
)


def has_pains(mol: Chem.Mol) -> bool:
    return _PAINS_CATALOG.HasMatch(mol)


def matched_bad_motifs(mol: Chem.Mol) -> list[str]:
    return [name for name, pat in _COMPILED_SMARTS.items() if mol.HasSubstructMatch(pat)]


def max_fused_benzene_rings(mol: Chem.Mol) -> int:
    ri = mol.GetRingInfo()
    if ri.NumRings() == 0:
        return 0
    atom_rings = ri.AtomRings()
    bond_rings = ri.BondRings()

    benzene_idx = []
    for i, aring in enumerate(atom_rings):
        if len(aring) == 6 and all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in aring):
            benzene_idx.append(i)

    n = len(benzene_idx)
    if n <= 1:
        return n

    bsets = [set(bond_rings[i]) for i in benzene_idx]
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if bsets[i] & bsets[j]:
                adj[i].add(j)
                adj[j].add(i)

    visited = [False] * n
    largest = 0
    for start in range(n):
        if visited[start]:
            continue
        stack, size = [start], 0
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            size += 1
            stack.extend(nb for nb in adj[node] if not visited[nb])
        largest = max(largest, size)
    return largest


def compute_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    mols = df["hit_smiles"].apply(Chem.MolFromSmiles)
    valid = mols.notna()
    n_invalid = (~valid).sum()
    if n_invalid > 0:
        print(f"  {n_invalid} SMILES failed to parse — dropped")

    df = df[valid].copy()
    mols = mols[valid]

    df["mw"] = mols.apply(Descriptors.ExactMolWt)
    df["logp"] = mols.apply(Descriptors.MolLogP)
    df["hbd"] = mols.apply(rdMolDescriptors.CalcNumHBD)
    df["hba"] = mols.apply(rdMolDescriptors.CalcNumHBA)
    df["heavy_atoms"] = mols.apply(lambda m: m.GetNumHeavyAtoms())
    df["num_rings"] = mols.apply(rdMolDescriptors.CalcNumRings)
    df["max_fused_rings"] = mols.apply(max_fused_ring_size)
    df["pains"] = mols.apply(has_pains)
    df["bad_motifs"] = mols.apply(lambda m: ",".join(matched_bad_motifs(m)) or "")
    df["fused_benzene_rings"] = mols.apply(max_fused_benzene_rings)
    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    n_start = len(df)
    filters = [
        ("MW >= {MW_MIN}", df["mw"] >= MW_MIN),
        ("MW <= {MW_MAX}", df["mw"] <= MW_MAX),
        ("LogP <= {LOGP_MAX}", df["logp"] <= LOGP_MAX),
        ("HBD <= {HBD_MAX}", df["hbd"] <= HBD_MAX),
        ("HBA <= {HBA_MAX}", df["hba"] <= HBA_MAX),
        ("HBD+HBA <= {HBD_HBA_SUM_MAX}", (df["hbd"] + df["hba"]) <= HBD_HBA_SUM_MAX),
        ("Heavy atoms >= {HEAVY_MIN}", df["heavy_atoms"] >= HEAVY_MIN),
        ("Heavy atoms <= {HEAVY_MAX}", df["heavy_atoms"] <= HEAVY_MAX),
        ("Rings <= {NUM_RINGS_MAX}", df["num_rings"] <= NUM_RINGS_MAX),
        ("Fused rings <= {NUM_FUSED_RINGS_MAX}", df["max_fused_rings"] <= NUM_FUSED_RINGS_MAX),
        ("No PAINS", ~df["pains"]),
        ("No bad motifs", df["bad_motifs"] == ""),
        (f"Fused benzene <= {FUSED_BENZENE_MAX}", df["fused_benzene_rings"] <= FUSED_BENZENE_MAX),
        ("Supplier risk = low", df["supplier_risk"] == "low"),
    ]

    cumulative_mask = pd.Series(True, index=df.index)
    print(f"\n{'Filter':<30} {'Fail':>8} {'Remaining':>10}")
    print("-" * 52)
    for label, mask in filters:
        fails = (~mask & cumulative_mask).sum()
        cumulative_mask &= mask
        remaining = cumulative_mask.sum()
        print(f"  {label:<28} {fails:>8,} {remaining:>10,}")

    df_out = df[cumulative_mask].copy()
    print(f"\n  Total: {n_start:,} → {len(df_out):,} ({len(df_out)/n_start:.1%} pass)")

    motif_hits = df["bad_motifs"][df["bad_motifs"] != ""].str.split(",").explode()
    if len(motif_hits) > 0:
        print("\n  Bad-motif breakdown (pre-filter counts):")
        for motif, count in motif_hits.value_counts().items():
            print(f"    {motif:<24} {count:>6,}")

    return df_out


def main() -> int:
    df = pd.read_csv(INPUT)
    assert len(df) > 0, f"Empty input: {INPUT}"
    print(f"Loaded {len(df):,} compounds from {INPUT.name}")

    print("Computing descriptors...")
    df = compute_descriptors(df)

    df_filtered = apply_filters(df)

    print(f"\nPer-pocket counts after filtering:")
    for pocket, group in df_filtered.groupby("pocket"):
        print(f"  {pocket}: {len(group):,}")

    df_filtered.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(df_filtered):,} compounds to {OUTPUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
