"""Query OnePot for 10k nearest neighbors per fragment, then filter.

Queries 10,000 neighbors per fragment (all pockets), deduplicates to
best-pocket assignment, then applies the full property + substructure
filter cascade.

Outputs:
  - data/onepot_10k_all_matches.csv      (all pocket matches preserved)
  - data/onepot_10k_assigned.csv         (one row per compound, best-pocket)
  - data/onepot_10k_filtered.csv         (after all filters)

Usage:
    export ONEPOT_API_KEY="your-key-here"
    uv run python scripts/query_and_filter_10k.py
    uv run python scripts/query_and_filter_10k.py --filter-only  # skip query, reuse existing assigned csv
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, FilterCatalog, rdMolDescriptors

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAGMENTS_CSV = REPO_ROOT / "data" / "structures" / "sgc_fragments.csv"
OUTPUT_ALL = REPO_ROOT / "data" / "onepot_10k_all_matches.csv"
OUTPUT_ASSIGNED = REPO_ROOT / "data" / "onepot_10k_assigned.csv"
OUTPUT_FILTERED = REPO_ROOT / "data" / "onepot_10k_filtered.csv"

MAX_RESULTS = 10_000
BATCH_SIZE = 5

# ── Property filter constants ──────────────────────────────────────────
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

_PAINS_CATALOG = FilterCatalog.FilterCatalog(
    FilterCatalog.FilterCatalogParams(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
)


# ── Query helpers ──────────────────────────────────────────────────────

def load_fragments(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    assert df.shape[0] > 0, f"No rows in {csv_path}"
    assert {"pocket", "pdb_id", "ligand_id", "smiles"}.issubset(df.columns)
    df = df.drop_duplicates(subset=["smiles"])
    print(f"Loaded {len(df)} unique fragments across pockets: "
          f"{dict(df['pocket'].value_counts().sort_index())}")
    return df


def query_onepot(fragments: pd.DataFrame) -> pd.DataFrame:
    from onepot import Client

    api_key = os.environ.get("ONEPOT_API_KEY")
    if not api_key:
        print("ERROR: Set ONEPOT_API_KEY environment variable.")
        sys.exit(1)

    client = Client(api_key=api_key)
    all_rows: list[dict] = []
    total_credits = 0

    for pocket, group in fragments.groupby("pocket"):
        smiles_list = group["smiles"].tolist()
        print(f"\nPocket {pocket}: {len(smiles_list)} fragments, max_results={MAX_RESULTS}")

        for batch_start in range(0, len(smiles_list), BATCH_SIZE):
            batch = smiles_list[batch_start : batch_start + BATCH_SIZE]
            batch_num = batch_start // BATCH_SIZE + 1
            total_batches = (len(smiles_list) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"  Batch {batch_num}/{total_batches}: {len(batch)} queries...")

            t0 = time.time()
            resp = client.search(smiles_list=batch, max_results=MAX_RESULTS)
            elapsed = time.time() - t0

            credits_used = resp.get("credits_used", 0)
            credits_remaining = resp.get("credits_remaining", "?")
            total_credits += credits_used
            print(f"    {elapsed:.1f}s | credits: {credits_used} | remaining: {credits_remaining}")

            for query_obj in resp.get("queries", []):
                query_smi = query_obj.get("query_smiles", "")
                frag_row = group[group["smiles"] == query_smi]
                if frag_row.empty:
                    print(f"    WARNING: no match for {query_smi[:60]}")
                    continue
                frag_info = frag_row.iloc[0]
                for r in query_obj.get("results", []):
                    all_rows.append({
                        "pocket": frag_info["pocket"],
                        "query_pdb": frag_info["pdb_id"],
                        "query_ligand": frag_info["ligand_id"],
                        "query_smiles": query_smi,
                        "hit_smiles": r.get("smiles", ""),
                        "hit_inchikey": r.get("inchikey", ""),
                        "similarity": r.get("similarity"),
                        "price_usd": r.get("price_usd"),
                        "supplier_risk": r.get("supplier_risk", ""),
                        "chemistry_risk": r.get("chemistry_risk", ""),
                    })

    print(f"\nTotal credits used: {total_credits}")
    df = pd.DataFrame(all_rows)
    print(f"Raw results: {len(df):,} rows")
    return df


def assign_best_pocket(df_all: pd.DataFrame) -> pd.DataFrame:
    all_pocket_labels = (
        df_all
        .groupby("hit_smiles")["pocket"]
        .apply(lambda x: ",".join(sorted(x.unique())))
        .rename("all_pockets")
    )
    df_assigned = (
        df_all
        .sort_values("similarity", ascending=False)
        .drop_duplicates(subset=["hit_smiles"], keep="first")
        .sort_values(["pocket", "similarity"], ascending=[True, False])
        .reset_index(drop=True)
    )
    df_assigned = df_assigned.merge(all_pocket_labels, on="hit_smiles", how="left")
    return df_assigned


# ── Filter helpers ─────────────────────────────────────────────────────

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
    print("Computing RDKit descriptors...")
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


# ── Main ───────────────────────────────────────────────────────────────

def main() -> int:
    filter_only = "--filter-only" in sys.argv

    if filter_only and OUTPUT_ASSIGNED.exists():
        print(f"Filter-only mode: loading {OUTPUT_ASSIGNED.name}")
        df_assigned = pd.read_csv(OUTPUT_ASSIGNED)
    else:
        fragments = load_fragments(FRAGMENTS_CSV)

        df_all = query_onepot(fragments)
        df_all.to_csv(OUTPUT_ALL, index=False)
        print(f"All matches saved to {OUTPUT_ALL.relative_to(REPO_ROOT)}")

        df_assigned = assign_best_pocket(df_all)
        print(f"After dedup/best-pocket: {len(df_assigned):,} unique compounds")
        print("\nPer-pocket counts (assigned):")
        for pocket, group in df_assigned.groupby("pocket"):
            sims = group["similarity"]
            print(f"  {pocket}: {len(group):,}  (sim: {sims.min():.3f}–{sims.max():.3f})")

        df_assigned.to_csv(OUTPUT_ASSIGNED, index=False)
        print(f"Assigned results saved to {OUTPUT_ASSIGNED.relative_to(REPO_ROOT)}")

    assert len(df_assigned) > 0, "No compounds to filter"
    print(f"\n{'='*52}")
    print(f"Filtering {len(df_assigned):,} compounds")
    print(f"{'='*52}")

    df_assigned = compute_descriptors(df_assigned)
    df_filtered = apply_filters(df_assigned)

    print(f"\nPer-pocket counts after filtering:")
    for pocket, group in df_filtered.groupby("pocket"):
        print(f"  {pocket}: {len(group):,}")

    df_filtered.to_csv(OUTPUT_FILTERED, index=False)
    print(f"\nSaved {len(df_filtered):,} compounds to {OUTPUT_FILTERED.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
