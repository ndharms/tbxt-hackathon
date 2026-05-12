"""
Fetch all TBXT TEP fragment SMILES from RCSB PDB and save with pocket assignments.

Pocket mapping (TEP site labels → Newman pocket labels → our working labels):
  - TEP Pocket A & A' → Newman A/A' → A_prime (2 submission slots)
  - TEP Pocket F → Newman D → F (1 submission slot, speculative)
  - TEP Pocket G → Newman C → G (1 submission slot)
  - TEP Pocket B → Newman B (subset) → DROPPED (bad Boltz pose)
  - TEP Pocket D → Newman B → DROPPED
  - TEP Pocket E → unclear → DROPPED
  - TEP Pocket C (5QRW loc2 only) → ambiguous → DROPPED

Also includes Newman-progressed thiazole series (7ZK2, 8A7N) not in original TEP.

Usage:
    uv run python scripts/fetch_fragment_smiles.py
"""

import time
from pathlib import Path
from typing import Dict, List

import requests
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]

# All fragment PDB entries from the TEP datasheet (Table 2) plus Newman 2025
# progressed compounds, grouped by our working pocket label.
#
# Format: {pocket_label: [(pdb_id, ligand_id_hint), ...]}
# ligand_id_hint is from the TEP — we use it to pick the right ligand from
# multi-ligand entries. If None, we take the first non-artifact ligand.
FRAGMENT_PDBS: Dict[str, List[tuple]] = {
    "A_prime": [
        # Crystal form 1 — TEP "Pocket A"
        ("5QRF", "F9000532"),
        ("5QRG", "XS115742"),
        ("5QRH", "FM001763"),
        ("5QRI", "F9000380"),
        ("5QRJ", "F9000536"),
        ("5QRK", "FM010104"),
        ("5QRL", "F9000949"),
        ("5QRN", "F9000950"),
        ("5QRO", "F9000403"),
        ("5QRP", "F9000414"),
        ("5QRQ", "FM002138"),
        ("5QRS", "F9000441"),
        ("5QRT", "FM010020"),
        ("5QRV", "FM002272"),
        ("5QRX", "F9000951"),
        ("5QRY", "FM010072"),
        ("5QRZ", "F9000591"),
        ("5QS2", "FM001886"),
        ("5QS3", "FM002333"),
        ("5QS4", "UB000200"),
        ("5QS5", "FM002032"),
        # Crystal form 2 — TEP "Pocket A'"
        ("5QS9", "FM002076"),
        ("5QSD", "FM002038"),
        # Newman 2025 progressed compounds (thiazole series)
        ("7ZK2", None),  # morpholino-thiazole
        ("8A7N", None),  # cyclopropylacetamide series
    ],
    "F": [
        # Crystal form 2 — TEP "Pocket F" = Newman Pocket D (Y88/D177)
        ("5QSA", "FM001580"),
        ("5QSC", "F9000560"),  # location 2 is pocket F
        ("5QSI", "FM001452"),
        ("5QSK", "FM002150"),
    ],
    "G": [
        # Crystal form 2 — TEP "Pocket G" = Newman Pocket C (E48/R54)
        ("5QS6", "FM010013"),
        ("5QS8", "F9000511"),
        ("5QSB", "XS022802"),
        ("5QSC", "F9000560"),  # location 1 is pocket G (same PDB, dual binding)
        ("5QSE", "F9000674"),
        ("5QSF", "DA000167"),
        ("5QSG", "F9000710"),
        ("5QT0", "XS092188"),
        ("5QSH", "F9000416"),
        ("5QSJ", "FM002214"),
    ],
}

# Common crystallization/buffer artifacts to exclude
ARTIFACT_LIGANDS = frozenset([
    "HOH", "SO4", "PO4", "GOL", "EDO", "ACT", "DMS", "MPD", "CD",
    "CL", "NA", "MG", "ZN", "CA", "K", "IOD", "BR", "FMT", "MN",
    "NI", "CO", "CU", "FE", "PEG", "PGE", "1PE", "P6G", "PE4",
    "TRS", "EPE", "MES", "CIT", "TAR", "SUC", "MLI", "IMD",
])


def fetch_ligand_smiles_graphql(pdb_id: str) -> Dict[str, str]:
    """Fetch ligand SMILES from RCSB PDB GraphQL API.

    Args:
        pdb_id: 4-character PDB identifier.

    Returns:
        Dict mapping ligand_id -> canonical SMILES string.

    Example:
        >>> fetch_ligand_smiles_graphql("5QS2")
        {'LV4': 'c1ccc(c(c1)NC(=S)N)OC(F)(F)F'}
    """
    url = "https://data.rcsb.org/graphql"

    query = """
    query getLigands($entryId: String!) {
      entry(entry_id: $entryId) {
        nonpolymer_entities {
          nonpolymer_comp {
            chem_comp {
              id
              name
              formula
            }
            rcsb_chem_comp_descriptor {
              SMILES
              SMILES_stereo
            }
          }
        }
      }
    }
    """

    variables = {"entryId": pdb_id}

    response = requests.post(
        url,
        json={"query": query, "variables": variables},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    ligands: Dict[str, str] = {}
    if "data" in data and data["data"]["entry"]:
        for entity in data["data"]["entry"].get("nonpolymer_entities", []):
            comp = entity.get("nonpolymer_comp", {})
            chem_comp = comp.get("chem_comp", {})
            desc = comp.get("rcsb_chem_comp_descriptor", {})

            lig_id = chem_comp.get("id")
            # Prefer stereo SMILES, fall back to canonical
            smiles = desc.get("SMILES_stereo") or desc.get("SMILES")

            if lig_id and smiles and lig_id.upper() not in ARTIFACT_LIGANDS:
                ligands[lig_id] = smiles

    return ligands


def fetch_all_fragments() -> list[dict]:
    """Fetch SMILES for all fragment PDBs across all pockets.

    Returns:
        List of dicts with keys: pocket, pdb_id, ligand_id, smiles
    """
    all_fragments: list[dict] = []
    seen_pdb_ligand: set[tuple[str, str]] = set()

    for pocket, entries in FRAGMENT_PDBS.items():
        logger.info(f"Processing pocket {pocket} ({len(entries)} PDB entries)...")

        for pdb_id, ligand_hint in entries:
            try:
                ligands = fetch_ligand_smiles_graphql(pdb_id)
            except Exception as e:
                logger.error(f"  Failed to fetch {pdb_id}: {e}")
                continue

            if not ligands:
                logger.warning(f"  {pdb_id}: no non-artifact ligands found")
                continue

            # Pick the relevant ligand
            if ligand_hint and ligand_hint in ligands:
                # Exact match on the TEP ligand ID
                selected = {ligand_hint: ligands[ligand_hint]}
            elif ligand_hint:
                # TEP ligand ID not found — take all non-artifacts
                logger.warning(
                    f"  {pdb_id}: hint '{ligand_hint}' not found in "
                    f"{list(ligands.keys())}; taking first"
                )
                first_key = next(iter(ligands))
                selected = {first_key: ligands[first_key]}
            else:
                # No hint (Newman progressed compounds) — take all
                selected = ligands

            for lig_id, smiles in selected.items():
                key = (pdb_id, lig_id)
                if key in seen_pdb_ligand:
                    continue
                seen_pdb_ligand.add(key)

                all_fragments.append({
                    "pocket": pocket,
                    "pdb_id": pdb_id,
                    "ligand_id": lig_id,
                    "smiles": smiles,
                })
                logger.info(f"  {pdb_id}/{lig_id} -> {smiles[:60]}")

            # Be polite to RCSB
            time.sleep(0.3)

    return all_fragments


def main() -> None:
    """Fetch all fragment SMILES and save to CSV."""
    import polars as pl

    fragments = fetch_all_fragments()

    if not fragments:
        logger.error("No fragments fetched. Check network connectivity.")
        return

    df = pl.DataFrame(fragments)

    output_path = REPO_ROOT / "data" / "structures" / "sgc_fragments.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path)

    logger.info(f"Saved {len(df)} fragments to {output_path}")

    # Summary by pocket
    summary = df.group_by("pocket").agg(pl.len().alias("count")).sort("pocket")
    logger.info(f"Per-pocket counts:\n{summary}")


if __name__ == "__main__":
    main()
