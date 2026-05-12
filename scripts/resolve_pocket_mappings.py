"""Resolve unresolved TEP pocket assignments by downloading PDBs and
computing ligand-residue contacts against Newman pocket signatures.

Downloads PDB structures for fragments with unknown/unresolved pocket
mappings, finds the ligand, computes distances to all protein residues,
and reports which Newman pocket signature the contacts best match.

Newman pocket signature residues (from brachyury-site-summary.md):
  A/A' (our A site): R180, V123, L91, I125, V173, I182, S89
  B (our D site, DROPPED): G112, H100, G113, P115, P111, K114, E116
  C (our G site): R54, E48, E50, L51, K76
  D (our F site): Y88, D177, V173, I182, M181, T183

Usage:
    uv run python scripts/resolve_pocket_mappings.py
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]

# Newman pocket signature residues (residue number -> pocket)
# These are the key residues that define each pocket
POCKET_SIGNATURES: Dict[str, List[int]] = {
    "A": [180, 123, 91, 125, 173, 182, 89],
    "D_site_newman_B": [112, 100, 113, 115, 111, 114, 116],
    "G": [54, 48, 50, 51, 76],
    "F": [88, 177, 173, 182, 181, 183],
}

# Contact distance threshold (Angstroms)
CONTACT_CUTOFF = 5.0

# Unresolved PDB entries: (pdb_id, TEP_pocket_label)
UNRESOLVED_ENTRIES = [
    ("5QSL", "TEP_B"),
    ("5QRU", "TEP_B"),
    ("5QRW", "TEP_B/C"),  # dual location: loc1=B, loc2=C
    ("5QS1", "TEP_B"),
    ("5QS0", "TEP_D"),
    ("5QRR", "TEP_E"),
    ("5QS7", "TEP_CHECK"),
    ("5QRM", "crystal_contact"),
]

PDB_CACHE_DIR = REPO_ROOT / "data" / "structures" / "pdb_cache"


def download_pdb(pdb_id: str) -> Path:
    """Download a PDB file from RCSB.

    Args:
        pdb_id: 4-character PDB identifier.

    Returns:
        Path to downloaded PDB file.
    """
    PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pdb_path = PDB_CACHE_DIR / f"{pdb_id.lower()}.pdb"

    if pdb_path.exists():
        logger.debug(f"  Using cached {pdb_path}")
        return pdb_path

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    logger.info(f"  Downloading {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    pdb_path.write_text(response.text)
    return pdb_path


def get_ligand_residues(structure: Structure) -> List[Residue]:
    """Extract non-water HETATM residues (ligands) from a structure.

    Filters out common crystallization artifacts.
    """
    artifacts = {
        "HOH", "SO4", "PO4", "GOL", "EDO", "ACT", "DMS", "MPD",
        "CL", "NA", "MG", "ZN", "CA", "K", "CD", "FMT", "MN",
        "PEG", "PGE", "1PE", "EPE", "MES", "TRS", "IMD",
    }
    ligands = []
    for model in structure:
        for chain in model:
            for residue in chain:
                het_flag = residue.get_id()[0]
                if het_flag.startswith("H_"):
                    resname = residue.get_resname().strip()
                    if resname not in artifacts:
                        ligands.append(residue)
    return ligands


def compute_contacts(
    structure: Structure,
    ligand: Residue,
    cutoff: float = CONTACT_CUTOFF,
) -> List[Tuple[int, str, float]]:
    """Compute protein residues within cutoff of any ligand atom.

    Args:
        structure: Parsed PDB structure.
        ligand: The ligand residue.
        cutoff: Distance threshold in Angstroms.

    Returns:
        List of (residue_number, residue_name, min_distance) sorted by distance.
    """
    # Collect all protein atoms
    protein_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                het_flag = residue.get_id()[0]
                if het_flag == " ":  # standard residue
                    for atom in residue:
                        protein_atoms.append(atom)

    if not protein_atoms:
        return []

    ns = NeighborSearch(protein_atoms)

    # Find all protein atoms near any ligand atom
    contacts: Dict[int, Tuple[str, float]] = {}
    for lig_atom in ligand.get_atoms():
        nearby = ns.search(lig_atom.get_vector().get_array(), cutoff)
        for prot_atom in nearby:
            parent_res = prot_atom.get_parent()
            res_num = parent_res.get_id()[1]
            res_name = parent_res.get_resname()
            dist = float(np.linalg.norm(
                lig_atom.get_vector().get_array() - prot_atom.get_vector().get_array()
            ))
            if res_num not in contacts or dist < contacts[res_num][1]:
                contacts[res_num] = (res_name, dist)

    result = [(num, name, dist) for num, (name, dist) in contacts.items()]
    result.sort(key=lambda x: x[2])
    return result


def score_pocket_match(
    contacts: List[Tuple[int, str, float]],
) -> Dict[str, Tuple[float, List[str]]]:
    """Score how well contacts match each Newman pocket signature.

    Args:
        contacts: List of (residue_number, residue_name, distance).

    Returns:
        Dict mapping pocket -> (score, matched_residues).
        Score = fraction of signature residues found in contacts.
    """
    contact_residues = {num for num, _, _ in contacts}
    contact_details = {num: (name, dist) for num, name, dist in contacts}

    results: Dict[str, Tuple[float, List[str]]] = {}
    for pocket, signature in POCKET_SIGNATURES.items():
        matched = []
        for sig_res in signature:
            if sig_res in contact_residues:
                name, dist = contact_details[sig_res]
                matched.append(f"{name}{sig_res}({dist:.1f}A)")
        score = len(matched) / len(signature)
        results[pocket] = (score, matched)

    return results


def main() -> None:
    """Download PDBs and resolve pocket mappings."""
    parser = PDBParser(QUIET=True)

    logger.info(
        f"Resolving {len(UNRESOLVED_ENTRIES)} unresolved TEP pocket entries..."
    )
    logger.info(f"Contact cutoff: {CONTACT_CUTOFF} A")
    logger.info("")

    for pdb_id, tep_label in UNRESOLVED_ENTRIES:
        logger.info(f"=== {pdb_id} (TEP: {tep_label}) ===")

        try:
            pdb_path = download_pdb(pdb_id)
            structure = parser.get_structure(pdb_id, pdb_path)
        except Exception as e:
            logger.error(f"  Failed to load {pdb_id}: {e}")
            continue

        ligands = get_ligand_residues(structure)
        if not ligands:
            logger.warning(f"  No ligands found in {pdb_id}")
            continue

        for ligand in ligands:
            lig_name = ligand.get_resname().strip()
            lig_id = ligand.get_id()
            logger.info(f"  Ligand: {lig_name} (chain {ligand.get_parent().id}, {lig_id})")

            contacts = compute_contacts(structure, ligand, CONTACT_CUTOFF)
            if not contacts:
                logger.warning(f"    No protein contacts within {CONTACT_CUTOFF} A")
                continue

            # Show top contacts
            logger.info(f"    Contacts ({len(contacts)} residues within {CONTACT_CUTOFF} A):")
            for res_num, res_name, dist in contacts[:10]:
                logger.info(f"      {res_name}{res_num}: {dist:.2f} A")

            # Score against pocket signatures
            pocket_scores = score_pocket_match(contacts)
            logger.info(f"    Pocket signature matches:")
            for pocket, (score, matched) in sorted(
                pocket_scores.items(), key=lambda x: -x[1][0]
            ):
                if score > 0:
                    logger.info(
                        f"      {pocket}: {score:.0%} "
                        f"({len(matched)}/{len(POCKET_SIGNATURES[pocket])}) "
                        f"— {', '.join(matched)}"
                    )

            # Best match
            best_pocket = max(pocket_scores, key=lambda p: pocket_scores[p][0])
            best_score, best_matched = pocket_scores[best_pocket]
            if best_score > 0:
                logger.info(
                    f"    → BEST MATCH: {best_pocket} ({best_score:.0%})"
                )
            else:
                logger.info(f"    → NO MATCH to any known pocket signature")

        logger.info("")


if __name__ == "__main__":
    main()
