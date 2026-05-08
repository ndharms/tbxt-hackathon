"""Calculate RMSD between two PDB structures.

Uses BioPython to load PDB files and compute the Root Mean Square Deviation
between corresponding atoms after optimal superposition.

Example:
    python scripts/calculate_rmsd.py data/structures/ref.pdb data/structures/target.pdb
    python scripts/calculate_rmsd.py ref.pdb target.pdb --chain A --align-backbone
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from Bio import PDB
from Bio.PDB import PDBIO, Superimposer
from loguru import logger


def calculate_rmsd(
    ref_pdb: Path,
    target_pdb: Path,
    chain_id: Optional[str] = None,
    align_backbone: bool = False,
    residue_range: Optional[tuple[int, int]] = None,
) -> dict[str, float]:
    """Calculate RMSD between two PDB structures.
    
    Args:
        ref_pdb: Path to reference PDB file
        target_pdb: Path to target PDB file to compare
        chain_id: Specific chain to compare (default: first chain in each structure)
        align_backbone: If True, only align on backbone atoms (N, CA, C, O)
        residue_range: Optional tuple (start, end) to limit comparison to residue range
        
    Returns:
        Dictionary with 'rmsd' and 'n_atoms' keys
        
    Example:
        >>> result = calculate_rmsd(
        ...     Path("ref.pdb"),
        ...     Path("target.pdb"),
        ...     align_backbone=True
        ... )
        >>> print(f"RMSD: {result['rmsd']:.2f} Å over {result['n_atoms']} atoms")
    """
    parser = PDB.PDBParser(QUIET=True)
    
    # Load structures
    logger.info(f"Loading reference structure: {ref_pdb}")
    ref_structure = parser.get_structure("reference", ref_pdb)
    
    logger.info(f"Loading target structure: {target_pdb}")
    target_structure = parser.get_structure("target", target_pdb)
    
    # Get chains
    ref_chain = _get_chain(ref_structure, chain_id)
    target_chain = _get_chain(target_structure, chain_id)
    
    logger.info(f"Using chain {ref_chain.id} from reference")
    logger.info(f"Using chain {target_chain.id} from target")
    
    # Get atoms for alignment
    ref_atoms = _get_atoms_for_alignment(
        ref_chain, 
        align_backbone=align_backbone,
        residue_range=residue_range
    )
    target_atoms = _get_atoms_for_alignment(
        target_chain,
        align_backbone=align_backbone,
        residue_range=residue_range
    )
    
    # Ensure same number of atoms
    if len(ref_atoms) != len(target_atoms):
        raise ValueError(
            f"Atom count mismatch: reference has {len(ref_atoms)} atoms, "
            f"target has {len(target_atoms)} atoms"
        )
    
    logger.info(f"Aligning {len(ref_atoms)} atoms")
    
    # Superimpose and calculate RMSD
    superimposer = Superimposer()
    superimposer.set_atoms(ref_atoms, target_atoms)
    
    rmsd = superimposer.rms
    logger.info(f"RMSD: {rmsd:.3f} Å")
    
    return {
        "rmsd": float(rmsd),
        "n_atoms": len(ref_atoms),
        "rotation_matrix": superimposer.rotran[0],
        "translation_vector": superimposer.rotran[1],
    }


def _get_chain(structure: PDB.Structure.Structure, chain_id: Optional[str] = None):
    """Get specified chain or first chain from structure."""
    chains = list(structure.get_chains())
    
    if not chains:
        raise ValueError("No chains found in structure")
    
    if chain_id is None:
        return chains[0]
    
    for chain in chains:
        if chain.id == chain_id:
            return chain
    
    raise ValueError(f"Chain {chain_id} not found in structure")


def _get_atoms_for_alignment(
    chain,
    align_backbone: bool = False,
    residue_range: Optional[tuple[int, int]] = None,
) -> list:
    """Extract atoms from chain for alignment.
    
    Args:
        chain: BioPython Chain object
        align_backbone: If True, only include backbone atoms (N, CA, C, O)
        residue_range: Optional (start, end) residue numbers to include
        
    Returns:
        List of Atom objects
    """
    atoms = []
    backbone_atoms = {"N", "CA", "C", "O"}
    
    for residue in chain:
        # Skip hetero residues (water, ligands, etc.)
        if residue.id[0] != " ":
            continue
        
        # Check residue range
        if residue_range is not None:
            res_num = residue.id[1]
            if res_num < residue_range[0] or res_num > residue_range[1]:
                continue
        
        for atom in residue:
            if align_backbone:
                if atom.name in backbone_atoms:
                    atoms.append(atom)
            else:
                atoms.append(atom)
    
    return atoms


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate RMSD between two PDB structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - compare all atoms
  python scripts/calculate_rmsd.py ref.pdb target.pdb
  
  # Compare only backbone atoms
  python scripts/calculate_rmsd.py ref.pdb target.pdb --align-backbone
  
  # Compare specific chain
  python scripts/calculate_rmsd.py ref.pdb target.pdb --chain A
  
  # Compare residue range (e.g., TBXT DNA-binding domain 42-219)
  python scripts/calculate_rmsd.py ref.pdb target.pdb --residue-range 42 219
        """,
    )
    
    parser.add_argument("ref_pdb", type=Path, help="Reference PDB file")
    parser.add_argument("target_pdb", type=Path, help="Target PDB file to compare")
    parser.add_argument(
        "--chain",
        type=str,
        default=None,
        help="Specific chain ID to compare (default: first chain)",
    )
    parser.add_argument(
        "--align-backbone",
        action="store_true",
        help="Only align backbone atoms (N, CA, C, O)",
    )
    parser.add_argument(
        "--residue-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Only compare residues in this range",
    )
    
    args = parser.parse_args()
    
    # Validate files exist
    if not args.ref_pdb.exists():
        parser.error(f"Reference PDB not found: {args.ref_pdb}")
    if not args.target_pdb.exists():
        parser.error(f"Target PDB not found: {args.target_pdb}")
    
    # Calculate RMSD
    residue_range = tuple(args.residue_range) if args.residue_range else None
    
    result = calculate_rmsd(
        ref_pdb=args.ref_pdb,
        target_pdb=args.target_pdb,
        chain_id=args.chain,
        align_backbone=args.align_backbone,
        residue_range=residue_range,
    )
    
    # Print results
    print(f"\nRMSD: {result['rmsd']:.3f} Å")
    print(f"Number of atoms aligned: {result['n_atoms']}")
    
    if args.align_backbone:
        print("Alignment method: Backbone atoms only (N, CA, C, O)")
    else:
        print("Alignment method: All atoms")
    
    if residue_range:
        print(f"Residue range: {residue_range[0]}-{residue_range[1]}")


if __name__ == "__main__":
    main()
