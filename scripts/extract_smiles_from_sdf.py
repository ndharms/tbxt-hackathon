"""Extract SMILES from SDF files.

Reads all SDF files from a directory and outputs SMILES strings to a text file.
Each line contains: SMILES, molecule name (from SDF or filename).

Example:
    python scripts/extract_smiles_from_sdf.py data/a-site-binders/ output.txt
    python scripts/extract_smiles_from_sdf.py data/a-site-binders/ --output data/a-site-binders.txt
"""

import argparse
from pathlib import Path
from typing import Optional

from loguru import logger
from rdkit import Chem


def extract_smiles_from_sdf(
    sdf_path: Path,
    mol_name: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    """Extract SMILES from a single SDF file.
    
    Args:
        sdf_path: Path to SDF file
        mol_name: Optional name to use (default: from SDF or filename)
        
    Returns:
        Tuple of (SMILES, name) or None if parsing fails
        
    Example:
        >>> result = extract_smiles_from_sdf(Path("molecule.sdf"))
        >>> if result:
        ...     smiles, name = result
        ...     print(f"{name}: {smiles}")
    """
    try:
        suppl = Chem.SDMolSupplier(str(sdf_path))
        mol = next(iter(suppl))
        
        if mol is None:
            logger.warning(f"Failed to parse molecule from {sdf_path}")
            return None
        
        # Get SMILES
        smiles = Chem.MolToSmiles(mol)
        
        # Get molecule name (priority: arg > SDF property > filename)
        name: str
        if mol_name is None:
            if mol.HasProp("_Name") and mol.GetProp("_Name"):
                name = mol.GetProp("_Name")
            else:
                name = sdf_path.stem
        else:
            name = mol_name
        
        logger.debug(f"Extracted {name}: {smiles}")
        return (smiles, name)
        
    except Exception as e:
        logger.error(f"Error processing {sdf_path}: {e}")
        return None


def extract_smiles_from_directory(
    input_dir: Path,
    output_file: Path,
    recursive: bool = False,
) -> int:
    """Extract SMILES from all SDF files in a directory.
    
    Args:
        input_dir: Directory containing SDF files
        output_file: Output text file path
        recursive: If True, search subdirectories recursively
        
    Returns:
        Number of molecules successfully extracted
        
    Example:
        >>> count = extract_smiles_from_directory(
        ...     Path("data/structures"),
        ...     Path("data/structures.txt")
        ... )
        >>> print(f"Extracted {count} molecules")
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")
    
    # Find all SDF files
    pattern = "**/*.sdf" if recursive else "*.sdf"
    sdf_files = sorted(input_dir.glob(pattern))
    
    if not sdf_files:
        logger.warning(f"No SDF files found in {input_dir}")
        return 0
    
    logger.info(f"Found {len(sdf_files)} SDF files in {input_dir}")
    
    # Extract SMILES
    results = []
    for sdf_path in sdf_files:
        result = extract_smiles_from_sdf(sdf_path)
        if result:
            results.append(result)
    
    # Write output
    logger.info(f"Writing {len(results)} SMILES to {output_file}")
    with open(output_file, "w") as f:
        for smiles, name in results:
            f.write(f"{smiles}\t{name}\n")
    
    logger.info(f"Successfully extracted {len(results)} molecules")
    return len(results)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract SMILES from SDF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from directory
  python scripts/extract_smiles_from_sdf.py data/a-site-binders/ output.txt
  
  # Specify output location
  python scripts/extract_smiles_from_sdf.py data/a-site-binders/ --output data/a-site-binders.txt
  
  # Search subdirectories recursively
  python scripts/extract_smiles_from_sdf.py data/structures/ output.txt --recursive
        """,
    )
    
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing SDF files",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        nargs="?",
        help="Output text file (default: <input_dir>.txt)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        dest="output_alt",
        help="Alternative way to specify output file",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Search subdirectories recursively",
    )
    
    args = parser.parse_args()
    
    # Determine output file
    if args.output_alt:
        output_file = args.output_alt
    elif args.output_file:
        output_file = args.output_file
    else:
        # Default: input_dir name + .txt
        output_file = args.input_dir.parent / f"{args.input_dir.name}.txt"
    
    # Extract SMILES
    count = extract_smiles_from_directory(
        input_dir=args.input_dir,
        output_file=output_file,
        recursive=args.recursive,
    )
    
    print(f"\nExtracted {count} molecules from {args.input_dir}")
    print(f"Output written to: {output_file}")


if __name__ == "__main__":
    main()
