"""Validate the pocket-assigner against the fragment CSV.

This script:
  1. Loads fragment SMILES + pocket assignments from data/structures/sgc_fragments.csv
  2. Builds the PocketAssigner (Morgan FPs + Mol objects for substructure)
  3. Validates by scoring the fragments themselves (each should assign to its own pocket)

Usage:
    uv run python scripts/build_pocket_assigner.py
"""

from pathlib import Path

from loguru import logger

from tbxt_hackathon.pocket_assigner import PocketAssigner

REPO_ROOT = Path(__file__).resolve().parents[1]

FRAGMENT_CSV = REPO_ROOT / "data" / "structures" / "sgc_fragments.csv"


def validate_self_assignment(assigner: PocketAssigner) -> None:
    """Sanity check: each fragment should assign to its own pocket.

    Logs warnings for any fragment that doesn't self-assign correctly.
    Fragments are small so some cross-pocket Tanimoto overlap is expected,
    but substructure self-match should always hold (Tc=1.0 + substruct=True).
    """
    misassigned = 0
    for frag in assigner.fragments:
        scores = assigner.score(frag.smiles)
        assigned = assigner.assign(frag.smiles)

        if assigned != frag.pocket:
            misassigned += 1
            logger.warning(
                f"Fragment {frag.pdb_id}/{frag.ligand_id} ({frag.pocket}) "
                f"assigned to {assigned}. "
                f"Scores: {', '.join(f'{k}={v.combined:.3f}' for k, v in scores.items())}"
            )

    total = len(assigner.fragments)
    correct = total - misassigned
    logger.info(
        f"Self-assignment validation: {correct}/{total} correct "
        f"({100*correct/total:.1f}%)"
    )

    if misassigned > 0:
        logger.warning(
            f"{misassigned} fragments misassigned — likely due to "
            f"shared substructures across pockets (expected for dual-binders)"
        )


def main() -> None:
    """Build and validate the pocket assigner."""
    if not FRAGMENT_CSV.exists():
        logger.error(
            f"Fragment CSV not found at {FRAGMENT_CSV}. "
            f"Run 'uv run python scripts/fetch_fragment_smiles.py' first."
        )
        return

    logger.info(f"Loading fragments from {FRAGMENT_CSV}")
    assigner = PocketAssigner.from_csv(FRAGMENT_CSV)

    logger.info("Running self-assignment validation...")
    validate_self_assignment(assigner)

    # Show per-pocket summary
    logger.info("Per-pocket fragment counts:")
    for pocket, fps in sorted(assigner.pocket_fps.items()):
        logger.info(f"  {pocket}: {len(fps)} fragments")


if __name__ == "__main__":
    main()
