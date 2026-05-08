"""Pocket-assigner: assigns compounds to TBXT binding pockets based on
fragment substructure matching and Tanimoto similarity.

Scoring logic (per pocket):
  1. Substructure match: does the query contain any pocket fragment as a
     substructure? If yes, that pocket gets a score of 1.0 + max_tanimoto
     (guarantees it outranks pure-similarity matches).
  2. Max ECFP4 Tanimoto: highest Tanimoto similarity to any fragment in
     the pocket. Used as tiebreaker and fallback when no substructure match.

Assignment: compound goes to the pocket with highest combined score,
subject to a minimum threshold (default 0.35 on the Tanimoto component).

This is a rule-based scorer (no learned parameters) that serves as a
pre-filter before per-pocket Boltz surrogates.

All four Newman pockets are scored (A, B, C, D). Pocket B is included
for completeness but is dropped from the downstream pipeline (bad Boltz
pose reliability). The caller decides which pockets to act on.

Pocket labels follow Newman et al. 2025 nomenclature:
    A  -> Newman pocket A/A' (R180 anchor, 26 fragments, 2 submission slots)
    B  -> Newman pocket B    (loop 112-120, 5 fragments, DROPPED downstream)
    C  -> Newman pocket C    (E48/R54, 10 fragments, 1 submission slot)
    D  -> Newman pocket D    (Y88/D177, induced, 4 fragments, 1 speculative slot)

Example:
    >>> from tbxt_hackathon.pocket_assigner import PocketAssigner
    >>> assigner = PocketAssigner.from_csv("data/structures/sgc_fragments.csv")
    >>> scores = assigner.score("CC(=O)Nc1ccc(cc1)c2csc(n2)N")
    >>> scores
    {'A': PocketScore(tanimoto=1.0, substruct=True, combined=2.0), ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from loguru import logger
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from typeguard import typechecked

from .exceptions import DataError


@dataclass(frozen=True)
class FragmentEntry:
    """A single fragment with pocket assignment."""

    pocket: str
    pdb_id: str
    ligand_id: str
    smiles: str


@dataclass(frozen=True)
class PocketScore:
    """Score for a single pocket.

    Attributes:
        tanimoto: Max ECFP4 Tanimoto to any fragment in the pocket.
        substruct: Whether the query contains any pocket fragment as
            a substructure (exact subgraph match).
        combined: Scoring value used for ranking. If substruct is True,
            combined = 1.0 + tanimoto (range 1.0-2.0). Otherwise,
            combined = tanimoto (range 0.0-1.0).
    """

    tanimoto: float
    substruct: bool
    combined: float


@dataclass
class PocketAssigner:
    """Assigns compounds to TBXT pockets via substructure match + Tanimoto.

    Scoring priority:
      1. Substructure containment (strongest signal: the compound literally
         contains a crystallographic fragment as a subgraph).
      2. Max ECFP4 Tanimoto similarity (fallback for non-substructure hits).

    Attributes:
        fragments: All fragment entries loaded from the CSV.
        pocket_fps: Dict mapping pocket label to list of Morgan fingerprints.
        pocket_mols: Dict mapping pocket label to list of RDKit Mol objects
            (used for substructure matching).
        n_bits: Fingerprint bit length.
        radius: Morgan fingerprint radius.
        threshold: Minimum max-Tanimoto to assign a compound to a pocket.

    Example:
        | smiles                         | A_tc | A_sub | B_tc | B_sub | C_tc | C_sub | D_tc | D_sub | assigned |
        |--------------------------------|------|-------|------|-------|------|-------|------|-------|----------|
        | CC(=O)Nc1ccc(cc1)c2csc(n2)N    | 1.00 | True  | 0.15 | False | 0.22 | False | 0.18 | False | A        |
        | c1ccc(c(c1)C(=O)O)OC(F)(F)F   | 0.23 | False | 0.11 | False | 0.19 | False | 1.00 | True  | D        |
        | c1ccc(cc1)NC(=O)Nc2cccnc2      | 0.25 | False | 0.14 | False | 1.00 | True  | 0.20 | False | C        |
    """

    fragments: list[FragmentEntry] = field(repr=False)
    pocket_fps: dict[str, list[ExplicitBitVect]] = field(repr=False)
    pocket_mols: dict[str, list[Mol]] = field(repr=False)
    n_bits: int = 2048
    radius: int = 2
    threshold: float = 0.35

    @classmethod
    @typechecked
    def from_csv(
        cls,
        csv_path: str | Path,
        n_bits: int = 2048,
        radius: int = 2,
        threshold: float = 0.35,
    ) -> "PocketAssigner":
        """Load fragments from CSV and build fingerprints + mol objects.

        Args:
            csv_path: Path to sgc_fragments.csv with columns:
                pocket, pdb_id, ligand_id, smiles.
            n_bits: Morgan fingerprint bit length.
            radius: Morgan fingerprint radius (2 = ECFP4).
            threshold: Minimum max-Tanimoto for pocket assignment.

        Returns:
            Configured PocketAssigner instance.
        """
        import polars as pl

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise DataError(f"Fragment CSV not found: {csv_path}")

        df = pl.read_csv(csv_path)
        required_cols = {"pocket", "pdb_id", "ligand_id", "smiles"}
        missing = required_cols - set(df.columns)
        if missing:
            raise DataError(f"Missing columns in fragment CSV: {missing}")

        fragments: list[FragmentEntry] = []
        pocket_fps: dict[str, list[ExplicitBitVect]] = {}
        pocket_mols: dict[str, list[Mol]] = {}
        gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)

        for row in df.iter_rows(named=True):
            pocket = row["pocket"]
            smiles = row["smiles"]

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(
                    f"Skipping invalid SMILES: {smiles} "
                    f"(PDB {row['pdb_id']}/{row['ligand_id']})"
                )
                continue

            fp = gen.GetFingerprint(mol)
            entry = FragmentEntry(
                pocket=pocket,
                pdb_id=row["pdb_id"],
                ligand_id=row["ligand_id"],
                smiles=smiles,
            )
            fragments.append(entry)

            if pocket not in pocket_fps:
                pocket_fps[pocket] = []
                pocket_mols[pocket] = []
            pocket_fps[pocket].append(fp)
            pocket_mols[pocket].append(mol)

        logger.info(
            f"PocketAssigner loaded: {len(fragments)} fragments across "
            f"{len(pocket_fps)} pockets "
            f"({', '.join(f'{k}={len(v)}' for k, v in sorted(pocket_fps.items()))})"
        )

        return cls(
            fragments=fragments,
            pocket_fps=pocket_fps,
            pocket_mols=pocket_mols,
            n_bits=n_bits,
            radius=radius,
            threshold=threshold,
        )

    @typechecked
    def _smiles_to_fp_and_mol(
        self, smiles: str
    ) -> tuple[ExplicitBitVect, Mol] | None:
        """Convert SMILES to fingerprint and Mol, or None if invalid."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        gen = AllChem.GetMorganGenerator(radius=self.radius, fpSize=self.n_bits)
        fp = gen.GetFingerprint(mol)
        return fp, mol

    @typechecked
    def score(self, smiles: str) -> dict[str, PocketScore]:
        """Score a single compound against all pockets.

        For each pocket, computes:
          - max Tanimoto similarity to pocket fragments
          - whether the compound contains any pocket fragment as substructure

        The combined score = 1.0 + tanimoto if substruct, else tanimoto.
        This ensures substructure matches always outrank pure similarity.

        Args:
            smiles: Query compound SMILES.

        Returns:
            Dict mapping pocket label to PocketScore.
            Returns empty dict if SMILES is invalid.
        """
        result = self._smiles_to_fp_and_mol(smiles)
        if result is None:
            logger.warning(f"Invalid SMILES for scoring: {smiles}")
            return {}

        fp, mol = result
        scores: dict[str, PocketScore] = {}

        for pocket in self.pocket_fps:
            # Tanimoto similarity
            sims = DataStructs.BulkTanimotoSimilarity(fp, self.pocket_fps[pocket])
            max_tc = float(max(sims)) if sims else 0.0

            # Substructure check: does the query contain any fragment?
            has_substruct = any(
                mol.HasSubstructMatch(frag_mol)
                for frag_mol in self.pocket_mols[pocket]
            )

            combined = (1.0 + max_tc) if has_substruct else max_tc
            scores[pocket] = PocketScore(
                tanimoto=max_tc,
                substruct=has_substruct,
                combined=combined,
            )

        return scores

    @typechecked
    def score_batch(
        self, smiles_list: Sequence[str]
    ) -> list[dict[str, PocketScore]]:
        """Score a batch of compounds against all pockets.

        Args:
            smiles_list: Sequence of SMILES strings.

        Returns:
            List of score dicts (same order as input). Invalid SMILES
            get empty dicts.
        """
        return [self.score(smi) for smi in smiles_list]

    @typechecked
    def assign(self, smiles: str) -> str | None:
        """Assign a compound to its best-matching pocket.

        Priority: substructure match > Tanimoto similarity.
        If multiple pockets have substructure matches, picks the one
        with highest Tanimoto (i.e. the fragment that covers the most
        of the query).

        Args:
            smiles: Query compound SMILES.

        Returns:
            Pocket label (e.g. "A_prime") if best combined score
            meets threshold criteria, None otherwise.
        """
        scores = self.score(smiles)
        if not scores:
            return None

        # Tiebreaker: prefer pocket with more fragments (more evidence)
        best_pocket = max(
            scores,
            key=lambda p: (scores[p].combined, len(self.pocket_fps.get(p, []))),
        )
        best_score = scores[best_pocket]

        # If there's a substructure match, always assign (strong signal)
        if best_score.substruct:
            return best_pocket

        # Otherwise require Tanimoto >= threshold
        if best_score.tanimoto >= self.threshold:
            return best_pocket

        return None

    @typechecked
    def assign_batch(
        self, smiles_list: Sequence[str]
    ) -> list[str | None]:
        """Assign a batch of compounds to pockets.

        Args:
            smiles_list: Sequence of SMILES strings.

        Returns:
            List of pocket labels or None (same order as input).
        """
        return [self.assign(smi) for smi in smiles_list]
