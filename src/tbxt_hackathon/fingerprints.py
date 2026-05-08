"""Morgan ECFP4-like fingerprints (2048 bits) for a list of SMILES."""

from __future__ import annotations

import numpy as np
from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from typeguard import typechecked

from .exceptions import DataError


@typechecked
def morgan_bitvects(
    smiles: list[str],
    n_bits: int = 2048,
    radius: int = 2,
) -> list[ExplicitBitVect]:
    """Compute RDKit ExplicitBitVect fingerprints. Preserves order.

    These are the objects to feed ``BulkTanimotoSimilarity``. Parallel
    to ``morgan_ndarray``; we return both representations to avoid
    repeatedly rebuilding them.
    """
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps: list[ExplicitBitVect] = []
    for idx, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise DataError(f"invalid SMILES at index {idx}: {smi!r}")
        fps.append(gen.GetFingerprint(mol))
    logger.info(f"built {len(fps)} Morgan bitvects (radius={radius}, {n_bits} bits)")
    return fps


@typechecked
def bitvects_to_ndarray(fps: list[ExplicitBitVect]) -> np.ndarray:
    """Pack a list of ExplicitBitVects into a dense uint8 matrix (n, n_bits)."""
    n_bits = fps[0].GetNumBits()
    arr = np.zeros((len(fps), n_bits), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


@typechecked
def morgan_ndarray(
    smiles: list[str],
    n_bits: int = 2048,
    radius: int = 2,
) -> tuple[np.ndarray, list[ExplicitBitVect]]:
    """Return both a (n, n_bits) uint8 array and the ExplicitBitVect list."""
    fps = morgan_bitvects(smiles, n_bits=n_bits, radius=radius)
    arr = bitvects_to_ndarray(fps)
    return arr, fps
