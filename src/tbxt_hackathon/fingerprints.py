"""Morgan, RDKit-path, and MACCS fingerprints for a list of SMILES."""

from __future__ import annotations

import numpy as np
from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, MACCSkeys, rdFingerprintGenerator
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


@typechecked
def rdkit_path_bitvects(
    smiles: list[str],
    n_bits: int = 2048,
    min_path: int = 1,
    max_path: int = 7,
) -> list[ExplicitBitVect]:
    """Compute RDKit path-based (Daylight-style) fingerprints. Preserves order.

    Path fingerprints enumerate linear subgraphs up to ``max_path`` bonds,
    in contrast to Morgan's circular neighborhood enumeration. They tend
    to capture different aspects of molecular topology (backbone shape
    vs. atom-environment diversity).
    """
    gen = rdFingerprintGenerator.GetRDKitFPGenerator(
        minPath=min_path, maxPath=max_path, fpSize=n_bits
    )
    fps: list[ExplicitBitVect] = []
    for idx, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise DataError(f"invalid SMILES at index {idx}: {smi!r}")
        fps.append(gen.GetFingerprint(mol))
    logger.info(
        f"built {len(fps)} RDKit-path bitvects "
        f"(min_path={min_path}, max_path={max_path}, {n_bits} bits)"
    )
    return fps


@typechecked
def rdkit_path_ndarray(
    smiles: list[str],
    n_bits: int = 2048,
    min_path: int = 1,
    max_path: int = 7,
) -> tuple[np.ndarray, list[ExplicitBitVect]]:
    """Return both a (n, n_bits) uint8 array and the ExplicitBitVect list."""
    fps = rdkit_path_bitvects(smiles, n_bits=n_bits, min_path=min_path, max_path=max_path)
    arr = bitvects_to_ndarray(fps)
    return arr, fps


@typechecked
def maccs_bitvects(smiles: list[str]) -> list[ExplicitBitVect]:
    """Compute RDKit MACCS-keys (167-bit) fingerprints. Preserves order.

    MACCS keys are a fixed 167-bit definition of well-known chemical
    substructures (rings, functional groups, specific atom-pair
    patterns). RDKit emits 167 bits but indexes from 1, so bit 0 is
    always unset; we expose all 167 bits for compatibility.
    """
    fps: list[ExplicitBitVect] = []
    for idx, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise DataError(f"invalid SMILES at index {idx}: {smi!r}")
        fps.append(MACCSkeys.GenMACCSKeys(mol))
    logger.info(f"built {len(fps)} MACCS bitvects (167 bits)")
    return fps


@typechecked
def maccs_ndarray(
    smiles: list[str],
) -> tuple[np.ndarray, list[ExplicitBitVect]]:
    """Return both a (n, 167) uint8 array and the ExplicitBitVect list."""
    fps = maccs_bitvects(smiles)
    arr = bitvects_to_ndarray(fps)
    return arr, fps


# ---- single-point dispatch helper -----------------------------------------

FP_TYPES = ("morgan", "rdkit_path", "maccs")


@typechecked
def fingerprint_ndarray(
    smiles: list[str],
    fp_type: str,
    n_bits: int = 2048,
) -> np.ndarray:
    """Dispatch to the right fingerprint builder by name.

    Args:
        smiles: list of canonical SMILES strings.
        fp_type: one of ``"morgan"`` (ECFP4, 2048 bits),
            ``"rdkit_path"`` (path-based, 2048 bits),
            or ``"maccs"`` (MACCS keys, fixed 167 bits).
        n_bits: ignored for ``"maccs"``; applied to the other two.

    Returns:
        (n, width) uint8 matrix of fingerprint bits.
    """
    if fp_type == "morgan":
        arr, _ = morgan_ndarray(smiles, n_bits=n_bits)
        return arr
    if fp_type == "rdkit_path":
        arr, _ = rdkit_path_ndarray(smiles, n_bits=n_bits)
        return arr
    if fp_type == "maccs":
        arr, _ = maccs_ndarray(smiles)
        return arr
    raise DataError(f"unknown fp_type={fp_type!r}; expected one of {FP_TYPES}")

