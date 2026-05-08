"""Structurally-distinct folds via 2048-bit Morgan FP -> PaCMAP -> KMeans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pacmap
import polars as pl
from loguru import logger
from sklearn.cluster import KMeans
from typeguard import typechecked

from .exceptions import FoldAssignmentError
from .fingerprints import morgan_ndarray


@dataclass(frozen=True)
class FoldAssignment:
    """Per-row fold assignments plus the embeddings used to produce them.

    Attributes:
        fold_id: int array of length n, values in [0, n_folds).
        embedding_2d: PaCMAP 2D coordinates (n, 2) for plotting.
        fingerprints: (n, 2048) uint8 Morgan FP matrix (bits) for downstream Tanimoto QC.
    """

    fold_id: np.ndarray
    embedding_2d: np.ndarray
    fingerprints: np.ndarray


@typechecked
def assign_folds_by_chemical_space(
    smiles: list[str],
    n_folds: int = 6,
    n_bits: int = 2048,
    radius: int = 2,
    random_state: int = 0,
    pacmap_n_neighbors: int = 15,
    pacmap_n_components: int = 2,
) -> FoldAssignment:
    """Bucket SMILES into ``n_folds`` structurally distinct groups.

    Pipeline: Morgan FP (``n_bits``, ``radius``) -> PaCMAP (2D) -> KMeans
    on the 2D embedding. Two-step embedding is used (rather than clustering
    on raw bits) to produce visually distinct, geometrically-compact folds
    that are easier to inspect and reason about.

    Raises:
        FoldAssignmentError: if any fold ends up empty.

    Example:
        >>> fa = assign_folds_by_chemical_space(["CCO", "c1ccccc1", ...], n_folds=6)
        >>> fa.fold_id.shape  # doctest: +SKIP
        (n,)
    """
    if n_folds < 2:
        raise FoldAssignmentError(f"n_folds must be >= 2, got {n_folds}")

    fp_matrix, _ = morgan_ndarray(smiles, n_bits=n_bits, radius=radius)
    logger.info(
        f"FP matrix shape={fp_matrix.shape}; fitting PaCMAP "
        f"(n_neighbors={pacmap_n_neighbors}, n_components={pacmap_n_components})",
    )
    reducer = pacmap.PaCMAP(
        n_components=pacmap_n_components,
        n_neighbors=pacmap_n_neighbors,
        random_state=random_state,
    )
    emb = reducer.fit_transform(fp_matrix.astype(np.float32))
    assert emb.shape == (fp_matrix.shape[0], pacmap_n_components), emb.shape

    logger.info(f"fitting KMeans k={n_folds} on 2D embedding")
    km = KMeans(n_clusters=n_folds, n_init="auto", random_state=random_state)
    labels = km.fit_predict(emb)

    counts = np.bincount(labels, minlength=n_folds)
    if (counts == 0).any():
        raise FoldAssignmentError(f"empty fold in KMeans output: counts={counts.tolist()}")
    logger.info(f"fold sizes: {dict(enumerate(counts.tolist()))}")
    return FoldAssignment(fold_id=labels.astype(np.int32), embedding_2d=emb, fingerprints=fp_matrix)


@typechecked
def attach_folds(
    frame: pl.DataFrame,
    fold_id: np.ndarray,
    column: str = "fold",
) -> pl.DataFrame:
    """Attach a fold column to a polars frame (order-preserving)."""
    if len(fold_id) != frame.shape[0]:
        raise FoldAssignmentError(
            f"length mismatch: fold_id has {len(fold_id)}, frame has {frame.shape[0]}",
        )
    return frame.with_columns(pl.Series(column, fold_id))
