"""Data loading and binder labeling.

Uses the already-aggregated ``pKD_global_mean`` column from
``data/processed/tbxt_compounds_clean.csv``. Binder threshold is the 75th
percentile of pKD across the labeled set (top-quartile binders).
"""

from dataclasses import dataclass
from pathlib import Path

import polars as pl
from loguru import logger
from typeguard import typechecked

from .exceptions import DataError


@dataclass(frozen=True)
class LabeledDataset:
    """Container for the labeled SMILES/pKD/binder dataset.

    Attributes:
        frame: polars DataFrame with at minimum columns
            ``compound_id``, ``canonical_smiles``, ``pKD_global_mean``, ``is_binder``.
        threshold: pKD cut used to define ``is_binder``.
        n_positives: number of rows with ``is_binder == True``.
        n_total: total number of rows.
    """

    frame: pl.DataFrame
    threshold: float
    n_positives: int
    n_total: int


REQUIRED_COLUMNS = ("compound_id", "canonical_smiles", "pKD_global_mean")


@typechecked
def load_compounds(path: str | Path) -> pl.DataFrame:
    """Load the cleaned compounds CSV as a polars DataFrame.

    Example input columns: ``compound_id``, ``canonical_smiles``,
    ``pKD_global_mean``, ... Passes through unchanged after validation.
    """
    path = Path(path)
    if not path.exists():
        raise DataError(f"Compounds file not found at {path}")
    df = pl.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"Compounds file missing required columns: {missing}")
    assert df.shape[0] > 0, "compounds file is empty"
    logger.info(f"loaded {df.shape[0]} compounds from {path}")
    return df


@typechecked
def label_binders(
    compounds: pl.DataFrame,
    quantile: float = 0.75,
) -> LabeledDataset:
    """Add a boolean ``is_binder`` column using the top-quantile pKD threshold.

    Top-quartile (``quantile=0.75``) gives a data-driven threshold that is
    robust to the long left tail of non-binders while still producing a
    meaningfully enriched positive class for a binary classifier.

    Args:
        compounds: polars frame with ``pKD_global_mean``.
        quantile: fraction in (0, 1). 0.75 -> top quartile as binders.

    Returns:
        LabeledDataset with added ``is_binder`` bool column and recorded threshold.

    Example:
        >>> lbl = label_binders(df, quantile=0.75)
        >>> lbl.threshold  # doctest: +SKIP
        3.837
    """
    if not 0.0 < quantile < 1.0:
        raise DataError(f"quantile must be in (0,1); got {quantile}")
    if compounds["pKD_global_mean"].is_null().any():
        raise DataError("pKD_global_mean contains nulls; cannot threshold")

    threshold: float = float(compounds["pKD_global_mean"].quantile(quantile))
    labeled = compounds.with_columns(
        (pl.col("pKD_global_mean") >= threshold).alias("is_binder"),
    )
    n_pos = int(labeled["is_binder"].sum())
    n_tot = labeled.shape[0]
    logger.info(
        f"binder threshold pKD >= {threshold:.3f} "
        f"(q={quantile:.2f}) -> {n_pos}/{n_tot} positives "
        f"({n_pos / n_tot:.1%})",
    )
    return LabeledDataset(
        frame=labeled,
        threshold=threshold,
        n_positives=n_pos,
        n_total=n_tot,
    )


@typechecked
def pkd_diagnostic(compounds: pl.DataFrame) -> dict[str, float]:
    """Return a small dict of pKD distribution statistics for logging."""
    s = compounds["pKD_global_mean"]
    return {
        "count": float(s.len()),
        "mean": float(s.mean()),  # type: ignore[arg-type]
        "std": float(s.std()),  # type: ignore[arg-type]
        "min": float(s.min()),  # type: ignore[arg-type]
        "q25": float(s.quantile(0.25)),  # type: ignore[arg-type]
        "median": float(s.median()),  # type: ignore[arg-type]
        "q75": float(s.quantile(0.75)),  # type: ignore[arg-type]
        "q90": float(s.quantile(0.90)),  # type: ignore[arg-type]
        "max": float(s.max()),  # type: ignore[arg-type]
    }
