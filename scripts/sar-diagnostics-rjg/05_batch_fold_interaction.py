"""Diagnostic 05: interaction between batches and chemical-space folds.

If certain batches produce compounds that cluster into specific folds,
and those batches systematically read higher/lower pKD than others, then
*fold* and *batch* are confounded — a cross-validation over folds is
really a cross-validation over batches, and our models have to learn a
batch-correction function, not pure chemistry.

Builds:
1. Batch x fold *composition* heatmap: count of compounds per
   (reference_date, fold) cell. Aggregated from the per-record SPR data
   by joining to fold assignments.
2. Batch x fold *pKD* heatmap: median pKD per (reference_date, fold)
   cell (cells with < 5 compounds blanked to avoid noise).

Writes:
    data/sar-diagnostics-rjg/batch_fold_composition.csv
    data/sar-diagnostics-rjg/batch_fold_median_pkd.csv
    docs/sar-diagnostics-rjg/batch_fold_composition.png
    docs/sar-diagnostics-rjg/batch_fold_median_pkd.png

Usage:
    uv run python scripts/sar-diagnostics-rjg/05_batch_fold_interaction.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
FOLDS_CSV = ROOT / "data" / "folds-pacmap-kmeans6" / "fold_assignments.csv"
SPR_CSV = ROOT / "data" / "zenodo" / "tbxt_spr_merged.csv"
ARTIFACT_DIR = ROOT / "data" / "sar-diagnostics-rjg"
DOC_FIG_DIR = ROOT / "docs" / "sar-diagnostics-rjg"


def _plot_heatmap(
    mat: np.ndarray,
    row_labels: list,
    col_labels: list,
    title: str,
    save_path: Path,
    cmap: str,
    value_fmt: str,
    cbar_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([f"fold {c}" for c in col_labels], rotation=0)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, value_fmt.format(val),
                        ha="center", va="center", fontsize=7,
                        color="white" if (np.nanmax(mat) - np.nanmin(mat) > 0
                                          and (val - np.nanmin(mat)) / (np.nanmax(mat) - np.nanmin(mat)) > 0.5)
                        else "black")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    ax.set_title(title)
    ax.set_xlabel("chemical-space fold (kmeans6)")
    ax.set_ylabel("reference_date (batch)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"wrote {save_path}")


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_FIG_DIR.mkdir(parents=True, exist_ok=True)

    spr = pl.read_csv(SPR_CSV).select(["compound_id", "reference_date", "pKD"])
    folds = pl.read_csv(FOLDS_CSV).select(["compound_id", "fold"])

    joined = spr.join(folds, on="compound_id", how="inner")
    logger.info(f"joined SPR x folds: {joined.shape} (vs. {spr.shape[0]} SPR records)")

    # Composition: one row per (reference_date, fold), value = distinct compounds
    comp = (
        joined
        .group_by(["reference_date", "fold"])
        .agg(pl.col("compound_id").n_unique().alias("n_compounds"),
             pl.col("pKD").median().alias("median_pKD"),
             pl.col("pKD").mean().alias("mean_pKD"),
             pl.col("pKD").len().alias("n_records"))
        .sort(["reference_date", "fold"])
    )
    comp.write_csv(ARTIFACT_DIR / "batch_fold_composition.csv")
    comp.write_csv(ARTIFACT_DIR / "batch_fold_median_pkd.csv")  # same data, kept for naming clarity

    batches = sorted(joined["reference_date"].unique().to_list())
    fold_ids = sorted(joined["fold"].unique().to_list())

    comp_mat = np.zeros((len(batches), len(fold_ids)), dtype=np.float64)
    median_mat = np.full((len(batches), len(fold_ids)), np.nan, dtype=np.float64)
    for row in comp.iter_rows(named=True):
        i = batches.index(row["reference_date"])
        j = fold_ids.index(row["fold"])
        comp_mat[i, j] = row["n_compounds"]
        if row["n_compounds"] >= 5:  # suppress low-count noise
            median_mat[i, j] = row["median_pKD"]

    # Batch composition (counts of distinct compounds)
    _plot_heatmap(
        comp_mat,
        row_labels=[str(b) for b in batches],
        col_labels=fold_ids,
        title=f"Batch x fold composition (n compounds, total = {int(comp_mat.sum())})",
        save_path=DOC_FIG_DIR / "batch_fold_composition.png",
        cmap="Blues",
        value_fmt="{:.0f}",
        cbar_label="# distinct compounds",
    )

    # Median pKD (cells with >= 5 compounds)
    _plot_heatmap(
        median_mat,
        row_labels=[str(b) for b in batches],
        col_labels=fold_ids,
        title="Batch x fold median pKD (cells with n_compounds >= 5)",
        save_path=DOC_FIG_DIR / "batch_fold_median_pkd.png",
        cmap="viridis",
        value_fmt="{:.2f}",
        cbar_label="median pKD",
    )


if __name__ == "__main__":
    main()
