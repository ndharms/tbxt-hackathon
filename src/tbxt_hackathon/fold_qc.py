"""Fold quality evaluation via bulk Tanimoto similarity.

For each molecule in fold A, compute its Tanimoto against every molecule in
fold B (B != A) and take the mean of the top-5 most-similar hits. Plot the
distribution of these per-molecule top-5 means per fold vs. every other
fold. The fold with the lowest median of this distribution (aggregated
across all other folds) is the most structurally distinct.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from typeguard import typechecked


@dataclass(frozen=True)
class FoldQC:
    """Results of inter-fold Tanimoto QC.

    Attributes:
        per_fold_top5: dict fold_id -> np.ndarray of per-molecule top-5 mean
            Tanimoto against the union of all other folds.
        pairwise_medians: (n_folds, n_folds) matrix; entry [i, j] is the
            median of per-molecule top-5 mean Tanimoto for fold i against fold j.
            Diagonal is nan.
        distinctness_ranking: list of fold ids sorted most -> least distinct
            (lowest -> highest median of per-fold top-5 distribution).
    """

    per_fold_top5: dict[int, np.ndarray]
    pairwise_medians: np.ndarray
    distinctness_ranking: list[int]


@typechecked
def _top_k_mean(values: np.ndarray, k: int = 5) -> float:
    """Mean of the top-k largest values in a 1D array."""
    if values.size == 0:
        return float("nan")
    k = min(k, values.size)
    idx = np.argpartition(values, -k)[-k:]
    return float(values[idx].mean())


@typechecked
def per_molecule_top5_against_fold(
    query_fps: list[ExplicitBitVect],
    ref_fps: list[ExplicitBitVect],
    k: int = 5,
) -> np.ndarray:
    """For each query molecule, mean of top-k Tanimoto against all ref molecules.

    Example (one query, three refs):
        query   ref0  ref1  ref2  -> top-2 mean = mean of best two
    """
    out = np.empty(len(query_fps), dtype=np.float64)
    for i, qfp in enumerate(query_fps):
        sims = np.asarray(BulkTanimotoSimilarity(qfp, ref_fps), dtype=np.float64)
        out[i] = _top_k_mean(sims, k=k)
    return out


@typechecked
def evaluate_fold_quality(
    fps: list[ExplicitBitVect],
    fold_id: np.ndarray,
    k: int = 5,
) -> FoldQC:
    """Compute per-molecule top-k inter-fold Tanimoto and pairwise medians.

    For the ``distinctness_ranking`` we use the median of the aggregated
    per-fold top-k distribution (one value per molecule in the fold,
    computed against the union of all other folds). Lower median = more
    structurally distinct.
    """
    if len(fps) != fold_id.shape[0]:
        raise ValueError("fps and fold_id length mismatch")
    unique_folds = sorted(np.unique(fold_id).tolist())
    n_folds = len(unique_folds)

    # precompute per-fold index lists + fp lists
    fold_to_fps: dict[int, list[ExplicitBitVect]] = {}
    fold_to_idx: dict[int, np.ndarray] = {}
    for f in unique_folds:
        idx = np.where(fold_id == f)[0]
        fold_to_idx[f] = idx
        fold_to_fps[f] = [fps[i] for i in idx.tolist()]

    pairwise = np.full((n_folds, n_folds), np.nan)
    per_fold_top5: dict[int, np.ndarray] = {}

    for i, fi in enumerate(unique_folds):
        logger.info(f"QC for fold {fi} ({len(fold_to_fps[fi])} mols)")
        # Against every other fold (union)
        other_fps: list[ExplicitBitVect] = []
        for fj in unique_folds:
            if fj == fi:
                continue
            other_fps.extend(fold_to_fps[fj])
        per_fold_top5[fi] = per_molecule_top5_against_fold(fold_to_fps[fi], other_fps, k=k)
        # Pairwise against each individual other fold
        for j, fj in enumerate(unique_folds):
            if fi == fj:
                continue
            vals = per_molecule_top5_against_fold(fold_to_fps[fi], fold_to_fps[fj], k=k)
            pairwise[i, j] = float(np.median(vals))

    ranking = sorted(unique_folds, key=lambda f: float(np.median(per_fold_top5[f])))
    logger.info(
        "distinctness ranking (most -> least distinct): "
        + ", ".join(f"fold {f} (median={np.median(per_fold_top5[f]):.3f})" for f in ranking),
    )
    return FoldQC(
        per_fold_top5=per_fold_top5,
        pairwise_medians=pairwise,
        distinctness_ranking=ranking,
    )


@typechecked
def plot_top5_distributions(
    qc: FoldQC,
    save_path: Path,
    title: str = "Per-molecule top-5 Tanimoto vs. other folds",
) -> None:
    """Plot violin+strip of per-molecule top-5 Tanimoto distributions per fold.

    Lower = more structurally distinct from the other folds.
    """
    fold_ids = sorted(qc.per_fold_top5.keys())
    # long-format arrays for seaborn
    data = []
    labels = []
    for f in fold_ids:
        vals = qc.per_fold_top5[f]
        data.extend(vals.tolist())
        labels.extend([f"fold {f}\n(n={vals.size})"] * vals.size)
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(fold_ids)), 4.5))
    sns.violinplot(x=labels, y=data, ax=ax, inner=None, cut=0, color="#cfd8dc")
    sns.stripplot(x=labels, y=data, ax=ax, size=1.5, color="#455a64", alpha=0.5, jitter=0.25)
    ax.set_ylabel("mean of top-5 Tanimoto vs. all other folds")
    ax.set_xlabel("")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"wrote distribution plot to {save_path}")


@typechecked
def plot_pairwise_heatmap(qc: FoldQC, save_path: Path) -> None:
    """Heatmap of pairwise median top-5 Tanimoto between every fold pair."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        qc.pairwise_medians,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "median per-mol top-5 Tanimoto"},
    )
    ax.set_xlabel("reference fold")
    ax.set_ylabel("query fold")
    ax.set_title("Pairwise inter-fold top-5 Tanimoto (median)")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"wrote pairwise heatmap to {save_path}")


@typechecked
def plot_embedding(
    embedding_2d: np.ndarray,
    fold_id: np.ndarray,
    save_path: Path,
    holdout_fold: int | None = None,
) -> None:
    """Scatter plot of PaCMAP embedding colored by fold id.

    If ``holdout_fold`` is provided, that fold is drawn with a heavier
    outline and annotated in the legend so it is easy to see which cluster
    the held-out evaluation set corresponds to.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))
    folds = sorted(np.unique(fold_id).tolist())
    palette = sns.color_palette("tab10", n_colors=max(len(folds), 3))
    for i, f in enumerate(folds):
        mask = fold_id == f
        is_holdout = holdout_fold is not None and f == holdout_fold
        label = f"fold {f} (n={int(mask.sum())})"
        if is_holdout:
            label += "  [holdout]"
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            s=22 if is_holdout else 10,
            alpha=0.9 if is_holdout else 0.7,
            color=palette[i],
            edgecolor="black" if is_holdout else "none",
            linewidth=0.6 if is_holdout else 0.0,
            label=label,
            zorder=3 if is_holdout else 2,
        )
        # centroid marker
        cx = float(embedding_2d[mask, 0].mean())
        cy = float(embedding_2d[mask, 1].mean())
        ax.scatter(
            [cx], [cy],
            marker="X" if is_holdout else "o",
            s=150 if is_holdout else 60,
            color=palette[i],
            edgecolor="black",
            linewidth=1.2,
            zorder=5,
        )
        ax.annotate(
            str(f),
            xy=(cx, cy),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center", va="center",
            fontsize=8, weight="bold", color="white", zorder=6,
        )
    ax.set_xlabel("PaCMAP 1")
    ax.set_ylabel("PaCMAP 2")
    title = "Morgan FP -> PaCMAP 2D, colored by KMeans fold"
    if holdout_fold is not None:
        title += f"\n(fold {holdout_fold} held out: most structurally distinct)"
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"wrote embedding plot to {save_path}")
