"""Diagnostic 04: pKD visualization in chemical space.

Two questions:
1. Does pKD have visible structure in the PaCMAP 2D embedding? If
   chemically-tight clusters show smoothly varying pKD color, there is
   local signal even if the global SAR is noisy. If colors are spatial
   noise, chemistry doesn't predict activity.
2. Do the 6 folds have similar pKD distributions? If fold 4 (the holdout)
   has a systematically different pKD median, our holdout is partly a
   covariate-shift problem, not just an OOD chemistry problem.

Writes:
    docs/sar-diagnostics-rjg/pacmap_colored_by_pkd.png
    docs/sar-diagnostics-rjg/fold_pkd_distributions.png
    data/sar-diagnostics-rjg/fold_pkd_stats.csv

Usage:
    uv run python scripts/sar-diagnostics-rjg/04_pacmap_pkd_viz.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
FOLDS_CSV = ROOT / "data" / "folds-pacmap-kmeans6" / "fold_assignments.csv"
ARTIFACT_DIR = ROOT / "data" / "sar-diagnostics-rjg"
DOC_FIG_DIR = ROOT / "docs" / "sar-diagnostics-rjg"
TARGET_COL = "pKD_global_mean"


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(FOLDS_CSV)
    pkd = df[TARGET_COL].to_numpy().astype(np.float64)
    x = df["pacmap_1"].to_numpy()
    y_c = df["pacmap_2"].to_numpy()
    fold = df["fold"].to_numpy()
    holdout_mask = df["is_holdout"].to_numpy()

    # ---- 1. PaCMAP colored by pKD ------------------------------------------
    # Two panels: (a) raw pKD color, (b) is_binder (top-quartile) color
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel (a): continuous pKD
    # Saturate colorbar at [1, 6] to avoid outlier compression
    vmin, vmax = float(np.quantile(pkd, 0.02)), float(np.quantile(pkd, 0.98))
    sc = axes[0].scatter(
        x, y_c, c=pkd, cmap="viridis", s=14,
        vmin=vmin, vmax=vmax, alpha=0.85, edgecolor="none",
    )
    # Circle holdout compounds for reference
    axes[0].scatter(
        x[holdout_mask], y_c[holdout_mask],
        s=30, facecolors="none", edgecolors="#c62828", linewidths=0.8,
        label=f"holdout (fold 4, n={int(holdout_mask.sum())})",
    )
    cbar = plt.colorbar(sc, ax=axes[0])
    cbar.set_label(f"pKD (color clipped to [{vmin:.1f}, {vmax:.1f}])")
    axes[0].set_xlabel("pacmap_1")
    axes[0].set_ylabel("pacmap_2")
    axes[0].set_title("PaCMAP colored by pKD")
    axes[0].legend(frameon=False, loc="best", fontsize=9)
    axes[0].grid(linestyle=":", alpha=0.4)

    # Panel (b): binder / non-binder
    is_binder = df["is_binder"].to_numpy().astype(bool)
    axes[1].scatter(x[~is_binder], y_c[~is_binder], s=10, color="#90a4ae", alpha=0.6,
                    label=f"non-binder (n={(~is_binder).sum()})")
    axes[1].scatter(x[is_binder], y_c[is_binder], s=14, color="#c62828", alpha=0.85,
                    label=f"binder (n={is_binder.sum()})")
    axes[1].set_xlabel("pacmap_1")
    axes[1].set_ylabel("pacmap_2")
    axes[1].set_title("PaCMAP colored by is_binder (top-quartile pKD)")
    axes[1].legend(frameon=False, loc="best", fontsize=9)
    axes[1].grid(linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(DOC_FIG_DIR / "pacmap_colored_by_pkd.png", dpi=150)
    plt.close(fig)
    logger.info(f"wrote {DOC_FIG_DIR / 'pacmap_colored_by_pkd.png'}")

    # ---- 2. per-fold pKD distribution ---------------------------------------
    folds_sorted = sorted(np.unique(fold).tolist())
    pkd_by_fold = [pkd[fold == f] for f in folds_sorted]

    fold_stats = pl.DataFrame({
        "fold": folds_sorted,
        "n": [len(v) for v in pkd_by_fold],
        "pKD_mean": [float(np.mean(v)) for v in pkd_by_fold],
        "pKD_median": [float(np.median(v)) for v in pkd_by_fold],
        "pKD_std": [float(np.std(v)) for v in pkd_by_fold],
        "is_holdout": [bool(holdout_mask[fold == f].all()) for f in folds_sorted],
    })
    fold_stats.write_csv(ARTIFACT_DIR / "fold_pkd_stats.csv")
    logger.info(f"fold pKD stats:\n{fold_stats}")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    parts = ax.violinplot(pkd_by_fold, showmedians=True, widths=0.8)
    for i, pc in enumerate(parts["bodies"]):
        is_hold = holdout_mask[fold == folds_sorted[i]].all()
        pc.set_facecolor("#c62828" if is_hold else "#90a4ae")
        pc.set_alpha(0.7)
    overall_median = float(np.median(pkd))
    ax.axhline(overall_median, color="black", linestyle=":", linewidth=1,
               label=f"overall median pKD = {overall_median:.2f}")
    ax.set_xticks(range(1, len(folds_sorted) + 1))
    ax.set_xticklabels([
        f"fold {f}\n(n={len(v)})" + ("\nHOLDOUT" if holdout_mask[fold == f].all() else "")
        for f, v in zip(folds_sorted, pkd_by_fold, strict=True)
    ], fontsize=9)
    ax.set_ylabel("pKD")
    ax.set_title("pKD distribution by chemical-space fold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(DOC_FIG_DIR / "fold_pkd_distributions.png", dpi=150)
    plt.close(fig)
    logger.info(f"wrote {DOC_FIG_DIR / 'fold_pkd_distributions.png'}")


if __name__ == "__main__":
    main()
