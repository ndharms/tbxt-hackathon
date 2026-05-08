"""Diagnostic: PaCMAP visualization colored by pocket assignment.

Plots the existing PaCMAP 2D embedding from fold_assignments.csv,
colored by predicted Newman pocket (A, B, C, D, or unassigned).

Usage:
    uv run python scripts/sar-diagnostics-rjg/07_pocket_pacmap_viz.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger

from tbxt_hackathon.pocket_assigner import PocketAssigner

ROOT = Path(__file__).resolve().parents[2]
FOLDS_CSV = ROOT / "data" / "folds-pacmap-kmeans6" / "fold_assignments.csv"
FRAGMENT_CSV = ROOT / "data" / "structures" / "sgc_fragments.csv"
DOC_FIG_DIR = ROOT / "docs" / "sar-diagnostics-rjg"


POCKET_COLORS = {
    "A": "#1565c0",  # blue
    "B": "#f57c00",  # orange
    "C": "#2e7d32",  # green
    "D": "#7b1fa2",  # purple
    None: "#bdbdbd",  # grey for unassigned
}


def main() -> None:
    DOC_FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(FOLDS_CSV)
    smiles = df["canonical_smiles"].to_list()
    pacmap_1 = df["pacmap_1"].to_numpy()
    pacmap_2 = df["pacmap_2"].to_numpy()
    pkd = df["pKD_global_mean"].to_numpy()
    is_binder = df["is_binder"].to_numpy().astype(bool)
    n = len(smiles)
    logger.info(f"Loaded {n} compounds")

    # Assign pockets
    assigner = PocketAssigner.from_csv(FRAGMENT_CSV)
    pocket_assignments = assigner.assign_batch(smiles)
    pocket_scores = assigner.score_batch(smiles)

    # Get max combined score for sizing
    max_combined = np.array([
        max((s[p].combined for p in s), default=0.0)
        for s in pocket_scores
    ])

    # ---- Figure 1: PaCMAP colored by pocket assignment ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: colored by pocket
    ax = axes[0]
    # Plot unassigned first (background)
    for pocket, color in POCKET_COLORS.items():
        mask = np.array([p == pocket for p in pocket_assignments])
        if not mask.any():
            continue
        label = pocket if pocket is not None else "Unassigned"
        count = int(mask.sum())
        alpha = 0.3 if pocket is None else 0.7
        size = 8 if pocket is None else 15
        ax.scatter(
            pacmap_1[mask], pacmap_2[mask],
            c=color, s=size, alpha=alpha,
            label=f"{label} (n={count})",
            edgecolors="none",
        )

    ax.set_xlabel("PaCMAP 1")
    ax.set_ylabel("PaCMAP 2")
    ax.set_title("PaCMAP colored by predicted Newman pocket")
    ax.legend(frameon=False, fontsize=9, markerscale=1.5)

    # Right: colored by pocket, sized by confidence (combined score)
    ax = axes[1]
    for pocket, color in POCKET_COLORS.items():
        if pocket is None:
            continue
        mask = np.array([p == pocket for p in pocket_assignments])
        if not mask.any():
            continue
        # Size by Tanimoto score
        sizes = max_combined[mask] * 30
        ax.scatter(
            pacmap_1[mask], pacmap_2[mask],
            c=color, s=sizes, alpha=0.6,
            label=f"{pocket}",
            edgecolors="none",
        )

    # Show unassigned as tiny dots
    mask_none = np.array([p is None for p in pocket_assignments])
    ax.scatter(
        pacmap_1[mask_none], pacmap_2[mask_none],
        c="#e0e0e0", s=4, alpha=0.3, label="Unassigned",
        edgecolors="none",
    )

    ax.set_xlabel("PaCMAP 1")
    ax.set_ylabel("PaCMAP 2")
    ax.set_title("Pocket assignments (size = confidence)")
    ax.legend(frameon=False, fontsize=9, markerscale=1.5)

    fig.tight_layout()
    fig.savefig(DOC_FIG_DIR / "pacmap_pocket_assignments.png", dpi=150)
    plt.close(fig)
    logger.info(f"Wrote {DOC_FIG_DIR / 'pacmap_pocket_assignments.png'}")

    # ---- Figure 2: PaCMAP pocket vs pKD side by side ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: pocket
    ax = axes[0]
    for pocket, color in POCKET_COLORS.items():
        mask = np.array([p == pocket for p in pocket_assignments])
        if not mask.any():
            continue
        label = pocket if pocket is not None else "Unassigned"
        alpha = 0.3 if pocket is None else 0.7
        size = 8 if pocket is None else 15
        ax.scatter(
            pacmap_1[mask], pacmap_2[mask],
            c=color, s=size, alpha=alpha,
            label=label, edgecolors="none",
        )
    ax.set_xlabel("PaCMAP 1")
    ax.set_ylabel("PaCMAP 2")
    ax.set_title("Predicted pocket")
    ax.legend(frameon=False, fontsize=8)

    # Right: pKD
    ax = axes[1]
    sc = ax.scatter(
        pacmap_1, pacmap_2,
        c=pkd, s=10, alpha=0.6,
        cmap="RdYlGn", edgecolors="none",
        vmin=np.percentile(pkd, 5), vmax=np.percentile(pkd, 95),
    )
    plt.colorbar(sc, ax=ax, label="pKD")
    ax.set_xlabel("PaCMAP 1")
    ax.set_ylabel("PaCMAP 2")
    ax.set_title("pKD (higher = more potent)")

    fig.suptitle("Chemical space: pocket assignment vs. binding affinity", fontsize=11)
    fig.tight_layout()
    fig.savefig(DOC_FIG_DIR / "pacmap_pocket_vs_pkd.png", dpi=150)
    plt.close(fig)
    logger.info(f"Wrote {DOC_FIG_DIR / 'pacmap_pocket_vs_pkd.png'}")

    # ---- Figure 3: Per-pocket substructure match highlights ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()

    for idx, pocket in enumerate(["A", "B", "C", "D"]):
        ax = axes_flat[idx]
        # Background: all points grey
        ax.scatter(pacmap_1, pacmap_2, c="#e0e0e0", s=5, alpha=0.3, edgecolors="none")

        # Substructure matches
        sub_mask = np.array([
            pocket in s and s[pocket].substruct for s in pocket_scores
        ])
        # Tanimoto-only (assigned but not substruct)
        tc_mask = np.array([
            pocket_assignments[i] == pocket and not sub_mask[i]
            for i in range(n)
        ])

        color = POCKET_COLORS[pocket]
        if tc_mask.any():
            ax.scatter(
                pacmap_1[tc_mask], pacmap_2[tc_mask],
                c=color, s=20, alpha=0.5, edgecolors="none",
                label=f"Tanimoto only ({int(tc_mask.sum())})",
            )
        if sub_mask.any():
            ax.scatter(
                pacmap_1[sub_mask], pacmap_2[sub_mask],
                c=color, s=40, alpha=0.9, edgecolors="black", linewidths=0.5,
                label=f"Substructure match ({int(sub_mask.sum())})",
            )

        ax.set_title(f"Pocket {pocket}", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        ax.set_xlabel("PaCMAP 1")
        ax.set_ylabel("PaCMAP 2")

    fig.suptitle("Per-pocket assignments in chemical space", fontsize=11)
    fig.tight_layout()
    fig.savefig(DOC_FIG_DIR / "pacmap_per_pocket_detail.png", dpi=150)
    plt.close(fig)
    logger.info(f"Wrote {DOC_FIG_DIR / 'pacmap_per_pocket_detail.png'}")


if __name__ == "__main__":
    main()
