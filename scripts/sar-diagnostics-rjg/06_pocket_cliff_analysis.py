"""Diagnostic: can pocket assignment explain activity cliffs?

Hypothesis: if two structurally similar compounds bind different pockets,
their different pKD values are expected (not a cliff — just different
targets). If pocket assignment disambiguates cliffs, we'd see:
  - Cross-pocket pairs: higher |dpKD|, higher cliff rate
  - Same-pocket pairs: lower |dpKD|, smoother SAR

This re-runs the SAR smoothness analysis from diagnostic 02, stratified
by pocket assignment (same pocket vs different pocket vs unassigned).

Usage:
    uv run python scripts/sar-diagnostics-rjg/06_pocket_cliff_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from rdkit.DataStructs import BulkTanimotoSimilarity

from tbxt_hackathon.fingerprints import morgan_bitvects
from tbxt_hackathon.pocket_assigner import PocketAssigner

ROOT = Path(__file__).resolve().parents[2]
FOLDS_CSV = ROOT / "data" / "folds-pacmap-kmeans6" / "fold_assignments.csv"
FRAGMENT_CSV = ROOT / "data" / "structures" / "sgc_fragments.csv"
ARTIFACT_DIR = ROOT / "data" / "sar-diagnostics-rjg"
DOC_FIG_DIR = ROOT / "docs" / "sar-diagnostics-rjg"
TARGET_COL = "pKD_global_mean"


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load compounds
    df = pl.read_csv(FOLDS_CSV)
    smiles = df["canonical_smiles"].to_list()
    pkd = df[TARGET_COL].to_numpy().astype(np.float64)
    n = len(smiles)
    logger.info(f"Loaded {n} compounds from fold assignments")

    # Assign pockets
    logger.info("Scoring pocket assignments...")
    assigner = PocketAssigner.from_csv(FRAGMENT_CSV)
    pocket_assignments = assigner.assign_batch(smiles)

    # Also get per-pocket scores for richer analysis
    pocket_scores = assigner.score_batch(smiles)

    # Summary of assignments
    assigned_count = sum(1 for p in pocket_assignments if p is not None)
    logger.info(f"Assigned: {assigned_count}/{n} ({100*assigned_count/n:.1f}%)")

    pocket_counts: dict[str, int] = {}
    for p in pocket_assignments:
        if p is not None:
            pocket_counts[p] = pocket_counts.get(p, 0) + 1
    for pocket, count in sorted(pocket_counts.items()):
        logger.info(f"  {pocket}: {count}")

    # Compute fingerprints
    logger.info("Computing Morgan fingerprints...")
    fps = morgan_bitvects(smiles)

    # Pairwise analysis on high-similarity pairs (Tanimoto >= 0.5)
    # Using a lower threshold than diagnostic 02 to get more pairs for statistics
    MIN_TANIMOTO = 0.5
    logger.info(f"Computing pairwise analysis (Tanimoto >= {MIN_TANIMOTO})...")

    same_pocket_dpkd: list[float] = []
    cross_pocket_dpkd: list[float] = []
    unassigned_dpkd: list[float] = []
    same_pocket_tan: list[float] = []
    cross_pocket_tan: list[float] = []
    unassigned_tan: list[float] = []

    # Also track at the cliff threshold
    cliff_threshold = 2.0  # |dpKD| >= 2 is a cliff
    same_pocket_pairs = 0
    same_pocket_cliffs = 0
    cross_pocket_pairs = 0
    cross_pocket_cliffs = 0
    unassigned_pairs = 0
    unassigned_cliffs = 0

    for i in range(n - 1):
        sims = np.asarray(
            BulkTanimotoSimilarity(fps[i], fps[i + 1:]), dtype=np.float32
        )
        # Only look at similar pairs
        high_sim_idx = np.where(sims >= MIN_TANIMOTO)[0]
        if len(high_sim_idx) == 0:
            continue

        p_i = pocket_assignments[i]
        for rel_j in high_sim_idx:
            j = i + 1 + rel_j
            p_j = pocket_assignments[j]
            tc = float(sims[rel_j])
            dpk = abs(pkd[i] - pkd[j])

            if p_i is None or p_j is None:
                unassigned_dpkd.append(dpk)
                unassigned_tan.append(tc)
                unassigned_pairs += 1
                if dpk >= cliff_threshold:
                    unassigned_cliffs += 1
            elif p_i == p_j:
                same_pocket_dpkd.append(dpk)
                same_pocket_tan.append(tc)
                same_pocket_pairs += 1
                if dpk >= cliff_threshold:
                    same_pocket_cliffs += 1
            else:
                cross_pocket_dpkd.append(dpk)
                cross_pocket_tan.append(tc)
                cross_pocket_pairs += 1
                if dpk >= cliff_threshold:
                    cross_pocket_cliffs += 1

    total_pairs = same_pocket_pairs + cross_pocket_pairs + unassigned_pairs
    logger.info(f"\nPairs with Tanimoto >= {MIN_TANIMOTO}: {total_pairs:,}")
    logger.info(f"  Same pocket:  {same_pocket_pairs:,}")
    logger.info(f"  Cross pocket: {cross_pocket_pairs:,}")
    logger.info(f"  Unassigned:   {unassigned_pairs:,}")

    # Statistics
    results: dict = {
        "min_tanimoto_threshold": MIN_TANIMOTO,
        "cliff_threshold_dpkd": cliff_threshold,
        "total_pairs": total_pairs,
    }

    for label, dpkd_list, tan_list, n_pairs, n_cliffs in [
        ("same_pocket", same_pocket_dpkd, same_pocket_tan, same_pocket_pairs, same_pocket_cliffs),
        ("cross_pocket", cross_pocket_dpkd, cross_pocket_tan, cross_pocket_pairs, cross_pocket_cliffs),
        ("unassigned", unassigned_dpkd, unassigned_tan, unassigned_pairs, unassigned_cliffs),
    ]:
        if n_pairs == 0:
            results[label] = {"n_pairs": 0}
            continue

        arr = np.array(dpkd_list)
        cliff_frac = n_cliffs / n_pairs
        results[label] = {
            "n_pairs": n_pairs,
            "median_abs_dpkd": float(np.median(arr)),
            "mean_abs_dpkd": float(np.mean(arr)),
            "std_abs_dpkd": float(np.std(arr)),
            "n_cliffs": n_cliffs,
            "cliff_fraction": cliff_frac,
            "median_tanimoto": float(np.median(tan_list)),
        }
        logger.info(
            f"\n  {label}:"
            f"\n    n_pairs = {n_pairs:,}"
            f"\n    median |dpKD| = {np.median(arr):.3f}"
            f"\n    mean |dpKD| = {np.mean(arr):.3f}"
            f"\n    cliff rate (|dpKD| >= {cliff_threshold}) = {100*cliff_frac:.1f}%"
            f"\n    median Tanimoto = {np.median(tan_list):.3f}"
        )

    # Also do this at Tanimoto >= 0.7 to compare directly with diagnostic 02
    logger.info(f"\n--- Restricted to Tanimoto >= 0.7 ---")
    for label, dpkd_list, tan_list in [
        ("same_pocket", same_pocket_dpkd, same_pocket_tan),
        ("cross_pocket", cross_pocket_dpkd, cross_pocket_tan),
        ("unassigned", unassigned_dpkd, unassigned_tan),
    ]:
        arr = np.array(dpkd_list)
        tan_arr = np.array(tan_list)
        mask_07 = tan_arr >= 0.7
        n_07 = int(mask_07.sum())
        if n_07 == 0:
            logger.info(f"  {label} (Tc>=0.7): 0 pairs")
            continue
        cliffs_07 = int((arr[mask_07] >= cliff_threshold).sum())
        cliff_frac_07 = cliffs_07 / n_07
        logger.info(
            f"  {label} (Tc>=0.7): {n_07} pairs, "
            f"median |dpKD| = {np.median(arr[mask_07]):.3f}, "
            f"cliff rate = {100*cliff_frac_07:.1f}%"
        )
        results[f"{label}_tc_ge_0p7"] = {
            "n_pairs": n_07,
            "median_abs_dpkd": float(np.median(arr[mask_07])),
            "n_cliffs": cliffs_07,
            "cliff_fraction": cliff_frac_07,
        }

    # Per-pocket pKD distributions (for compounds assigned to each pocket)
    logger.info("\n--- Per-pocket pKD distributions ---")
    for pocket in sorted(pocket_counts.keys()):
        mask = [pocket_assignments[i] == pocket for i in range(n)]
        pocket_pkd = pkd[mask]
        logger.info(
            f"  {pocket}: n={len(pocket_pkd)}, "
            f"mean pKD={np.mean(pocket_pkd):.3f}, "
            f"median={np.median(pocket_pkd):.3f}, "
            f"std={np.std(pocket_pkd):.3f}"
        )
        results[f"pocket_{pocket}_pkd"] = {
            "n": int(len(pocket_pkd)),
            "mean": float(np.mean(pocket_pkd)),
            "median": float(np.median(pocket_pkd)),
            "std": float(np.std(pocket_pkd)),
        }

    # ---- Figures ----

    # Figure 1: |dpKD| distributions by pocket pairing category
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, (label, data, color) in zip(axes, [
        ("Same pocket", same_pocket_dpkd, "#2e7d32"),
        ("Cross pocket", cross_pocket_dpkd, "#c62828"),
        ("Unassigned (>=1)", unassigned_dpkd, "#757575"),
    ]):
        if len(data) > 0:
            ax.hist(data, bins=30, color=color, alpha=0.7, edgecolor="white",
                    density=True)
            ax.axvline(np.median(data), color="black", linestyle="--",
                       label=f"median = {np.median(data):.2f}")
            ax.axvline(cliff_threshold, color="#e65100", linestyle=":",
                       label=f"cliff threshold = {cliff_threshold}")
        ax.set_xlabel("|pKD_i - pKD_j|")
        ax.set_title(f"{label}\n(n = {len(data):,})")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(linestyle=":", alpha=0.3)

    axes[0].set_ylabel("Density")
    fig.suptitle(
        f"Activity cliff analysis by pocket assignment (Tanimoto >= {MIN_TANIMOTO})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(DOC_FIG_DIR / "pocket_cliff_dpkd_distributions.png", dpi=150)
    plt.close(fig)
    logger.info(f"Wrote {DOC_FIG_DIR / 'pocket_cliff_dpkd_distributions.png'}")

    # Figure 2: cliff rate comparison bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    categories = []
    rates = []
    counts = []
    for label, n_p, n_c in [
        ("Same pocket", same_pocket_pairs, same_pocket_cliffs),
        ("Cross pocket", cross_pocket_pairs, cross_pocket_cliffs),
        ("Unassigned", unassigned_pairs, unassigned_cliffs),
    ]:
        if n_p > 0:
            categories.append(label)
            rates.append(100 * n_c / n_p)
            counts.append(n_p)

    colors = ["#2e7d32", "#c62828", "#757575"][:len(categories)]
    bars = ax.bar(categories, rates, color=colors, edgecolor="white", width=0.6)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"n={count:,}", ha="center", fontsize=9,
        )
    ax.set_ylabel(f"% pairs with |dpKD| >= {cliff_threshold}")
    ax.set_title(
        f"Activity cliff rate by pocket pairing\n"
        f"(Tanimoto >= {MIN_TANIMOTO}, {total_pairs:,} total pairs)"
    )
    ax.grid(linestyle=":", alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(DOC_FIG_DIR / "pocket_cliff_rate_comparison.png", dpi=150)
    plt.close(fig)
    logger.info(f"Wrote {DOC_FIG_DIR / 'pocket_cliff_rate_comparison.png'}")

    # Figure 3: per-pocket pKD violins
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pocket_data = []
    pocket_labels = []
    for pocket in sorted(pocket_counts.keys()):
        mask = [pocket_assignments[i] == pocket for i in range(n)]
        pocket_data.append(pkd[mask])
        pocket_labels.append(f"{pocket}\n(n={sum(mask)})")

    # Add unassigned
    mask_none = [pocket_assignments[i] is None for i in range(n)]
    pocket_data.append(pkd[mask_none])
    pocket_labels.append(f"None\n(n={sum(mask_none)})")

    parts = ax.violinplot(pocket_data, showmedians=True, showextrema=False)
    ax.set_xticks(range(1, len(pocket_labels) + 1))
    ax.set_xticklabels(pocket_labels)
    ax.set_ylabel("pKD")
    ax.set_title("pKD distribution by assigned Newman pocket")
    ax.grid(linestyle=":", alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(DOC_FIG_DIR / "pocket_pkd_violins.png", dpi=150)
    plt.close(fig)
    logger.info(f"Wrote {DOC_FIG_DIR / 'pocket_pkd_violins.png'}")

    # Save results
    output_path = ARTIFACT_DIR / "pocket_cliff_analysis.json"
    output_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Wrote {output_path}")

    # Save per-compound pocket assignments
    pocket_df = pl.DataFrame({
        "compound_id": df["compound_id"],
        "canonical_smiles": df["canonical_smiles"],
        "pKD_global_mean": pkd,
        "pocket_best": pocket_assignments,
    })
    # Add per-pocket scores
    for pocket in sorted(assigner.pocket_fps.keys()):
        pocket_df = pocket_df.with_columns(
            pl.Series(
                f"pocket_{pocket}_tc",
                [s[pocket].tanimoto if pocket in s else 0.0 for s in pocket_scores],
            )
        )
        pocket_df = pocket_df.with_columns(
            pl.Series(
                f"pocket_{pocket}_sub",
                [s[pocket].substruct if pocket in s else False for s in pocket_scores],
            )
        )

    pocket_df.write_csv(ARTIFACT_DIR / "pocket_assignments_zenodo.csv")
    logger.info(f"Wrote {ARTIFACT_DIR / 'pocket_assignments_zenodo.csv'}")


if __name__ == "__main__":
    main()
