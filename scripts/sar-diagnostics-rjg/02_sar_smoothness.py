"""Diagnostic 02: structure-activity smoothness.

For every pair of compounds compute Tanimoto (Morgan FP, r=2, 2048 bits)
and |pKD_i - pKD_j|. Smooth, learnable SAR: |dpKD| -> 0 as Tanimoto -> 1.
Pathological SAR (lots of activity cliffs): high |dpKD| at high Tanimoto.

Also computes the per-compound "k=5 nearest-neighbor pKD std" in Morgan
space. If this is comparable to the global pKD std, structure doesn't
predict activity locally and no model can learn a smooth regressor.

Writes:
    data/sar-diagnostics-rjg/sar_pairwise_summary.json
    data/sar-diagnostics-rjg/sar_knn_pkd_std.csv
    docs/sar-diagnostics-rjg/sar_tanimoto_vs_dpkd.png
    docs/sar-diagnostics-rjg/sar_sali_cdf.png
    docs/sar-diagnostics-rjg/sar_knn_pkd_std.png

Usage:
    uv run python scripts/sar-diagnostics-rjg/02_sar_smoothness.py
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

ROOT = Path(__file__).resolve().parents[2]
FOLDS_CSV = ROOT / "data" / "folds-pacmap-kmeans6" / "fold_assignments.csv"
ARTIFACT_DIR = ROOT / "data" / "sar-diagnostics-rjg"
DOC_FIG_DIR = ROOT / "docs" / "sar-diagnostics-rjg"
TARGET_COL = "pKD_global_mean"


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(FOLDS_CSV)
    smiles = df["canonical_smiles"].to_list()
    pkd = df[TARGET_COL].to_numpy().astype(np.float64)
    n = len(smiles)
    logger.info(f"loaded {n} compounds")

    fps = morgan_bitvects(smiles)

    # ---- 1. Pairwise Tanimoto vs |dpKD| ------------------------------------
    # Upper-triangle only (no self-pairs, no double count).
    # Computing 1599 x 1599 is ~2.5M pairs; bulk helper handles it in seconds.
    logger.info("computing pairwise Tanimoto + |dpKD| ...")
    tan_all: list[np.ndarray] = []
    dpkd_all: list[np.ndarray] = []
    for i in range(n - 1):
        # Tanimoto(fp_i, fps[i+1:]) -> list[float] length n - i - 1
        sims = np.asarray(BulkTanimotoSimilarity(fps[i], fps[i + 1:]), dtype=np.float32)
        dpk = np.abs(pkd[i + 1:] - pkd[i]).astype(np.float32)
        tan_all.append(sims)
        dpkd_all.append(dpk)
    tanimoto = np.concatenate(tan_all)
    dpkd = np.concatenate(dpkd_all)
    n_pairs = len(tanimoto)
    logger.info(f"computed {n_pairs:,} unique pairs")

    # Binned smoothness: median |dpKD| in Tanimoto bins
    edges = np.linspace(0.0, 1.0, 21)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(tanimoto, edges) - 1, 0, len(centers) - 1)
    median_dpkd_per_bin = np.full(len(centers), np.nan)
    q25_dpkd_per_bin = np.full(len(centers), np.nan)
    q75_dpkd_per_bin = np.full(len(centers), np.nan)
    n_per_bin = np.zeros(len(centers), dtype=np.int64)
    for b in range(len(centers)):
        mask = bin_idx == b
        n_per_bin[b] = int(mask.sum())
        if n_per_bin[b] >= 10:
            median_dpkd_per_bin[b] = float(np.median(dpkd[mask]))
            q25_dpkd_per_bin[b] = float(np.quantile(dpkd[mask], 0.25))
            q75_dpkd_per_bin[b] = float(np.quantile(dpkd[mask], 0.75))

    # Spearman of Tanimoto vs |dpKD| - smooth SAR -> strong negative rho
    # scipy's spearmanr chokes on 1.3M pairs; we'll just compute on a random sample
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n_pairs, size=min(200_000, n_pairs), replace=False)
    from scipy.stats import spearmanr  # local import
    rho_sample, pval_sample = spearmanr(tanimoto[sample_idx], dpkd[sample_idx])

    # Activity cliffs: pairs with Tanimoto >= 0.7 and |dpKD| >= 2
    high_sim_mask = tanimoto >= 0.7
    high_sim_cliff_mask = high_sim_mask & (dpkd >= 2.0)
    n_high_sim_pairs = int(high_sim_mask.sum())
    n_cliffs = int(high_sim_cliff_mask.sum())
    cliff_frac = n_cliffs / max(n_high_sim_pairs, 1)
    logger.info(
        f"pairs with Tanimoto >= 0.7: {n_high_sim_pairs:,}  "
        f"of which |dpKD| >= 2: {n_cliffs:,}  ({100*cliff_frac:.1f}%)"
    )

    # SALI: |dpKD| / (1 - Tanimoto). Clip similarity to avoid div-by-zero.
    sali_similarity_floor = 1e-3
    denom = np.maximum(1.0 - tanimoto, sali_similarity_floor)
    sali = dpkd / denom

    # ---- 2. kNN pKD std in Morgan space ------------------------------------
    logger.info("computing kNN (k=5) pKD std in Morgan space ...")
    k = 5
    knn_std = np.zeros(n, dtype=np.float64)
    knn_mean_tanimoto = np.zeros(n, dtype=np.float64)
    for i in range(n):
        sims = np.asarray(BulkTanimotoSimilarity(fps[i], fps), dtype=np.float32)
        sims[i] = -1.0  # exclude self
        top_k = np.argsort(-sims)[:k]
        knn_std[i] = float(np.std(pkd[top_k]))
        knn_mean_tanimoto[i] = float(sims[top_k].mean())
    global_pkd_std = float(np.std(pkd))

    knn_frame = pl.DataFrame(
        {
            "compound_id": df["compound_id"],
            "pKD": pkd,
            "knn_mean_tanimoto": knn_mean_tanimoto,
            "knn_pKD_std": knn_std,
        }
    )
    knn_frame.write_csv(ARTIFACT_DIR / "sar_knn_pkd_std.csv")

    logger.info(
        f"kNN pKD std: median={np.median(knn_std):.3f}  mean={np.mean(knn_std):.3f}  "
        f"global pKD std={global_pkd_std:.3f}"
    )

    # ---- figures ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(centers, median_dpkd_per_bin, marker="o", color="#1565c0",
            label="median |dpKD|")
    ax.fill_between(centers, q25_dpkd_per_bin, q75_dpkd_per_bin,
                    alpha=0.25, color="#1565c0", label="IQR")
    ax.axhline(float(np.median(dpkd)), color="#c62828", linestyle=":",
               label=f"global median |dpKD| = {np.median(dpkd):.2f}")
    ax.set_xlabel("Tanimoto similarity (Morgan r=2, 2048 bits)")
    ax.set_ylabel("|pKD_i - pKD_j|")
    ax.set_title(
        f"Pair SAR smoothness  (n_pairs = {n_pairs:,}, "
        f"Spearman rho[Tan, |dpKD|] on 200k sample = {rho_sample:+.3f})"
    )
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(DOC_FIG_DIR / "sar_tanimoto_vs_dpkd.png", dpi=150)
    plt.close(fig)
    logger.info(f"wrote {DOC_FIG_DIR / 'sar_tanimoto_vs_dpkd.png'}")

    # SALI CDF: cumulative distribution of SALI values
    # SALI is heavy-tailed; plot CDF on the log scale for high-tail visibility
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sali_sorted = np.sort(sali[high_sim_mask])  # focus on the high-similarity pairs
    if sali_sorted.size > 0:
        cdf = np.arange(1, sali_sorted.size + 1) / sali_sorted.size
        ax.plot(sali_sorted, cdf, color="#1565c0")
        ax.set_xscale("log")
    ax.set_xlabel("SALI = |dpKD| / (1 - Tanimoto)  (Tanimoto >= 0.7 pairs only)")
    ax.set_ylabel("Cumulative fraction of pairs")
    ax.set_title(
        f"SALI distribution (high-similarity pairs, n = {n_high_sim_pairs:,})\n"
        f"{100*cliff_frac:.1f}% of Tanimoto >= 0.7 pairs have |dpKD| >= 2 (activity cliffs)"
    )
    ax.grid(linestyle=":", alpha=0.4, which="both")
    fig.tight_layout()
    fig.savefig(DOC_FIG_DIR / "sar_sali_cdf.png", dpi=150)
    plt.close(fig)
    logger.info(f"wrote {DOC_FIG_DIR / 'sar_sali_cdf.png'}")

    # kNN std distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(knn_std, bins=40, color="#37474f", edgecolor="white")
    ax.axvline(float(np.median(knn_std)), color="#c62828", linestyle="--",
               label=f"median = {np.median(knn_std):.2f}")
    ax.axvline(global_pkd_std, color="#2e7d32", linestyle=":",
               label=f"global pKD std = {global_pkd_std:.2f}")
    ax.set_xlabel(f"Std of pKD over k={k} nearest Morgan-FP neighbors")
    ax.set_ylabel(f"# compounds (n = {n})")
    ax.set_title("Local pKD variability in Morgan FP space")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(DOC_FIG_DIR / "sar_knn_pkd_std.png", dpi=150)
    plt.close(fig)
    logger.info(f"wrote {DOC_FIG_DIR / 'sar_knn_pkd_std.png'}")

    # ---- summary JSON -------------------------------------------------------
    summary = {
        "n_compounds": n,
        "n_pairs": int(n_pairs),
        "spearman_tanimoto_vs_abs_dpKD_sampled": {
            "sample_size": int(sample_idx.size),
            "rho": float(rho_sample),
            "pvalue": float(pval_sample),
        },
        "global_median_abs_dpKD": float(np.median(dpkd)),
        "binned_median_abs_dpKD_by_tanimoto": {
            f"{centers[b]:.3f}": {
                "n_pairs": int(n_per_bin[b]),
                "median_abs_dpKD": (
                    float(median_dpkd_per_bin[b])
                    if not np.isnan(median_dpkd_per_bin[b])
                    else None
                ),
            }
            for b in range(len(centers))
        },
        "high_similarity_pairs_tanimoto_ge_0p7": {
            "n_pairs": n_high_sim_pairs,
            "n_activity_cliffs_abs_dpKD_ge_2": n_cliffs,
            "fraction_cliffs": float(cliff_frac),
        },
        "knn_pKD_std_k5_morgan": {
            "median": float(np.median(knn_std)),
            "mean": float(np.mean(knn_std)),
            "p90": float(np.quantile(knn_std, 0.9)),
            "global_pKD_std": global_pkd_std,
            "median_ratio_to_global": float(np.median(knn_std) / global_pkd_std),
        },
    }
    (ARTIFACT_DIR / "sar_pairwise_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"wrote summary -> {ARTIFACT_DIR / 'sar_pairwise_summary.json'}")


if __name__ == "__main__":
    main()
