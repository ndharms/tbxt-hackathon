"""Diagnostic 01: batch-to-batch consistency of the pKD target.

Uses the full 2,153-record SPR table ``data/zenodo/tbxt_spr_merged.csv``
which preserves every individual measurement (one row = one SPR run) with
a ``reference_date`` batch identifier. Asks three questions:

1. **Within-compound reproducibility.** For the 72 compounds measured in
   >=2 batches, what is the std of their pKD across replicates? A perfectly
   consistent target has within-compound std ~0.
2. **Batch-level pKD shifts.** Does median pKD per batch vary? If so, the
   merged pKD is partially a batch effect, not purely a molecular property.
3. **Batch ANOVA.** How much of total pKD variance is explained by
   ``reference_date`` alone (ignoring chemistry)?

Writes:
    data/sar-diagnostics-rjg/batch_stats_per_compound.csv
    data/sar-diagnostics-rjg/batch_stats_per_batch.csv
    data/sar-diagnostics-rjg/batch_consistency_summary.json
    docs/sar-diagnostics-rjg/batch_within_compound_std.png
    docs/sar-diagnostics-rjg/batch_pkd_per_batch.png

Usage:
    uv run python scripts/sar-diagnostics-rjg/01_batch_consistency.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
SPR_CSV = ROOT / "data" / "zenodo" / "tbxt_spr_merged.csv"
ARTIFACT_DIR = ROOT / "data" / "sar-diagnostics-rjg"
DOC_FIG_DIR = ROOT / "docs" / "sar-diagnostics-rjg"


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw = pl.read_csv(SPR_CSV)
    logger.info(f"loaded SPR records: {raw.shape}")
    assert {"compound_id", "reference_date", "pKD"}.issubset(raw.columns)

    # ---- 1. Within-compound consistency ------------------------------------
    per_compound = (
        raw.group_by("compound_id")
        .agg(
            pl.len().alias("n_measurements"),
            pl.col("reference_date").n_unique().alias("n_batches"),
            pl.col("pKD").mean().alias("pKD_mean"),
            pl.col("pKD").std().alias("pKD_std"),
            pl.col("pKD").min().alias("pKD_min"),
            pl.col("pKD").max().alias("pKD_max"),
        )
        .sort("n_batches", descending=True)
    )
    per_compound = per_compound.with_columns(
        (pl.col("pKD_max") - pl.col("pKD_min")).alias("pKD_range"),
    )
    per_compound.write_csv(ARTIFACT_DIR / "batch_stats_per_compound.csv")

    multi = per_compound.filter(pl.col("n_measurements") >= 2)
    multi_batches = per_compound.filter(pl.col("n_batches") >= 2)
    logger.info(f"compounds with >=2 measurements: {multi.shape[0]}")
    logger.info(f"compounds spanning >=2 batches:  {multi_batches.shape[0]}")

    within_std = multi["pKD_std"].to_numpy()
    within_std = within_std[~np.isnan(within_std)]
    within_range = multi["pKD_range"].to_numpy()
    within_range = within_range[~np.isnan(within_range)]

    # Overall pKD std (between compounds) as the reference scale
    global_std = float(raw["pKD"].std())

    # ---- 2. Per-batch pKD distribution -------------------------------------
    per_batch = (
        raw.group_by("reference_date")
        .agg(
            pl.len().alias("n_records"),
            pl.col("compound_id").n_unique().alias("n_unique_compounds"),
            pl.col("pKD").mean().alias("pKD_mean"),
            pl.col("pKD").median().alias("pKD_median"),
            pl.col("pKD").std().alias("pKD_std"),
            pl.col("pKD").quantile(0.25).alias("pKD_q25"),
            pl.col("pKD").quantile(0.75).alias("pKD_q75"),
        )
        .sort("reference_date")
    )
    per_batch.write_csv(ARTIFACT_DIR / "batch_stats_per_batch.csv")

    batch_median_spread = float(per_batch["pKD_median"].max() - per_batch["pKD_median"].min())
    logger.info(f"median-pKD spread across batches: {batch_median_spread:.3f} pKD units")

    # ---- 3. Batch-only ANOVA (one-way) -------------------------------------
    # SSB / SST where groups are reference_date.
    overall_mean = float(raw["pKD"].mean())
    sst = float(((raw["pKD"] - overall_mean) ** 2).sum())
    batch_stats = raw.group_by("reference_date").agg(
        pl.len().alias("n"),
        pl.col("pKD").mean().alias("mu"),
    )
    ssb = float((batch_stats["n"] * (batch_stats["mu"] - overall_mean) ** 2).sum())
    r2_batch = ssb / sst if sst > 0 else float("nan")
    logger.info(
        f"variance of pKD explained by reference_date alone: R2 = {r2_batch:.3f}  "
        f"(SSB={ssb:.1f}, SST={sst:.1f})"
    )

    # ---- figures ------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].hist(within_std, bins=30, color="#37474f", edgecolor="white")
    axes[0].axvline(np.median(within_std), color="#c62828", linestyle="--",
                    label=f"median = {np.median(within_std):.3f}")
    axes[0].axvline(global_std, color="#2e7d32", linestyle=":",
                    label=f"global pKD std = {global_std:.3f}")
    axes[0].set_xlabel("Within-compound pKD std across replicates")
    axes[0].set_ylabel(f"# compounds (n = {len(within_std)})")
    axes[0].set_title("Within-compound measurement noise")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(linestyle=":", alpha=0.4)

    axes[1].hist(within_range, bins=30, color="#37474f", edgecolor="white")
    axes[1].axvline(np.median(within_range), color="#c62828", linestyle="--",
                    label=f"median = {np.median(within_range):.3f}")
    axes[1].set_xlabel("Within-compound pKD range (max - min)")
    axes[1].set_ylabel(f"# compounds (n = {len(within_range)})")
    axes[1].set_title("Within-compound peak-to-peak variability")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(DOC_FIG_DIR / "batch_within_compound_std.png", dpi=150)
    plt.close(fig)
    logger.info(f"wrote {DOC_FIG_DIR / 'batch_within_compound_std.png'}")

    # Per-batch violin / box
    batches = per_batch["reference_date"].to_list()
    pkd_by_batch = [raw.filter(pl.col("reference_date") == b)["pKD"].to_numpy() for b in batches]

    fig, ax = plt.subplots(figsize=(11, 5))
    parts = ax.violinplot(pkd_by_batch, showmedians=True, widths=0.8)
    for pc in parts["bodies"]:
        pc.set_facecolor("#90a4ae")
        pc.set_alpha(0.7)
    ax.set_xticks(range(1, len(batches) + 1))
    ax.set_xticklabels([str(b) for b in batches], rotation=45, ha="right", fontsize=8)
    ax.axhline(overall_mean, color="#c62828", linestyle=":", linewidth=1,
               label=f"overall mean pKD = {overall_mean:.2f}")
    ax.set_ylabel("pKD")
    ax.set_xlabel("reference_date (batch)")
    ax.set_title(
        f"pKD distribution by batch (n={len(batches)} batches, "
        f"median-pKD spread = {batch_median_spread:.2f} pKD units)"
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(DOC_FIG_DIR / "batch_pkd_per_batch.png", dpi=150)
    plt.close(fig)
    logger.info(f"wrote {DOC_FIG_DIR / 'batch_pkd_per_batch.png'}")

    # ---- summary JSON ------------------------------------------------------
    summary = {
        "n_records": int(raw.shape[0]),
        "n_compounds": int(raw["compound_id"].n_unique()),
        "n_batches": int(raw["reference_date"].n_unique()),
        "n_compounds_multi_measurement": int(multi.shape[0]),
        "n_compounds_multi_batch": int(multi_batches.shape[0]),
        "within_compound_pKD_std": {
            "n": int(len(within_std)),
            "median": float(np.median(within_std)),
            "mean": float(np.mean(within_std)),
            "p90": float(np.quantile(within_std, 0.9)),
            "max": float(np.max(within_std)),
        },
        "within_compound_pKD_range": {
            "median": float(np.median(within_range)),
            "mean": float(np.mean(within_range)),
            "p90": float(np.quantile(within_range, 0.9)),
            "max": float(np.max(within_range)),
        },
        "global_pKD_std": global_std,
        "batch_median_pKD_spread": batch_median_spread,
        "r2_batch_only_anova": r2_batch,
    }
    (ARTIFACT_DIR / "batch_consistency_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"wrote summary -> {ARTIFACT_DIR / 'batch_consistency_summary.json'}")


if __name__ == "__main__":
    main()
