"""Step 04 (try4): rank-based comparison of all 9 XGB variants.

Adapts the ``08_compare_ensembles_by_rank.py`` template from try1/2/3 to a
9-ensemble matrix. Column names in the holdout CSVs follow the pattern
``p_binder_xgb_<variant>_ensemble_mean``.

Outputs
-------
    data/classification-models-try4-rjg/holdout_rank_per_positive.csv
    data/classification-models-try4-rjg/holdout_rank_summary.json
    data/classification-models-try4-rjg/holdout_rank_tukey.txt
    data/classification-models-try4-rjg/holdout_rank_wilcoxon.txt
    docs/classification-models-try4-rjg/holdout_rank_tukey.png
    docs/classification-models-try4-rjg/holdout_rank_box.png

Usage
-----
    uv run python scripts/classification-models-try4-rjg/04_compare_ensembles_by_rank.py
"""

from __future__ import annotations

import json
from itertools import combinations, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from scipy.stats import wilcoxon
from statsmodels.stats.multicomp import pairwise_tukeyhsd

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "data" / "classification-models-try4-rjg"
DOC_FIG_DIR = ROOT / "docs" / "classification-models-try4-rjg"

# Build the variant list from the same dimensions as 03_train_xgb_matrix.py
FP_NAMES = ("morgan", "rdkit", "maccs")
FEAT_SETS = ("fp", "fp_pocket", "fp_pocket_phys")
VARIANTS = [f"{fp}_{fs}" for fp, fs in product(FP_NAMES, FEAT_SETS)]


def _load_predictions() -> pl.DataFrame:
    """Return a wide frame: compound_id, is_binder, + one column per variant."""
    base = None
    for variant in VARIANTS:
        fp = ARTIFACT_DIR / f"xgb_{variant}_holdout.csv"
        col = f"p_binder_xgb_{variant}_ensemble_mean"
        if not fp.exists():
            raise FileNotFoundError(f"missing ensemble predictions: {fp}")
        df = pl.read_csv(fp).select(
            ["compound_id", "is_binder", pl.col(col).alias(variant)],
        )
        base = df if base is None else base.join(df, on=["compound_id", "is_binder"])
    assert base is not None
    return base


def _fractional_ranks(preds: pl.DataFrame, col: str) -> np.ndarray:
    scores = preds[col].to_numpy()
    y = preds["is_binder"].to_numpy().astype(bool)
    n = scores.size
    pos_scores = scores[y]
    frac = np.empty(pos_scores.size, dtype=np.float64)
    for i, s in enumerate(pos_scores):
        above = int((scores > s).sum())
        tied = int((scores == s).sum()) - 1
        frac[i] = (above + 0.5 * tied) / max(n - 1, 1)
    return frac


def _wilcoxon_pairs(
    ranks: dict[str, np.ndarray], variants: list[str],
) -> list[dict]:
    rows: list[dict] = []
    n_pos = len(next(iter(ranks.values())))
    for a, b in combinations(variants, 2):
        diff = ranks[a] - ranks[b]
        if n_pos < 2 or np.allclose(diff, 0):
            rows.append({
                "group1": a, "group2": b, "n_positives": int(n_pos),
                "median_diff": float(np.median(diff)) if n_pos else float("nan"),
                "wilcoxon_stat": float("nan"),
                "wilcoxon_p": float("nan"),
                "note": "insufficient data or no differences",
            })
            continue
        try:
            res = wilcoxon(ranks[a], ranks[b], zero_method="wilcox",
                           alternative="two-sided")
            rows.append({
                "group1": a, "group2": b, "n_positives": int(n_pos),
                "median_diff": float(np.median(diff)),
                "wilcoxon_stat": float(res.statistic),
                "wilcoxon_p": float(res.pvalue),
                "note": "",
            })
        except ValueError as e:
            rows.append({
                "group1": a, "group2": b, "n_positives": int(n_pos),
                "median_diff": float(np.median(diff)),
                "wilcoxon_stat": float("nan"),
                "wilcoxon_p": float("nan"),
                "note": str(e),
            })
    return rows


def _tukey_plot(
    tukey_result,
    variants: list[str],
    means: dict[str, float],
    save_path: Path,
    random_line: float = 0.5,
) -> None:
    best_name = min(variants, key=lambda n: means[n])
    fig = tukey_result.plot_simultaneous(
        comparison_name=best_name,
        ylabel="Variant",
        xlabel="Mean per-positive fractional rank (95% family-wise CI)",
    )
    ax = fig.axes[0]
    fig.set_size_inches(9, 6)
    cur_xmin, cur_xmax = ax.get_xlim()
    ax.set_xlim(min(cur_xmin, 0.0), max(cur_xmax, random_line + 0.05))
    ax.axvline(
        random_line,
        linestyle=":", color="#c62828", linewidth=1.5,
        label=f"random ordering (mean rank = {random_line:.2f})",
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_title(
        "TukeyHSD on per-positive fractional rank (try4 holdout, 9 variants)\n"
        f"comparison_name='{best_name}' (best = lowest mean): red = significantly different"
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"wrote Tukey rank plot to {save_path}")


def _box_plot(
    ranks: dict[str, np.ndarray],
    means: dict[str, float],
    save_path: Path,
    random_line: float = 0.5,
) -> None:
    # Color by fingerprint type for quick visual scanning
    fp_color = {"morgan": "#1f77b4", "rdkit": "#2ca02c", "maccs": "#d62728"}

    ordered = sorted(means, key=lambda n: means[n])
    best = ordered[0]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.boxplot(
        [ranks[n] for n in ordered],
        tick_labels=ordered,
        widths=0.6,
        boxprops=dict(linewidth=1.0),
        medianprops=dict(color="black", linewidth=1.5),
        showfliers=False,
    )
    rng = np.random.default_rng(0)
    for i, n in enumerate(ordered, start=1):
        vals = ranks[n]
        fp_prefix = n.split("_", 1)[0]
        color = fp_color.get(fp_prefix, "#78909c")
        jitter = rng.normal(0, 0.06, size=vals.size)
        ax.scatter(
            np.full_like(vals, i, dtype=float) + jitter,
            vals,
            s=16, alpha=0.55, color=color,
            edgecolor="black" if n == best else "none",
            linewidth=0.6,
        )
        ax.scatter([i], [float(np.mean(vals))], marker="D", color="black", s=28, zorder=5)
    ax.axhline(
        random_line,
        linestyle=":", color="#c62828", linewidth=1.2,
        label=f"random (mean rank = {random_line:.2f})",
    )
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Fractional rank of positive (lower = better)")
    ax.set_title(
        f"Holdout per-positive rank, 9 XGB variants (best: {best}). "
        f"Colors: blue=morgan, green=rdkit, red=maccs"
    )
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"wrote rank box plot to {save_path}")


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_FIG_DIR.mkdir(parents=True, exist_ok=True)

    preds = _load_predictions()
    n_total = preds.shape[0]
    n_positives = int(preds["is_binder"].sum())
    logger.info(
        f"loaded 9-variant holdout predictions: n={n_total} positives={n_positives}"
    )
    if n_positives < 2:
        raise ValueError(
            f"need >= 2 positives for rank comparison; got {n_positives}"
        )

    # Fractional ranks
    ranks: dict[str, np.ndarray] = {}
    for variant in VARIANTS:
        ranks[variant] = _fractional_ranks(preds, variant)
    logger.info("per-variant rank stats (mean, median):")
    for variant in VARIANTS:
        logger.info(
            f"  {variant:<28s} mean={np.mean(ranks[variant]):.3f} "
            f"median={np.median(ranks[variant]):.3f} "
            f"min={np.min(ranks[variant]):.3f} max={np.max(ranks[variant]):.3f}"
        )

    # Wide per-positive table
    pos_df = preds.filter(pl.col("is_binder")).select(["compound_id", "is_binder"])
    pos_df = pos_df.with_columns(
        [pl.Series(f"rank_{v}", ranks[v]) for v in VARIANTS]
    )
    pos_df.write_csv(ARTIFACT_DIR / "holdout_rank_per_positive.csv")

    # Long format for TukeyHSD
    long = pl.concat(
        [
            pl.DataFrame({
                "compound_id": pos_df["compound_id"],
                "model": [v] * n_positives,
                "rank": ranks[v],
            })
            for v in VARIANTS
        ]
    )
    values = long["rank"].to_numpy()
    groups = long["model"].to_numpy()
    tukey = pairwise_tukeyhsd(endog=values, groups=groups, alpha=0.05)
    tukey_text = str(tukey.summary())
    (ARTIFACT_DIR / "holdout_rank_tukey.txt").write_text(tukey_text)

    tukey_rows = []
    for row in tukey.summary().data[1:]:
        tukey_rows.append({
            "group1": row[0], "group2": row[1],
            "meandiff": float(row[2]), "p_adj": float(row[3]),
            "lower": float(row[4]), "upper": float(row[5]),
            "reject": bool(row[6]),
        })
    n_tukey_reject = sum(1 for r in tukey_rows if r["reject"])
    logger.info(f"Tukey: {n_tukey_reject}/{len(tukey_rows)} pairs reject at FWER=0.05")

    means = {v: float(np.mean(ranks[v])) for v in VARIANTS}
    medians = {v: float(np.median(ranks[v])) for v in VARIANTS}

    _tukey_plot(tukey, VARIANTS, means, DOC_FIG_DIR / "holdout_rank_tukey.png")
    _box_plot(ranks, means, DOC_FIG_DIR / "holdout_rank_box.png")

    # Paired Wilcoxon (36 pairs for 9 variants)
    wilcoxon_rows = _wilcoxon_pairs(ranks, VARIANTS)
    # Bonferroni: alpha / n_pairs = 0.05 / 36 = 0.00139
    n_pairs = len(wilcoxon_rows)
    bonf = 0.05 / n_pairs
    lines = [
        f"Paired Wilcoxon signed-rank, n_positives = {n_positives}, "
        f"{n_pairs} pairs (9 variants choose 2).",
        f"alpha = 0.05 (uncorrected); Bonferroni = 0.05 / {n_pairs} = {bonf:.4f}.",
        "",
        f"{'group1':<28s} {'group2':<28s} {'median_diff':>12s} "
        f"{'w_stat':>10s} {'p':>10s} {'sig':>5s} {'bonf':>5s}",
    ]
    n_sig_uncorrected = 0
    n_sig_bonferroni = 0
    for row in wilcoxon_rows:
        p = row["wilcoxon_p"]
        sig = ("yes" if (not np.isnan(p) and p < 0.05) else "no")
        bonf_sig = ("yes" if (not np.isnan(p) and p < bonf) else "no")
        if sig == "yes":
            n_sig_uncorrected += 1
        if bonf_sig == "yes":
            n_sig_bonferroni += 1
        lines.append(
            f"{row['group1']:<28s} {row['group2']:<28s} "
            f"{row['median_diff']:>12.4f} {row['wilcoxon_stat']:>10.3f} "
            f"{p:>10.4f} {sig:>5s} {bonf_sig:>5s}"
            + (f"  # {row['note']}" if row["note"] else "")
        )
    lines.append("")
    lines.append(
        f"Summary: {n_sig_uncorrected}/{n_pairs} pairs p < 0.05 (uncorrected), "
        f"{n_sig_bonferroni}/{n_pairs} pass Bonferroni."
    )
    wilcoxon_text = "\n".join(lines)
    (ARTIFACT_DIR / "holdout_rank_wilcoxon.txt").write_text(wilcoxon_text + "\n")
    logger.info(
        "\nWilcoxon: " + str(n_sig_uncorrected) + "/" + str(n_pairs)
        + " uncorrected, " + str(n_sig_bonferroni) + "/" + str(n_pairs)
        + " Bonferroni."
    )

    best_by_mean = min(means, key=lambda v: means[v])
    best_by_median = min(medians, key=lambda v: medians[v])
    summary = {
        "n_holdout": n_total,
        "n_positives": n_positives,
        "n_negatives": n_total - n_positives,
        "random_baseline_mean_rank": 0.5,
        "variants": VARIANTS,
        "ensemble_mean_rank": means,
        "ensemble_median_rank": medians,
        "ensemble_min_rank": {v: float(np.min(ranks[v])) for v in VARIANTS},
        "ensemble_max_rank": {v: float(np.max(ranks[v])) for v in VARIANTS},
        "best_by_mean_rank": best_by_mean,
        "best_by_median_rank": best_by_median,
        "tukeyhsd_alpha": 0.05,
        "tukeyhsd_n_reject": n_tukey_reject,
        "tukeyhsd_pairs": tukey_rows,
        "wilcoxon_n_sig_uncorrected": n_sig_uncorrected,
        "wilcoxon_n_sig_bonferroni": n_sig_bonferroni,
        "wilcoxon_pairs": wilcoxon_rows,
    }
    (ARTIFACT_DIR / "holdout_rank_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        f"best by mean rank: {best_by_mean} ({means[best_by_mean]:.3f}); "
        f"best by median: {best_by_median} ({medians[best_by_median]:.3f})"
    )


if __name__ == "__main__":
    main()
