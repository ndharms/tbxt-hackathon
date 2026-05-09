"""Step 08: rank-based ensemble comparison on the holdout set.

Why this replaces 07's Brier comparison:

- For this project the deployment task is "pick top-K compounds from a
  3.4B library to send for SPR". The metric that matches that task is
  how highly the model ranks true positives, not how calibrated its
  probabilities are on true negatives.
- Brier (squared error) on low-prevalence holdouts is dominated by the
  large negative class; 157 true negatives drown out 29 true positives
  in the mean. TukeyHSD on Brier has failed to reject across try1, try2,
  and try3 while AUROC / AUPRC / top-K precision clearly separated the
  ensembles.

Fractional rank of a positive
-----------------------------
For each true positive, compute:

    fractional_rank = (# holdout compounds with strictly higher score) / (N - 1)

A perfect model gets fractional_rank = 0 for every positive; random
ordering gets ~0.5; worst possible is 1. Ties are broken by averaging.

Values are bounded in [0, 1] regardless of prevalence, so rank
distributions from try1 (20 positives), try2 (1 positive), and try3
(29 positives) are on comparable scales.

Statistical tests
-----------------
Each holdout positive contributes one fractional rank per ensemble; the
values are paired (same compound under four models). We report:

1. Paired Wilcoxon signed-rank on each ensemble pair (accounts for the
   within-compound pairing; more power than Tukey when positives vary
   in intrinsic rankability).
2. TukeyHSD (unpaired) on the flat per-positive-per-ensemble values,
   for visual consistency with the existing 07_* plots.

With only 1 positive (try2), neither test runs meaningfully; that script
writes a note-only summary.

Outputs
-------
    <artifact_dir>/holdout_rank_per_positive.csv     # wide: pos_id x ensemble
    <artifact_dir>/holdout_rank_summary.json         # medians, CIs, tests
    <artifact_dir>/holdout_rank_tukey.txt            # statsmodels TukeyHSD
    <artifact_dir>/holdout_rank_wilcoxon.txt         # paired pairwise table
    <doc_fig_dir>/holdout_rank_tukey.png             # Tukey simultaneous CI
    <doc_fig_dir>/holdout_rank_box.png               # per-positive rank box+strip

Usage
-----
    uv run python scripts/classification-models-tryN-rjg/08_compare_ensembles_by_rank.py

This file is the shared template; the per-try copies only override
ARTIFACT_DIR / DOC_FIG_DIR constants. Re-run freely -- reads saved
holdout CSVs, writes new artifacts, touches nothing else.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from scipy.stats import wilcoxon
from statsmodels.stats.multicomp import pairwise_tukeyhsd

ROOT = Path(__file__).resolve().parents[2]
# The two path constants below are the only values the per-try copies override.
ARTIFACT_DIR = ROOT / "data" / "classification-models-try3-rjg"
DOC_FIG_DIR = ROOT / "docs" / "classification-models-try3-rjg"

MODEL_LABELS = {
    "chemeleon_with_val": ("chemeleon_with_val_holdout.csv", "p_binder_ensemble_mean"),
    "chemeleon_no_val": ("chemeleon_no_val_holdout.csv", "p_binder_novalid_ensemble_mean"),
    "xgb_with_val": ("xgb_with_val_holdout.csv", "p_binder_xgb_ensemble_mean"),
    "xgb_no_val": ("xgb_no_val_holdout.csv", "p_binder_xgb_novalid_ensemble_mean"),
}


def _load_predictions() -> pl.DataFrame:
    """Return a wide frame: compound_id, is_binder, + one column per model."""
    base = None
    for name, (fname, col) in MODEL_LABELS.items():
        path = ARTIFACT_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"missing ensemble predictions: {path}")
        df = pl.read_csv(path).select(
            ["compound_id", "is_binder", pl.col(col).alias(name)],
        )
        base = df if base is None else base.join(df, on=["compound_id", "is_binder"])
    assert base is not None
    return base


def _fractional_ranks(preds: pl.DataFrame, col: str) -> np.ndarray:
    """Fractional rank of each positive among all holdout compounds.

    Returns array of length n_positives. 0 = top, 1 = bottom.
    Ties are resolved with "average" method (compatible with AUROC's
    tie handling).
    """
    scores = preds[col].to_numpy()
    y = preds["is_binder"].to_numpy().astype(bool)
    n = scores.size

    # For each positive: count compounds with strictly greater score (above)
    # and compounds with equal score (tie). Rank = (above + 0.5 * ties_excl_self)
    pos_scores = scores[y]
    frac = np.empty(pos_scores.size, dtype=np.float64)
    for i, s in enumerate(pos_scores):
        above = int((scores > s).sum())
        tied = int((scores == s).sum()) - 1  # exclude self
        # average-rank tie handling: distribute ties evenly; divide by N-1 to
        # keep the range in [0, 1].
        frac[i] = (above + 0.5 * tied) / max(n - 1, 1)
    return frac


def _wilcoxon_pairs(
    ranks: dict[str, np.ndarray], ordered_names: list[str]
) -> list[dict]:
    """Paired Wilcoxon signed-rank for every ensemble pair.

    With < 2 positives we can't test; returns p = nan rows.
    """
    rows: list[dict] = []
    n_pos = len(next(iter(ranks.values())))
    for a, b in combinations(ordered_names, 2):
        da = ranks[a]
        db = ranks[b]
        diff = da - db
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
            res = wilcoxon(da, db, zero_method="wilcox", alternative="two-sided")
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
    ordered_names: list[str],
    medians: dict[str, float],
    save_path: Path,
    random_line: float = 0.5,
) -> None:
    """Tukey simultaneous CIs on fractional rank, lower = better."""
    best_name = min(ordered_names, key=lambda n: medians[n])
    fig = tukey_result.plot_simultaneous(
        comparison_name=best_name,
        ylabel="Ensemble",
        xlabel="Mean per-positive fractional rank (95% family-wise CI)",
    )
    ax = fig.axes[0]
    # Make sure the random-ordering line is visible
    cur_xmin, cur_xmax = ax.get_xlim()
    ax.set_xlim(min(cur_xmin, 0.0), max(cur_xmax, random_line + 0.05))
    ax.axvline(
        random_line,
        linestyle=":",
        color="#c62828",
        linewidth=1.5,
        label=f"random ordering (mean rank = {random_line:.2f})",
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_title(
        "TukeyHSD on per-positive fractional rank (holdout)\n"
        f"comparison_name='{best_name}' (best = lowest mean rank): "
        "red = significantly different"
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"wrote TukeyHSD rank plot to {save_path}")


def _box_plot(
    ranks: dict[str, np.ndarray],
    medians: dict[str, float],
    save_path: Path,
    random_line: float = 0.5,
) -> None:
    """Box + strip plot of per-positive fractional rank."""
    ordered = sorted(medians, key=lambda n: medians[n])
    best = ordered[0]
    palette = {n: ("#2e7d32" if n == best else "#78909c") for n in ordered}

    fig, ax = plt.subplots(figsize=(7, 4.5))
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
        jitter = rng.normal(0, 0.06, size=vals.size)
        ax.scatter(
            np.full_like(vals, i, dtype=float) + jitter,
            vals,
            s=16, alpha=0.6, color=palette[n],
        )
        # mean marker
        ax.scatter([i], [float(np.mean(vals))], marker="D", color="black", s=30, zorder=5)
    ax.axhline(
        random_line,
        linestyle=":", color="#c62828", linewidth=1.2,
        label=f"random ordering (mean rank = {random_line:.2f})",
    )
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Fractional rank of positive (0 = top, 1 = bottom; lower = better)")
    ax.set_title(f"Holdout fractional-rank distributions by ensemble (best: {best})")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
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
    n_negatives = n_total - n_positives
    logger.info(
        f"loaded holdout predictions: n={n_total} "
        f"(positives={n_positives}, negatives={n_negatives})"
    )
    if n_positives == 0:
        raise ValueError("holdout has no positives; cannot compute rank of positives")

    # Compute fractional ranks per ensemble
    ranks: dict[str, np.ndarray] = {}
    for name in MODEL_LABELS:
        ranks[name] = _fractional_ranks(preds, name)
        logger.info(
            f"{name}: median_rank={np.median(ranks[name]):.3f}, "
            f"mean_rank={np.mean(ranks[name]):.3f}, "
            f"best={np.min(ranks[name]):.3f}, worst={np.max(ranks[name]):.3f}"
        )

    # Wide per-positive table
    pos_df = preds.filter(pl.col("is_binder")).select(["compound_id", "is_binder"])
    pos_df = pos_df.with_columns(
        [pl.Series(f"rank_{n}", ranks[n]) for n in MODEL_LABELS]
    )
    pos_df.write_csv(ARTIFACT_DIR / "holdout_rank_per_positive.csv")

    # Long format for TukeyHSD
    long = pl.concat(
        [
            pl.DataFrame(
                {
                    "compound_id": pos_df["compound_id"],
                    "model": [n] * n_positives,
                    "rank": ranks[n],
                }
            )
            for n in MODEL_LABELS
        ]
    )

    ordered_names = list(MODEL_LABELS.keys())
    medians = {n: float(np.median(ranks[n])) for n in ordered_names}
    means = {n: float(np.mean(ranks[n])) for n in ordered_names}

    # Tukey only valid with >= 2 positives (groups need variance; 1 per group
    # is degenerate). For n=1 we skip Tukey entirely.
    tukey_rows: list[dict] = []
    tukey_summary_text = ""
    if n_positives >= 2:
        values = long["rank"].to_numpy()
        groups = long["model"].to_numpy()
        tukey = pairwise_tukeyhsd(endog=values, groups=groups, alpha=0.05)
        tukey_summary_text = str(tukey.summary())
        (ARTIFACT_DIR / "holdout_rank_tukey.txt").write_text(tukey_summary_text)
        logger.info("\n" + tukey_summary_text)
        for row in tukey.summary().data[1:]:
            tukey_rows.append({
                "group1": row[0], "group2": row[1],
                "meandiff": float(row[2]), "p_adj": float(row[3]),
                "lower": float(row[4]), "upper": float(row[5]),
                "reject": bool(row[6]),
            })
        _tukey_plot(tukey, ordered_names, means, DOC_FIG_DIR / "holdout_rank_tukey.png")
    else:
        logger.warning(
            f"only {n_positives} positive(s) on holdout; "
            "skipping TukeyHSD (requires >= 2)"
        )
        (ARTIFACT_DIR / "holdout_rank_tukey.txt").write_text(
            f"Skipped: holdout has only {n_positives} positive.\n"
            "TukeyHSD requires >= 2 per-positive values per group.\n"
        )

    # Paired Wilcoxon on the rank pairs
    wilcoxon_rows = _wilcoxon_pairs(ranks, ordered_names)
    # Also a small text table for the README
    lines = [
        "Paired Wilcoxon signed-rank on per-positive fractional rank.",
        f"n_positives = {n_positives}. alpha = 0.05 (uncorrected; Bonferroni = 0.05/6 = 0.0083).",
        "",
        f"{'group1':<22s} {'group2':<22s} {'median_diff':>12s} {'w_stat':>10s} {'p':>10s}",
    ]
    for row in wilcoxon_rows:
        lines.append(
            f"{row['group1']:<22s} {row['group2']:<22s} "
            f"{row['median_diff']:>12.4f} {row['wilcoxon_stat']:>10.3f} "
            f"{row['wilcoxon_p']:>10.4f}"
            + (f"  # {row['note']}" if row["note"] else "")
        )
    wilcoxon_text = "\n".join(lines)
    (ARTIFACT_DIR / "holdout_rank_wilcoxon.txt").write_text(wilcoxon_text + "\n")
    logger.info("\n" + wilcoxon_text)

    # Box plot (always works, even with n=1)
    _box_plot(ranks, medians, DOC_FIG_DIR / "holdout_rank_box.png")

    # JSON summary
    best_by_mean = min(means, key=lambda n: means[n])
    best_by_median = min(medians, key=lambda n: medians[n])
    summary = {
        "n_holdout": n_total,
        "n_positives": n_positives,
        "n_negatives": n_negatives,
        "random_baseline_mean_rank": 0.5,
        "ensemble_mean_rank": means,
        "ensemble_median_rank": medians,
        "ensemble_min_rank": {n: float(np.min(ranks[n])) for n in ordered_names},
        "ensemble_max_rank": {n: float(np.max(ranks[n])) for n in ordered_names},
        "best_by_mean_rank": best_by_mean,
        "best_by_median_rank": best_by_median,
        "tukeyhsd_alpha": 0.05,
        "tukeyhsd_pairs": tukey_rows,
        "wilcoxon_pairs": wilcoxon_rows,
    }
    (ARTIFACT_DIR / "holdout_rank_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        f"best by mean rank: {best_by_mean} ({means[best_by_mean]:.3f}); "
        f"best by median rank: {best_by_median} ({medians[best_by_median]:.3f})"
    )
    logger.info(f"wrote rank summary -> {ARTIFACT_DIR / 'holdout_rank_summary.json'}")


if __name__ == "__main__":
    main()
