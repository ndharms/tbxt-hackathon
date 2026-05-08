"""Step 07: TukeyHSD comparison of all four ensembles on the holdout set.

For each of the 131 holdout compounds we compute per-compound Brier score
(squared error vs. binary label) for every ensemble. Lower = better
calibrated prediction. TukeyHSD then tests whether pairwise mean-Brier
differences are significant at family-wise alpha=0.05.

Ensembles compared:
    - chemeleon_with_val:    3-train / 1-val (early-stop) / 1-test
    - chemeleon_no_val:      4-train (15 fixed epochs) / 1-test
    - xgb_with_val:          3-train / 1-val (early-stop) / 1-test
    - xgb_no_val:            4-train (100 fixed trees) / 1-test

Writes:
    data/models/holdout_comparison_brier.csv   (per-compound per-model)
    data/models/holdout_tukey_hsd.txt          (TukeyHSD summary)
    data/models/holdout_comparison_summary.json
    docs/figures/holdout_tukey_hsd.png         (statsmodels plot; best annotated)
    docs/figures/holdout_brier_box.png         (box+strip per ensemble)

Usage:
    uv run python scripts/07_compare_ensembles.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score
from statsmodels.stats.multicomp import pairwise_tukeyhsd

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "data" / "classification-models-try1-rjg"
DOC_FIG_DIR = ROOT / "docs" / "classification-models-try1-rjg"

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


def _tukey_plot_with_best(
    tukey_result,
    ordered_names: list[str],
    means: dict[str, float],
    save_path: Path,
    baseline_brier: float,
    metric: str = "Brier (squared error)",
) -> None:
    """Render Tukey simultaneous CIs using statsmodels' built-in callout.

    Passes the best group (lowest mean Brier) as ``comparison_name`` so
    statsmodels color-codes groups significantly different from it in red
    and insignificant overlaps in gray. Adds a dotted vertical line at
    ``baseline_brier`` (the expected Brier of a prevalence-only predictor)
    so model performance can be judged against the no-information floor.
    """
    best_name = min(ordered_names, key=lambda n: means[n])
    fig = tukey_result.plot_simultaneous(
        comparison_name=best_name,
        ylabel="Ensemble",
        xlabel=f"Mean per-compound {metric} (95% family-wise CI)",
    )
    ax = fig.axes[0]
    # Ensure the baseline line is visible (it may fall outside the auto xlim)
    cur_xmin, cur_xmax = ax.get_xlim()
    new_xmin = min(cur_xmin, baseline_brier - 0.01)
    ax.set_xlim(new_xmin, cur_xmax)
    ax.axvline(
        baseline_brier,
        linestyle=":",
        color="#c62828",
        linewidth=1.5,
        label=f"prevalence-only baseline (Brier = {baseline_brier:.3f})",
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_title(
        "TukeyHSD on per-compound Brier (holdout fold)\n"
        f"comparison_name='{best_name}' (lowest mean): "
        "red = significantly different, gray = not"
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"wrote TukeyHSD plot to {save_path}")


def _box_plot(
    brier_long: pl.DataFrame, means: dict[str, float], save_path: Path,
) -> None:
    """Box + strip plot of per-compound Brier, best model highlighted."""
    ordered = sorted(means, key=lambda n: means[n])
    best = ordered[0]
    palette = {n: ("#2e7d32" if n == best else "#78909c") for n in ordered}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals_by_model = {n: brier_long.filter(pl.col("model") == n)["brier"].to_numpy() for n in ordered}
    ax.boxplot(
        [vals_by_model[n] for n in ordered],
        tick_labels=ordered,
        widths=0.6,
        boxprops=dict(linewidth=1.0),
        medianprops=dict(color="black", linewidth=1.5),
        showfliers=False,
    )
    for i, n in enumerate(ordered, start=1):
        vals = vals_by_model[n]
        jitter = np.random.default_rng(0).normal(0, 0.06, size=vals.size)
        ax.scatter(
            np.full_like(vals, i, dtype=float) + jitter,
            vals,
            s=8, alpha=0.5, color=palette[n],
        )
        ax.scatter([i], [means[n]], marker="D", color="black", s=30, zorder=5)
    ax.set_ylabel("Per-compound Brier score (lower = better)")
    ax.set_title(f"Holdout Brier distributions by ensemble (best: {best})")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"wrote box plot to {save_path}")


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_FIG_DIR.mkdir(parents=True, exist_ok=True)

    preds = _load_predictions()
    logger.info(f"loaded holdout predictions: {preds.shape}")
    y = preds["is_binder"].to_numpy().astype(np.int64)

    # Per-compound Brier = (p - y)^2 per model; AUROC/AUPRC at ensemble level
    brier_cols = {}
    ensemble_metrics: dict[str, dict[str, float]] = {}
    for name in MODEL_LABELS:
        p = preds[name].to_numpy().astype(np.float64)
        brier_cols[name] = (p - y) ** 2
        ensemble_metrics[name] = {
            "mean_brier": float(np.mean(brier_cols[name])),
            "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
            "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        }
        logger.info(
            f"{name}: mean_brier={ensemble_metrics[name]['mean_brier']:.4f} "
            f"AUROC={ensemble_metrics[name]['auroc']:.3f} "
            f"AUPRC={ensemble_metrics[name]['auprc']:.3f}"
        )

    # Wide per-compound Brier table
    wide = preds.select(["compound_id", "is_binder"]).with_columns(
        [pl.Series(f"brier_{n}", brier_cols[n]) for n in MODEL_LABELS]
    )
    wide.write_csv(ARTIFACT_DIR / "holdout_comparison_brier.csv")

    # Long format for TukeyHSD
    long = pl.concat(
        [
            pl.DataFrame(
                {
                    "compound_id": preds["compound_id"],
                    "model": [n] * preds.shape[0],
                    "brier": brier_cols[n],
                }
            )
            for n in MODEL_LABELS
        ]
    )
    values = long["brier"].to_numpy()
    groups = long["model"].to_numpy()

    tukey = pairwise_tukeyhsd(endog=values, groups=groups, alpha=0.05)
    summary_text = str(tukey.summary())
    (ARTIFACT_DIR / "holdout_tukey_hsd.txt").write_text(summary_text)
    logger.info("\n" + summary_text)

    ordered_names = list(MODEL_LABELS.keys())
    means = {n: ensemble_metrics[n]["mean_brier"] for n in ordered_names}

    # Baseline Brier for a constant-prevalence predictor:
    # a model that always predicts p = P(y=1) has expected Brier = p * (1-p).
    prevalence = float(y.mean())
    baseline_brier = prevalence * (1.0 - prevalence)
    logger.info(
        f"holdout prevalence = {prevalence:.3f} -> "
        f"prevalence-only baseline Brier = {baseline_brier:.4f}"
    )

    _tukey_plot_with_best(
        tukey, ordered_names, means,
        DOC_FIG_DIR / "holdout_tukey_hsd.png",
        baseline_brier=baseline_brier,
    )
    _box_plot(long, means, DOC_FIG_DIR / "holdout_brier_box.png")

    # Persist a JSON summary for easy downstream consumption.
    # TukeyHSD results are accessed via .summary()'s tabular data.
    tukey_rows = []
    for row in tukey.summary().data[1:]:
        tukey_rows.append({
            "group1": row[0],
            "group2": row[1],
            "meandiff": float(row[2]),
            "p_adj": float(row[3]),
            "lower": float(row[4]),
            "upper": float(row[5]),
            "reject": bool(row[6]),
        })

    best = min(means, key=means.get)  # type: ignore[arg-type]
    summary = {
        "n_holdout": int(preds.shape[0]),
        "n_holdout_positives": int(y.sum()),
        "holdout_prevalence": prevalence,
        "baseline_brier_prevalence_only": baseline_brier,
        "ensemble_metrics": ensemble_metrics,
        "best_by_mean_brier": best,
        "tukeyhsd_alpha": 0.05,
        "tukeyhsd_pairs": tukey_rows,
    }
    (ARTIFACT_DIR / "holdout_comparison_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"best ensemble by mean Brier: {best}")
    logger.info(f"wrote summary -> {ARTIFACT_DIR / 'holdout_comparison_summary.json'}")


if __name__ == "__main__":
    main()
