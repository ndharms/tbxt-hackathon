"""Step 07: TukeyHSD comparison of all four regression ensembles on holdout.

Per-compound squared error ((pred - pKD)^2) is computed for every ensemble
on the 131 holdout compounds; TukeyHSD tests pairwise mean-squared-error
differences at family-wise alpha=0.05.

Ensembles compared:
    - chemeleon_with_val:    3-train / 1-val (early-stop) / 1-test
    - chemeleon_no_val:      4-train (fixed epochs) / 1-test
    - xgb_with_val:          3-train / 1-val (early-stop) / 1-test
    - xgb_no_val:            4-train (fixed n_estimators) / 1-test

Baseline: predict the mean pKD of the full non-holdout pool. The reported
baseline MSE is evaluated on the same holdout compounds, so a model with
holdout MSE above the baseline is worse than a constant predictor.

Writes:
    data/regression-models-try1-rjg/holdout_comparison_squared_error.csv
    data/regression-models-try1-rjg/holdout_tukey_hsd.txt
    data/regression-models-try1-rjg/holdout_comparison_summary.json
    docs/regression-models-try1-rjg/holdout_tukey_hsd.png
    docs/regression-models-try1-rjg/holdout_squared_error_box.png
    docs/regression-models-try1-rjg/holdout_parity.png

Usage:
    uv run python scripts/regression-models-try1-rjg/07_compare_ensembles.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.multicomp import pairwise_tukeyhsd

ROOT = Path(__file__).resolve().parents[2]
FOLDS_CSV = ROOT / "data" / "folds-pacmap-kmeans6" / "fold_assignments.csv"
ARTIFACT_DIR = ROOT / "data" / "regression-models-try1-rjg"
DOC_FIG_DIR = ROOT / "docs" / "regression-models-try1-rjg"
TARGET_COL = "pKD_global_mean"

# Each entry: (holdout csv, ensemble-mean prediction column).
MODEL_LABELS: dict[str, tuple[str, str]] = {
    "chemeleon_with_val": ("chemeleon_with_val_holdout.csv", "pred_pKD_ensemble_mean"),
    "chemeleon_no_val": ("chemeleon_no_val_holdout.csv", "pred_pKD_novalid_ensemble_mean"),
    "xgb_with_val": ("xgb_with_val_holdout.csv", "pred_pKD_xgb_ensemble_mean"),
    "xgb_no_val": ("xgb_no_val_holdout.csv", "pred_pKD_xgb_novalid_ensemble_mean"),
}


def _load_predictions() -> pl.DataFrame:
    """Return a wide frame: compound_id, pKD target, + one column per model."""
    base: pl.DataFrame | None = None
    for name, (fname, col) in MODEL_LABELS.items():
        path = ARTIFACT_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"missing ensemble predictions: {path}")
        df = pl.read_csv(path).select(
            ["compound_id", TARGET_COL, pl.col(col).alias(name)],
        )
        base = df if base is None else base.join(df, on=["compound_id", TARGET_COL])
    assert base is not None
    return base


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, pred)))


def _tukey_plot_with_best(
    tukey_result,
    ordered_names: list[str],
    means: dict[str, float],
    save_path: Path,
    baseline_mse: float,
    metric: str = "squared error",
) -> None:
    """TukeyHSD simultaneous CIs with statsmodels' built-in best-group callout.

    Passes the best group (lowest mean squared error) as ``comparison_name``
    so statsmodels color-codes significantly-different groups in red and
    insignificant overlaps in gray. Adds a dotted vertical line at the
    train-mean baseline MSE so model performance is judged against the
    no-information floor.
    """
    best_name = min(ordered_names, key=lambda n: means[n])
    fig = tukey_result.plot_simultaneous(
        comparison_name=best_name,
        ylabel="Ensemble",
        xlabel=f"Mean per-compound {metric} (95% family-wise CI)",
    )
    ax = fig.axes[0]
    cur_xmin, cur_xmax = ax.get_xlim()
    new_xmin = min(cur_xmin, baseline_mse - 0.05 * abs(baseline_mse))
    new_xmax = max(cur_xmax, baseline_mse + 0.05 * abs(baseline_mse))
    ax.set_xlim(new_xmin, new_xmax)
    ax.axvline(
        baseline_mse,
        linestyle=":",
        color="#c62828",
        linewidth=1.5,
        label=f"train-mean baseline (MSE = {baseline_mse:.3f})",
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_title(
        "TukeyHSD on per-compound squared error (holdout fold)\n"
        f"comparison_name='{best_name}' (lowest mean): "
        "red = significantly different, gray = not"
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"wrote TukeyHSD plot to {save_path}")


def _box_plot(
    se_long: pl.DataFrame, means: dict[str, float], save_path: Path,
) -> None:
    """Box + strip plot of per-compound squared error, best model highlighted."""
    ordered = sorted(means, key=lambda n: means[n])
    best = ordered[0]
    palette = {n: ("#2e7d32" if n == best else "#78909c") for n in ordered}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals_by_model = {n: se_long.filter(pl.col("model") == n)["sq_err"].to_numpy() for n in ordered}
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
    ax.set_ylabel("Per-compound squared error (lower = better)")
    ax.set_title(f"Holdout squared-error distributions by ensemble (best: {best})")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"wrote box plot to {save_path}")


def _parity_plot(
    preds: pl.DataFrame,
    means: dict[str, float],
    rmses: dict[str, float],
    save_path: Path,
) -> None:
    """2x2 grid of predicted-vs-observed pKD (parity) with y=x reference.

    Best ensemble (lowest mean squared error) gets a green accent; others
    neutral. Titles show per-ensemble RMSE on the holdout.
    """
    names = list(MODEL_LABELS.keys())
    y = preds[TARGET_COL].to_numpy().astype(np.float64)
    lo = float(min(y.min(), min(preds[n].min() for n in names)))
    hi = float(max(y.max(), max(preds[n].max() for n in names)))
    pad = (hi - lo) * 0.05
    lo, hi = lo - pad, hi + pad
    best = min(means, key=lambda n: means[n])

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.0), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, names, strict=True):
        p = preds[name].to_numpy().astype(np.float64)
        color = "#2e7d32" if name == best else "#37474f"
        ax.scatter(y, p, s=15, alpha=0.6, color=color)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="#c62828")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{name}  RMSE={rmses[name]:.3f}")
        ax.grid(linestyle=":", alpha=0.4)
    for ax in axes[-1, :]:
        ax.set_xlabel("observed pKD")
    for ax in axes[:, 0]:
        ax.set_ylabel("predicted pKD")
    fig.suptitle("Holdout parity plots (best ensemble in green)")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"wrote parity plot to {save_path}")


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_FIG_DIR.mkdir(parents=True, exist_ok=True)

    preds = _load_predictions()
    logger.info(f"loaded holdout predictions: {preds.shape}")
    y = preds[TARGET_COL].to_numpy().astype(np.float64)

    se_cols: dict[str, np.ndarray] = {}
    ensemble_metrics: dict[str, dict[str, float]] = {}
    for name in MODEL_LABELS:
        p = preds[name].to_numpy().astype(np.float64)
        se_cols[name] = (p - y) ** 2
        ensemble_metrics[name] = {
            "mean_squared_error": float(np.mean(se_cols[name])),
            "rmse": _rmse(y, p),
            "mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p)),
            "spearman": float(spearmanr(y, p).statistic),
        }
        logger.info(
            f"{name}: MSE={ensemble_metrics[name]['mean_squared_error']:.4f} "
            f"RMSE={ensemble_metrics[name]['rmse']:.3f} "
            f"MAE={ensemble_metrics[name]['mae']:.3f} "
            f"R2={ensemble_metrics[name]['r2']:.3f} "
            f"rho={ensemble_metrics[name]['spearman']:.3f}"
        )

    # Per-compound squared error table (wide)
    wide = preds.select(["compound_id", TARGET_COL]).with_columns(
        [pl.Series(f"sq_err_{n}", se_cols[n]) for n in MODEL_LABELS]
    )
    wide.write_csv(ARTIFACT_DIR / "holdout_comparison_squared_error.csv")

    # Long format for TukeyHSD
    long = pl.concat(
        [
            pl.DataFrame(
                {
                    "compound_id": preds["compound_id"],
                    "model": [n] * preds.shape[0],
                    "sq_err": se_cols[n],
                }
            )
            for n in MODEL_LABELS
        ]
    )
    values = long["sq_err"].to_numpy()
    groups = long["model"].to_numpy()

    tukey = pairwise_tukeyhsd(endog=values, groups=groups, alpha=0.05)
    summary_text = str(tukey.summary())
    (ARTIFACT_DIR / "holdout_tukey_hsd.txt").write_text(summary_text)
    logger.info("\n" + summary_text)

    ordered_names = list(MODEL_LABELS.keys())
    means = {n: ensemble_metrics[n]["mean_squared_error"] for n in ordered_names}
    rmses = {n: ensemble_metrics[n]["rmse"] for n in ordered_names}

    # Baseline: predict train-pool mean pKD on the holdout.
    # Read the full fold assignments to compute train-pool mean without
    # relying on one of the prediction CSVs.
    folds_df = pl.read_csv(FOLDS_CSV)
    train_mean = float(folds_df.filter(~pl.col("is_holdout"))[TARGET_COL].mean())
    baseline_preds = np.full_like(y, train_mean)
    baseline_mse = float(mean_squared_error(y, baseline_preds))
    baseline_rmse = float(np.sqrt(baseline_mse))
    baseline_mae = float(mean_absolute_error(y, baseline_preds))
    logger.info(
        f"train-pool mean pKD = {train_mean:.3f} -> "
        f"baseline MSE={baseline_mse:.4f} RMSE={baseline_rmse:.3f} MAE={baseline_mae:.3f}"
    )

    _tukey_plot_with_best(
        tukey, ordered_names, means,
        DOC_FIG_DIR / "holdout_tukey_hsd.png",
        baseline_mse=baseline_mse,
    )
    _box_plot(long, means, DOC_FIG_DIR / "holdout_squared_error_box.png")
    _parity_plot(preds, means, rmses, DOC_FIG_DIR / "holdout_parity.png")

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
        "target": TARGET_COL,
        "train_pool_mean_pKD": train_mean,
        "baseline_mse_train_mean": baseline_mse,
        "baseline_rmse_train_mean": baseline_rmse,
        "baseline_mae_train_mean": baseline_mae,
        "ensemble_metrics": ensemble_metrics,
        "best_by_mean_squared_error": best,
        "tukeyhsd_alpha": 0.05,
        "tukeyhsd_pairs": tukey_rows,
    }
    (ARTIFACT_DIR / "holdout_comparison_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"best ensemble by mean squared error: {best}")
    logger.info(f"wrote summary -> {ARTIFACT_DIR / 'holdout_comparison_summary.json'}")


if __name__ == "__main__":
    main()
