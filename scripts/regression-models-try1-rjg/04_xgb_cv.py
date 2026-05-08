"""Step 04: XGBoost regression baseline with the same 6 chemical-space folds.

Regression counterpart to ``classification-models-try1-rjg/04_xgb_cv.py``.
Target is continuous pKD (not thresholded). Feature matrix: Morgan FP (2048)
+ physchem (8) = 2056 cols. Fold rotation matches the other ensembles.

Writes:
    data/regression-models-try1-rjg/xgb_with_val_cv_fold_{k}.ubj
    data/regression-models-try1-rjg/xgb_with_val_oof.csv
    data/regression-models-try1-rjg/xgb_with_val_holdout.csv
    data/regression-models-try1-rjg/xgb_with_val_metrics.json

Usage:
    uv run python scripts/regression-models-try1-rjg/04_xgb_cv.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tbxt_hackathon.xgb_baseline import (
    PHYSCHEM_COLUMNS,
    XGBConfig,
    build_features,
    predict_xgb_regression,
    save_xgb_regression_model,
    train_one_xgb_regression,
)

ROOT = Path(__file__).resolve().parents[2]
FOLDS_CSV = ROOT / "data" / "folds-pacmap-kmeans6" / "fold_assignments.csv"
COMPOUNDS_CSV = ROOT / "data" / "processed" / "tbxt_compounds_clean.csv"
ARTIFACT_DIR = ROOT / "data" / "regression-models-try1-rjg"
TARGET_COL = "pKD_global_mean"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-estimators", type=int, default=1000)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--early-stopping-rounds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, pred)))


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(FOLDS_CSV)
    physchem = pl.read_csv(COMPOUNDS_CSV).select(
        ["compound_id", *PHYSCHEM_COLUMNS],
    )
    df = df.join(physchem, on="compound_id", how="left")
    assert df.shape[0] == pl.read_csv(FOLDS_CSV).shape[0]
    for c in PHYSCHEM_COLUMNS:
        if df[c].is_null().any():
            raise ValueError(f"null physchem values after join in column {c!r}")
    feat = build_features(df, physchem_cols=PHYSCHEM_COLUMNS)
    y_all = df[TARGET_COL].to_numpy().astype(np.float64)

    holdout_mask = df["is_holdout"].to_numpy()
    holdout_fold = int(df.filter(pl.col("is_holdout"))["fold"].unique().to_list()[0])
    train_mask_global = ~holdout_mask
    cv_folds = sorted(df.filter(~pl.col("is_holdout"))["fold"].unique().to_list())
    assert len(cv_folds) == 5

    cfg = XGBConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.seed,
    )

    oof = np.full(df.shape[0], np.nan)
    hold_preds_per_model: dict[int, np.ndarray] = {}
    metrics_per_model: list[dict] = []
    importances_by_fold: list[np.ndarray] = []

    fold_arr = df["fold"].to_numpy()
    X = feat.X

    for order, test_fold in enumerate(cv_folds):
        val_fold = cv_folds[(order + 1) % 5]
        train_folds = [f for f in cv_folds if f not in (test_fold, val_fold)]

        tr_mask = np.isin(fold_arr, train_folds) & train_mask_global
        va_mask = (fold_arr == val_fold) & train_mask_global
        te_mask = (fold_arr == test_fold) & train_mask_global

        X_tr, y_tr = X[tr_mask], y_all[tr_mask]
        X_va, y_va = X[va_mask], y_all[va_mask]
        X_te, y_te = X[te_mask], y_all[te_mask]

        logger.info(
            f"\n--- XGB regression CV {order + 1}/5: test fold={test_fold} "
            f"(n={te_mask.sum()}), val fold={val_fold} (n={va_mask.sum()}), "
            f"train folds={train_folds} (n={tr_mask.sum()})"
        )
        res = train_one_xgb_regression(X_tr, y_tr, X_va, y_va, cfg)

        model_path = ARTIFACT_DIR / f"xgb_with_val_cv_fold_{test_fold}.ubj"
        save_xgb_regression_model(res.booster, model_path)

        te_pred = predict_xgb_regression(res.booster, X_te)
        oof[np.where(te_mask)[0]] = te_pred
        hold_pred = predict_xgb_regression(res.booster, X[holdout_mask])
        hold_preds_per_model[test_fold] = hold_pred

        te_rmse = _rmse(y_te, te_pred)
        te_mae = float(mean_absolute_error(y_te, te_pred))
        te_r2 = float(r2_score(y_te, te_pred))
        te_rho = float(spearmanr(y_te, te_pred).statistic)

        run_metrics = {
            "test_fold": int(test_fold),
            "val_fold": int(val_fold),
            "train_folds": [int(f) for f in train_folds],
            "n_train": int(tr_mask.sum()),
            "n_val": int(va_mask.sum()),
            "n_test": int(te_mask.sum()),
            "best_iteration": res.best_iteration,
            "best_val_rmse": res.best_val_rmse,
            "test_rmse": te_rmse,
            "test_mae": te_mae,
            "test_r2": te_r2,
            "test_spearman": te_rho,
            "model_path": str(model_path.relative_to(ROOT)),
        }
        metrics_per_model.append(run_metrics)
        importances_by_fold.append(res.feature_importances)
        logger.info(
            f"fold {test_fold} XGB OOF RMSE={te_rmse:.3f} MAE={te_mae:.3f} "
            f"R2={te_r2:.3f} rho={te_rho:.3f} best_iter={res.best_iteration}"
        )

    finite = ~np.isnan(oof) & train_mask_global
    oof_rmse = _rmse(y_all[finite], oof[finite])
    oof_mae = float(mean_absolute_error(y_all[finite], oof[finite]))
    oof_r2 = float(r2_score(y_all[finite], oof[finite]))
    oof_rho = float(spearmanr(y_all[finite], oof[finite]).statistic)
    logger.info(
        f"XGB OOF RMSE={oof_rmse:.3f} MAE={oof_mae:.3f} "
        f"R2={oof_r2:.3f} rho={oof_rho:.3f}"
    )

    hold_matrix = np.column_stack([hold_preds_per_model[f] for f in cv_folds])
    hold_mean = hold_matrix.mean(axis=1)
    y_hold = y_all[holdout_mask]
    hold_rmse = _rmse(y_hold, hold_mean)
    hold_mae = float(mean_absolute_error(y_hold, hold_mean))
    hold_r2 = float(r2_score(y_hold, hold_mean))
    hold_rho = float(spearmanr(y_hold, hold_mean).statistic)
    logger.info(
        f"Holdout fold {holdout_fold} XGB ensemble RMSE={hold_rmse:.3f} "
        f"MAE={hold_mae:.3f} R2={hold_r2:.3f} rho={hold_rho:.3f}"
    )

    train_df = df.filter(~pl.col("is_holdout"))
    hold_df = df.filter(pl.col("is_holdout"))

    oof_out = train_df.with_columns(
        pl.Series("oof_pred_pKD_xgb", oof[train_mask_global]),
    ).select(
        ["compound_id", "canonical_smiles", "fold", "is_binder",
         TARGET_COL, "oof_pred_pKD_xgb"]
    )
    oof_out.write_csv(ARTIFACT_DIR / "xgb_with_val_oof.csv")

    hold_cols = {f"pred_pKD_xgb_fold_{f}": hold_preds_per_model[f] for f in cv_folds}
    hold_out = hold_df.select(
        ["compound_id", "canonical_smiles", "fold", "is_binder", TARGET_COL]
    ).with_columns(
        [pl.Series(k, v) for k, v in hold_cols.items()]
        + [pl.Series("pred_pKD_xgb_ensemble_mean", hold_mean)],
    )
    hold_out.write_csv(ARTIFACT_DIR / "xgb_with_val_holdout.csv")

    mean_importance = np.mean(np.stack(importances_by_fold, axis=0), axis=0)
    fp_importance_sum = float(mean_importance[: feat.fp_end].sum())
    phys_importance = {
        col: float(mean_importance[feat.fp_end + i])
        for i, col in enumerate(PHYSCHEM_COLUMNS)
    }

    summary = {
        "holdout_fold": holdout_fold,
        "target": TARGET_COL,
        "cv_runs": metrics_per_model,
        "oof_rmse": oof_rmse,
        "oof_mae": oof_mae,
        "oof_r2": oof_r2,
        "oof_spearman": oof_rho,
        "holdout_ensemble_rmse": hold_rmse,
        "holdout_ensemble_mae": hold_mae,
        "holdout_ensemble_r2": hold_r2,
        "holdout_ensemble_spearman": hold_rho,
        "feature_importance_morgan_sum": fp_importance_sum,
        "feature_importance_physchem": phys_importance,
        "config": {
            "n_estimators": cfg.n_estimators,
            "max_depth": cfg.max_depth,
            "learning_rate": cfg.learning_rate,
            "early_stopping_rounds": cfg.early_stopping_rounds,
            "subsample": cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "random_state": cfg.random_state,
        },
    }
    (ARTIFACT_DIR / "xgb_with_val_metrics.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"wrote XGB regression metrics + predictions to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
