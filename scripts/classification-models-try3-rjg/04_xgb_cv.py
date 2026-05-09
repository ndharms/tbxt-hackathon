"""Step 04: XGBoost baseline with the same 6 chemical-space folds.

Feature matrix: Morgan FP (2048) + physchem (8) = 2056 cols.
Fold rotation matches the chemeleon CV: test fold k, val fold (k+1) mod 5,
remaining three folds train. Val fold drives XGBoost ``early_stopping_rounds``.

Writes:
    data/models/xgb_oof_predictions.csv
    data/models/xgb_holdout_predictions.csv
    data/models/xgb_cv_metrics.json

Usage:
    uv run python scripts/04_xgb_cv.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score

from tbxt_hackathon.xgb_baseline import (
    PHYSCHEM_COLUMNS,
    XGBConfig,
    build_features,
    predict_proba_xgb,
    save_xgb_model,
    train_one_xgb,
)

ROOT = Path(__file__).resolve().parents[2]
FOLDS_CSV = ROOT / "data" / "classification-models-try3-rjg" / "fold_assignments_filtered.csv"
COMPOUNDS_CSV = ROOT / "data" / "processed" / "tbxt_compounds_clean.csv"
ARTIFACT_DIR = ROOT / "data" / "classification-models-try3-rjg"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-estimators", type=int, default=1000)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--early-stopping-rounds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(FOLDS_CSV)
    physchem = pl.read_csv(COMPOUNDS_CSV).select(
        ["compound_id", *PHYSCHEM_COLUMNS],
    )
    df = df.join(physchem, on="compound_id", how="left")
    # preserve original row order (join should already, but assert)
    assert df.shape[0] == pl.read_csv(FOLDS_CSV).shape[0]
    for c in PHYSCHEM_COLUMNS:
        if df[c].is_null().any():
            raise ValueError(f"null physchem values after join in column {c!r}")
    feat = build_features(df, physchem_cols=PHYSCHEM_COLUMNS)
    y_all = df["is_binder"].to_numpy().astype(np.int64)

    holdout_mask = df["is_holdout"].to_numpy()
    holdout_fold = int(df.filter(pl.col("is_holdout"))["fold"].unique().to_list()[0])
    train_mask_global = ~holdout_mask
    train_idx_all = np.where(train_mask_global)[0]

    train_fold_values = df.filter(~pl.col("is_holdout"))["fold"].to_numpy()
    cv_folds = sorted(np.unique(train_fold_values).tolist())
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
            f"\n--- XGB CV {order + 1}/5: test fold={test_fold} "
            f"(n={te_mask.sum()}, pos={int(y_te.sum())}), "
            f"val fold={val_fold} (n={va_mask.sum()}, pos={int(y_va.sum())}), "
            f"train folds={train_folds} (n={tr_mask.sum()}, pos={int(y_tr.sum())})"
        )
        res = train_one_xgb(X_tr, y_tr, X_va, y_va, cfg)

        # Persist booster for later rehydration/scoring.
        model_path = ARTIFACT_DIR / f"xgb_with_val_cv_fold_{test_fold}.ubj"
        save_xgb_model(res.booster, model_path)

        # OOF
        te_probs = predict_proba_xgb(res.booster, X_te)
        oof[np.where(te_mask)[0]] = te_probs

        # Holdout
        hold_probs = predict_proba_xgb(res.booster, X[holdout_mask])
        hold_preds_per_model[test_fold] = hold_probs

        try:
            auroc_te = float(roc_auc_score(y_te, te_probs))
        except ValueError:
            auroc_te = float("nan")
        try:
            auprc_te = float(average_precision_score(y_te, te_probs))
        except ValueError:
            auprc_te = float("nan")

        run_metrics = {
            "test_fold": int(test_fold),
            "val_fold": int(val_fold),
            "train_folds": [int(f) for f in train_folds],
            "n_train": int(tr_mask.sum()),
            "n_val": int(va_mask.sum()),
            "n_test": int(te_mask.sum()),
            "pos_train": int(y_tr.sum()),
            "pos_val": int(y_va.sum()),
            "pos_test": int(y_te.sum()),
            "best_iteration": res.best_iteration,
            "best_val_logloss": res.best_val_logloss,
            "test_auroc": auroc_te,
            "test_auprc": auprc_te,
            "model_path": str(model_path.relative_to(ROOT)),
        }
        metrics_per_model.append(run_metrics)
        importances_by_fold.append(res.feature_importances)
        logger.info(
            f"fold {test_fold} XGB OOF AUROC={auroc_te:.3f} AUPRC={auprc_te:.3f} "
            f"best_iter={res.best_iteration}",
        )

    finite = ~np.isnan(oof) & train_mask_global
    oof_auroc = float(roc_auc_score(y_all[finite], oof[finite]))
    oof_auprc = float(average_precision_score(y_all[finite], oof[finite]))
    logger.info(f"XGB OOF AUROC={oof_auroc:.3f} AUPRC={oof_auprc:.3f}")

    hold_matrix = np.column_stack([hold_preds_per_model[f] for f in cv_folds])
    hold_mean = hold_matrix.mean(axis=1)
    y_hold = y_all[holdout_mask]
    try:
        hold_auroc = float(roc_auc_score(y_hold, hold_mean))
    except ValueError:
        hold_auroc = float("nan")
    try:
        hold_auprc = float(average_precision_score(y_hold, hold_mean))
    except ValueError:
        hold_auprc = float("nan")
    logger.info(
        f"Holdout fold {holdout_fold} XGB ensemble AUROC={hold_auroc:.3f} "
        f"AUPRC={hold_auprc:.3f} (n={len(y_hold)}, pos={int(y_hold.sum())})"
    )

    # Write predictions
    train_df = df.filter(~pl.col("is_holdout"))
    hold_df = df.filter(pl.col("is_holdout"))

    oof_out = train_df.with_columns(
        pl.Series("oof_p_binder_xgb", oof[train_mask_global]),
    ).select(
        ["compound_id", "canonical_smiles", "fold", "is_binder",
         "pKD_global_mean", "oof_p_binder_xgb"]
    )
    oof_out.write_csv(ARTIFACT_DIR / "xgb_with_val_oof.csv")

    hold_cols = {f"p_binder_xgb_fold_{f}": hold_preds_per_model[f] for f in cv_folds}
    hold_out = hold_df.select(
        ["compound_id", "canonical_smiles", "fold", "is_binder", "pKD_global_mean"]
    ).with_columns(
        [pl.Series(k, v) for k, v in hold_cols.items()]
        + [pl.Series("p_binder_xgb_ensemble_mean", hold_mean)],
    )
    hold_out.write_csv(ARTIFACT_DIR / "xgb_with_val_holdout.csv")

    # Mean importance across folds for interpretability
    mean_importance = np.mean(np.stack(importances_by_fold, axis=0), axis=0)
    # split fp vs physchem portions
    fp_importance_sum = float(mean_importance[: feat.fp_end].sum())
    phys_importance = {
        col: float(mean_importance[feat.fp_end + i])
        for i, col in enumerate(PHYSCHEM_COLUMNS)
    }

    summary = {
        "holdout_fold": holdout_fold,
        "cv_runs": metrics_per_model,
        "oof_auroc": oof_auroc,
        "oof_auprc": oof_auprc,
        "holdout_ensemble_auroc": hold_auroc,
        "holdout_ensemble_auprc": hold_auprc,
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
    logger.info(f"wrote XGB metrics + predictions to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
