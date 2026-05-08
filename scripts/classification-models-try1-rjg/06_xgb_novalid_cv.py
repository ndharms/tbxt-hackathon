"""Step 06: XGBoost no-validation ensemble (4 train folds, fixed n_estimators).

Counterpart to ``05_chemeleon_novalid_cv.py`` for XGBoost. Rationale for
the fixed tree count: the val-variant's best_iteration across the 5 folds
was [97, 109, 26, 42, 110]; median = 97. We default to 100 trees.

Writes:
    data/models/xgb_novalid_oof_predictions.csv
    data/models/xgb_novalid_holdout_predictions.csv
    data/models/xgb_novalid_cv_metrics.json

Usage:
    uv run python scripts/06_xgb_novalid_cv.py
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
    train_one_xgb_novalid,
)

ROOT = Path(__file__).resolve().parents[2]
FOLDS_CSV = ROOT / "data" / "folds-pacmap-kmeans6" / "fold_assignments.csv"
COMPOUNDS_CSV = ROOT / "data" / "processed" / "tbxt_compounds_clean.csv"
ARTIFACT_DIR = ROOT / "data" / "classification-models-try1-rjg"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-estimators", type=int, default=100,
                    help="fixed tree count (default 100, median of val-variant best_iter)")
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.05)
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
    for c in PHYSCHEM_COLUMNS:
        if df[c].is_null().any():
            raise ValueError(f"null physchem values after join in column {c!r}")
    feat = build_features(df, physchem_cols=PHYSCHEM_COLUMNS)
    y_all = df["is_binder"].to_numpy().astype(np.int64)

    holdout_mask = df["is_holdout"].to_numpy()
    holdout_fold = int(df.filter(pl.col("is_holdout"))["fold"].unique().to_list()[0])
    train_mask_global = ~holdout_mask
    cv_folds = sorted(df.filter(~pl.col("is_holdout"))["fold"].unique().to_list())
    assert len(cv_folds) == 5

    cfg = XGBConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        random_state=args.seed,
    )

    oof = np.full(df.shape[0], np.nan)
    hold_preds_per_model: dict[int, np.ndarray] = {}
    metrics_per_model: list[dict] = []

    fold_arr = df["fold"].to_numpy()
    X = feat.X

    for order, test_fold in enumerate(cv_folds):
        train_folds = [f for f in cv_folds if f != test_fold]
        tr_mask = np.isin(fold_arr, train_folds) & train_mask_global
        te_mask = (fold_arr == test_fold) & train_mask_global

        X_tr, y_tr = X[tr_mask], y_all[tr_mask]
        X_te, y_te = X[te_mask], y_all[te_mask]

        logger.info(
            f"\n--- XGB no-val CV {order + 1}/5: test fold={test_fold} "
            f"(n={te_mask.sum()}, pos={int(y_te.sum())}), "
            f"train folds={train_folds} (n={tr_mask.sum()}, pos={int(y_tr.sum())})"
        )
        res = train_one_xgb_novalid(X_tr, y_tr, cfg)

        # Persist booster for later rehydration/scoring.
        model_path = ARTIFACT_DIR / f"xgb_no_val_cv_fold_{test_fold}.ubj"
        save_xgb_model(res.booster, model_path)

        te_probs = predict_proba_xgb(res.booster, X_te)
        oof[np.where(te_mask)[0]] = te_probs
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

        metrics_per_model.append(
            {
                "test_fold": int(test_fold),
                "train_folds": [int(f) for f in train_folds],
                "n_train": int(tr_mask.sum()),
                "n_test": int(te_mask.sum()),
                "pos_train": int(y_tr.sum()),
                "pos_test": int(y_te.sum()),
                "n_estimators": cfg.n_estimators,
                "test_auroc": auroc_te,
                "test_auprc": auprc_te,
                "model_path": str(model_path.relative_to(ROOT)),
            }
        )
        logger.info(f"fold {test_fold} XGB no-val OOF AUROC={auroc_te:.3f} AUPRC={auprc_te:.3f}")

    finite = ~np.isnan(oof) & train_mask_global
    oof_auroc = float(roc_auc_score(y_all[finite], oof[finite]))
    oof_auprc = float(average_precision_score(y_all[finite], oof[finite]))
    logger.info(f"XGB no-val OOF AUROC={oof_auroc:.3f} AUPRC={oof_auprc:.3f}")

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
        f"Holdout fold {holdout_fold} XGB no-val ensemble "
        f"AUROC={hold_auroc:.3f} AUPRC={hold_auprc:.3f}"
    )

    train_df = df.filter(~pl.col("is_holdout"))
    hold_df = df.filter(pl.col("is_holdout"))

    oof_out = train_df.with_columns(
        pl.Series("oof_p_binder_xgb_novalid", oof[train_mask_global]),
    ).select(
        ["compound_id", "canonical_smiles", "fold", "is_binder",
         "pKD_global_mean", "oof_p_binder_xgb_novalid"]
    )
    oof_out.write_csv(ARTIFACT_DIR / "xgb_no_val_oof.csv")

    hold_cols = {f"p_binder_xgb_novalid_fold_{f}": hold_preds_per_model[f] for f in cv_folds}
    hold_out = hold_df.select(
        ["compound_id", "canonical_smiles", "fold", "is_binder", "pKD_global_mean"]
    ).with_columns(
        [pl.Series(k, v) for k, v in hold_cols.items()]
        + [pl.Series("p_binder_xgb_novalid_ensemble_mean", hold_mean)],
    )
    hold_out.write_csv(ARTIFACT_DIR / "xgb_no_val_holdout.csv")

    summary = {
        "holdout_fold": holdout_fold,
        "cv_runs": metrics_per_model,
        "oof_auroc": oof_auroc,
        "oof_auprc": oof_auprc,
        "holdout_ensemble_auroc": hold_auroc,
        "holdout_ensemble_auprc": hold_auprc,
        "config": {
            "n_estimators": cfg.n_estimators,
            "max_depth": cfg.max_depth,
            "learning_rate": cfg.learning_rate,
            "subsample": cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "random_state": cfg.random_state,
        },
    }
    (ARTIFACT_DIR / "xgb_no_val_metrics.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"wrote XGB no-val metrics + predictions to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
