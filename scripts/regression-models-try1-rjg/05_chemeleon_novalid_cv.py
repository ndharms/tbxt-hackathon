"""Step 05: No-validation CheMeleon-transfer regression ensemble.

Regression counterpart to ``classification-models-try1-rjg/05_chemeleon_novalid_cv.py``.
Trains each model on 4 folds (~33% more data than the with-val variant) for
a fixed number of epochs. Initial default of 15 epochs mirrors the
classification setup; tune via ``--epochs`` after inspecting the with-val
``best_epoch`` distribution.

Writes:
    data/regression-models-try1-rjg/chemeleon_no_val_cv_fold_{k}.ckpt
    data/regression-models-try1-rjg/chemeleon_no_val_oof.csv
    data/regression-models-try1-rjg/chemeleon_no_val_holdout.csv
    data/regression-models-try1-rjg/chemeleon_no_val_metrics.json

Usage:
    uv run python scripts/regression-models-try1-rjg/05_chemeleon_novalid_cv.py [--accelerator mps]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from loguru import logger
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tbxt_hackathon.chemeleon_transfer import (
    ClassifierConfig,
    predict_regression,
    train_one_novalid_regression,
)

ROOT = Path(__file__).resolve().parents[2]
FOLDS_CSV = ROOT / "data" / "folds-pacmap-kmeans6" / "fold_assignments.csv"
ARTIFACT_DIR = ROOT / "data" / "regression-models-try1-rjg"
TARGET_COL = "pKD_global_mean"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--accelerator",
        default="auto",
        choices=["auto", "cpu", "mps", "gpu"],
    )
    ap.add_argument("--epochs", type=int, default=15,
                    help="fixed epoch count (default 15; tune to median of with-val best_epoch)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, pred)))


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(FOLDS_CSV)
    train_df = df.filter(~pl.col("is_holdout"))
    hold_df = df.filter(pl.col("is_holdout"))
    holdout_fold = int(hold_df["fold"].unique().to_list()[0])
    cv_folds = sorted(train_df["fold"].unique().to_list())
    assert len(cv_folds) == 5

    cfg = ClassifierConfig(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        max_epochs=args.epochs,
        patience=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        random_state=args.seed,
    )

    oof = np.full(train_df.shape[0], np.nan)
    hold_preds_per_model: dict[int, np.ndarray] = {}
    metrics_per_model: list[dict] = []

    for order, test_fold in enumerate(cv_folds):
        train_folds = [f for f in cv_folds if f != test_fold]

        train_mask = train_df["fold"].is_in(train_folds).to_numpy()
        test_mask = (train_df["fold"] == test_fold).to_numpy()

        tr_smiles = train_df.filter(pl.Series(train_mask))["canonical_smiles"].to_list()
        tr_y = train_df.filter(pl.Series(train_mask))[TARGET_COL].to_numpy().astype(np.float32)
        te_smiles = train_df.filter(pl.Series(test_mask))["canonical_smiles"].to_list()
        te_y = train_df.filter(pl.Series(test_mask))[TARGET_COL].to_numpy().astype(np.float32)

        logger.info(
            f"\n--- no-val CV {order + 1}/5: test fold={test_fold} (n={test_mask.sum()}), "
            f"train folds={train_folds} (n={train_mask.sum()}), epochs={cfg.max_epochs}"
        )

        ckpt_path = ARTIFACT_DIR / f"chemeleon_no_val_cv_fold_{test_fold}.ckpt"
        model = train_one_novalid_regression(
            train_smiles=tr_smiles,
            train_targets=tr_y,
            cfg=cfg,
            save_path=ckpt_path,
            accelerator=args.accelerator,
        )

        te_pred = predict_regression(model, te_smiles, accelerator=args.accelerator)
        oof[np.where(test_mask)[0]] = te_pred

        hold_smiles = hold_df["canonical_smiles"].to_list()
        hold_pred = predict_regression(model, hold_smiles, accelerator=args.accelerator)
        hold_preds_per_model[test_fold] = hold_pred

        te_rmse = _rmse(te_y, te_pred)
        te_mae = float(mean_absolute_error(te_y, te_pred))
        te_r2 = float(r2_score(te_y, te_pred))
        te_rho = float(spearmanr(te_y, te_pred).statistic)

        run_metrics = {
            "test_fold": int(test_fold),
            "train_folds": [int(f) for f in train_folds],
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "epochs": cfg.max_epochs,
            "test_rmse": te_rmse,
            "test_mae": te_mae,
            "test_r2": te_r2,
            "test_spearman": te_rho,
            "ckpt": str(ckpt_path.relative_to(ROOT)),
        }
        metrics_per_model.append(run_metrics)
        logger.info(
            f"fold {test_fold} no-val OOF RMSE={te_rmse:.3f} MAE={te_mae:.3f} "
            f"R2={te_r2:.3f} rho={te_rho:.3f}"
        )
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    y_all = train_df[TARGET_COL].to_numpy().astype(np.float64)
    finite = ~np.isnan(oof)
    oof_rmse = _rmse(y_all[finite], oof[finite])
    oof_mae = float(mean_absolute_error(y_all[finite], oof[finite]))
    oof_r2 = float(r2_score(y_all[finite], oof[finite]))
    oof_rho = float(spearmanr(y_all[finite], oof[finite]).statistic)
    logger.info(
        f"no-val OOF RMSE={oof_rmse:.3f} MAE={oof_mae:.3f} "
        f"R2={oof_r2:.3f} rho={oof_rho:.3f}"
    )

    hold_matrix = np.column_stack([hold_preds_per_model[f] for f in cv_folds])
    hold_mean = hold_matrix.mean(axis=1)
    y_hold = hold_df[TARGET_COL].to_numpy().astype(np.float64)
    hold_rmse = _rmse(y_hold, hold_mean)
    hold_mae = float(mean_absolute_error(y_hold, hold_mean))
    hold_r2 = float(r2_score(y_hold, hold_mean))
    hold_rho = float(spearmanr(y_hold, hold_mean).statistic)
    logger.info(
        f"Holdout fold {holdout_fold} no-val ensemble RMSE={hold_rmse:.3f} "
        f"MAE={hold_mae:.3f} R2={hold_r2:.3f} rho={hold_rho:.3f}"
    )

    oof_out = train_df.with_columns(
        pl.Series("oof_pred_pKD_novalid", oof),
    ).select(
        ["compound_id", "canonical_smiles", "fold", "is_binder",
         TARGET_COL, "oof_pred_pKD_novalid"]
    )
    oof_out.write_csv(ARTIFACT_DIR / "chemeleon_no_val_oof.csv")

    hold_cols = {f"pred_pKD_novalid_fold_{f}": hold_preds_per_model[f] for f in cv_folds}
    hold_out = hold_df.select(
        ["compound_id", "canonical_smiles", "fold", "is_binder", TARGET_COL]
    ).with_columns(
        [pl.Series(k, v) for k, v in hold_cols.items()]
        + [pl.Series("pred_pKD_novalid_ensemble_mean", hold_mean)],
    )
    hold_out.write_csv(ARTIFACT_DIR / "chemeleon_no_val_holdout.csv")

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
        "config": {
            "hidden_dim": cfg.hidden_dim,
            "dropout": cfg.dropout,
            "epochs": cfg.max_epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "random_state": cfg.random_state,
            "accelerator": args.accelerator,
        },
    }
    (ARTIFACT_DIR / "chemeleon_no_val_metrics.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"wrote no-val regression metrics + predictions to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
