"""Step 03: 5-fold CV ensemble of CheMeleon-transfer classifiers.

Strategy:
    - Hold out the most structurally distinct fold (``is_holdout == True``).
    - For the remaining 5 folds, train 5 models. Each model:
        train on 3 folds, validate (early-stop) on 1 fold,
        ignore the 5th fold (OOF prediction target).
    - Actually: the standard setup here is 5-fold CV where for each fold k,
      fold k is the OOF test fold, another fold is the val/early-stop fold,
      and the remaining 3 are training. That gives us 5 models and OOF preds
      for all non-holdout rows plus an ensemble prediction on the holdout.

Writes:
    data/models/cv_fold_{k}/best-*.ckpt
    data/models/oof_predictions.csv     (non-holdout rows + OOF P(binder))
    data/models/holdout_predictions.csv (holdout rows with per-model + mean preds)
    data/models/cv_metrics.json         (val + OOF + holdout metrics)

Usage:
    uv run python scripts/03_train_cv.py [--accelerator mps|cpu|auto]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score

from tbxt_hackathon.chemeleon_transfer import (
    ClassifierConfig,
    predict_proba,
    train_one,
)

ROOT = Path(__file__).resolve().parents[2]
FOLDS_CSV = ROOT / "data" / "classification-models-try3-rjg" / "fold_assignments_filtered.csv"
ARTIFACT_DIR = ROOT / "data" / "classification-models-try3-rjg"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--accelerator",
        default="auto",
        choices=["auto", "cpu", "mps", "gpu"],
        help="Lightning accelerator (default: auto)",
    )
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(FOLDS_CSV)
    assert {"fold", "is_holdout", "is_binder", "canonical_smiles"}.issubset(df.columns)

    holdout_fold_values = df.filter(pl.col("is_holdout"))["fold"].unique().to_list()
    assert len(holdout_fold_values) == 1, f"expected one holdout fold, got {holdout_fold_values}"
    holdout_fold = int(holdout_fold_values[0])
    train_df = df.filter(~pl.col("is_holdout"))
    hold_df = df.filter(pl.col("is_holdout"))
    logger.info(
        f"holdout fold = {holdout_fold} ({hold_df.shape[0]} mols); "
        f"train pool = {train_df.shape[0]} mols across folds "
        f"{sorted(train_df['fold'].unique().to_list())}"
    )

    cv_folds = sorted(train_df["fold"].unique().to_list())
    assert len(cv_folds) == 5, f"expected 5 CV folds after holdout, got {cv_folds}"

    cfg = ClassifierConfig(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        random_state=args.seed,
    )

    # Per-row OOF prediction storage (keyed by row idx within train_df)
    oof = np.full(train_df.shape[0], np.nan)
    # Per-model holdout prediction storage
    hold_preds_per_model: dict[int, np.ndarray] = {}
    metrics_per_model: list[dict] = []

    # For each CV fold k: k = OOF test, (k+1) mod 5 = val, other three = train.
    # Rotating val ensures every non-test fold takes a turn as validation.
    for order, test_fold in enumerate(cv_folds):
        val_fold = cv_folds[(order + 1) % 5]
        train_folds = [f for f in cv_folds if f not in (test_fold, val_fold)]

        train_mask = train_df["fold"].is_in(train_folds).to_numpy()
        val_mask = (train_df["fold"] == val_fold).to_numpy()
        test_mask = (train_df["fold"] == test_fold).to_numpy()

        tr_smiles = train_df.filter(pl.Series(train_mask))["canonical_smiles"].to_list()
        tr_y = train_df.filter(pl.Series(train_mask))["is_binder"].to_numpy().astype(np.int64)
        va_smiles = train_df.filter(pl.Series(val_mask))["canonical_smiles"].to_list()
        va_y = train_df.filter(pl.Series(val_mask))["is_binder"].to_numpy().astype(np.int64)
        te_smiles = train_df.filter(pl.Series(test_mask))["canonical_smiles"].to_list()
        te_y = train_df.filter(pl.Series(test_mask))["is_binder"].to_numpy().astype(np.int64)

        logger.info(
            f"\n--- CV run {order + 1}/5: test fold={test_fold} "
            f"(n={test_mask.sum()}, pos={int(te_y.sum())}), "
            f"val fold={val_fold} (n={val_mask.sum()}, pos={int(va_y.sum())}), "
            f"train folds={train_folds} (n={train_mask.sum()}, pos={int(tr_y.sum())})"
        )

        ckpt_dir = ARTIFACT_DIR / f"chemeleon_with_val_cv_fold_{test_fold}"
        result = train_one(
            train_smiles=tr_smiles,
            train_labels=tr_y,
            val_smiles=va_smiles,
            val_labels=va_y,
            cfg=cfg,
            checkpoint_dir=ckpt_dir,
            accelerator=args.accelerator,
        )

        # Predict on this fold's test rows (OOF)
        te_probs = predict_proba(result.model, te_smiles, accelerator=args.accelerator)
        # scatter back to oof array
        test_idx = np.where(test_mask)[0]
        oof[test_idx] = te_probs

        # Predict on global holdout
        hold_smiles = hold_df["canonical_smiles"].to_list()
        hold_probs = predict_proba(result.model, hold_smiles, accelerator=args.accelerator)
        hold_preds_per_model[test_fold] = hold_probs

        # fold-level OOF metrics (on this test fold)
        try:
            auroc_te = float(roc_auc_score(te_y, te_probs))
        except ValueError:
            auroc_te = float("nan")
        try:
            auprc_te = float(average_precision_score(te_y, te_probs))
        except ValueError:
            auprc_te = float("nan")

        run_metrics = {
            "test_fold": int(test_fold),
            "val_fold": int(val_fold),
            "train_folds": [int(f) for f in train_folds],
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "n_test": int(test_mask.sum()),
            "pos_train": int(tr_y.sum()),
            "pos_val": int(va_y.sum()),
            "pos_test": int(te_y.sum()),
            "best_val_loss": result.best_val_loss,
            "best_val_auroc": result.best_val_auroc,
            "best_val_auprc": result.best_val_auprc,
            "test_auroc": auroc_te,
            "test_auprc": auprc_te,
            "best_epoch": result.best_epoch,
            "ckpt": str(result.ckpt_path),
        }
        metrics_per_model.append(run_metrics)
        logger.info(
            f"fold {test_fold} OOF AUROC={auroc_te:.3f} AUPRC={auprc_te:.3f} "
            f"(val AUROC={result.best_val_auroc}, AUPRC={result.best_val_auprc})",
        )
        # free memory between runs
        del result
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Aggregate OOF metrics (on all non-holdout rows)
    y_all = train_df["is_binder"].to_numpy().astype(np.int64)
    finite = ~np.isnan(oof)
    oof_auroc = float(roc_auc_score(y_all[finite], oof[finite]))
    oof_auprc = float(average_precision_score(y_all[finite], oof[finite]))
    logger.info(f"OOF (5-fold CV) AUROC={oof_auroc:.3f} AUPRC={oof_auprc:.3f}")

    # Ensemble mean on holdout
    hold_matrix = np.column_stack([hold_preds_per_model[f] for f in cv_folds])
    hold_mean = hold_matrix.mean(axis=1)
    y_hold = hold_df["is_binder"].to_numpy().astype(np.int64)
    try:
        hold_auroc = float(roc_auc_score(y_hold, hold_mean))
    except ValueError:
        hold_auroc = float("nan")
    try:
        hold_auprc = float(average_precision_score(y_hold, hold_mean))
    except ValueError:
        hold_auprc = float("nan")
    logger.info(
        f"Holdout fold {holdout_fold} ensemble AUROC={hold_auroc:.3f} AUPRC={hold_auprc:.3f} "
        f"(n={len(y_hold)}, pos={int(y_hold.sum())})"
    )

    # Write OOF predictions table
    oof_out = train_df.with_columns(
        pl.Series("oof_p_binder", oof),
    ).select(
        [
            "compound_id",
            "canonical_smiles",
            "fold",
            "is_binder",
            "pKD_global_mean",
            "oof_p_binder",
        ]
    )
    oof_out.write_csv(ARTIFACT_DIR / "chemeleon_with_val_oof.csv")

    # Holdout predictions: per-model + ensemble
    hold_cols = {
        f"p_binder_fold_{f}_model": hold_preds_per_model[f] for f in cv_folds
    }
    hold_out = hold_df.select(
        [
            "compound_id",
            "canonical_smiles",
            "fold",
            "is_binder",
            "pKD_global_mean",
        ]
    ).with_columns(
        [pl.Series(k, v) for k, v in hold_cols.items()]
        + [pl.Series("p_binder_ensemble_mean", hold_mean)],
    )
    hold_out.write_csv(ARTIFACT_DIR / "chemeleon_with_val_holdout.csv")

    metrics_summary = {
        "holdout_fold": holdout_fold,
        "cv_runs": metrics_per_model,
        "oof_auroc": oof_auroc,
        "oof_auprc": oof_auprc,
        "holdout_ensemble_auroc": hold_auroc,
        "holdout_ensemble_auprc": hold_auprc,
        "config": {
            "hidden_dim": cfg.hidden_dim,
            "dropout": cfg.dropout,
            "max_epochs": cfg.max_epochs,
            "patience": cfg.patience,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "random_state": cfg.random_state,
            "accelerator": args.accelerator,
        },
    }
    (ARTIFACT_DIR / "chemeleon_with_val_metrics.json").write_text(json.dumps(metrics_summary, indent=2))
    logger.info(f"wrote metrics + predictions to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
