"""Step 05: No-validation CheMeleon-transfer ensemble (fixed-epoch training).

Motivation: in the val-fold variant the best val_loss epoch clustered at
12-14 (median 13) across the 5 folds. Here we drop the validation set so
each model trains on 4 folds (~33% more data) and we cap at 15 epochs
(median best + small cushion) to avoid drifting into the overfit regime
without the early-stop signal.

Writes:
    data/models/novalid_cv_fold_{k}/ (empty; no checkpoints by design)
    data/models/novalid_oof_predictions.csv
    data/models/novalid_holdout_predictions.csv
    data/models/novalid_cv_metrics.json

Usage:
    uv run python scripts/05_chemeleon_novalid_cv.py [--accelerator mps]
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
    train_one_novalid,
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
    )
    ap.add_argument("--epochs", type=int, default=15,
                    help="fixed epoch count (default 15, ~median of val-variant best_epoch)")
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
    train_df = df.filter(~pl.col("is_holdout"))
    hold_df = df.filter(pl.col("is_holdout"))
    holdout_fold = int(hold_df["fold"].unique().to_list()[0])
    cv_folds = sorted(train_df["fold"].unique().to_list())
    assert len(cv_folds) == 5

    cfg = ClassifierConfig(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        max_epochs=args.epochs,
        patience=args.epochs,  # unused here but kept consistent
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
        tr_y = train_df.filter(pl.Series(train_mask))["is_binder"].to_numpy().astype(np.int64)
        te_smiles = train_df.filter(pl.Series(test_mask))["canonical_smiles"].to_list()
        te_y = train_df.filter(pl.Series(test_mask))["is_binder"].to_numpy().astype(np.int64)

        logger.info(
            f"\n--- no-val CV {order + 1}/5: test fold={test_fold} "
            f"(n={test_mask.sum()}, pos={int(te_y.sum())}), "
            f"train folds={train_folds} (n={train_mask.sum()}, pos={int(tr_y.sum())}), "
            f"epochs={cfg.max_epochs}"
        )

        ckpt_path = ARTIFACT_DIR / f"chemeleon_no_val_cv_fold_{test_fold}.ckpt"
        model = train_one_novalid(
            train_smiles=tr_smiles,
            train_labels=tr_y,
            cfg=cfg,
            save_path=ckpt_path,
            accelerator=args.accelerator,
        )

        te_probs = predict_proba(model, te_smiles, accelerator=args.accelerator)
        oof[np.where(test_mask)[0]] = te_probs

        hold_smiles = hold_df["canonical_smiles"].to_list()
        hold_probs = predict_proba(model, hold_smiles, accelerator=args.accelerator)
        hold_preds_per_model[test_fold] = hold_probs

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
            "train_folds": [int(f) for f in train_folds],
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "pos_train": int(tr_y.sum()),
            "pos_test": int(te_y.sum()),
            "epochs": cfg.max_epochs,
            "test_auroc": auroc_te,
            "test_auprc": auprc_te,
            "ckpt": str(ckpt_path.relative_to(ROOT)),
        }
        metrics_per_model.append(run_metrics)
        logger.info(f"fold {test_fold} no-val OOF AUROC={auroc_te:.3f} AUPRC={auprc_te:.3f}")

        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    y_all = train_df["is_binder"].to_numpy().astype(np.int64)
    finite = ~np.isnan(oof)
    oof_auroc = float(roc_auc_score(y_all[finite], oof[finite]))
    oof_auprc = float(average_precision_score(y_all[finite], oof[finite]))
    logger.info(f"no-val OOF AUROC={oof_auroc:.3f} AUPRC={oof_auprc:.3f}")

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
        f"Holdout fold {holdout_fold} no-val ensemble "
        f"AUROC={hold_auroc:.3f} AUPRC={hold_auprc:.3f}"
    )

    oof_out = train_df.with_columns(
        pl.Series("oof_p_binder_novalid", oof),
    ).select(
        ["compound_id", "canonical_smiles", "fold", "is_binder",
         "pKD_global_mean", "oof_p_binder_novalid"]
    )
    oof_out.write_csv(ARTIFACT_DIR / "chemeleon_no_val_oof.csv")

    hold_cols = {f"p_binder_novalid_fold_{f}": hold_preds_per_model[f] for f in cv_folds}
    hold_out = hold_df.select(
        ["compound_id", "canonical_smiles", "fold", "is_binder", "pKD_global_mean"]
    ).with_columns(
        [pl.Series(k, v) for k, v in hold_cols.items()]
        + [pl.Series("p_binder_novalid_ensemble_mean", hold_mean)],
    )
    hold_out.write_csv(ARTIFACT_DIR / "chemeleon_no_val_holdout.csv")

    summary = {
        "holdout_fold": holdout_fold,
        "cv_runs": metrics_per_model,
        "oof_auroc": oof_auroc,
        "oof_auprc": oof_auprc,
        "holdout_ensemble_auroc": hold_auroc,
        "holdout_ensemble_auprc": hold_auprc,
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
    logger.info(f"wrote no-val metrics + predictions to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
