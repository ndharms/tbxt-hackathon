"""Diagnostic 08: XGBoost classifier with pocket features vs without.

Compares two models:
  1. Baseline: Morgan FP (2048) + physchem (8) = 2056 features
  2. + Pocket: same + per-pocket Tanimoto (4) + per-pocket substruct (4) = 2064 features

Both use the same 5-fold CV with val-fold early stopping, identical to
scripts/classification-models-try1-rjg/04_xgb_cv.py.

If pocket assignment provides signal beyond what Morgan FP already captures,
the +Pocket model should show higher OOF AUROC.

Usage:
    uv run python scripts/sar-diagnostics-rjg/08_xgb_with_pocket_features.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score

from tbxt_hackathon.fingerprints import morgan_ndarray
from tbxt_hackathon.pocket_assigner import PocketAssigner
from tbxt_hackathon.xgb_baseline import (
    PHYSCHEM_COLUMNS,
    XGBConfig,
    predict_proba_xgb,
    train_one_xgb,
)

ROOT = Path(__file__).resolve().parents[2]
FOLDS_CSV = ROOT / "data" / "folds-pacmap-kmeans6" / "fold_assignments.csv"
COMPOUNDS_CSV = ROOT / "data" / "processed" / "tbxt_compounds_clean.csv"
FRAGMENT_CSV = ROOT / "data" / "structures" / "sgc_fragments.csv"
ARTIFACT_DIR = ROOT / "data" / "sar-diagnostics-rjg"


def build_pocket_features(
    smiles_list: list[str],
    assigner: PocketAssigner,
) -> np.ndarray:
    """Build pocket feature columns for a list of SMILES.

    Returns an (n, 8) float32 array:
      - 4 columns: max Tanimoto to each pocket (A, B, C, D)
      - 4 columns: substructure match to each pocket (0/1)

    Args:
        smiles_list: List of SMILES strings.
        assigner: Loaded PocketAssigner.

    Returns:
        (n, 8) float32 array of pocket features.
    """
    pockets = sorted(assigner.pocket_fps.keys())  # A, B, C, D
    n = len(smiles_list)
    n_pockets = len(pockets)

    tc_arr = np.zeros((n, n_pockets), dtype=np.float32)
    sub_arr = np.zeros((n, n_pockets), dtype=np.float32)

    scores = assigner.score_batch(smiles_list)
    for i, s in enumerate(scores):
        for j, pocket in enumerate(pockets):
            if pocket in s:
                tc_arr[i, j] = s[pocket].tanimoto
                sub_arr[i, j] = float(s[pocket].substruct)

    return np.concatenate([tc_arr, sub_arr], axis=1)


def run_cv(
    X: np.ndarray,
    y_all: np.ndarray,
    fold_arr: np.ndarray,
    train_mask_global: np.ndarray,
    holdout_mask: np.ndarray,
    cv_folds: list[int],
    cfg: XGBConfig,
    label: str,
) -> dict:
    """Run 5-fold CV and return metrics dict."""
    oof = np.full(len(y_all), np.nan)
    hold_preds: dict[int, np.ndarray] = {}

    for order, test_fold in enumerate(cv_folds):
        val_fold = cv_folds[(order + 1) % 5]
        train_folds = [f for f in cv_folds if f not in (test_fold, val_fold)]

        tr_mask = np.isin(fold_arr, train_folds) & train_mask_global
        va_mask = (fold_arr == val_fold) & train_mask_global
        te_mask = (fold_arr == test_fold) & train_mask_global

        X_tr, y_tr = X[tr_mask], y_all[tr_mask]
        X_va, y_va = X[va_mask], y_all[va_mask]
        X_te = X[te_mask]

        res = train_one_xgb(X_tr, y_tr, X_va, y_va, cfg)

        te_probs = predict_proba_xgb(res.booster, X_te)
        oof[np.where(te_mask)[0]] = te_probs

        hold_probs = predict_proba_xgb(res.booster, X[holdout_mask])
        hold_preds[test_fold] = hold_probs

    # OOF metrics
    finite = ~np.isnan(oof) & train_mask_global
    oof_auroc = float(roc_auc_score(y_all[finite], oof[finite]))
    oof_auprc = float(average_precision_score(y_all[finite], oof[finite]))

    # Holdout metrics
    hold_matrix = np.column_stack([hold_preds[f] for f in cv_folds])
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
        f"  {label}: OOF AUROC={oof_auroc:.4f} AUPRC={oof_auprc:.4f} | "
        f"Holdout AUROC={hold_auroc:.4f} AUPRC={hold_auprc:.4f}"
    )

    return {
        "label": label,
        "oof_auroc": oof_auroc,
        "oof_auprc": oof_auprc,
        "holdout_auroc": hold_auroc,
        "holdout_auprc": hold_auprc,
        "n_features": int(X.shape[1]),
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pl.read_csv(FOLDS_CSV)
    physchem = pl.read_csv(COMPOUNDS_CSV).select(
        ["compound_id", *PHYSCHEM_COLUMNS],
    )
    df = df.join(physchem, on="compound_id", how="left")

    smiles = df["canonical_smiles"].to_list()
    y_all = df["is_binder"].to_numpy().astype(np.int64)
    fold_arr = df["fold"].to_numpy()
    holdout_mask = df["is_holdout"].to_numpy().astype(bool)
    train_mask_global = ~holdout_mask

    cv_folds = sorted(
        np.unique(fold_arr[train_mask_global]).tolist()
    )
    # Exclude the holdout fold from cv_folds
    holdout_fold = int(df.filter(pl.col("is_holdout"))["fold"].unique().to_list()[0])
    cv_folds = [f for f in cv_folds if f != holdout_fold]
    assert len(cv_folds) == 5, f"Expected 5 CV folds, got {len(cv_folds)}"

    n = len(smiles)
    logger.info(f"Loaded {n} compounds, {len(cv_folds)} CV folds")

    cfg = XGBConfig(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        early_stopping_rounds=30,
        random_state=0,
    )

    # Build base features: Morgan FP + physchem
    logger.info("Building Morgan FP + physchem features...")
    fp_arr, _ = morgan_ndarray(smiles, n_bits=2048, radius=2)
    phys_arr = df.select(list(PHYSCHEM_COLUMNS)).to_numpy().astype(np.float32)
    X_base = np.concatenate([fp_arr.astype(np.float32), phys_arr], axis=1)
    logger.info(f"  Base features: {X_base.shape}")

    # Build pocket features
    logger.info("Building pocket features...")
    assigner = PocketAssigner.from_csv(FRAGMENT_CSV)
    pocket_feat = build_pocket_features(smiles, assigner)
    logger.info(f"  Pocket features: {pocket_feat.shape}")

    # Concatenate for +Pocket model
    X_pocket = np.concatenate([X_base, pocket_feat], axis=1)
    logger.info(f"  Base+Pocket features: {X_pocket.shape}")

    # Run both models
    logger.info("\n=== Running baseline (Morgan + physchem) ===")
    result_base = run_cv(
        X_base, y_all, fold_arr, train_mask_global,
        holdout_mask, cv_folds, cfg, "baseline"
    )

    logger.info("\n=== Running +Pocket (Morgan + physchem + pocket features) ===")
    result_pocket = run_cv(
        X_pocket, y_all, fold_arr, train_mask_global,
        holdout_mask, cv_folds, cfg, "+pocket"
    )

    # Compare
    delta_oof = result_pocket["oof_auroc"] - result_base["oof_auroc"]
    delta_hold = result_pocket["holdout_auroc"] - result_base["holdout_auroc"]
    logger.info(f"\n=== Comparison ===")
    logger.info(f"  OOF AUROC delta:     {delta_oof:+.4f}")
    logger.info(f"  Holdout AUROC delta: {delta_hold:+.4f}")

    summary = {
        "baseline": result_base,
        "with_pocket": result_pocket,
        "delta_oof_auroc": delta_oof,
        "delta_holdout_auroc": delta_hold,
        "config": {
            "n_estimators": cfg.n_estimators,
            "max_depth": cfg.max_depth,
            "learning_rate": cfg.learning_rate,
            "early_stopping_rounds": cfg.early_stopping_rounds,
            "random_state": cfg.random_state,
        },
        "pocket_feature_columns": [
            "pocket_A_tc", "pocket_B_tc", "pocket_C_tc", "pocket_D_tc",
            "pocket_A_sub", "pocket_B_sub", "pocket_C_sub", "pocket_D_sub",
        ],
    }

    output_path = ARTIFACT_DIR / "xgb_pocket_comparison.json"
    output_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
