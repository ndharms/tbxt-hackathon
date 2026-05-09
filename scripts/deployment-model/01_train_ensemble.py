"""Step 01 (deployment): train the full 3-model rank-norm-mean ensemble.

Trains three model families under a shared 6-fold leave-one-out
cross-validation regime, so every compound has an out-of-fold (OOF)
prediction from each model family (the booster/checkpoint that did not
see it). The final deployment score for a novel compound is the mean of
the three models' rank-normalized probabilities (see
``src/tbxt_hackathon/deployment.py``).

Model families
--------------
    1. deploy_xgb: XGBoost on MACCS (167) + pocket (8) + physchem (8)
       = 183 features. Winner of the try4 feature ablation.
    2. morgan_xgb: XGBoost on Morgan ECFP4 (2048) + physchem (8)
       = 2056 features. The try3 ``xgb_no_val`` recipe.
    3. chemeleon: CheMeleon transfer-learning MPNN. The try3
       ``chemeleon_no_val`` recipe.

Each fold k in {0, 1, 2, 3, 4, 5} is held out in turn; one model per
family is fit on the other 5 folds. Every compound contributes to 5/6
boosters/checkpoints in each family, and each compound has exactly one
OOF prediction per family (from the model that did not see it).

Outputs
-------
    data/deployment-model/
    |-- xgb_deploy_fold_{0..5}.ubj             # deploy XGB boosters
    |-- xgb_morgan_fold_{0..5}.ubj             # Morgan XGB boosters
    |-- chemeleon_fold_{0..5}.ckpt             # CheMeleon MPNN checkpoints
    |-- training_predictions.csv               # per-compound OOF for all 3 models + ensemble
    |-- metrics.json                           # per-model + combined metrics
    |-- feature_spec.json                      # feature pipeline fingerprints
    |-- fold_assignments_used.csv              # frozen training folds

Usage
-----
    uv run python scripts/deployment-model/01_train_ensemble.py

    # Skip CheMeleon (slowest step) for quick iteration on the XGB halves:
    uv run python scripts/deployment-model/01_train_ensemble.py --skip-chemeleon

    # Force CPU for CheMeleon (default: torch picks MPS/GPU if available):
    uv run python scripts/deployment-model/01_train_ensemble.py --accelerator cpu
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score

from tbxt_hackathon.deployment import (
    CHEMELEON_SPEC,
    DEPLOYMENT_FEATURE_SPEC,
    ENSEMBLE_SPEC,
    MORGAN_FEATURE_SPEC,
    build_deployment_features,
    featurize_smiles_morgan,
    rank_norm,
)
from tbxt_hackathon.xgb_baseline import (
    PHYSCHEM_COLUMNS,
    XGBConfig,
    predict_proba_xgb,
    save_xgb_model,
    train_one_xgb_novalid,
)

ROOT = Path(__file__).resolve().parents[2]

FOLDS_CSV = ROOT / "data" / "classification-models-try4-rjg" / "fold_assignments_filtered.csv"
POCKET_CSV = ROOT / "data" / "classification-models-try4-rjg" / "pocket_features.csv"
COMPOUNDS_CSV = ROOT / "data" / "processed" / "tbxt_compounds_clean.csv"
OUT_DIR = ROOT / "data" / "deployment-model"

# XGB hyperparameters: match try3/try4 (n_estimators = 61 tracks median best_iter)
XGB_N_ESTIMATORS = 61
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.05

# CheMeleon hyperparameters: match try3 recipe
CHEMELEON_EPOCHS = 15
CHEMELEON_BATCH_SIZE = 32
CHEMELEON_HIDDEN_DIM = 256
CHEMELEON_DROPOUT = 0.2
CHEMELEON_LR = 1e-3
CHEMELEON_SEED = 0


# ---- helpers ---------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--skip-chemeleon",
        action="store_true",
        help="Don't train the CheMeleon arm. Useful during iteration.",
    )
    ap.add_argument(
        "--skip-morgan",
        action="store_true",
        help="Don't train the Morgan XGB arm.",
    )
    ap.add_argument(
        "--skip-deploy",
        action="store_true",
        help="Don't retrain the deploy XGB arm (keep existing boosters if any).",
    )
    ap.add_argument(
        "--accelerator",
        default="auto",
        choices=["auto", "cpu", "mps", "gpu"],
        help="Accelerator for CheMeleon training.",
    )
    return ap.parse_args()


def _safe_metric(fn, y_true, y_score) -> float:
    try:
        return float(fn(y_true, y_score))
    except ValueError:
        return float("nan")


@dataclass
class TrainedFamily:
    """Results from training one model family across 6 folds."""

    name: str
    oof: np.ndarray                        # (n_compounds,) one prediction per compound
    per_fold: list[dict]                   # per-fold diagnostics
    artifact_paths: list[str]              # saved model paths (relative to repo root)


# ---- training loops per family --------------------------------------------


def train_deploy_xgb(
    df: pl.DataFrame,
    y_all: np.ndarray,
    fold_arr: np.ndarray,
    all_folds: list[int],
) -> TrainedFamily:
    """Train 6-fold LOFO XGB on MACCS+pocket+physchem (183 features)."""
    feats = build_deployment_features(df)
    logger.info(
        f"[deploy_xgb] feature matrix: {feats.X.shape} "
        f"(maccs=[0:{feats.maccs_end}], pocket=[{feats.maccs_end}:{feats.pocket_end}], "
        f"physchem=[{feats.pocket_end}:{feats.phys_end}])",
    )
    cfg = XGBConfig(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        random_state=0,
    )
    return _train_xgb_family(
        family_name="deploy_xgb",
        X=feats.X,
        y_all=y_all,
        fold_arr=fold_arr,
        all_folds=all_folds,
        cfg=cfg,
        model_prefix="xgb_deploy_fold",
    )


def train_morgan_xgb(
    df: pl.DataFrame,
    y_all: np.ndarray,
    fold_arr: np.ndarray,
    all_folds: list[int],
) -> TrainedFamily:
    """Train 6-fold LOFO XGB on Morgan ECFP4 (2048) + physchem (8) = 2056."""
    smiles = df["canonical_smiles"].to_list()
    X = featurize_smiles_morgan(smiles)
    logger.info(f"[morgan_xgb] feature matrix: {X.shape}")
    cfg = XGBConfig(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        random_state=0,
    )
    return _train_xgb_family(
        family_name="morgan_xgb",
        X=X,
        y_all=y_all,
        fold_arr=fold_arr,
        all_folds=all_folds,
        cfg=cfg,
        model_prefix="xgb_morgan_fold",
    )


def _train_xgb_family(
    family_name: str,
    X: np.ndarray,
    y_all: np.ndarray,
    fold_arr: np.ndarray,
    all_folds: list[int],
    cfg: XGBConfig,
    model_prefix: str,
) -> TrainedFamily:
    """Shared 6-fold LOFO training loop for an XGB family."""
    oof = np.full(X.shape[0], np.nan)
    per_fold: list[dict] = []
    artifact_paths: list[str] = []

    for held_out in all_folds:
        train_mask = fold_arr != held_out
        test_mask = fold_arr == held_out

        X_tr, y_tr = X[train_mask], y_all[train_mask]
        X_te, y_te = X[test_mask], y_all[test_mask]

        logger.info(
            f"[{family_name}] fold {held_out} held out: "
            f"n_train={int(train_mask.sum())} (pos={int(y_tr.sum())}) "
            f"n_test={int(test_mask.sum())} (pos={int(y_te.sum())})",
        )
        res = train_one_xgb_novalid(X_tr, y_tr, cfg)

        model_path = OUT_DIR / f"{model_prefix}_{held_out}.ubj"
        save_xgb_model(res.booster, model_path)
        artifact_paths.append(str(model_path.relative_to(ROOT)))

        te_probs = predict_proba_xgb(res.booster, X_te)
        oof[test_mask] = te_probs

        per_fold.append({
            "held_out_fold": int(held_out),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "pos_train": int(y_tr.sum()),
            "pos_test": int(y_te.sum()),
            "oof_auroc": _safe_metric(roc_auc_score, y_te, te_probs),
            "oof_auprc": _safe_metric(average_precision_score, y_te, te_probs),
            "model_path": str(model_path.relative_to(ROOT)),
        })

    return TrainedFamily(
        name=family_name, oof=oof, per_fold=per_fold, artifact_paths=artifact_paths,
    )


def train_chemeleon(
    df: pl.DataFrame,
    y_all: np.ndarray,
    fold_arr: np.ndarray,
    all_folds: list[int],
    accelerator: str,
) -> TrainedFamily:
    """Train 6-fold LOFO CheMeleon transfer-learning ensemble."""
    # Deferred import: torch + chemprop cost ~10 s to import
    import torch

    from tbxt_hackathon.chemeleon_transfer import (
        ClassifierConfig,
        predict_proba,
        train_one_novalid,
    )

    smiles_all = df["canonical_smiles"].to_list()
    oof = np.full(len(smiles_all), np.nan)
    per_fold: list[dict] = []
    artifact_paths: list[str] = []

    cfg = ClassifierConfig(
        hidden_dim=CHEMELEON_HIDDEN_DIM,
        dropout=CHEMELEON_DROPOUT,
        max_epochs=CHEMELEON_EPOCHS,
        patience=CHEMELEON_EPOCHS,  # unused in novalid
        batch_size=CHEMELEON_BATCH_SIZE,
        learning_rate=CHEMELEON_LR,
        random_state=CHEMELEON_SEED,
    )

    for held_out in all_folds:
        train_idx = np.where(fold_arr != held_out)[0]
        test_idx = np.where(fold_arr == held_out)[0]
        tr_smiles = [smiles_all[i] for i in train_idx]
        tr_y = y_all[train_idx]
        te_smiles = [smiles_all[i] for i in test_idx]
        te_y = y_all[test_idx]

        logger.info(
            f"[chemeleon] fold {held_out} held out: "
            f"n_train={len(tr_smiles)} (pos={int(tr_y.sum())}) "
            f"n_test={len(te_smiles)} (pos={int(te_y.sum())}) "
            f"epochs={cfg.max_epochs}",
        )
        ckpt_path = OUT_DIR / f"chemeleon_fold_{held_out}.ckpt"
        model = train_one_novalid(
            train_smiles=tr_smiles,
            train_labels=tr_y,
            cfg=cfg,
            save_path=ckpt_path,
            accelerator=accelerator,
        )
        te_probs = predict_proba(model, te_smiles, accelerator=accelerator)
        oof[test_idx] = te_probs
        artifact_paths.append(str(ckpt_path.relative_to(ROOT)))

        per_fold.append({
            "held_out_fold": int(held_out),
            "n_train": int(len(tr_smiles)),
            "n_test": int(len(te_smiles)),
            "pos_train": int(tr_y.sum()),
            "pos_test": int(te_y.sum()),
            "oof_auroc": _safe_metric(roc_auc_score, te_y, te_probs),
            "oof_auprc": _safe_metric(average_precision_score, te_y, te_probs),
            "ckpt_path": str(ckpt_path.relative_to(ROOT)),
        })

        # Free VRAM / RAM before the next fold
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return TrainedFamily(
        name="chemeleon", oof=oof, per_fold=per_fold, artifact_paths=artifact_paths,
    )


# ---- main -----------------------------------------------------------------


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(FOLDS_CSV)
    pocket = pl.read_csv(POCKET_CSV)
    compounds = pl.read_csv(COMPOUNDS_CSV).select(["compound_id", *PHYSCHEM_COLUMNS])
    df = df.join(pocket, on="compound_id", how="left")
    df = df.join(compounds, on="compound_id", how="left")
    logger.info(f"loaded training frame: {df.shape}")

    y_all = df["is_binder"].to_numpy().astype(np.int64)
    fold_arr = df["fold"].to_numpy()
    all_folds = sorted(np.unique(fold_arr).tolist())
    assert len(all_folds) == 6, f"expected 6 folds, got {all_folds}"
    logger.info(f"positives per fold: {dict(zip(*np.unique(fold_arr[y_all == 1], return_counts=True)))}")

    families: list[TrainedFamily] = []
    if not args.skip_deploy:
        families.append(train_deploy_xgb(df, y_all, fold_arr, all_folds))
    else:
        logger.info("--skip-deploy: skipping deploy_xgb family")
    if not args.skip_morgan:
        families.append(train_morgan_xgb(df, y_all, fold_arr, all_folds))
    else:
        logger.info("--skip-morgan: skipping morgan_xgb family")
    if not args.skip_chemeleon:
        families.append(train_chemeleon(df, y_all, fold_arr, all_folds, args.accelerator))
    else:
        logger.info("--skip-chemeleon: skipping chemeleon family")

    # ---- per-compound OOF table for every family + combined rank-norm mean --
    train_out = df.select(
        ["compound_id", "canonical_smiles", "fold", "is_binder", "pKD_global_mean"],
    )
    oof_cols = {}
    for fam in families:
        oof_cols[f"oof_p_binder_{fam.name}"] = fam.oof

    # Combined score: rank-normalize each family's OOF, then mean.
    if families:
        rn_stack = np.column_stack([rank_norm(fam.oof) for fam in families])
        oof_cols["oof_ensemble_rank_score"] = rn_stack.mean(axis=1)
        for fam, rn in zip(families, rn_stack.T):
            oof_cols[f"oof_rank_norm_{fam.name}"] = rn

    train_out = train_out.with_columns([pl.Series(k, v) for k, v in oof_cols.items()])
    train_out.write_csv(OUT_DIR / "training_predictions.csv")
    logger.info(f"wrote {len(train_out)} OOF rows to training_predictions.csv")

    # ---- per-family + combined metrics --------------------------------------
    overall_per_family: dict[str, dict] = {}
    for fam in families:
        finite = ~np.isnan(fam.oof)
        overall_per_family[fam.name] = {
            "overall_oof_auroc": _safe_metric(
                roc_auc_score, y_all[finite], fam.oof[finite],
            ),
            "overall_oof_auprc": _safe_metric(
                average_precision_score, y_all[finite], fam.oof[finite],
            ),
            "per_fold": fam.per_fold,
            "artifact_paths": fam.artifact_paths,
        }

    combined_metrics: dict[str, float] = {}
    if families:
        ens = oof_cols["oof_ensemble_rank_score"]
        finite = ~np.isnan(ens)
        combined_metrics = {
            "overall_oof_auroc": _safe_metric(roc_auc_score, y_all[finite], ens[finite]),
            "overall_oof_auprc": _safe_metric(
                average_precision_score, y_all[finite], ens[finite],
            ),
        }
        logger.info(
            "=== ensemble OOF (rank-norm mean of "
            f"{len(families)} families): "
            f"AUROC={combined_metrics['overall_oof_auroc']:.3f} "
            f"AUPRC={combined_metrics['overall_oof_auprc']:.3f} ===",
        )

    # Freeze the fold assignment used for training
    df.select(["compound_id", "canonical_smiles", "fold", "is_binder"]).write_csv(
        OUT_DIR / "fold_assignments_used.csv",
    )

    # Feature pipeline spec (all families)
    (OUT_DIR / "feature_spec.json").write_text(
        json.dumps(
            {
                "deploy_xgb": DEPLOYMENT_FEATURE_SPEC,
                "morgan_xgb": MORGAN_FEATURE_SPEC,
                "chemeleon": CHEMELEON_SPEC,
                "ensemble": ENSEMBLE_SPEC,
            },
            indent=2,
        ),
    )

    metrics = {
        "model_name": "tbxt_3model_rank_ensemble_deploy",
        "ensemble_spec": ENSEMBLE_SPEC,
        "feature_specs": {
            "deploy_xgb": DEPLOYMENT_FEATURE_SPEC,
            "morgan_xgb": MORGAN_FEATURE_SPEC,
            "chemeleon": CHEMELEON_SPEC,
        },
        "n_train_total": int(df.shape[0]),
        "n_positives_total": int(y_all.sum()),
        "n_folds": len(all_folds),
        "ensemble_size_per_family": len(all_folds),
        "configs": {
            "xgb": {
                "n_estimators": XGB_N_ESTIMATORS,
                "max_depth": XGB_MAX_DEPTH,
                "learning_rate": XGB_LEARNING_RATE,
                "random_state": 0,
                "scale_pos_weight_strategy": "neg/pos per training fold",
            },
            "chemeleon": {
                "epochs": CHEMELEON_EPOCHS,
                "batch_size": CHEMELEON_BATCH_SIZE,
                "hidden_dim": CHEMELEON_HIDDEN_DIM,
                "dropout": CHEMELEON_DROPOUT,
                "learning_rate": CHEMELEON_LR,
                "random_state": CHEMELEON_SEED,
            },
        },
        "per_family": overall_per_family,
        "ensemble_overall": combined_metrics,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info(f"wrote metrics + specs to {OUT_DIR}")


if __name__ == "__main__":
    main()
