"""Step 03 (try4): 9-variant XGBoost matrix with no-val CV.

Matrix
------
Three fingerprint types x three feature sets = nine ensemble variants.
Each variant follows the try3 no-val recipe: 5-fold CV on the filtered
label, n_estimators = 61 (fixed), holdout = fold 3.

    FP type       | fp_only | fp + pocket | fp + pocket + physchem |
    ------------- | ------- | ----------- | ---------------------- |
    morgan        | (M0)    | (MP)        | (MPP)                  |
    rdkit_path    | (R0)    | (RP)        | (RPP)                  |
    maccs         | (K0)    | (KP)        | (KPP)                  |

Variant names in artifacts follow ``<fp>_<featset>`` where:
    fp in {morgan, rdkit, maccs}
    featset in {fp, fp_pocket, fp_pocket_phys}

Writes per variant (9 sets total)
---------------------------------
    data/classification-models-try4-rjg/xgb_<variant>_cv_fold_{0,1,2,4,5}.ubj
    data/classification-models-try4-rjg/xgb_<variant>_oof.csv
    data/classification-models-try4-rjg/xgb_<variant>_holdout.csv
    data/classification-models-try4-rjg/xgb_<variant>_metrics.json

Usage
-----
    uv run python scripts/classification-models-try4-rjg/03_train_xgb_matrix.py \\
        --n-estimators 61
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score

from tbxt_hackathon.fingerprints import FP_TYPES, fingerprint_ndarray
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
ARTIFACT_DIR = ROOT / "data" / "classification-models-try4-rjg"

FP_NAME_MAP = {"morgan": "morgan", "rdkit_path": "rdkit", "maccs": "maccs"}
FEAT_SETS = ("fp", "fp_pocket", "fp_pocket_phys")
POCKET_COLUMNS = tuple(
    f"pocket_{p}_{kind}" for p in ("A", "B", "C", "D") for kind in ("tanimoto", "substruct")
)


@dataclass
class VariantFeatures:
    """Feature matrix + column-group metadata for one variant."""

    X: np.ndarray
    column_names: list[str]
    fp_end: int  # X[:, :fp_end] = fingerprint slice
    pocket_end: int | None  # X[:, fp_end:pocket_end] = pocket slice (if any)
    phys_end: int | None  # X[:, pocket_end:phys_end] = physchem slice (if any)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--n-estimators",
        type=int,
        default=61,
        help="Fixed tree count (default 61, matching try3's median best_iter)",
    )
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--fp-types",
        nargs="+",
        default=list(FP_TYPES),
        choices=list(FP_TYPES),
        help="Subset of fingerprint types to train (default: all three)",
    )
    ap.add_argument(
        "--featsets",
        nargs="+",
        default=list(FEAT_SETS),
        choices=list(FEAT_SETS),
        help="Subset of feature combinations to train (default: all three)",
    )
    return ap.parse_args()


def _build_variant_features(
    df: pl.DataFrame,
    fp_type: str,
    featset: str,
) -> VariantFeatures:
    """Construct the feature matrix for one variant.

    df must already carry pocket_*_* and the PHYSCHEM_COLUMNS columns.
    """
    smiles = df["canonical_smiles"].to_list()
    fp = fingerprint_ndarray(smiles, fp_type=fp_type, n_bits=2048).astype(np.float32)
    blocks = [fp]
    cols = [f"fp_{i}" for i in range(fp.shape[1])]
    fp_end = fp.shape[1]
    pocket_end: int | None = None
    phys_end: int | None = None

    if featset in ("fp_pocket", "fp_pocket_phys"):
        pocket = df.select(list(POCKET_COLUMNS)).to_numpy().astype(np.float32)
        assert not np.isnan(pocket).any(), "NaN in pocket features after join"
        blocks.append(pocket)
        cols.extend(POCKET_COLUMNS)
        pocket_end = fp_end + pocket.shape[1]

    if featset == "fp_pocket_phys":
        phys = df.select(list(PHYSCHEM_COLUMNS)).to_numpy().astype(np.float32)
        assert not np.isnan(phys).any(), "NaN in physchem after join"
        blocks.append(phys)
        cols.extend(PHYSCHEM_COLUMNS)
        phys_end = (pocket_end or fp_end) + phys.shape[1]

    X = np.concatenate(blocks, axis=1)
    return VariantFeatures(
        X=X, column_names=cols, fp_end=fp_end,
        pocket_end=pocket_end, phys_end=phys_end,
    )


def _train_one_variant(
    feat: VariantFeatures,
    df: pl.DataFrame,
    variant_name: str,
    cfg: XGBConfig,
    holdout_fold: int,
) -> dict:
    """Run 5-fold no-val CV for one feature-set variant. Write artifacts."""
    y_all = df["is_binder"].to_numpy().astype(np.int64)
    holdout_mask = df["is_holdout"].to_numpy()
    train_mask_global = ~holdout_mask
    cv_folds = sorted(
        df.filter(~pl.col("is_holdout"))["fold"].unique().to_list()
    )
    assert len(cv_folds) == 5, f"expected 5 CV folds, got {cv_folds}"

    oof = np.full(df.shape[0], np.nan)
    hold_preds_per_model: dict[int, np.ndarray] = {}
    per_fold: list[dict] = []
    importances: list[np.ndarray] = []

    fold_arr = df["fold"].to_numpy()
    X = feat.X

    for test_fold in cv_folds:
        train_folds = [f for f in cv_folds if f != test_fold]
        tr_mask = np.isin(fold_arr, train_folds) & train_mask_global
        te_mask = (fold_arr == test_fold) & train_mask_global

        X_tr, y_tr = X[tr_mask], y_all[tr_mask]
        X_te, y_te = X[te_mask], y_all[te_mask]

        logger.info(
            f"[{variant_name}] fold={test_fold} "
            f"n_train={tr_mask.sum()} (pos={int(y_tr.sum())}) "
            f"n_test={te_mask.sum()} (pos={int(y_te.sum())})"
        )
        res = train_one_xgb_novalid(X_tr, y_tr, cfg)

        model_path = ARTIFACT_DIR / f"xgb_{variant_name}_cv_fold_{test_fold}.ubj"
        save_xgb_model(res.booster, model_path)

        te_probs = predict_proba_xgb(res.booster, X_te)
        oof[np.where(te_mask)[0]] = te_probs
        hold_probs = predict_proba_xgb(res.booster, X[holdout_mask])
        hold_preds_per_model[test_fold] = hold_probs
        importances.append(res.feature_importances)

        try:
            auroc_te = float(roc_auc_score(y_te, te_probs))
        except ValueError:
            auroc_te = float("nan")
        try:
            auprc_te = float(average_precision_score(y_te, te_probs))
        except ValueError:
            auprc_te = float("nan")

        per_fold.append({
            "test_fold": int(test_fold),
            "train_folds": [int(f) for f in train_folds],
            "n_train": int(tr_mask.sum()),
            "n_test": int(te_mask.sum()),
            "pos_train": int(y_tr.sum()),
            "pos_test": int(y_te.sum()),
            "test_auroc": auroc_te,
            "test_auprc": auprc_te,
            "model_path": str(model_path.relative_to(ROOT)),
        })

    # OOF + holdout aggregates
    finite = ~np.isnan(oof) & train_mask_global
    oof_auroc = float(roc_auc_score(y_all[finite], oof[finite]))
    oof_auprc = float(average_precision_score(y_all[finite], oof[finite]))

    hold_matrix = np.column_stack([hold_preds_per_model[f] for f in cv_folds])
    hold_mean = hold_matrix.mean(axis=1)
    y_hold = y_all[holdout_mask]
    hold_auroc = float(roc_auc_score(y_hold, hold_mean))
    hold_auprc = float(average_precision_score(y_hold, hold_mean))

    logger.info(
        f"[{variant_name}] OOF AUROC={oof_auroc:.3f} AUPRC={oof_auprc:.3f} | "
        f"Holdout AUROC={hold_auroc:.3f} AUPRC={hold_auprc:.3f}"
    )

    # Write predictions
    train_df = df.filter(~pl.col("is_holdout"))
    hold_df = df.filter(pl.col("is_holdout"))

    oof_out = train_df.with_columns(
        pl.Series(f"oof_p_binder_xgb_{variant_name}", oof[train_mask_global]),
    ).select([
        "compound_id", "canonical_smiles", "fold", "is_binder",
        "pKD_global_mean", f"oof_p_binder_xgb_{variant_name}",
    ])
    oof_out.write_csv(ARTIFACT_DIR / f"xgb_{variant_name}_oof.csv")

    hold_cols = {
        f"p_binder_xgb_{variant_name}_fold_{f}": hold_preds_per_model[f]
        for f in cv_folds
    }
    hold_out = hold_df.select([
        "compound_id", "canonical_smiles", "fold", "is_binder", "pKD_global_mean",
    ]).with_columns(
        [pl.Series(k, v) for k, v in hold_cols.items()]
        + [pl.Series(f"p_binder_xgb_{variant_name}_ensemble_mean", hold_mean)]
    )
    hold_out.write_csv(ARTIFACT_DIR / f"xgb_{variant_name}_holdout.csv")

    # Feature-group importances (rolled up via gain sum)
    mean_importance = np.mean(np.stack(importances, axis=0), axis=0)
    fp_imp = float(mean_importance[: feat.fp_end].sum())
    importance_by_group = {"fingerprint": fp_imp}
    if feat.pocket_end is not None:
        pocket_imp = float(mean_importance[feat.fp_end : feat.pocket_end].sum())
        importance_by_group["pocket"] = pocket_imp
    if feat.phys_end is not None:
        phys_imp = float(
            mean_importance[(feat.pocket_end or feat.fp_end) : feat.phys_end].sum()
        )
        importance_by_group["physchem"] = phys_imp
    total = sum(importance_by_group.values()) or 1.0
    importance_share = {k: v / total for k, v in importance_by_group.items()}

    summary = {
        "variant": variant_name,
        "holdout_fold": holdout_fold,
        "n_features": int(feat.X.shape[1]),
        "feature_groups": {
            "fingerprint": [0, feat.fp_end],
            "pocket": [feat.fp_end, feat.pocket_end] if feat.pocket_end else None,
            "physchem": [feat.pocket_end or feat.fp_end, feat.phys_end] if feat.phys_end else None,
        },
        "cv_runs": per_fold,
        "oof_auroc": oof_auroc,
        "oof_auprc": oof_auprc,
        "holdout_ensemble_auroc": hold_auroc,
        "holdout_ensemble_auprc": hold_auprc,
        "feature_importance_group_gain_sum": importance_by_group,
        "feature_importance_group_share": importance_share,
        "config": {
            "n_estimators": cfg.n_estimators,
            "max_depth": cfg.max_depth,
            "learning_rate": cfg.learning_rate,
            "subsample": cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "random_state": cfg.random_state,
        },
    }
    (ARTIFACT_DIR / f"xgb_{variant_name}_metrics.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Load everything once
    df = pl.read_csv(FOLDS_CSV)
    holdout_fold = int(df.filter(pl.col("is_holdout"))["fold"].unique().to_list()[0])

    pocket = pl.read_csv(POCKET_CSV)
    compounds = pl.read_csv(COMPOUNDS_CSV).select(["compound_id", *PHYSCHEM_COLUMNS])

    df = df.join(pocket, on="compound_id", how="left")
    df = df.join(compounds, on="compound_id", how="left")
    for c in POCKET_COLUMNS:
        if df[c].is_null().any():
            raise ValueError(f"null pocket column after join: {c}")
    for c in PHYSCHEM_COLUMNS:
        if df[c].is_null().any():
            raise ValueError(f"null physchem column after join: {c}")
    logger.info(f"loaded merged frame: {df.shape}; holdout fold = {holdout_fold}")

    cfg = XGBConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        random_state=args.seed,
    )

    # Train matrix
    matrix_summary: dict[str, dict] = {}
    for fp_type, featset in product(args.fp_types, args.featsets):
        variant_name = f"{FP_NAME_MAP[fp_type]}_{featset}"
        logger.info(f"\n=== variant: {variant_name} (fp={fp_type}, featset={featset}) ===")
        feat = _build_variant_features(df, fp_type=fp_type, featset=featset)
        logger.info(
            f"[{variant_name}] feature matrix shape={feat.X.shape} "
            f"fp_end={feat.fp_end} pocket_end={feat.pocket_end} phys_end={feat.phys_end}"
        )
        summary = _train_one_variant(feat, df, variant_name, cfg, holdout_fold)
        matrix_summary[variant_name] = {
            "fp_type": fp_type,
            "featset": featset,
            "n_features": summary["n_features"],
            "oof_auroc": summary["oof_auroc"],
            "oof_auprc": summary["oof_auprc"],
            "holdout_ensemble_auroc": summary["holdout_ensemble_auroc"],
            "holdout_ensemble_auprc": summary["holdout_ensemble_auprc"],
            "feature_importance_group_share": summary["feature_importance_group_share"],
        }

    (ARTIFACT_DIR / "matrix_summary.json").write_text(json.dumps(matrix_summary, indent=2))

    # Compact log at the end
    logger.info("\n=== matrix summary ===")
    logger.info(
        f"{'variant':<28s} {'n_feat':>7s} {'OOF_AUROC':>10s} {'OOF_AUPRC':>10s} "
        f"{'HO_AUROC':>10s} {'HO_AUPRC':>10s}"
    )
    for name, m in matrix_summary.items():
        logger.info(
            f"{name:<28s} {m['n_features']:>7d} {m['oof_auroc']:>10.3f} "
            f"{m['oof_auprc']:>10.3f} {m['holdout_ensemble_auroc']:>10.3f} "
            f"{m['holdout_ensemble_auprc']:>10.3f}"
        )


if __name__ == "__main__":
    main()
