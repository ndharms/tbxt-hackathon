"""Step 01 (deployment): train the final MACCS+pocket+physchem ensemble.

Produces a 6-booster leave-one-fold-out ensemble of XGBoost classifiers
using the best feature set from try4 (``maccs_fp_pocket_phys``):
    MACCS keys (167 bits) + pocket features (8 cols) + physchem (8 cols)
    = 183 features total.

Each fold k in {0, 1, 2, 3, 4, 5} is held out in turn; one booster is
fit on the other 5 folds. Every compound contributes to training in 5 of
the 6 boosters, and each compound has exactly one OOF prediction (from
the model that did not see it). For novel compounds at inference time,
all 6 boosters are averaged -- see ``src/tbxt_hackathon/deployment.py``.

Feature provenance at inference
-------------------------------
Downstream inference must use the same featurization pipeline, which is
why we also save ``fold_ids.json`` (the fold assignments that were used)
and ``feature_spec.json`` (the exact column ordering and feature recipe).

Rationale for "leave one fold out" rather than a single all-data model
---------------------------------------------------------------------
Every training compound contributes to 5/6 boosters. Averaging 6 boosters
gives variance reduction comparable to a 5-fold CV ensemble, while not
discarding any fold's data. The single-model alternative would train on
all 708 compounds but we'd lose the built-in "predictions diversity" that
an ensemble provides for scoring novel out-of-distribution compounds
(exactly the onepot 3.4B screening task).

Outputs
-------
    data/deployment-model/xgb_deploy_fold_{0..5}.ubj           # 6 boosters
    data/deployment-model/training_predictions.csv              # OOF preds per compound
    data/deployment-model/metrics.json                          # aggregated metrics
    data/deployment-model/feature_spec.json                     # feature pipeline fingerprint
    data/deployment-model/fold_assignments_used.csv             # frozen training folds

Usage
-----
    uv run python scripts/deployment-model/01_train_ensemble.py

    # to rebuild with a different fingerprint or feature set, instead
    # use the more general try4 training script:
    #   scripts/classification-models-try4-rjg/03_train_xgb_matrix.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score

from tbxt_hackathon.deployment import (
    DEPLOYMENT_FEATURE_SPEC,
    build_deployment_features,
)
from tbxt_hackathon.xgb_baseline import (
    PHYSCHEM_COLUMNS,
    XGBConfig,
    predict_proba_xgb,
    save_xgb_model,
    train_one_xgb_novalid,
)

ROOT = Path(__file__).resolve().parents[2]

# Training data is the same filtered label+folds used by try4, which is
# fold 3 as the evaluation holdout. For the deployment model we still use
# the filtered-label rows, but every fold gets rotated through the
# held-out position so no fold is uniquely held out. In other words:
# training set = all 708 filtered rows, and is_holdout is ignored here.
FOLDS_CSV = ROOT / "data" / "classification-models-try4-rjg" / "fold_assignments_filtered.csv"
POCKET_CSV = ROOT / "data" / "classification-models-try4-rjg" / "pocket_features.csv"
COMPOUNDS_CSV = ROOT / "data" / "processed" / "tbxt_compounds_clean.csv"
OUT_DIR = ROOT / "data" / "deployment-model"

N_ESTIMATORS = 61  # matches try3/try4 median best_iter
MAX_DEPTH = 6
LEARNING_RATE = 0.05


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(FOLDS_CSV)
    pocket = pl.read_csv(POCKET_CSV)
    compounds = pl.read_csv(COMPOUNDS_CSV).select(["compound_id", *PHYSCHEM_COLUMNS])
    df = df.join(pocket, on="compound_id", how="left")
    df = df.join(compounds, on="compound_id", how="left")
    logger.info(f"loaded training frame: {df.shape}")

    feats = build_deployment_features(df)
    y_all = df["is_binder"].to_numpy().astype(np.int64)
    fold_arr = df["fold"].to_numpy()
    all_folds = sorted(np.unique(fold_arr).tolist())
    assert len(all_folds) == 6, f"expected 6 folds, got {all_folds}"
    logger.info(
        f"feature matrix: shape={feats.X.shape} "
        f"(maccs={feats.maccs_end}, pocket={feats.maccs_end}-{feats.pocket_end}, "
        f"physchem={feats.pocket_end}-{feats.phys_end})"
    )

    cfg = XGBConfig(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        random_state=0,
    )

    oof = np.full(df.shape[0], np.nan)
    per_fold_metrics: list[dict] = []
    saved_models: list[str] = []

    for held_out in all_folds:
        train_mask = fold_arr != held_out
        test_mask = fold_arr == held_out

        X_tr, y_tr = feats.X[train_mask], y_all[train_mask]
        X_te, y_te = feats.X[test_mask], y_all[test_mask]

        logger.info(
            f"\n--- fold {held_out} held out: "
            f"n_train={int(train_mask.sum())} (pos={int(y_tr.sum())}) "
            f"n_test={int(test_mask.sum())} (pos={int(y_te.sum())})"
        )
        res = train_one_xgb_novalid(X_tr, y_tr, cfg)

        model_path = OUT_DIR / f"xgb_deploy_fold_{held_out}.ubj"
        save_xgb_model(res.booster, model_path)
        saved_models.append(str(model_path.relative_to(ROOT)))

        te_probs = predict_proba_xgb(res.booster, X_te)
        oof[test_mask] = te_probs

        try:
            auroc_te = float(roc_auc_score(y_te, te_probs))
            auprc_te = float(average_precision_score(y_te, te_probs))
        except ValueError:
            auroc_te = float("nan")
            auprc_te = float("nan")

        per_fold_metrics.append({
            "held_out_fold": int(held_out),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "pos_train": int(y_tr.sum()),
            "pos_test": int(y_te.sum()),
            "oof_auroc": auroc_te,
            "oof_auprc": auprc_te,
            "model_path": str(model_path.relative_to(ROOT)),
        })
        logger.info(
            f"fold {held_out}: OOF AUROC={auroc_te:.3f} AUPRC={auprc_te:.3f}"
        )

    finite = ~np.isnan(oof)
    overall_auroc = float(roc_auc_score(y_all[finite], oof[finite]))
    overall_auprc = float(average_precision_score(y_all[finite], oof[finite]))
    logger.info(
        f"\n=== deployment ensemble overall OOF: "
        f"AUROC={overall_auroc:.3f} AUPRC={overall_auprc:.3f} "
        f"(n={int(finite.sum())}, pos={int(y_all[finite].sum())})"
    )

    # Training-set OOF predictions file
    train_out = df.select([
        "compound_id", "canonical_smiles", "fold", "is_binder", "pKD_global_mean",
    ]).with_columns(pl.Series("oof_p_binder", oof))
    train_out.write_csv(OUT_DIR / "training_predictions.csv")

    # Freeze the fold assignment used for training
    df.select(["compound_id", "canonical_smiles", "fold", "is_binder"]).write_csv(
        OUT_DIR / "fold_assignments_used.csv"
    )

    # Feature spec for inference reproducibility
    (OUT_DIR / "feature_spec.json").write_text(
        json.dumps(DEPLOYMENT_FEATURE_SPEC, indent=2)
    )

    metrics = {
        "model_name": "xgb_maccs_fp_pocket_phys_deploy",
        "feature_spec": DEPLOYMENT_FEATURE_SPEC,
        "n_train_total": int(df.shape[0]),
        "n_positives_total": int(y_all.sum()),
        "n_folds": len(all_folds),
        "ensemble_size": len(all_folds),
        "config": {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "learning_rate": LEARNING_RATE,
            "random_state": 0,
            "scale_pos_weight_strategy": "neg/pos per training fold",
        },
        "per_fold": per_fold_metrics,
        "overall_oof_auroc": overall_auroc,
        "overall_oof_auprc": overall_auprc,
        "saved_models": saved_models,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info(f"wrote {len(saved_models)} boosters + metadata to {OUT_DIR}")


if __name__ == "__main__":
    main()
