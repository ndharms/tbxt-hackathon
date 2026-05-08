"""XGBoost baseline: Morgan FP (2048) + physchem features.

Uses the same 6-fold chemical-space splits as the chemeleon ensemble so
OOF/holdout metrics are directly comparable. Validation fold is used for
``early_stopping_rounds`` on log loss, which caps tree count robustly
without needing a fixed ``n_estimators``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from loguru import logger
from typeguard import typechecked

from .exceptions import DataError
from .fingerprints import morgan_ndarray

PHYSCHEM_COLUMNS: tuple[str, ...] = (
    "mw",
    "logp",
    "hbd",
    "hba",
    "heavy_atoms",
    "num_rings",
    "tpsa",
    "rotatable_bonds",
)


@dataclass(frozen=True)
class XGBConfig:
    """Hyperparameters for the XGBoost baseline.

    Defaults chosen to be sensible-but-generic for a ~1.5k-row, 2056-dim,
    ~25%-positive binary task. Not tuned.
    """

    n_estimators: int = 1000
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.6
    reg_lambda: float = 1.0
    min_child_weight: float = 1.0
    early_stopping_rounds: int = 30
    random_state: int = 0
    n_jobs: int = -1


@dataclass
class XGBFeatureMatrix:
    """Feature matrix + column provenance.

    Attributes:
        X: (n, 2048 + len(physchem)) float32 matrix.
        column_names: ordered list of column names (fp_0..fp_2047, then physchem).
        fp_end: index where physchem columns start; fp slice is X[:, :fp_end].
    """

    X: np.ndarray
    column_names: list[str]
    fp_end: int


@typechecked
def build_features(
    frame: pl.DataFrame,
    smiles_col: str = "canonical_smiles",
    physchem_cols: tuple[str, ...] = PHYSCHEM_COLUMNS,
    n_bits: int = 2048,
    radius: int = 2,
) -> XGBFeatureMatrix:
    """Concatenate Morgan fingerprints with physchem descriptor columns."""
    missing = [c for c in physchem_cols if c not in frame.columns]
    if missing:
        raise DataError(f"frame missing physchem columns: {missing}")
    smiles = frame[smiles_col].to_list()
    fp_arr, _ = morgan_ndarray(smiles, n_bits=n_bits, radius=radius)
    phys = frame.select(list(physchem_cols)).to_numpy().astype(np.float32)
    if np.isnan(phys).any():
        raise DataError("physchem matrix contains NaNs")
    X = np.concatenate([fp_arr.astype(np.float32), phys], axis=1)
    cols = [f"fp_{i}" for i in range(n_bits)] + list(physchem_cols)
    logger.info(f"built XGB feature matrix: shape={X.shape}")
    return XGBFeatureMatrix(X=X, column_names=cols, fp_end=n_bits)


@dataclass
class XGBRunResult:
    """Outputs from a single fold's XGBoost fit.

    Attributes:
        booster: trained xgboost classifier.
        best_iteration: iteration chosen by early stopping (or n_estimators - 1).
        best_val_logloss: validation log loss at best_iteration.
        feature_importances: gain-based importances (flat float array, aligned to X columns).
    """

    booster: xgb.XGBClassifier
    best_iteration: int
    best_val_logloss: float
    feature_importances: np.ndarray = field(default_factory=lambda: np.zeros(0))


@typechecked
def train_one_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: XGBConfig,
) -> XGBRunResult:
    """Fit one XGBoost classifier with val-based early stopping on logloss.

    Uses ``scale_pos_weight`` derived from the training set to counter the
    ~3:1 negative:positive class imbalance.
    """
    pos = int(y_train.sum())
    neg = int(y_train.size - pos)
    scale_pos_weight = neg / max(pos, 1)

    clf = xgb.XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        min_child_weight=cfg.min_child_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        early_stopping_rounds=cfg.early_stopping_rounds,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    best_iter = int(clf.best_iteration) if clf.best_iteration is not None else cfg.n_estimators - 1
    results = clf.evals_result()
    # evals_result format: {"validation_0": {"logloss": [...]}}
    val_logloss_history = next(iter(results.values()))["logloss"]
    best_ll = float(val_logloss_history[best_iter])
    logger.info(
        f"xgb fit: best_iter={best_iter}/{cfg.n_estimators} val_logloss={best_ll:.4f} "
        f"(scale_pos_weight={scale_pos_weight:.2f})",
    )
    return XGBRunResult(
        booster=clf,
        best_iteration=best_iter,
        best_val_logloss=best_ll,
        feature_importances=clf.feature_importances_.astype(np.float64),
    )


@typechecked
def predict_proba_xgb(clf: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
    """Return P(binder) in [0, 1]."""
    return clf.predict_proba(X)[:, 1]


@typechecked
def save_xgb_model(clf: xgb.XGBClassifier, path: Path) -> None:
    """Persist an XGBoost classifier to UBJSON (``.ubj``).

    XGBoost's native binary format is portable across versions and
    languages. Saving uses the sklearn wrapper's ``save_model``, which
    writes the booster + learner metadata (including ``best_iteration``
    set by early stopping) but *not* the sklearn wrapper config. On
    reload via :func:`load_xgb_model` the wrapper is reconstructed fresh.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(path))


@typechecked
def load_xgb_model(path: Path) -> xgb.XGBClassifier:
    """Rehydrate an XGBoost classifier from :func:`save_xgb_model`."""
    if not path.exists():
        raise FileNotFoundError(f"xgb model not found: {path}")
    clf = xgb.XGBClassifier()
    clf.load_model(str(path))
    return clf


@typechecked
def train_one_xgb_novalid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: XGBConfig,
) -> XGBRunResult:
    """Fit one XGBoost classifier for a fixed ``n_estimators`` with no val set.

    Counterpart to ``train_one_xgb`` for the no-validation ensemble. Uses
    the same class-imbalance weighting.
    """
    pos = int(y_train.sum())
    neg = int(y_train.size - pos)
    scale_pos_weight = neg / max(pos, 1)

    clf = xgb.XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        min_child_weight=cfg.min_child_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )
    clf.fit(X_train, y_train, verbose=False)
    logger.info(
        f"xgb no-val fit: n_estimators={cfg.n_estimators} "
        f"(scale_pos_weight={scale_pos_weight:.2f})",
    )
    return XGBRunResult(
        booster=clf,
        best_iteration=cfg.n_estimators - 1,
        best_val_logloss=float("nan"),
        feature_importances=clf.feature_importances_.astype(np.float64),
    )
