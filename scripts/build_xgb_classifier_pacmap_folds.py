#!/usr/bin/env python3
"""
TBXT XGBoost binary classifier using PaCMAP-KMeans6 folds (Ray's splits).

Same features and labeling as build_xgb_classifier.py:
  Features: MACCS keys (167) + RDKit 2D descriptors (217)
  Labels:  pKD >= 4.5 active, pKD <= 3.5 inactive, ambiguous zone excluded
  Negatives: SPR inactives only (no Enamine decoys)
  Weights: positives upweighted to balance total positive/negative weight
  HPO: Optuna 50-trial search over XGBoost parameter space, optimizing AUC-PR

Fold strategy changed from Butina clustering to Ray's PaCMAP-KMeans6 folds
(6 folds, fold 4 held out as structurally distinct OOD test set).
CV runs on folds 0, 1, 2, 3, 5.
"""

import base64
import json
import time
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm
import umap

optuna.logging.set_verbosity(optuna.logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parent.parent
SPR_PATH = REPO_ROOT / "data" / "zenodo" / "tbxt_spr_merged.csv"
FOLDS_PATH = REPO_ROOT / "data" / "folds-pacmap-kmeans6" / "fold_assignments.csv"
OUTPUT_DIR = REPO_ROOT / "models" / "xgb_classifier_pacmap_folds"

ACTIVE_PKD = 4.5
INACTIVE_PKD = 3.5
HOLDOUT_FOLD = 4
CV_FOLDS = [0, 1, 2, 3, 5]
N_CV_FOLDS = len(CV_FOLDS)
SEED = int(np.random.default_rng().integers(0, 2**31))
N_OPTUNA_TRIALS = 50

_DESC_NAMES = [d[0] for d in Descriptors.descList]
_DESC_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(_DESC_NAMES)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load SPR data, apply labels, join with PaCMAP-KMeans6 folds.

    Returns (cv_df, holdout_df) — both with columns:
      compound_id, smiles, pKD, label, source, fold, mol
    """
    spr = pd.read_csv(SPR_PATH)
    assert spr.shape[0] > 0, "SPR file is empty"

    spr_agg = (
        spr.groupby("compound_id")
        .agg(smiles=("smiles", "first"), pKD=("pKD", "median"))
        .reset_index()
    )

    actives = spr_agg.loc[spr_agg["pKD"] >= ACTIVE_PKD].copy()
    actives["label"] = 1
    actives["source"] = "spr_active"

    inactives = spr_agg.loc[spr_agg["pKD"] <= INACTIVE_PKD].copy()
    inactives["label"] = 0
    inactives["source"] = "spr_inactive"

    excluded = spr_agg.loc[
        (spr_agg["pKD"] > INACTIVE_PKD) & (spr_agg["pKD"] < ACTIVE_PKD)
    ]
    print(f"SPR: {len(actives)} actives, {len(inactives)} inactives, "
          f"{len(excluded)} excluded (ambiguous zone)")

    df = pd.concat([actives, inactives], ignore_index=True)
    df = df[df["smiles"].notna() & (df["smiles"] != "")].reset_index(drop=True)

    folds_df = pd.read_csv(FOLDS_PATH, usecols=["compound_id", "fold"])
    df = df.merge(folds_df, on="compound_id", how="inner")
    print(f"After joining with PaCMAP-KMeans6 folds: {len(df)} compounds "
          f"({len(spr_agg) - len(excluded) - len(df)} dropped — no fold assignment)")

    mols = []
    for s in tqdm(df["smiles"], desc="Parsing SMILES"):
        mols.append(Chem.MolFromSmiles(str(s)))
    df["mol"] = mols
    valid = df["mol"].notna()
    n_bad = (~valid).sum()
    if n_bad:
        print(f"Dropped {n_bad} unparseable SMILES")
    df = df[valid].reset_index(drop=True)

    holdout = df[df["fold"] == HOLDOUT_FOLD].reset_index(drop=True)
    cv = df[df["fold"] != HOLDOUT_FOLD].reset_index(drop=True)

    print(f"CV set: {len(cv)} compounds "
          f"({cv['label'].sum()} pos / {(cv['label'] == 0).sum()} neg)")
    print(f"Holdout set (fold {HOLDOUT_FOLD}): {len(holdout)} compounds "
          f"({holdout['label'].sum()} pos / {(holdout['label'] == 0).sum()} neg)")

    for f in sorted(cv["fold"].unique()):
        mask = cv["fold"] == f
        print(f"  CV fold {f}: {mask.sum()} compounds, {cv.loc[mask, 'label'].sum()} pos")

    return cv, holdout


# ═══════════════════════════════════════════════════════════════════════════════
#  2. FEATURIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_features(mols: list) -> tuple[np.ndarray, dict]:
    n = len(mols)

    maccs = np.zeros((n, 167), dtype=np.float32)
    for i, mol in enumerate(tqdm(mols, desc="MACCS")):
        fp = MACCSkeys.GenMACCSKeys(mol)
        DataStructs.ConvertToNumpyArray(fp, maccs[i])

    descs = np.zeros((n, len(_DESC_NAMES)), dtype=np.float64)
    for i, mol in enumerate(tqdm(mols, desc="RDKit descriptors")):
        descs[i] = _DESC_CALC.CalcDescriptors(mol)
    descs = np.nan_to_num(descs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    X = np.hstack([maccs, descs])

    info = dict(
        maccs=167,
        rdkit_desc=len(_DESC_NAMES),
        rdkit_desc_names=_DESC_NAMES,
        total=X.shape[1],
    )
    print(f"Feature matrix: {X.shape[0]} x {X.shape[1]}  "
          f"(MACCS {info['maccs']} + Desc {info['rdkit_desc']})")
    return X, info


# ═══════════════════════════════════════════════════════════════════════════════
#  3. SAMPLE WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_weights(df: pd.DataFrame) -> np.ndarray:
    w = np.ones(len(df), dtype=np.float64)
    pos = df["label"] == 1
    n_neg = (~pos).sum()
    n_pos = pos.sum()
    pos_w = n_neg / n_pos
    w[pos] = pos_w
    print(f"Weights -- negative: 1.0, positive: {pos_w:.2f}")
    return w


# ═══════════════════════════════════════════════════════════════════════════════
#  4. HYPERPARAMETER OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def _cv_score(params: dict, X, y, folds, weights) -> float:
    aucs = []
    for fold in CV_FOLDS:
        tr = folds != fold
        va = folds == fold
        if y[va].sum() == 0 or (y[va] == 0).sum() == 0:
            continue
        clf = xgb.XGBClassifier(**params)
        clf.fit(X[tr], y[tr], sample_weight=weights[tr], verbose=False)
        p = clf.predict_proba(X[va])[:, 1]
        prec, rec, _ = precision_recall_curve(y[va], p)
        aucs.append(auc(rec, prec))
    return float(np.mean(aucs)) if aucs else 0.0


def optimize_hyperparameters(X, y, folds, weights) -> tuple[dict, optuna.Study]:
    def objective(trial: optuna.Trial) -> float:
        params = dict(
            objective="binary:logistic",
            eval_metric="aucpr",
            verbosity=0,
            n_jobs=-1,
            random_state=SEED,
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=50),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        )
        return _cv_score(params, X, y, folds, weights)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

    best = study.best_trial
    print(f"Best trial #{best.number}: AUC-PR = {best.value:.4f}")
    print(f"Best params: {json.dumps(best.params, indent=2)}")

    best_params = dict(
        objective="binary:logistic",
        eval_metric="aucpr",
        verbosity=0,
        n_jobs=-1,
        random_state=SEED,
        **best.params,
    )
    return best_params, study


# ═══════════════════════════════════════════════════════════════════════════════
#  5. TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_cv(X, y, folds, weights, params):
    models, fold_metrics = [], []
    oof = np.full(len(y), np.nan)

    for fold in CV_FOLDS:
        tr = folds != fold
        va = folds == fold

        clf = xgb.XGBClassifier(**params)
        clf.fit(X[tr], y[tr], sample_weight=weights[tr],
                eval_set=[(X[va], y[va])], verbose=False)

        p = clf.predict_proba(X[va])[:, 1]
        oof[va] = p
        models.append(clf)

        auc_roc = roc_auc_score(y[va], p) if y[va].sum() > 0 and (y[va] == 0).sum() > 0 else float("nan")
        prec, rec, _ = precision_recall_curve(y[va], p)
        auc_pr = auc(rec, prec)
        yhat = (p >= 0.5).astype(int)
        f1 = f1_score(y[va], yhat, zero_division=0)
        ba = balanced_accuracy_score(y[va], yhat)

        m = dict(fold=fold, n_train=int(tr.sum()), n_val=int(va.sum()),
                 n_pos_train=int(y[tr].sum()), n_pos_val=int(y[va].sum()),
                 auc_roc=auc_roc, auc_pr=auc_pr, f1=f1, balanced_acc=ba)
        fold_metrics.append(m)
        print(f"  Fold {fold}: AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  "
              f"F1={f1:.4f}  BA={ba:.4f}  "
              f"(val {m['n_val']}, pos {m['n_pos_val']})")

    return models, oof, fold_metrics


def train_final(X, y, weights, params):
    clf = xgb.XGBClassifier(**params)
    clf.fit(X, y, sample_weight=weights, verbose=False)
    return clf


def evaluate_holdout(models, X_holdout, y_holdout):
    preds = np.array([m.predict_proba(X_holdout)[:, 1] for m in models])
    mean_pred = preds.mean(axis=0)

    results = {"n": len(y_holdout), "n_pos": int(y_holdout.sum()),
               "prevalence": float(y_holdout.mean())}

    if y_holdout.sum() > 0 and (y_holdout == 0).sum() > 0:
        results["auroc"] = float(roc_auc_score(y_holdout, mean_pred))
        prec, rec, _ = precision_recall_curve(y_holdout, mean_pred)
        results["auprc"] = float(auc(rec, prec))
    else:
        results["auroc"] = float("nan")
        results["auprc"] = float("nan")

    results["brier"] = float(brier_score_loss(y_holdout, mean_pred))

    baseline_brier = float(brier_score_loss(
        y_holdout, np.full(len(y_holdout), y_holdout.mean())))
    results["baseline_brier_prevalence"] = baseline_brier

    yhat = (mean_pred >= 0.5).astype(int)
    results["f1_at_05"] = float(f1_score(y_holdout, yhat, zero_division=0))
    results["balanced_acc_at_05"] = float(balanced_accuracy_score(y_holdout, yhat))

    return mean_pred, results


# ═══════════════════════════════════════════════════════════════════════════════
#  6. PREDICTION API
# ═══════════════════════════════════════════════════════════════════════════════

class TBXTClassifier:
    def __init__(self, cv_models, final_model, feature_info):
        self.cv_models = cv_models
        self.final_model = final_model
        self.feature_info = feature_info

    def _featurize(self, smiles_list: list[str]):
        mols = []
        for s in smiles_list:
            m = Chem.MolFromSmiles(s)
            if m is None:
                raise ValueError(f"Unparseable SMILES: {s}")
            mols.append(m)
        X, _ = compute_features(mols)
        return X

    def predict_ensemble(self, smiles_list: list[str]):
        X = self._featurize(smiles_list)
        preds = np.array([m.predict_proba(X)[:, 1] for m in self.cv_models])
        return preds.mean(axis=0), preds.std(axis=0)

    def predict_final(self, smiles_list: list[str]):
        X = self._featurize(smiles_list)
        return self.final_model.predict_proba(X)[:, 1]

    def predict(self, smiles_list: list[str], mode: str = "both") -> pd.DataFrame:
        results = pd.DataFrame({"smiles": smiles_list})
        if mode in ("both", "ensemble"):
            mu, sd = self.predict_ensemble(smiles_list)
            results["ensemble_mean"] = mu
            results["ensemble_std"] = sd
        if mode in ("both", "final"):
            results["final_prob"] = self.predict_final(smiles_list)
        return results

    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(self.cv_models):
            m.save_model(str(path / f"cv_model_{i}.json"))
        self.final_model.save_model(str(path / "final_model.json"))
        info = {k: v for k, v in self.feature_info.items() if k != "rdkit_desc_names"}
        info["rdkit_desc_names"] = self.feature_info["rdkit_desc_names"]
        with open(path / "feature_info.json", "w") as f:
            json.dump(info, f, indent=2)

    @classmethod
    def load(cls, path: Path):
        path = Path(path)
        cv = []
        for i in range(N_CV_FOLDS):
            m = xgb.XGBClassifier()
            m.load_model(str(path / f"cv_model_{i}.json"))
            cv.append(m)
        final = xgb.XGBClassifier()
        final.load_model(str(path / "final_model.json"))
        with open(path / "feature_info.json") as f:
            info = json.load(f)
        return cls(cv, final, info)


# ═══════════════════════════════════════════════════════════════════════════════
#  7. HTML REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _plot_roc(y, oof, folds, fold_list) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    for fold in fold_list:
        mask = folds == fold
        if y[mask].sum() == 0 or (y[mask] == 0).sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y[mask], oof[mask])
        a = roc_auc_score(y[mask], oof[mask])
        ax.plot(fpr, tpr, alpha=0.3, label=f"Fold {fold} ({a:.3f})")
    valid = ~np.isnan(oof)
    if y[valid].sum() > 0:
        fpr, tpr, _ = roc_curve(y[valid], oof[valid])
        a = roc_auc_score(y[valid], oof[valid])
        ax.plot(fpr, tpr, "k-", lw=2, label=f"Overall ({a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC -- OOF predictions")
    ax.legend(fontsize=8)
    return _fig_to_b64(fig)


def _plot_pr(y, oof, folds, fold_list) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    for fold in fold_list:
        mask = folds == fold
        if y[mask].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y[mask], oof[mask])
        a = auc(rec, prec)
        ax.plot(rec, prec, alpha=0.3, label=f"Fold {fold} ({a:.3f})")
    valid = ~np.isnan(oof)
    if y[valid].sum() > 0:
        prec, rec, _ = precision_recall_curve(y[valid], oof[valid])
        a = auc(rec, prec)
        ax.plot(rec, prec, "k-", lw=2, label=f"Overall ({a:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="PR -- OOF predictions")
    ax.legend(fontsize=8)
    return _fig_to_b64(fig)


def _plot_hist(y, oof, title="OOF score distribution") -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    valid = ~np.isnan(oof)
    ax.hist(oof[valid & (y == 0)], bins=50, alpha=0.6, label="Negative", color="#4878d0")
    ax.hist(oof[valid & (y == 1)], bins=50, alpha=0.6, label="Positive", color="#ee854a")
    ax.set(xlabel="Predicted probability", ylabel="Count", title=title)
    ax.legend()
    return _fig_to_b64(fig)


def _plot_fold_composition(df, fold_list) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    sources = ["spr_active", "spr_inactive"]
    colors = ["#ee854a", "#4878d0"]
    bottom = np.zeros(len(fold_list))
    for src, col in zip(sources, colors):
        vals = [len(df[(df["fold"] == f) & (df["source"] == src)]) for f in fold_list]
        ax.bar(range(len(fold_list)), vals, bottom=bottom, label=src, color=col)
        bottom += vals
    ax.set(xlabel="Fold", ylabel="Compounds", title="Fold composition by source")
    ax.set_xticks(range(len(fold_list)))
    ax.set_xticklabels([str(f) for f in fold_list])
    ax.legend(fontsize=8)
    return _fig_to_b64(fig)


def _plot_umap(X, df, fold_list) -> str:
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="jaccard", random_state=SEED)
    emb = reducer.fit_transform(X)

    fold_colors = {0: "#4878d0", 1: "#ee854a", 2: "#6acc64",
                   3: "#d65f5f", 5: "#956cb4"}
    markers = {"spr_active": "^", "spr_inactive": "o"}
    labels_seen = set()

    fig, ax = plt.subplots(figsize=(8, 7))
    for source, marker in markers.items():
        for fold in fold_list:
            mask = (df["source"].values == source) & (df["fold"].values == fold)
            if mask.sum() == 0:
                continue
            label = f"{source.replace('_', ' ')}, fold {fold}"
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=fold_colors.get(fold, "#999999"), marker=marker, s=18, alpha=0.5,
                       edgecolors="none", label=label if label not in labels_seen else None)
            labels_seen.add(label)

    from matplotlib.lines import Line2D
    legend_elements = []
    for fold in fold_list:
        legend_elements.append(Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=fold_colors.get(fold, "#999"), markersize=8,
                               label=f"Fold {fold}"))
    for source, marker in markers.items():
        legend_elements.append(Line2D([0], [0], marker=marker, color="w",
                               markerfacecolor="gray", markersize=8,
                               label=source.replace("_", " ")))
    ax.legend(handles=legend_elements, fontsize=8, loc="best", ncol=2)
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2",
           title="UMAP of training features (color=fold, shape=source)")
    ax.set_xticks([])
    ax.set_yticks([])
    return _fig_to_b64(fig)


def _plot_optuna_history(study) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    vals = [t.value for t in trials]
    nums = [t.number for t in trials]

    ax1.scatter(nums, vals, s=15, alpha=0.6, color="#4878d0")
    best_so_far = np.maximum.accumulate(vals)
    ax1.plot(nums, best_so_far, "r-", lw=1.5, label="Best so far")
    ax1.set(xlabel="Trial", ylabel="AUC-PR", title="Optuna optimization history")
    ax1.legend(fontsize=8)

    importances = optuna.importance.get_param_importances(study)
    names = list(importances.keys())[:10]
    scores = [importances[n] for n in names]
    ax2.barh(range(len(names)), scores, color="#ee854a")
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set(xlabel="Importance", title="Hyperparameter importance")
    ax2.invert_yaxis()

    fig.tight_layout()
    return _fig_to_b64(fig)


def generate_report(cv_df, holdout_df, fold_metrics, oof, holdout_pred,
                    holdout_results, feature_info, best_params, study, X_cv,
                    output_path, threshold_info=None):
    y = cv_df["label"].values
    folds = cv_df["fold"].values

    roc_img = _plot_roc(y, oof, folds, CV_FOLDS)
    pr_img = _plot_pr(y, oof, folds, CV_FOLDS)
    hist_img = _plot_hist(y, oof)
    comp_img = _plot_fold_composition(cv_df, CV_FOLDS)
    optuna_img = _plot_optuna_history(study)
    print("  Computing UMAP embedding ...")
    umap_img = _plot_umap(X_cv, cv_df, CV_FOLDS)

    holdout_hist_img = ""
    if len(holdout_df) > 0:
        holdout_hist_img = _plot_hist(
            holdout_df["label"].values, holdout_pred,
            title=f"Holdout (fold {HOLDOUT_FOLD}) score distribution")

    valid = ~np.isnan(oof)
    overall_auc = roc_auc_score(y[valid], oof[valid]) if y[valid].sum() > 0 else float("nan")
    overall_yhat = (oof[valid] >= 0.5).astype(int)
    cm = confusion_matrix(y[valid], overall_yhat)

    metrics_rows = ""
    for m in fold_metrics:
        metrics_rows += (
            f"<tr><td>{m['fold']}</td><td>{m['n_train']}</td><td>{m['n_val']}</td>"
            f"<td>{m['n_pos_train']}</td><td>{m['n_pos_val']}</td>"
            f"<td>{m['auc_roc']:.4f}</td><td>{m['auc_pr']:.4f}</td>"
            f"<td>{m['f1']:.4f}</td><td>{m['balanced_acc']:.4f}</td></tr>\n"
        )
    mean_auc = np.nanmean([m["auc_roc"] for m in fold_metrics])
    mean_pr = np.nanmean([m["auc_pr"] for m in fold_metrics])
    mean_f1 = np.nanmean([m["f1"] for m in fold_metrics])
    mean_ba = np.nanmean([m["balanced_acc"] for m in fold_metrics])
    metrics_rows += (
        f"<tr style='font-weight:bold'><td>Mean</td><td></td><td></td><td></td><td></td>"
        f"<td>{mean_auc:.4f}</td><td>{mean_pr:.4f}</td>"
        f"<td>{mean_f1:.4f}</td><td>{mean_ba:.4f}</td></tr>\n"
    )

    n_spr_neg = int((cv_df["source"] == "spr_inactive").sum())
    n_pos = int(cv_df["label"].sum())
    tunable_params = {k: v for k, v in best_params.items()
                      if k not in ("objective", "eval_metric", "verbosity", "n_jobs", "random_state")}

    threshold_html = ""
    if threshold_info:
        ti = threshold_info
        threshold_html = (
            f'<h2>Optimal threshold (F1-maximizing on OOF)</h2>\n'
            f'<div class="summary">\n'
            f'<p><strong>Threshold:</strong> {ti["threshold"]:.3f}<br>\n'
            f'<strong>Precision:</strong> {ti["precision"]:.4f}<br>\n'
            f'<strong>Recall:</strong> {ti["recall"]:.4f}<br>\n'
            f'<strong>F1:</strong> {ti["f1"]:.4f}</p>\n'
            f'</div>\n'
            f'<h3>Confusion matrix at optimal threshold ({ti["threshold"]:.3f})</h3>\n'
            f'<table style="width:auto">\n'
            f'<tr><th></th><th>Pred Neg</th><th>Pred Pos</th></tr>\n'
            f'<tr><th>True Neg</th><td>{ti["cm"][0,0]}</td><td>{ti["cm"][0,1]}</td></tr>\n'
            f'<tr><th>True Pos</th><td>{ti["cm"][1,0]}</td><td>{ti["cm"][1,1]}</td></tr>\n'
            f'</table>\n'
        )

    hr = holdout_results
    holdout_html = f"""
<h2>Holdout evaluation (fold {HOLDOUT_FOLD})</h2>
<div class="summary">
<p><strong>N compounds:</strong> {hr['n']}<br>
<strong>N positives:</strong> {hr['n_pos']} ({hr['prevalence']:.1%} prevalence)<br>
<strong>AUROC:</strong> {hr['auroc']:.4f}<br>
<strong>AUPRC:</strong> {hr['auprc']:.4f}<br>
<strong>Brier score:</strong> {hr['brier']:.4f} (prevalence-only baseline: {hr['baseline_brier_prevalence']:.4f})<br>
<strong>F1 @ 0.5:</strong> {hr['f1_at_05']:.4f}<br>
<strong>Balanced accuracy @ 0.5:</strong> {hr['balanced_acc_at_05']:.4f}</p>
<p><em>Note: {hr['n_pos']} positives in holdout — metrics are high-variance.</em></p>
</div>
<h3>Holdout score distribution</h3>
<img src="data:image/png;base64,{holdout_hist_img}" alt="holdout scores">
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TBXT XGBoost Classifier (PaCMAP folds) -- CV Report</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1000px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
  h1 {{ border-bottom: 2px solid #1D0258; padding-bottom: .3rem; }}
  h2 {{ color: #1D0258; margin-top: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
  th {{ background: #f5f5f5; text-align: center; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
  .grid img {{ width: 100%; }}
  img {{ max-width: 100%; }}
  .summary {{ background: #f8f8fc; padding: 1rem; border-radius: 6px; }}
  code {{ background: #eee; padding: 2px 5px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>TBXT XGBoost Binary Classifier (PaCMAP-KMeans6 folds)</h1>
<p>Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>

<h2>Dataset summary</h2>
<div class="summary">
<p><strong>Total CV compounds:</strong> {len(cv_df)}<br>
<strong>Positives (pKD &ge; {ACTIVE_PKD}):</strong> {n_pos}<br>
<strong>Negatives (pKD &le; {INACTIVE_PKD}):</strong> {n_spr_neg} (SPR only)<br>
<strong>Fold strategy:</strong> PaCMAP-KMeans6 (Ray's chemical-space folds)<br>
<strong>CV folds:</strong> {CV_FOLDS}<br>
<strong>Holdout fold:</strong> {HOLDOUT_FOLD} ({len(holdout_df)} compounds, {holdout_df['label'].sum()} pos)<br>
<strong>Features:</strong> {feature_info['total']}
  (MACCS {feature_info['maccs']} + RDKit desc {feature_info['rdkit_desc']})</p>
<p><strong>Sample weights:</strong> negative = 1.0,
  positive = {n_spr_neg / max(n_pos, 1):.2f}
  (balanced to total negative weight)</p>
</div>

<h2>Fold composition</h2>
<img src="data:image/png;base64,{comp_img}" alt="fold composition">

<h2>UMAP of training data</h2>
<img src="data:image/png;base64,{umap_img}" alt="UMAP">

<h2>Hyperparameter optimization (Optuna, {N_OPTUNA_TRIALS} trials)</h2>
<img src="data:image/png;base64,{optuna_img}" alt="optuna history">
<p>Best trial: #{study.best_trial.number}, AUC-PR = {study.best_trial.value:.4f}</p>
<pre>{json.dumps(tunable_params, indent=2)}</pre>

<h2>Cross-validation metrics (best params)</h2>
<table>
<tr><th>Fold</th><th>N train</th><th>N val</th><th>Pos train</th><th>Pos val</th>
    <th>AUC-ROC</th><th>AUC-PR</th><th>F1</th><th>Balanced Acc</th></tr>
{metrics_rows}
</table>

<h2>OOF curves</h2>
<div class="grid">
<img src="data:image/png;base64,{roc_img}" alt="ROC">
<img src="data:image/png;base64,{pr_img}" alt="PR">
</div>

<h2>OOF score distribution</h2>
<img src="data:image/png;base64,{hist_img}" alt="score histogram">

<h2>OOF confusion matrix (threshold 0.5)</h2>
<table style="width:auto">
<tr><th></th><th>Pred Neg</th><th>Pred Pos</th></tr>
<tr><th>True Neg</th><td>{cm[0,0]}</td><td>{cm[0,1]}</td></tr>
<tr><th>True Pos</th><td>{cm[1,0]}</td><td>{cm[1,1]}</td></tr>
</table>
<p>Overall OOF AUC-ROC: <strong>{overall_auc:.4f}</strong></p>

{threshold_html}

{holdout_html}

<h2>Usage</h2>
<pre>
from scripts.build_xgb_classifier_pacmap_folds import TBXTClassifier

clf = TBXTClassifier.load("models/xgb_classifier_pacmap_folds")

# ensemble ({N_CV_FOLDS} CV models) -- mean +/- std
mu, sd = clf.predict_ensemble(["CCO", "c1ccccc1"])

# single final model (trained on all CV+holdout data)
probs = clf.predict_final(["CCO", "c1ccccc1"])

# convenience DataFrame
df = clf.predict(["CCO", "c1ccccc1"], mode="both")
</pre>
</body>
</html>"""

    output_path = Path(output_path)
    output_path.write_text(html)
    print(f"Report saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TBXT XGBoost Binary Classifier (PaCMAP-KMeans6 folds)")
    print(f"SEED = {SEED}")
    print("=" * 60)

    cv_df, holdout_df = load_data()

    print("\n-- Featurization (CV set) --")
    X_cv, feat_info = compute_features(cv_df["mol"].tolist())

    print("\n-- Featurization (holdout set) --")
    X_holdout, _ = compute_features(holdout_df["mol"].tolist())

    weights = compute_weights(cv_df)
    y_cv = cv_df["label"].values
    folds = cv_df["fold"].values
    y_holdout = holdout_df["label"].values

    print(f"\n-- Optuna hyperparameter optimization ({N_OPTUNA_TRIALS} trials) --")
    best_params, study = optimize_hyperparameters(X_cv, y_cv, folds, weights)

    print(f"\n-- {N_CV_FOLDS}-Fold Cross-Validation (best params) --")
    cv_models, oof, fold_metrics = train_cv(X_cv, y_cv, folds, weights, best_params)

    valid = ~np.isnan(oof)
    prec_curve, rec_curve, thresholds = precision_recall_curve(y_cv[valid], oof[valid])
    f1_scores = 2 * prec_curve[:-1] * rec_curve[:-1] / np.maximum(prec_curve[:-1] + rec_curve[:-1], 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    best_prec = float(prec_curve[best_idx])
    best_rec = float(rec_curve[best_idx])
    best_f1 = float(f1_scores[best_idx])
    print(f"\n  Optimal OOF threshold: {best_threshold:.3f}  "
          f"(Precision={best_prec:.4f}, Recall={best_rec:.4f}, F1={best_f1:.4f})")

    print(f"\n-- Holdout evaluation (fold {HOLDOUT_FOLD}) --")
    holdout_pred, holdout_results = evaluate_holdout(cv_models, X_holdout, y_holdout)
    print(f"  AUROC: {holdout_results['auroc']:.4f}")
    print(f"  AUPRC: {holdout_results['auprc']:.4f}")
    print(f"  Brier: {holdout_results['brier']:.4f} "
          f"(baseline: {holdout_results['baseline_brier_prevalence']:.4f})")

    print("\n-- Training final model (all data including holdout, best params) --")
    X_all = np.vstack([X_cv, X_holdout])
    y_all = np.concatenate([y_cv, y_holdout])
    all_df = pd.concat([cv_df, holdout_df], ignore_index=True)
    weights_all = compute_weights(all_df)
    final_model = train_final(X_all, y_all, weights_all, best_params)

    classifier = TBXTClassifier(cv_models, final_model, feat_info)
    classifier.save(OUTPUT_DIR)

    run_info = {
        **best_params,
        "seed": SEED,
        "fold_strategy": "pacmap-kmeans6",
        "cv_folds": CV_FOLDS,
        "holdout_fold": HOLDOUT_FOLD,
        "optimal_threshold": best_threshold,
        "optimal_precision": best_prec,
        "optimal_recall": best_rec,
        "optimal_f1": best_f1,
        "holdout": holdout_results,
    }
    with open(OUTPUT_DIR / "best_params.json", "w") as f:
        json.dump(run_info, f, indent=2)

    cv_out = cv_df.drop(columns=["mol"]).copy()
    cv_out["oof_pred"] = oof
    cv_out.to_csv(OUTPUT_DIR / "cv_predictions.csv", index=False)

    ho_out = holdout_df.drop(columns=["mol"]).copy()
    ho_out["holdout_pred"] = holdout_pred
    ho_out.to_csv(OUTPUT_DIR / "holdout_predictions.csv", index=False)

    print(f"\nModels saved to {OUTPUT_DIR}")

    print("\n-- Generating report --")
    oof_valid = ~np.isnan(oof)
    opt_yhat = (oof[oof_valid] >= best_threshold).astype(int)
    opt_cm = confusion_matrix(y_cv[oof_valid], opt_yhat)
    threshold_info = dict(
        threshold=best_threshold, precision=best_prec,
        recall=best_rec, f1=best_f1, cm=opt_cm,
    )
    generate_report(cv_df, holdout_df, fold_metrics, oof, holdout_pred,
                    holdout_results, feat_info, best_params, study, X_cv,
                    OUTPUT_DIR / "report.html", threshold_info=threshold_info)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    return classifier


if __name__ == "__main__":
    main()
