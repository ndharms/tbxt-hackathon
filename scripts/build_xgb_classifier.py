#!/usr/bin/env python3
"""
TBXT XGBoost binary classifier with Butina-clustered 5-fold CV.

Features: MACCS keys (167) + RDKit 2D descriptors (217)
Labels:  pKD >= 4.5 active, pKD <= 3.5 inactive, ambiguous zone excluded
Negatives: SPR inactives only (no Enamine decoys)
Weights: positives upweighted to balance total positive/negative weight
HPO: Optuna 50-trial search over XGBoost parameter space, optimizing AUC-PR
"""

import base64
import json
import sys
import time
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.ML.Cluster import Butina
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm
import umap

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── paths ────────────────────────────────────────────────────────────────────
SPR_PATH = Path(__file__).resolve().parent.parent / "data" / "zenodo" / "tbxt_spr_merged.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "xgb_classifier"

# ── thresholds & params ──────────────────────────────────────────────────────
ACTIVE_PKD = 4.5
INACTIVE_PKD = 3.5
BUTINA_DIST = 0.4
N_FOLDS = 5
SEED = int(np.random.default_rng().integers(0, 2**31))
N_OPTUNA_TRIALS = 50

# ── descriptor calculator (module-level so it's reusable) ────────────────────
_DESC_NAMES = [d[0] for d in Descriptors.descList]
_DESC_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(_DESC_NAMES)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
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

    mols = []
    for s in tqdm(df["smiles"], desc="Parsing SMILES"):
        mols.append(Chem.MolFromSmiles(str(s)))
    df["mol"] = mols
    valid = df["mol"].notna()
    n_bad = (~valid).sum()
    if n_bad:
        print(f"Dropped {n_bad} unparseable SMILES")
    df = df[valid].reset_index(drop=True)

    assert df.shape[0] > 0, "No valid molecules after parsing"
    print(f"Final dataset: {len(df)} compounds "
          f"({df['label'].sum()} pos / {(df['label'] == 0).sum()} neg)")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  2. CLUSTERING & FOLD ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

def butina_cluster(mols: list, threshold: float = BUTINA_DIST):
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols]
    n = len(fps)
    print(f"Computing {n * (n - 1) // 2:,} pairwise Tanimoto distances ...")
    dists = []
    for i in tqdm(range(1, n), desc="Tanimoto"):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])

    clusters = Butina.ClusterData(dists, n, distThresh=threshold, isDistData=True)
    cluster_ids = np.zeros(n, dtype=int)
    for cid, members in enumerate(clusters):
        for m in members:
            cluster_ids[m] = cid
    print(f"Butina (d={threshold}): {len(clusters)} clusters from {n} compounds")
    return cluster_ids


def assign_folds(cluster_ids: np.ndarray, n_folds: int = N_FOLDS) -> np.ndarray:
    unique, counts = np.unique(cluster_ids, return_counts=True)
    order = np.argsort(-counts)

    fold_sizes = np.zeros(n_folds, dtype=int)
    c2f = {}
    for idx in order:
        fold = int(np.argmin(fold_sizes))
        c2f[unique[idx]] = fold
        fold_sizes[fold] += counts[idx]

    folds = np.array([c2f[c] for c in cluster_ids])
    return folds


# ═══════════════════════════════════════════════════════════════════════════════
#  3. FEATURIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_features(mols: list) -> tuple[sp.csr_matrix, dict]:
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
#  4. SAMPLE WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_weights(df: pd.DataFrame) -> np.ndarray:
    w = np.ones(len(df), dtype=np.float64)

    pos = df["label"] == 1
    n_neg = (~pos).sum()
    n_pos = pos.sum()
    pos_w = n_neg / n_pos
    w[pos] = pos_w

    print(f"Weights -- negative: 1.0, positive: {pos_w:.2f} "
          f"(total neg weight {n_neg} = total pos weight {n_pos}*{pos_w:.2f} = {n_pos * pos_w:.0f})")
    return w


# ═══════════════════════════════════════════════════════════════════════════════
#  5. HYPERPARAMETER OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def _cv_score(params: dict, X, y, folds, weights) -> float:
    aucs = []
    for fold in range(N_FOLDS):
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


def optimize_hyperparameters(X, y, folds, weights) -> dict:
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

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
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
#  6. TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_cv(X, y, folds, weights, params):
    models, fold_metrics = [], []
    oof = np.full(len(y), np.nan)

    for fold in range(N_FOLDS):
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


# ═══════════════════════════════════════════════════════════════════════════════
#  7. PREDICTION API
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
        """Return (mean_prob, std_prob) from the 5 CV models."""
        X = self._featurize(smiles_list)
        preds = np.array([m.predict_proba(X)[:, 1] for m in self.cv_models])
        return preds.mean(axis=0), preds.std(axis=0)

    def predict_final(self, smiles_list: list[str]):
        """Return prob from the single final model."""
        X = self._featurize(smiles_list)
        return self.final_model.predict_proba(X)[:, 1]

    def predict(self, smiles_list: list[str], mode: str = "both") -> pd.DataFrame:
        """Convenience: return a DataFrame with ensemble and/or final predictions."""
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
        for i in range(N_FOLDS):
            m = xgb.XGBClassifier()
            m.load_model(str(path / f"cv_model_{i}.json"))
            cv.append(m)
        final = xgb.XGBClassifier()
        final.load_model(str(path / "final_model.json"))
        with open(path / "feature_info.json") as f:
            info = json.load(f)
        return cls(cv, final, info)


# ═══════════════════════════════════════════════════════════════════════════════
#  8. HTML REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _plot_roc(y, oof, folds) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    for fold in range(N_FOLDS):
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


def _plot_pr(y, oof, folds) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    for fold in range(N_FOLDS):
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


def _plot_hist(y, oof) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    valid = ~np.isnan(oof)
    ax.hist(oof[valid & (y == 0)], bins=50, alpha=0.6, label="Negative", color="#4878d0")
    ax.hist(oof[valid & (y == 1)], bins=50, alpha=0.6, label="Positive", color="#ee854a")
    ax.set(xlabel="Predicted probability", ylabel="Count",
           title="OOF score distribution")
    ax.legend()
    return _fig_to_b64(fig)


def _plot_fold_composition(df) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    sources = ["spr_active", "spr_inactive"]
    colors = ["#ee854a", "#4878d0"]
    bottom = np.zeros(N_FOLDS)
    for src, col in zip(sources, colors):
        vals = [len(df[(df["fold"] == f) & (df["source"] == src)]) for f in range(N_FOLDS)]
        ax.bar(range(N_FOLDS), vals, bottom=bottom, label=src, color=col)
        bottom += vals
    ax.set(xlabel="Fold", ylabel="Compounds", title="Fold composition by source")
    ax.set_xticks(range(N_FOLDS))
    ax.legend(fontsize=8)
    return _fig_to_b64(fig)


def _plot_cluster_sizes(cluster_ids) -> str:
    _, counts = np.unique(cluster_ids, return_counts=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(counts, bins=min(50, len(set(counts))), color="#4878d0", edgecolor="white")
    ax.set(xlabel="Cluster size", ylabel="Number of clusters",
           title=f"Butina cluster size distribution (n={len(set(cluster_ids))})")
    return _fig_to_b64(fig)


def _plot_umap(X, df) -> str:
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="jaccard", random_state=SEED)
    emb = reducer.fit_transform(X)

    fold_colors = ["#4878d0", "#ee854a", "#6acc64", "#d65f5f", "#956cb4"]
    markers = {"spr_active": "^", "spr_inactive": "o"}
    labels_seen = set()

    fig, ax = plt.subplots(figsize=(8, 7))
    for source, marker in markers.items():
        for fold in range(N_FOLDS):
            mask = (df["source"].values == source) & (df["fold"].values == fold)
            if mask.sum() == 0:
                continue
            src_label = source.replace("_", " ")
            fold_label = f"fold {fold}"
            label = f"{src_label}, {fold_label}"
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=fold_colors[fold], marker=marker, s=18, alpha=0.5,
                       edgecolors="none", label=label if label not in labels_seen else None)
            labels_seen.add(label)

    legend_elements = []
    from matplotlib.lines import Line2D
    for fold in range(N_FOLDS):
        legend_elements.append(Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=fold_colors[fold], markersize=8,
                               label=f"Fold {fold}"))
    for source, marker in markers.items():
        legend_elements.append(Line2D([0], [0], marker=marker, color="w",
                               markerfacecolor="gray", markersize=8,
                               label=source.replace("_", " ")))
    ax.legend(handles=legend_elements, fontsize=8, loc="best", ncol=2)
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2", title="UMAP of training features (color=fold, shape=source)")
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
    ax1.set(xlabel="Trial", ylabel="AUC-ROC", title="Optuna optimization history")
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


def generate_report(df, fold_metrics, oof, feature_info, cluster_ids,
                    best_params, study, X, output_path, threshold_info=None):
    y = df["label"].values
    folds = df["fold"].values

    roc_img = _plot_roc(y, oof, folds)
    pr_img = _plot_pr(y, oof, folds)
    hist_img = _plot_hist(y, oof)
    comp_img = _plot_fold_composition(df)
    clust_img = _plot_cluster_sizes(cluster_ids)
    optuna_img = _plot_optuna_history(study)
    print("  Computing UMAP embedding ...")
    umap_img = _plot_umap(X, df)

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

    n_spr_neg = int((df["source"] == "spr_inactive").sum())
    n_pos = int(df["label"].sum())

    tunable_params = {k: v for k, v in best_params.items()
                      if k not in ("objective", "eval_metric", "verbosity", "n_jobs", "random_state")}

    if threshold_info:
        ti = threshold_info
        threshold_html = (
            f'<h2>Optimal threshold (F1-maximizing)</h2>\n'
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
    else:
        threshold_html = ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TBXT XGBoost Classifier -- CV Report</title>
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
<h1>TBXT XGBoost Binary Classifier</h1>
<p>Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>

<h2>Dataset summary</h2>
<div class="summary">
<p><strong>Total compounds:</strong> {len(df)}<br>
<strong>Positives (pKD &ge; {ACTIVE_PKD}):</strong> {n_pos}<br>
<strong>Negatives (pKD &le; {INACTIVE_PKD}):</strong> {n_spr_neg} (SPR only, no Enamine decoys)<br>
<strong>Excluded (ambiguous {INACTIVE_PKD} &lt; pKD &lt; {ACTIVE_PKD}):</strong>
  see load_data() output<br>
<strong>Butina clusters:</strong> {len(set(cluster_ids))} (threshold {BUTINA_DIST})<br>
<strong>Features:</strong> {feature_info['total']}
  (MACCS {feature_info['maccs']} + RDKit desc {feature_info['rdkit_desc']})</p>
<p><strong>Sample weights:</strong> negative = 1.0,
  positive = {n_spr_neg / max(n_pos, 1):.2f}
  (balanced to total negative weight)</p>
</div>

<h2>Clustering</h2>
<img src="data:image/png;base64,{clust_img}" alt="cluster sizes">

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
<h2>Usage</h2>
<pre>
from scripts.build_xgb_classifier import TBXTClassifier

clf = TBXTClassifier.load("models/xgb_classifier")

# ensemble (5 CV models) -- mean +/- std
mu, sd = clf.predict_ensemble(["CCO", "c1ccccc1"])

# single final model
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
    print("TBXT XGBoost Binary Classifier")
    print(f"SEED = {SEED}")
    print("=" * 60)

    # 1. Data
    df = load_data()

    # 2. Butina clustering
    print("\n-- Butina clustering --")
    cluster_ids = butina_cluster(df["mol"].tolist(), BUTINA_DIST)
    df["cluster"] = cluster_ids

    # 3. Fold assignment
    folds = assign_folds(cluster_ids)
    df["fold"] = folds
    for f in range(N_FOLDS):
        mask = df["fold"] == f
        print(f"  Fold {f}: {mask.sum()} compounds, {df.loc[mask, 'label'].sum()} pos")

    # 4. Features
    print("\n-- Featurization --")
    X, feat_info = compute_features(df["mol"].tolist())

    # 5. Weights
    weights = compute_weights(df)

    y = df["label"].values

    # 6. Optuna HPO
    print(f"\n-- Optuna hyperparameter optimization ({N_OPTUNA_TRIALS} trials) --")
    best_params, study = optimize_hyperparameters(X, y, folds, weights)

    # 7. CV with best params
    print("\n-- 5-Fold Cross-Validation (best params) --")
    cv_models, oof, fold_metrics = train_cv(X, y, folds, weights, best_params)

    valid = ~np.isnan(oof)
    prec_curve, rec_curve, thresholds = precision_recall_curve(y[valid], oof[valid])
    f1_scores = 2 * prec_curve[:-1] * rec_curve[:-1] / np.maximum(prec_curve[:-1] + rec_curve[:-1], 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    best_prec = float(prec_curve[best_idx])
    best_rec = float(rec_curve[best_idx])
    best_f1 = float(f1_scores[best_idx])
    print(f"\n  Optimal threshold: {best_threshold:.3f}  "
          f"(Precision={best_prec:.4f}, Recall={best_rec:.4f}, F1={best_f1:.4f})")

    # 8. Final model
    print("\n-- Training final model (all data, best params) --")
    final_model = train_final(X, y, weights, best_params)

    # 9. Save
    classifier = TBXTClassifier(cv_models, final_model, feat_info)
    classifier.save(OUTPUT_DIR)
    run_info = {
        **best_params,
        "seed": SEED,
        "optimal_threshold": best_threshold,
        "optimal_precision": best_prec,
        "optimal_recall": best_rec,
        "optimal_f1": best_f1,
    }
    with open(OUTPUT_DIR / "best_params.json", "w") as f:
        json.dump(run_info, f, indent=2)
    df.drop(columns=["mol"]).to_csv(OUTPUT_DIR / "fold_assignments.csv", index=False)
    print(f"\nModels saved to {OUTPUT_DIR}")

    # 10. Report
    print("\n-- Generating report --")
    oof_valid = ~np.isnan(oof)
    opt_yhat = (oof[oof_valid] >= best_threshold).astype(int)
    opt_cm = confusion_matrix(y[oof_valid], opt_yhat)
    threshold_info = dict(
        threshold=best_threshold, precision=best_prec,
        recall=best_rec, f1=best_f1, cm=opt_cm,
    )
    generate_report(df, fold_metrics, oof, feat_info, cluster_ids,
                    best_params, study, X, OUTPUT_DIR / "report.html",
                    threshold_info=threshold_info)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    return classifier


if __name__ == "__main__":
    main()
