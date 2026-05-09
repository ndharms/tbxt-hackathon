#!/usr/bin/env python3
"""
TBXT pairwise comparison XGBoost classifier.

Generates all within-fold molecule pairs, labels by pKD difference >= 1,
and trains a binary classifier on difference fingerprints [f(A) - f(B)].
Evaluation: pair predictions -> per-molecule ranking scores -> correlation with true pKD.
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
import umap
import xgboost as xgb
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.ML.Cluster import Butina
from rdkit.ML.Descriptors import MoleculeDescriptors
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── paths ────────────────────────────────────────────────────────────────────
SPR_PATH = Path(__file__).resolve().parent.parent / "data" / "zenodo" / "tbxt_spr_merged.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "pairwise_classifier"

# ── params ───────────────────────────────────────────────────────────────────
PKD_DIFF_THRESHOLD = 1.0
BUTINA_DIST = 0.4
N_FOLDS = 5
SEED = int(np.random.default_rng().integers(0, 2**31))
GOBBI_HASH_BITS = 256
N_OPTUNA_TRIALS = 32

# ── descriptor calculator ────────────────────────────────────────────────────
_DESC_NAMES = [d[0] for d in Descriptors.descList]
_DESC_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(_DESC_NAMES)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    spr = pd.read_csv(SPR_PATH)
    assert spr.shape[0] > 0

    df = (
        spr.groupby("compound_id")
        .agg(smiles=("smiles", "first"), pKD=("pKD", "median"))
        .reset_index()
    )
    df = df[df["smiles"].notna() & (df["smiles"] != "") & df["pKD"].notna()].reset_index(drop=True)

    mols = []
    for s in tqdm(df["smiles"], desc="Parsing SMILES"):
        mols.append(Chem.MolFromSmiles(str(s)))
    df["mol"] = mols
    valid = df["mol"].notna()
    n_bad = (~valid).sum()
    if n_bad:
        print(f"Dropped {n_bad} unparseable SMILES")
    df = df[valid].reset_index(drop=True)

    print(f"SPR dataset: {len(df)} unique compounds, "
          f"pKD range [{df['pKD'].min():.2f}, {df['pKD'].max():.2f}], "
          f"median {df['pKD'].median():.2f}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  2. CLUSTERING & FOLD ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

def butina_cluster(mols, threshold=BUTINA_DIST):
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


def assign_folds(cluster_ids, n_folds=N_FOLDS):
    unique, counts = np.unique(cluster_ids, return_counts=True)
    order = np.argsort(-counts)
    fold_sizes = np.zeros(n_folds, dtype=int)
    c2f = {}
    for idx in order:
        fold = int(np.argmin(fold_sizes))
        c2f[unique[idx]] = fold
        fold_sizes[fold] += counts[idx]
    return np.array([c2f[c] for c in cluster_ids])


# ═══════════════════════════════════════════════════════════════════════════════
#  3. FEATURIZATION (per-molecule)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mol_features(mols):
    n = len(mols)

    maccs = np.zeros((n, 167), dtype=np.float32)
    for i, mol in enumerate(tqdm(mols, desc="MACCS")):
        fp = MACCSkeys.GenMACCSKeys(mol)
        DataStructs.ConvertToNumpyArray(fp, maccs[i])

    descs = np.zeros((n, len(_DESC_NAMES)), dtype=np.float64)
    for i, mol in enumerate(tqdm(mols, desc="RDKit descriptors")):
        descs[i] = _DESC_CALC.CalcDescriptors(mol)
    descs = np.nan_to_num(descs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    gobbi = np.zeros((n, GOBBI_HASH_BITS), dtype=np.float32)
    for i, mol in enumerate(tqdm(mols, desc="Gobbi 2D")):
        fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
        for bit in fp.GetOnBits():
            gobbi[i, bit % GOBBI_HASH_BITS] = 1

    X = np.hstack([maccs, descs, gobbi])
    feat_dim = X.shape[1]
    print(f"Per-molecule features: {n} x {feat_dim}  "
          f"(MACCS 167 + Desc {len(_DESC_NAMES)} + Gobbi {GOBBI_HASH_BITS})")
    return X


# ═══════════════════════════════════════════════════════════════════════════════
#  4. PAIR GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_pairs_for_fold(fold_indices, pKDs):
    """Generate all ordered pairs within a single fold's molecules.

    Returns (idx_a, idx_b, labels) where indices refer to the original
    molecule array, and label=1 if pKD(A) - pKD(B) >= PKD_DIFF_THRESHOLD.
    """
    fold_indices = np.asarray(fold_indices)
    n = len(fold_indices)
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    mask = ii != jj
    idx_a = fold_indices[ii[mask]]
    idx_b = fold_indices[jj[mask]]
    labels = (pKDs[idx_a] - pKDs[idx_b] >= PKD_DIFF_THRESHOLD).astype(np.int32)
    return idx_a, idx_b, labels


def build_pair_features(X_mol, idx_a, idx_b):
    """[f(A) - f(B)] only"""
    return X_mol[idx_a] - X_mol[idx_b]


def pairs_to_ranking(mol_indices, idx_a, idx_b, preds):
    """Convert pair predictions to per-molecule scores.

    For each molecule, score = mean P(mol > other) across all others.
    """
    scores = {}
    counts = {}
    for m in mol_indices:
        scores[m] = 0.0
        counts[m] = 0
    for a, b, p in zip(idx_a, idx_b, preds):
        scores[a] += p
        counts[a] += 1
    ranking = {m: scores[m] / max(counts[m], 1) for m in mol_indices}
    return ranking


# ═══════════════════════════════════════════════════════════════════════════════
#  5. HYPERPARAMETER OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def _balance_pairs(idx_a, idx_b, labels, rng):
    """Subsample negative pairs to match positive count."""
    pos_mask = labels == 1
    neg_mask = ~pos_mask
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_neg <= n_pos:
        return idx_a, idx_b, labels
    neg_indices = np.where(neg_mask)[0]
    keep_neg = rng.choice(neg_indices, size=n_pos, replace=False)
    keep = np.sort(np.concatenate([np.where(pos_mask)[0], keep_neg]))
    return idx_a[keep], idx_b[keep], labels[keep]


def precompute_cv_data(X_mol, pKDs, folds):
    """Pre-generate all pair features and labels for each CV split."""
    rng = np.random.default_rng(SEED)
    cv_data = []
    for fold in range(N_FOLDS):
        train_folds = [f for f in range(N_FOLDS) if f != fold]
        tr_a, tr_b, tr_y = [], [], []
        for tf in train_folds:
            idxs = np.where(folds == tf)[0]
            a, b, y = generate_pairs_for_fold(idxs, pKDs)
            tr_a.append(a)
            tr_b.append(b)
            tr_y.append(y)
        tr_a = np.concatenate(tr_a)
        tr_b = np.concatenate(tr_b)
        tr_y = np.concatenate(tr_y)

        n_pos_before = int(tr_y.sum())
        n_neg_before = len(tr_y) - n_pos_before
        tr_a, tr_b, tr_y = _balance_pairs(tr_a, tr_b, tr_y, rng)
        n_pos_after = int(tr_y.sum())
        n_neg_after = len(tr_y) - n_pos_after
        print(f"  Fold {fold} train: {n_pos_before + n_neg_before:,} pairs "
              f"-> balanced to {len(tr_y):,} ({n_pos_after:,} pos, {n_neg_after:,} neg)")

        va_idxs = np.where(folds == fold)[0]
        va_a, va_b, va_y = generate_pairs_for_fold(va_idxs, pKDs)

        X_tr = build_pair_features(X_mol, tr_a, tr_b)
        X_va = build_pair_features(X_mol, va_a, va_b)

        cv_data.append(dict(
            X_tr=X_tr, y_tr=tr_y, X_va=X_va, y_va=va_y,
            va_idxs=va_idxs, va_a=va_a, va_b=va_b,
        ))
        print(f"  Fold {fold} test:  {len(va_y):,} pairs ({int(va_y.sum()):,} pos)")
    return cv_data


def _cv_score_pairwise(params, cv_data, trial_num=None):
    """Run pairwise CV using pre-computed data, return mean AUC-PR."""
    aucprs = []
    for fold_i, d in enumerate(cv_data):
        t_start = time.time()
        clf = xgb.XGBClassifier(**params)
        clf.fit(d["X_tr"], d["y_tr"], verbose=False)
        p = clf.predict_proba(d["X_va"])[:, 1]
        if d["y_va"].sum() > 0 and (d["y_va"] == 0).sum() > 0:
            prec, rec, _ = precision_recall_curve(d["y_va"], p)
            aucprs.append(auc(rec, prec))
        elapsed = time.time() - t_start
        prefix = f"  [Trial {trial_num}]" if trial_num is not None else "  "
        print(f"{prefix} Fold {fold_i}: AUC-PR={aucprs[-1]:.4f}  ({elapsed:.1f}s)",
              flush=True)
    mean_val = float(np.mean(aucprs)) if aucprs else 0.0
    return mean_val


def optimize_hyperparameters(cv_data):
    def objective(trial):
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
        t_trial = time.time()
        score = _cv_score_pairwise(params, cv_data, trial_num=trial.number)
        print(f"  Trial {trial.number}/{N_OPTUNA_TRIALS}: AUC-PR = {score:.4f}  "
              f"({time.time() - t_trial:.1f}s total)", flush=True)
        return score

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

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
#  6. TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_cv(cv_data, pKDs, folds, params):
    models = []
    fold_metrics = []
    all_rankings = {}

    for fold in range(N_FOLDS):
        d = cv_data[fold]
        n_pos_tr = int(d["y_tr"].sum())
        n_pos_va = int(d["y_va"].sum())
        print(f"\n  Fold {fold}:")
        print(f"    Train: {len(d['y_tr']):,} pairs ({n_pos_tr:,} pos)")
        print(f"    Test:  {len(d['y_va']):,} pairs ({n_pos_va:,} pos)")

        clf = xgb.XGBClassifier(**params)
        clf.fit(d["X_tr"], d["y_tr"],
                eval_set=[(d["X_va"], d["y_va"])], verbose=False)
        models.append(clf)

        p = clf.predict_proba(d["X_va"])[:, 1]

        auc_roc = roc_auc_score(d["y_va"], p) if n_pos_va > 0 and (d["y_va"] == 0).sum() > 0 else float("nan")
        prec_curve, rec_curve, _ = precision_recall_curve(d["y_va"], p)
        auc_pr = auc(rec_curve, prec_curve)

        y_pred = (p >= 0.5).astype(int)
        pair_precision = precision_score(d["y_va"], y_pred, zero_division=0.0)
        pair_recall = recall_score(d["y_va"], y_pred, zero_division=0.0)
        pair_f1 = f1_score(d["y_va"], y_pred, zero_division=0.0)

        ranking = pairs_to_ranking(d["va_idxs"], d["va_a"], d["va_b"], p)
        pred_scores = np.array([ranking[m] for m in d["va_idxs"]])
        true_pkds = pKDs[d["va_idxs"]]

        r_pearson, _ = pearsonr(pred_scores, true_pkds)
        tau, _ = kendalltau(pred_scores, true_pkds)
        rho, _ = spearmanr(pred_scores, true_pkds)

        m = dict(fold=fold, n_mols=len(d["va_idxs"]),
                 n_pairs_train=len(d["y_tr"]), n_pos_train=n_pos_tr,
                 n_pairs_test=len(d["y_va"]), n_pos_test=n_pos_va,
                 auc_roc=auc_roc, auc_pr=auc_pr,
                 pair_precision=pair_precision, pair_recall=pair_recall, pair_f1=pair_f1,
                 pearson_r=r_pearson, kendall_tau=tau, spearman_rho=rho)
        fold_metrics.append(m)

        for idx in d["va_idxs"]:
            all_rankings[idx] = ranking[idx]

        print(f"    Pair metrics: AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  "
              f"Prec={pair_precision:.4f}  Rec={pair_recall:.4f}  F1={pair_f1:.4f}")
        print(f"    Rank metrics: Pearson={r_pearson:.4f}  "
              f"Kendall={tau:.4f}  Spearman={rho:.4f}")

    return models, fold_metrics, all_rankings


def train_final(X_mol, pKDs, folds, params):
    """Train final model on all within-fold pairs (balanced)."""
    rng = np.random.default_rng(SEED)
    all_a, all_b, all_y = [], [], []
    for fold in range(N_FOLDS):
        idxs = np.where(folds == fold)[0]
        a, b, y = generate_pairs_for_fold(idxs, pKDs)
        a, b, y = _balance_pairs(a, b, y, rng)
        all_a.append(a)
        all_b.append(b)
        all_y.append(y)
    all_a = np.concatenate(all_a)
    all_b = np.concatenate(all_b)
    all_y = np.concatenate(all_y)

    X = build_pair_features(X_mol, all_a, all_b)
    print(f"Final model: {len(all_y):,} pairs "
          f"({int(all_y.sum()):,} pos / {int((all_y == 0).sum()):,} neg)")

    clf = xgb.XGBClassifier(**params)
    clf.fit(X, all_y, verbose=False)
    return clf


# ═══════════════════════════════════════════════════════════════════════════════
#  7. HTML REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _plot_ranking_scatter(pKDs, all_rankings, folds):
    fig, axes = plt.subplots(1, N_FOLDS, figsize=(4 * N_FOLDS, 4), sharey=True)
    fold_colors = ["#4878d0", "#ee854a", "#6acc64", "#d65f5f", "#956cb4"]
    for fold in range(N_FOLDS):
        ax = axes[fold]
        idxs = np.where(folds == fold)[0]
        true = pKDs[idxs]
        pred = np.array([all_rankings.get(i, 0) for i in idxs])
        ax.scatter(true, pred, s=12, alpha=0.5, color=fold_colors[fold])
        r, _ = pearsonr(true, pred)
        ax.set_title(f"Fold {fold} (r={r:.3f})", fontsize=10)
        ax.set_xlabel("True pKD")
        if fold == 0:
            ax.set_ylabel("Predicted pair score")
    fig.suptitle("Predicted ranking score vs true pKD", fontsize=12, y=1.02)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_umap(X_mol, df):
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="jaccard",
                        random_state=SEED)
    emb = reducer.fit_transform(X_mol)

    fig, ax = plt.subplots(figsize=(7, 6))
    fold_colors = ["#4878d0", "#ee854a", "#6acc64", "#d65f5f", "#956cb4"]
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=[fold_colors[f] for f in df["fold"]],
                    s=10, alpha=0.5)

    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], marker="o", color="w",
              markerfacecolor=fold_colors[f], markersize=8,
              label=f"Fold {f}") for f in range(N_FOLDS)]
    ax.legend(handles=legend, fontsize=8)
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2",
           title="UMAP of per-molecule features (color = fold)")
    ax.set_xticks([])
    ax.set_yticks([])
    return _fig_to_b64(fig)


def _plot_pair_label_dist(pKDs, folds):
    fig, ax = plt.subplots(figsize=(7, 4))
    fold_colors = ["#4878d0", "#ee854a", "#6acc64", "#d65f5f", "#956cb4"]
    for fold in range(N_FOLDS):
        idxs = np.where(folds == fold)[0]
        pkds_fold = pKDs[idxs]
        diffs = (pkds_fold[:, None] - pkds_fold[None, :]).ravel()
        mask = np.ones(len(pkds_fold) ** 2, dtype=bool)
        mask[:: len(pkds_fold) + 1] = False
        diffs = diffs[mask]
        ax.hist(diffs, bins=80, alpha=0.4, color=fold_colors[fold],
                label=f"Fold {fold}", density=True)
    ax.axvline(PKD_DIFF_THRESHOLD, color="red", ls="--", lw=1.5,
               label=f"threshold (+{PKD_DIFF_THRESHOLD})")
    ax.axvline(-PKD_DIFF_THRESHOLD, color="red", ls="--", lw=1.5,
               label=f"threshold (-{PKD_DIFF_THRESHOLD})")
    ax.set(xlabel="pKD(A) - pKD(B)", ylabel="Density",
           title="Distribution of pKD differences across pairs")
    ax.legend(fontsize=7)
    return _fig_to_b64(fig)


def _plot_optuna_history(study):
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


def generate_report(df, fold_metrics, all_rankings, X_mol, best_params,
                    study, cluster_ids, output_path):
    pKDs = df["pKD"].values
    folds = df["fold"].values

    scatter_img = _plot_ranking_scatter(pKDs, all_rankings, folds)
    print("  Computing UMAP ...")
    umap_img = _plot_umap(X_mol, df)
    pair_dist_img = _plot_pair_label_dist(pKDs, folds)
    optuna_img = _plot_optuna_history(study)

    metrics_rows = ""
    for m in fold_metrics:
        metrics_rows += (
            f"<tr><td>{m['fold']}</td><td>{m['n_mols']}</td>"
            f"<td>{m['n_pairs_train']:,}</td><td>{m['n_pos_train']:,}</td>"
            f"<td>{m['n_pairs_test']:,}</td><td>{m['n_pos_test']:,}</td>"
            f"<td>{m['auc_roc']:.4f}</td><td>{m['auc_pr']:.4f}</td>"
            f"<td>{m['pair_precision']:.4f}</td><td>{m['pair_recall']:.4f}</td>"
            f"<td>{m['pair_f1']:.4f}</td>"
            f"<td>{m['pearson_r']:.4f}</td><td>{m['kendall_tau']:.4f}</td>"
            f"<td>{m['spearman_rho']:.4f}</td></tr>\n"
        )
    mean_keys = ["auc_roc", "auc_pr", "pair_precision", "pair_recall", "pair_f1",
                 "pearson_r", "kendall_tau", "spearman_rho"]
    means = {k: np.nanmean([m[k] for m in fold_metrics]) for k in mean_keys}
    metrics_rows += (
        f"<tr style='font-weight:bold'><td>Mean</td><td></td><td></td><td></td>"
        f"<td></td><td></td>"
        f"<td>{means['auc_roc']:.4f}</td><td>{means['auc_pr']:.4f}</td>"
        f"<td>{means['pair_precision']:.4f}</td><td>{means['pair_recall']:.4f}</td>"
        f"<td>{means['pair_f1']:.4f}</td>"
        f"<td>{means['pearson_r']:.4f}</td><td>{means['kendall_tau']:.4f}</td>"
        f"<td>{means['spearman_rho']:.4f}</td></tr>\n"
    )

    tunable = {k: v for k, v in best_params.items()
               if k not in ("objective", "eval_metric", "verbosity", "n_jobs", "random_state")}
    feat_dim = X_mol.shape[1]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TBXT Pairwise Classifier -- CV Report</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
  h1 {{ border-bottom: 2px solid #1D0258; padding-bottom: .3rem; }}
  h2 {{ color: #1D0258; margin-top: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
  th {{ background: #f5f5f5; text-align: center; }}
  img {{ max-width: 100%; }}
  .summary {{ background: #f8f8fc; padding: 1rem; border-radius: 6px; }}
</style>
</head>
<body>
<h1>TBXT Pairwise Comparison Classifier</h1>
<p>Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>

<h2>Approach</h2>
<div class="summary">
<p>For each pair (A, B) of molecules within a fold, a binary classifier predicts whether
molecule A is more potent than molecule B by &ge; {PKD_DIFF_THRESHOLD} pKD unit.
Features: <code>[f(A) - f(B)]</code> where f() = MACCS + RDKit descriptors + hashed Gobbi
({feat_dim} features per pair).</p>
<p>Pairs are generated within each fold separately (no cross-fold pairs), preserving
the Butina scaffold split. Training uses pairs from 4 folds; evaluation uses pairs
from the held-out fold. Pair predictions are aggregated into per-molecule ranking scores,
then compared to true pKD via correlation metrics.</p>
</div>

<h2>Dataset</h2>
<div class="summary">
<p><strong>Compounds:</strong> {len(df)} (all SPR, median-aggregated pKD)<br>
<strong>pKD range:</strong> [{df['pKD'].min():.2f}, {df['pKD'].max():.2f}], median {df['pKD'].median():.2f}<br>
<strong>Butina clusters:</strong> {len(set(cluster_ids))} (threshold {BUTINA_DIST})<br>
<strong>Positive pair criterion:</strong> pKD(A) - pKD(B) &ge; {PKD_DIFF_THRESHOLD}</p>
</div>

<h2>pKD difference distribution</h2>
<img src="data:image/png;base64,{pair_dist_img}" alt="pair diffs">

<h2>UMAP of per-molecule features</h2>
<img src="data:image/png;base64,{umap_img}" alt="UMAP">

<h2>Hyperparameter optimization (Optuna, {N_OPTUNA_TRIALS} trials)</h2>
<img src="data:image/png;base64,{optuna_img}" alt="optuna">
<p>Best trial: #{study.best_trial.number}, AUC-PR = {study.best_trial.value:.4f}</p>
<pre>{json.dumps(tunable, indent=2)}</pre>

<h2>Cross-validation metrics</h2>
<table>
<tr><th>Fold</th><th>Mols</th><th>Train pairs</th><th>Train pos</th>
    <th>Test pairs</th><th>Test pos</th>
    <th>AUC-ROC</th><th>AUC-PR</th>
    <th>Precision</th><th>Recall</th><th>F1</th>
    <th>Pearson r</th><th>Kendall &tau;</th><th>Spearman &rho;</th></tr>
{metrics_rows}
</table>

<h2>Predicted ranking vs true pKD</h2>
<img src="data:image/png;base64,{scatter_img}" alt="ranking scatter">

</body>
</html>"""

    Path(output_path).write_text(html)
    print(f"Report saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TBXT Pairwise Comparison Classifier")
    print(f"SEED = {SEED}")
    print("=" * 60)

    df = load_data()

    print("\n-- Butina clustering --")
    cluster_ids = butina_cluster(df["mol"].tolist(), BUTINA_DIST)
    df["cluster"] = cluster_ids
    folds = assign_folds(cluster_ids)
    df["fold"] = folds
    for f in range(N_FOLDS):
        mask = df["fold"] == f
        print(f"  Fold {f}: {mask.sum()} compounds")

    print("\n-- Per-molecule featurization --")
    X_mol = compute_mol_features(df["mol"].tolist())

    pKDs = df["pKD"].values

    print("\n-- Pre-computing CV pair data --")
    cv_data = precompute_cv_data(X_mol, pKDs, folds)

    print(f"\n-- Optuna HPO ({N_OPTUNA_TRIALS} trials) --")
    best_params, study = optimize_hyperparameters(cv_data)

    print("\n-- 5-Fold CV (best params) --")
    cv_models, fold_metrics, all_rankings = run_cv(cv_data, pKDs, folds, best_params)

    print("\n-- Training final model (all data) --")
    final_model = train_final(X_mol, pKDs, folds, best_params)

    # save
    for i, m in enumerate(cv_models):
        m.save_model(str(OUTPUT_DIR / f"cv_model_{i}.json"))
    final_model.save_model(str(OUTPUT_DIR / "final_model.json"))
    run_info = {**best_params, "seed": SEED}
    with open(OUTPUT_DIR / "best_params.json", "w") as f:
        json.dump(run_info, f, indent=2)
    df.drop(columns=["mol"]).to_csv(OUTPUT_DIR / "fold_assignments.csv", index=False)
    print(f"Models saved to {OUTPUT_DIR}")

    print("\n-- Generating report --")
    generate_report(df, fold_metrics, all_rankings, X_mol, best_params,
                    study, cluster_ids, OUTPUT_DIR / "report.html")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
