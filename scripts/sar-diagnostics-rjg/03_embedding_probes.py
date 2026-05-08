"""Diagnostic 03: CheMeleon embedding probes (Ridge + kNN).

Extract 2048-dim CheMeleon graph embeddings (no fine-tuning, just the
pretrained encoder's mean-aggregated output) for every compound, and
probe them with two simple regressors using the same 5-fold CV + fold-4
holdout as the main experiments:

1. **Ridge regression** of pKD on the embedding. If the embedding encodes
   useful chemistry for this target, Ridge should have positive OOF R2.
   If even Ridge fails, the problem is not architectural.
2. **kNN regressor** on the embedding (k=5, cosine distance). If kNN
   cannot beat the mean predictor, pKD is not a smooth function of the
   learned embedding space.

Also runs Ridge on the Morgan FP (2048 bits) baseline for comparison.

Writes:
    data/sar-diagnostics-rjg/chemeleon_embeddings.npy
    data/sar-diagnostics-rjg/probe_predictions.csv
    data/sar-diagnostics-rjg/probe_results.json
    docs/sar-diagnostics-rjg/probe_results.png

Usage:
    uv run python scripts/sar-diagnostics-rjg/03_embedding_probes.py [--accelerator mps]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from chemprop import data as cp_data
from chemprop import featurizers, nn
from loguru import logger
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

from tbxt_hackathon.chemeleon_transfer import build_chemeleon_encoder
from tbxt_hackathon.fingerprints import morgan_ndarray

ROOT = Path(__file__).resolve().parents[2]
FOLDS_CSV = ROOT / "data" / "folds-pacmap-kmeans6" / "fold_assignments.csv"
ARTIFACT_DIR = ROOT / "data" / "sar-diagnostics-rjg"
DOC_FIG_DIR = ROOT / "docs" / "sar-diagnostics-rjg"
TARGET_COL = "pKD_global_mean"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--accelerator",
        default="auto",
        choices=["auto", "cpu", "mps", "gpu"],
    )
    ap.add_argument("--batch-size", type=int, default=64)
    return ap.parse_args()


def _device_from_accelerator(acc: str) -> torch.device:
    if acc == "mps" or (acc == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    if acc == "gpu" or (acc == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def extract_chemeleon_embeddings(
    smiles: list[str],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Return (n, 2048) CheMeleon graph embeddings using the pretrained encoder.

    No fine-tuning; we just forward-pass through BondMessagePassing +
    MeanAggregation. Matches what a "linear probe on the foundation model"
    paper would do.
    """
    mp = build_chemeleon_encoder().to(device)
    agg = nn.MeanAggregation().to(device)
    mp.eval()
    agg.eval()

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    placeholder = np.zeros(len(smiles), dtype=np.float32)
    dps = [
        cp_data.MoleculeDatapoint.from_smi(s, np.array([placeholder[i]], dtype=np.float32))
        for i, s in enumerate(smiles)
    ]
    dset = cp_data.MoleculeDataset(dps, featurizer)
    loader = cp_data.build_dataloader(dset, batch_size=batch_size, shuffle=False)

    out: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            # Note: chemprop's BatchMolGraph.to() mutates in place and returns
            # None, unlike torch.nn.Module.to. Don't reassign.
            batch.bmg.to(device)
            h_v = mp(batch.bmg, None)
            h_g = agg(h_v, batch.bmg.batch)
            out.append(h_g.detach().cpu().numpy())
    emb = np.concatenate(out, axis=0).astype(np.float32)
    logger.info(f"extracted chemeleon embeddings: shape={emb.shape}")
    return emb


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, pred)))


def cv_probe(
    X: np.ndarray,
    y: np.ndarray,
    fold: np.ndarray,
    holdout_mask: np.ndarray,
    make_model,
    n_folds: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Run 5-fold CV on the non-holdout pool + ensemble on holdout.

    Args:
        X: (n, d) feature matrix for all compounds.
        y: (n,) target.
        fold: (n,) fold assignment.
        holdout_mask: (n,) boolean mask marking holdout rows.
        make_model: zero-arg factory returning a fresh estimator.

    Returns:
        oof: (n,) array of OOF predictions; nan for holdout rows.
        hold_mean: (n_holdout,) mean of 5-model predictions on holdout.
        per_fold_metrics: list of per-fold metric dicts.
    """
    cv_folds = sorted(np.unique(fold[~holdout_mask]).tolist())
    assert len(cv_folds) == n_folds, f"expected {n_folds} CV folds, got {cv_folds}"

    oof = np.full(X.shape[0], np.nan)
    hold_idx = np.where(holdout_mask)[0]
    hold_preds_per_model: list[np.ndarray] = []
    per_fold: list[dict] = []

    for test_fold in cv_folds:
        train_mask = (~holdout_mask) & (fold != test_fold)
        test_mask = (~holdout_mask) & (fold == test_fold)
        model = make_model()
        model.fit(X[train_mask], y[train_mask])
        te_pred = model.predict(X[test_mask])
        oof[test_mask] = te_pred
        hold_preds_per_model.append(model.predict(X[hold_idx]))

        te_y = y[test_mask]
        per_fold.append({
            "test_fold": int(test_fold),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "test_rmse": _rmse(te_y, te_pred),
            "test_r2": float(r2_score(te_y, te_pred)),
            "test_spearman": float(spearmanr(te_y, te_pred).statistic),
        })

    hold_mean = np.mean(np.stack(hold_preds_per_model, axis=0), axis=0)
    return oof, hold_mean, per_fold


def _aggregate(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": _rmse(y, pred),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "spearman": float(spearmanr(y, pred).statistic),
    }


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(FOLDS_CSV)
    smiles = df["canonical_smiles"].to_list()
    y = df[TARGET_COL].to_numpy().astype(np.float64)
    fold = df["fold"].to_numpy()
    holdout_mask = df["is_holdout"].to_numpy()
    n = len(smiles)
    logger.info(f"loaded {n} compounds; holdout n = {int(holdout_mask.sum())}")

    # ---- features ----------------------------------------------------------
    emb_path = ARTIFACT_DIR / "chemeleon_embeddings.npy"
    if emb_path.exists():
        emb = np.load(emb_path)
        logger.info(f"loaded cached chemeleon embeddings: {emb.shape}")
    else:
        device = _device_from_accelerator(args.accelerator)
        logger.info(f"extracting chemeleon embeddings on {device}")
        emb = extract_chemeleon_embeddings(smiles, device, args.batch_size)
        np.save(emb_path, emb)
        logger.info(f"wrote {emb_path}")

    morgan_arr, _ = morgan_ndarray(smiles)
    morgan_arr = morgan_arr.astype(np.float32)

    # ---- probes ------------------------------------------------------------
    probes: dict[str, tuple[np.ndarray, Callable]] = {
        "chemeleon_embed_ridge": (emb, lambda: Ridge(alpha=1.0, random_state=0)),
        "chemeleon_embed_knn5":  (emb, lambda: KNeighborsRegressor(n_neighbors=5, metric="cosine")),
        "morgan_fp_ridge":       (morgan_arr, lambda: Ridge(alpha=1.0, random_state=0)),
        "morgan_fp_knn5":        (morgan_arr, lambda: KNeighborsRegressor(n_neighbors=5, metric="jaccard")),
    }

    results: dict[str, dict] = {}
    all_oof: dict[str, np.ndarray] = {}
    all_hold: dict[str, np.ndarray] = {}

    for name, (X, factory) in probes.items():
        logger.info(f"\n--- probe: {name}  (features shape {X.shape}) ---")
        oof, hold_mean, per_fold = cv_probe(X, y, fold, holdout_mask, factory)
        finite = ~np.isnan(oof) & (~holdout_mask)
        oof_metrics = _aggregate(y[finite], oof[finite])
        hold_metrics = _aggregate(y[holdout_mask], hold_mean)
        logger.info(
            f"{name} OOF: RMSE={oof_metrics['rmse']:.3f} R2={oof_metrics['r2']:+.3f} "
            f"rho={oof_metrics['spearman']:+.3f}"
        )
        logger.info(
            f"{name} HOL: RMSE={hold_metrics['rmse']:.3f} R2={hold_metrics['r2']:+.3f} "
            f"rho={hold_metrics['spearman']:+.3f}"
        )
        results[name] = {
            "oof": oof_metrics,
            "holdout": hold_metrics,
            "per_fold": per_fold,
        }
        all_oof[name] = oof
        all_hold[name] = hold_mean

    # Baseline: predict train-pool mean on everything
    train_mean = float(y[~holdout_mask].mean())
    baseline_hold = np.full(int(holdout_mask.sum()), train_mean)
    baseline_oof = np.full(int((~holdout_mask).sum()), train_mean)
    baseline_oof_metrics = _aggregate(y[~holdout_mask], baseline_oof)
    baseline_hold_metrics = _aggregate(y[holdout_mask], baseline_hold)
    results["baseline_train_mean"] = {
        "oof": baseline_oof_metrics,
        "holdout": baseline_hold_metrics,
        "train_mean_pKD": train_mean,
    }
    logger.info(f"\nbaseline train-mean pKD = {train_mean:.3f}")
    logger.info(
        f"baseline OOF RMSE={baseline_oof_metrics['rmse']:.3f} | "
        f"HOL RMSE={baseline_hold_metrics['rmse']:.3f}"
    )

    # ---- output table -------------------------------------------------------
    # Wide per-compound predictions for inspection
    oof_full = {
        name: np.where(holdout_mask, np.nan, arr) for name, arr in all_oof.items()
    }
    hold_full = {
        name: np.full(n, np.nan) for name in all_hold
    }
    for name, arr in all_hold.items():
        hold_full[name][holdout_mask] = arr

    preds_frame = df.select([
        "compound_id", "canonical_smiles", "fold", "is_holdout", "is_binder",
        pl.col(TARGET_COL),
    ]).with_columns(
        [pl.Series(f"oof_{k}", v) for k, v in oof_full.items()]
        + [pl.Series(f"hold_{k}", v) for k, v in hold_full.items()]
    )
    preds_frame.write_csv(ARTIFACT_DIR / "probe_predictions.csv")

    (ARTIFACT_DIR / "probe_results.json").write_text(json.dumps(results, indent=2))
    logger.info(f"wrote probe_results.json")

    # ---- summary bar chart -------------------------------------------------
    probe_names = list(probes.keys())
    oof_rmses = [results[n]["oof"]["rmse"] for n in probe_names]
    hold_rmses = [results[n]["holdout"]["rmse"] for n in probe_names]
    base_oof = baseline_oof_metrics["rmse"]
    base_hold = baseline_hold_metrics["rmse"]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    xx = np.arange(len(probe_names))
    w = 0.38
    ax.bar(xx - w/2, oof_rmses, w, label="OOF RMSE", color="#1565c0")
    ax.bar(xx + w/2, hold_rmses, w, label="Holdout RMSE", color="#ef6c00")
    ax.axhline(base_oof, color="#1565c0", linestyle=":", linewidth=1,
               label=f"baseline OOF = {base_oof:.3f}")
    ax.axhline(base_hold, color="#ef6c00", linestyle=":", linewidth=1,
               label=f"baseline HOL = {base_hold:.3f}")
    ax.set_xticks(xx)
    ax.set_xticklabels(probe_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("RMSE on pKD")
    ax.set_title(
        "Linear / kNN probes on CheMeleon embedding vs Morgan FP\n"
        "(lower = better; dotted lines = train-mean baseline)"
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(DOC_FIG_DIR / "probe_results.png", dpi=150)
    plt.close(fig)
    logger.info(f"wrote {DOC_FIG_DIR / 'probe_results.png'}")


if __name__ == "__main__":
    main()
