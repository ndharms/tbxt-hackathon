"""Score the combined screening library, assign pockets, and cluster for Boltz.

Pipeline:
  1. Score all compounds with the deployment ensemble (or XGB-only with --skip-chemeleon)
  2. Filter to top-scoring compounds
  3. Assign each to its best pocket among {A, C, D}
  4. Agglomerative-cluster each site group on ECFP4 Tanimoto distance
  5. Pick one representative per cluster (highest ensemble score)
  6. Write per-site CSVs to --output-dir

Usage:
    uv run python scripts/score_and_select.py \
        --input data/screening_library_combined.csv \
        --top-n 5000 \
        --output-dir data/boltz_candidates

    # XGB-only (before CheMeleon checkpoints are available):
    uv run python scripts/score_and_select.py \
        --input data/screening_library_combined.csv \
        --top-n 5000 \
        --skip-chemeleon \
        --output-dir data/boltz_candidates
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from rdkit import Chem
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering

from tbxt_hackathon.deployment import (
    DeploymentXGBModel,
    EnsembleModel,
    MorganXGBModel,
    rank_norm,
)
from tbxt_hackathon.fingerprints import morgan_ndarray
from tbxt_hackathon.pocket_assigner import PocketAssigner

ALLOWED_POCKETS = ("A", "C", "D")

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = _REPO_ROOT / "data" / "screening_library_combined.csv"
DEFAULT_FRAGMENTS = _REPO_ROOT / "data" / "structures" / "sgc_fragments.csv"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "data" / "boltz_candidates"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--id-col", default="identifier")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    score_g = ap.add_mutually_exclusive_group()
    score_g.add_argument(
        "--top-n", type=int, default=50000,
        help="Keep the top-N compounds by ensemble score (default 50000).",
    )
    score_g.add_argument(
        "--threshold", type=float, default=None,
        help="Keep compounds with ensemble_rank_score >= this value.",
    )

    ap.add_argument("--pocket-threshold", type=float, default=0.25,
                     help="Min best-pocket Tanimoto to keep a compound.")
    ap.add_argument("--site-a-n", type=int, default=1000)
    ap.add_argument("--site-c-n", type=int, default=500)
    ap.add_argument("--site-d-n", type=int, default=500)

    ap.add_argument("--skip-chemeleon", action="store_true",
                     help="Run XGB-only (2-model) scoring without CheMeleon.")
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--fragments-csv", type=Path, default=DEFAULT_FRAGMENTS)
    ap.add_argument("--accelerator", default="auto",
                     choices=["auto", "cpu", "mps", "gpu"])
    return ap.parse_args()


# ── Step 1: Scoring ─────────────────────────────────────────────────────────


def score_library(
    smiles: list[str],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    """Score all SMILES and return a dict of score arrays."""
    if args.skip_chemeleon:
        logger.info("--skip-chemeleon: loading deploy-XGB + Morgan-XGB only")
        deploy = DeploymentXGBModel.load(
            model_dir=args.model_dir, fragments_csv=args.fragments_csv,
        )
        morgan = MorganXGBModel.load(model_dir=args.model_dir)

        p_deploy = deploy.predict(smiles)
        p_morgan = morgan.predict(smiles)

        rn_deploy = rank_norm(p_deploy)
        rn_morgan = rank_norm(p_morgan)
        ensemble = (rn_deploy + rn_morgan) / 2.0

        return {
            "p_deploy": p_deploy,
            "p_morgan": p_morgan,
            "score_deploy_rn": rn_deploy,
            "score_morgan_rn": rn_morgan,
            "ensemble_rank_score": ensemble,
        }

    model = EnsembleModel.load(
        model_dir=args.model_dir,
        fragments_csv=args.fragments_csv,
        accelerator=args.accelerator,
    )
    pred = model.predict(smiles)
    return {
        "p_deploy": pred.p_deploy,
        "p_morgan": pred.p_morgan,
        "p_chemeleon": pred.p_chemeleon,
        "score_deploy_rn": pred.score_deploy_rn,
        "score_morgan_rn": pred.score_morgan_rn,
        "score_chemeleon_rn": pred.score_chemeleon_rn,
        "ensemble_rank_score": pred.ensemble_rank_score,
    }


# ── Step 3: Pocket assignment ───────────────────────────────────────────────


def assign_pockets(
    smiles: list[str],
    fragments_csv: Path,
    pocket_threshold: float,
) -> tuple[list[str | None], list[float], list[float]]:
    """Return (pocket_label, best_tanimoto, best_combined) per compound."""
    assigner = PocketAssigner.from_csv(fragments_csv, threshold=pocket_threshold)
    pockets: list[str | None] = []
    tanimotos: list[float] = []
    combineds: list[float] = []

    for i, smi in enumerate(smiles):
        if i > 0 and i % 5000 == 0:
            logger.info(f"pocket assignment: {i}/{len(smiles)}")
        scores = assigner.score(smi)
        best_pocket = None
        best_tc = 0.0
        best_comb = 0.0
        for p in ALLOWED_POCKETS:
            if p in scores and scores[p].combined > best_comb:
                best_pocket = p
                best_tc = scores[p].tanimoto
                best_comb = scores[p].combined
        if best_tc < pocket_threshold:
            best_pocket = None
        pockets.append(best_pocket)
        tanimotos.append(best_tc)
        combineds.append(best_comb)

    return pockets, tanimotos, combineds


# ── Step 4: Clustering ──────────────────────────────────────────────────────


def tanimoto_distance_matrix(fp_array: np.ndarray) -> np.ndarray:
    """Pairwise Tanimoto distance from a dense bit-fingerprint matrix."""
    fp = fp_array.astype(np.float64)
    dot = fp @ fp.T
    norms = np.sum(fp, axis=1)
    union = norms[:, None] + norms[None, :] - dot
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, dot / union, 0.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


def cluster_and_select(
    df: pl.DataFrame,
    n_clusters: int,
    smiles_col: str = "smiles",
    score_col: str = "ensemble_rank_score",
) -> pl.DataFrame:
    """Agglomerative-cluster on ECFP4 Tanimoto, pick best-scoring per cluster."""
    n = df.shape[0]
    if n <= n_clusters:
        logger.info(f"  site has {n} compounds <= target {n_clusters}; taking all")
        return df.with_columns(pl.lit(list(range(n))).explode().cast(pl.Int32).alias("cluster_id"))

    smiles_list = df[smiles_col].to_list()
    logger.info(f"  computing ECFP4 for {n} compounds")
    fp_arr, _ = morgan_ndarray(smiles_list, n_bits=2048, radius=2)

    logger.info(f"  computing {n}x{n} Tanimoto distance matrix")
    dist = tanimoto_distance_matrix(fp_arr)

    logger.info(f"  agglomerative clustering into {n_clusters} clusters")
    clust = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )
    labels = clust.fit_predict(dist)

    df = df.with_columns(pl.Series("cluster_id", labels, dtype=pl.Int32))

    reps = (
        df.sort(score_col, descending=True)
        .group_by("cluster_id")
        .first()
        .sort("cluster_id")
    )
    logger.info(f"  selected {reps.shape[0]} cluster representatives")
    return reps


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    site_targets = {"A": args.site_a_n, "C": args.site_c_n, "D": args.site_d_n}

    # Load input
    logger.info(f"loading {args.input}")
    df = pl.read_csv(args.input)
    assert args.smiles_col in df.columns, f"missing column {args.smiles_col}"
    n_in = df.shape[0]
    logger.info(f"loaded {n_in} compounds")

    # Validate SMILES
    smiles_all = df[args.smiles_col].to_list()
    valid_mask = np.array(
        [Chem.MolFromSmiles(s) is not None for s in smiles_all], dtype=bool,
    )
    n_invalid = int((~valid_mask).sum())
    if n_invalid:
        logger.warning(f"{n_invalid} invalid SMILES dropped")
    df = df.filter(pl.Series(valid_mask))
    smiles_valid = df[args.smiles_col].to_list()
    logger.info(f"{len(smiles_valid)} valid SMILES")

    # Step 1: Score
    logger.info("step 1: scoring with ensemble")
    scores = score_library(smiles_valid, args)
    for col_name, arr in scores.items():
        df = df.with_columns(pl.Series(col_name, arr))

    # Step 2: Filter to high-scoring
    df = df.sort("ensemble_rank_score", descending=True)
    if args.threshold is not None:
        df = df.filter(pl.col("ensemble_rank_score") >= args.threshold)
        logger.info(f"step 2: {df.shape[0]} compounds above threshold {args.threshold}")
    else:
        df = df.head(args.top_n)
        logger.info(f"step 2: top {df.shape[0]} compounds by ensemble_rank_score")

    # Step 3: Pocket assignment
    logger.info("step 3: pocket assignment")
    top_smiles = df[args.smiles_col].to_list()
    pockets, tanimotos, combineds = assign_pockets(
        top_smiles, args.fragments_csv, args.pocket_threshold,
    )
    df = df.with_columns([
        pl.Series("pocket", pockets),
        pl.Series("pocket_tanimoto", tanimotos),
        pl.Series("pocket_combined", combineds),
    ])
    n_before = df.shape[0]
    df = df.filter(pl.col("pocket").is_not_null())
    logger.info(
        f"  assigned pockets: {df.shape[0]} of {n_before} "
        f"(dropped {n_before - df.shape[0]} below pocket threshold {args.pocket_threshold})"
    )
    for p in ALLOWED_POCKETS:
        ct = df.filter(pl.col("pocket") == p).shape[0]
        logger.info(f"  site {p}: {ct} compounds")

    # Step 4: Cluster per site
    logger.info("step 4: agglomerative clustering per site")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_reps: list[pl.DataFrame] = []

    for pocket in ALLOWED_POCKETS:
        site_df = df.filter(pl.col("pocket") == pocket)
        if site_df.shape[0] == 0:
            logger.warning(f"  site {pocket}: no compounds, skipping")
            continue
        target = site_targets[pocket]
        logger.info(f"  site {pocket}: {site_df.shape[0]} compounds -> target {target} clusters")
        reps = cluster_and_select(
            site_df, target, smiles_col=args.smiles_col,
        )

        # Step 5: Write per-site output
        out_path = args.output_dir / f"site_{pocket}.csv"
        output_cols = [
            args.smiles_col, args.id_col,
            "ensemble_rank_score", "pocket", "pocket_tanimoto",
            "pocket_combined", "cluster_id",
        ]
        output_cols = [c for c in output_cols if c in reps.columns]
        reps.select(output_cols).write_csv(out_path)
        logger.info(f"  wrote {reps.shape[0]} rows to {out_path}")
        all_reps.append(reps.select(output_cols))

    # Combined output
    if all_reps:
        combined = pl.concat(all_reps)
        combined_path = args.output_dir / "all_sites.csv"
        combined.write_csv(combined_path)
        logger.info(f"wrote {combined.shape[0]} total representatives to {combined_path}")

        # Summary
        for pocket in ALLOWED_POCKETS:
            site = combined.filter(pl.col("pocket") == pocket)
            if site.shape[0] > 0:
                scores_arr = site["ensemble_rank_score"].to_numpy()
                logger.info(
                    f"  site {pocket}: n={site.shape[0]}, "
                    f"score range [{scores_arr.min():.4f}, {scores_arr.max():.4f}], "
                    f"median={np.median(scores_arr):.4f}"
                )


if __name__ == "__main__":
    main()
