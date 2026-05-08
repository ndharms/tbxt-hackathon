"""Step 02: assign 6 folds via Morgan FP -> PaCMAP -> KMeans, QC via Tanimoto.

Writes:
    data/folds-pacmap-kmeans6/fold_assignments.csv
    data/folds-pacmap-kmeans6/fold_qc_summary.json
    docs/folds-pacmap-kmeans6/folds_pacmap.png
    docs/folds-pacmap-kmeans6/folds_top5_tanimoto.png
    docs/folds-pacmap-kmeans6/folds_pairwise_heatmap.png

Usage:
    uv run python scripts/folds-pacmap-kmeans6/02_make_folds.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from tbxt_hackathon.fingerprints import bitvects_to_ndarray, morgan_bitvects
from tbxt_hackathon.fold_qc import (
    evaluate_fold_quality,
    plot_embedding,
    plot_pairwise_heatmap,
    plot_top5_distributions,
)
from tbxt_hackathon.folds import assign_folds_by_chemical_space

ROOT = Path(__file__).resolve().parents[2]
LABELED_CSV = ROOT / "data" / "processed" / "tbxt_compounds_labeled.csv"
ARTIFACT_DIR = ROOT / "data" / "folds-pacmap-kmeans6"
DOC_FIG_DIR = ROOT / "docs" / "folds-pacmap-kmeans6"

N_FOLDS = 6
RANDOM_STATE = 0


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(LABELED_CSV)
    assert df.shape[0] > 0, "labeled compounds file is empty"
    smiles = df["canonical_smiles"].to_list()
    logger.info(f"loaded {len(smiles)} SMILES")

    fa = assign_folds_by_chemical_space(
        smiles, n_folds=N_FOLDS, random_state=RANDOM_STATE,
    )

    # Reconstruct bitvect objects for QC (keeps API clean; same FPs as in FoldAssignment)
    fps = morgan_bitvects(smiles, n_bits=2048, radius=2)
    # Sanity: array should match
    assert np.array_equal(fa.fingerprints, bitvects_to_ndarray(fps))

    qc = evaluate_fold_quality(fps, fa.fold_id, k=5)

    # Plots
    plot_embedding(
        fa.embedding_2d, fa.fold_id, DOC_FIG_DIR / "folds_pacmap.png",
        holdout_fold=int(qc.distinctness_ranking[0]),
    )
    plot_top5_distributions(qc, DOC_FIG_DIR / "folds_top5_tanimoto.png")
    plot_pairwise_heatmap(qc, DOC_FIG_DIR / "folds_pairwise_heatmap.png")

    holdout_fold = qc.distinctness_ranking[0]
    logger.info(f"holdout (most structurally distinct) fold: {holdout_fold}")

    # Write fold assignments
    out = df.with_columns(
        pl.Series("fold", fa.fold_id.astype(np.int64)),
        pl.Series("pacmap_1", fa.embedding_2d[:, 0].astype(np.float64)),
        pl.Series("pacmap_2", fa.embedding_2d[:, 1].astype(np.float64)),
        pl.lit(holdout_fold).alias("holdout_fold"),
        (pl.Series("fold", fa.fold_id.astype(np.int64)) == holdout_fold).alias("is_holdout"),
    )
    out_path = ARTIFACT_DIR / "fold_assignments.csv"
    out.write_csv(out_path)
    logger.info(f"wrote fold assignments to {out_path}")

    summary = {
        "n_folds": N_FOLDS,
        "random_state": RANDOM_STATE,
        "holdout_fold": int(holdout_fold),
        "distinctness_ranking": [int(x) for x in qc.distinctness_ranking],
        "per_fold_top5_median": {
            str(k): float(np.median(v)) for k, v in qc.per_fold_top5.items()
        },
        "per_fold_top5_mean": {
            str(k): float(np.mean(v)) for k, v in qc.per_fold_top5.items()
        },
        "fold_sizes": {
            str(k): int((fa.fold_id == k).sum()) for k in sorted(np.unique(fa.fold_id).tolist())
        },
    }
    summary_path = ARTIFACT_DIR / "fold_qc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"wrote QC summary to {summary_path}")


if __name__ == "__main__":
    main()
