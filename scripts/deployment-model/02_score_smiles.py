"""Step 02 (deployment): score new SMILES with the 3-model rank ensemble.

Takes a CSV or SDF of compounds, runs all three model families
(deploy-XGB, Morgan-XGB, CheMeleon), rank-normalizes each model's
probabilities within the batch, and ranks compounds by the mean of the
three rank-normalized scores.

**Important**: because rank-normalization is performed within the input
batch, the ``ensemble_rank_score`` depends on what else is in the batch.
For onepot screening, pass the FULL candidate set in one call and take
the top-K afterward. Scoring the same compound in two different batches
will give two different scores.

Output columns
--------------
    compound_id
    canonical_smiles
    ensemble_rank_score         # mean of the three rank-norm scores, [0, 1]
    rank                        # 1 = best ensemble_rank_score
    p_deploy                    # raw P(binder) from deploy-XGB
    p_morgan                    # raw P(binder) from Morgan-XGB
    p_chemeleon                 # raw P(binder) from CheMeleon
    rank_deploy                 # within-batch rank from deploy-XGB (1 = best)
    rank_morgan                 # within-batch rank from Morgan-XGB
    rank_chemeleon              # within-batch rank from CheMeleon
    score_deploy_rn             # rank-normalized deploy-XGB score in [0, 1]
    score_morgan_rn             # rank-normalized Morgan-XGB score
    score_chemeleon_rn          # rank-normalized CheMeleon score

Usage
-----
    uv run python scripts/deployment-model/02_score_smiles.py \\
        --input data/some-new-compounds.csv \\
        --smiles-col smiles \\
        --id-col compound_id \\
        --output /tmp/scored.csv

    # or from an SDF:
    uv run python scripts/deployment-model/02_score_smiles.py \\
        --input library.sdf \\
        --sdf \\
        --output /tmp/library_scored.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from rdkit import Chem

from tbxt_hackathon.deployment import EnsembleModel


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="Input CSV or SDF.")
    ap.add_argument(
        "--sdf",
        action="store_true",
        help="Treat --input as SDF. Otherwise, CSV is assumed.",
    )
    ap.add_argument(
        "--smiles-col",
        default="smiles",
        help="Column name containing SMILES (CSV mode; default 'smiles').",
    )
    ap.add_argument(
        "--id-col",
        default=None,
        help="Column name containing compound IDs (CSV mode; default: use row index).",
    )
    ap.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Override model directory (default: data/deployment-model/).",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Only write the top-N rows ranked by ensemble_rank_score.",
    )
    ap.add_argument(
        "--accelerator",
        default="auto",
        choices=["auto", "cpu", "mps", "gpu"],
        help="Torch accelerator for CheMeleon prediction.",
    )
    ap.add_argument(
        "--chemeleon-batch-size",
        type=int,
        default=128,
        help="Inference batch size for CheMeleon (higher = faster, more RAM).",
    )
    return ap.parse_args()


def _read_input(args: argparse.Namespace) -> pl.DataFrame:
    """Normalize CSV or SDF input to a polars frame with compound_id + canonical_smiles."""
    if args.sdf:
        mols = [m for m in Chem.SDMolSupplier(str(args.input)) if m is not None]
        logger.info(f"loaded {len(mols)} mols from {args.input}")
        rows = []
        for i, mol in enumerate(mols):
            cid = mol.GetProp("_Name") if mol.HasProp("_Name") else f"row_{i}"
            rows.append({
                "compound_id": str(cid) or f"row_{i}",
                "canonical_smiles": Chem.MolToSmiles(mol),
            })
        return pl.DataFrame(rows)

    df = pl.read_csv(args.input)
    if args.smiles_col not in df.columns:
        raise ValueError(
            f"SMILES column {args.smiles_col!r} not in input. "
            f"Got columns: {df.columns}",
        )
    if args.id_col is None:
        df = df.with_row_index("compound_id").with_columns(
            pl.col("compound_id").cast(pl.Utf8).str.zfill(8),
        )
        id_col = "compound_id"
    elif args.id_col not in df.columns:
        raise ValueError(
            f"id column {args.id_col!r} not in input. Got: {df.columns}",
        )
    else:
        id_col = args.id_col

    return df.select([
        pl.col(id_col).cast(pl.Utf8).alias("compound_id"),
        pl.col(args.smiles_col).alias("canonical_smiles"),
    ])


def main() -> None:
    args = parse_args()

    frame = _read_input(args)
    n_in = frame.shape[0]
    logger.info(f"loaded {n_in} input compounds; dropping nulls / non-parseable SMILES")
    frame = frame.filter(pl.col("canonical_smiles").is_not_null())
    smiles = frame["canonical_smiles"].to_list()

    # Filter invalid SMILES up front; invalid rows get NaN columns in the output.
    valid_mask = np.array(
        [Chem.MolFromSmiles(s) is not None for s in smiles], dtype=bool,
    )
    n_invalid = int((~valid_mask).sum())
    if n_invalid:
        logger.warning(f"{n_invalid} SMILES failed to parse; they will get NaN scores")

    valid_smiles = [s for s, ok in zip(smiles, valid_mask) if ok]

    logger.info("loading 3-model ensemble (deploy XGB, Morgan XGB, CheMeleon)")
    model = EnsembleModel.load(
        model_dir=args.model_dir, accelerator=args.accelerator,
    )
    pred = model.predict(valid_smiles, chemeleon_batch_size=args.chemeleon_batch_size)

    # Scatter predictions back to the full input length (np.nan for invalid).
    n = len(smiles)
    full_cols: dict[str, np.ndarray] = {}

    def scatter(vec: np.ndarray) -> np.ndarray:
        out = np.full(n, np.nan, dtype=np.float64)
        out[valid_mask] = vec
        return out

    full_cols["p_deploy"] = scatter(pred.p_deploy)
    full_cols["p_morgan"] = scatter(pred.p_morgan)
    full_cols["p_chemeleon"] = scatter(pred.p_chemeleon)
    full_cols["rank_deploy"] = scatter(pred.rank_deploy)
    full_cols["rank_morgan"] = scatter(pred.rank_morgan)
    full_cols["rank_chemeleon"] = scatter(pred.rank_chemeleon)
    full_cols["score_deploy_rn"] = scatter(pred.score_deploy_rn)
    full_cols["score_morgan_rn"] = scatter(pred.score_morgan_rn)
    full_cols["score_chemeleon_rn"] = scatter(pred.score_chemeleon_rn)
    full_cols["ensemble_rank_score"] = scatter(pred.ensemble_rank_score)

    # Compose output frame.  Column order: id, smiles, ensemble_rank_score,
    # rank, then per-model columns.
    out = frame.with_columns([pl.Series(k, v) for k, v in full_cols.items()])

    # Final rank: 1 = best ensemble_rank_score. NaN goes to the bottom.
    out = out.with_columns(
        pl.col("ensemble_rank_score")
        .rank(method="ordinal", descending=True)
        .alias("rank"),
    )

    # Reorder for readability
    column_order = [
        "compound_id",
        "canonical_smiles",
        "ensemble_rank_score",
        "rank",
        "p_deploy",
        "p_morgan",
        "p_chemeleon",
        "rank_deploy",
        "rank_morgan",
        "rank_chemeleon",
        "score_deploy_rn",
        "score_morgan_rn",
        "score_chemeleon_rn",
    ]
    out = out.select([c for c in column_order if c in out.columns])

    out = out.sort("rank")
    if args.top_n is not None:
        out = out.head(args.top_n)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(args.output)
    logger.info(
        f"wrote {out.shape[0]} rows to {args.output} "
        f"(from {n_in} input; {n_invalid} invalid flagged as NaN)",
    )

    # Quick summary
    head = out.head(5).select([
        "compound_id", "ensemble_rank_score", "rank",
        "p_deploy", "p_morgan", "p_chemeleon",
    ])
    logger.info(f"top 5 by ensemble_rank_score:\n{head}")


if __name__ == "__main__":
    main()
