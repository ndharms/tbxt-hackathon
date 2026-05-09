"""Step 02 (deployment): score new SMILES with the deployment ensemble.

Takes a CSV or SDF of compounds, builds the MACCS+pocket+physchem feature
matrix for each, runs the 6-booster ensemble, and writes ranked output.

Output columns:
    compound_id (or smiles, if no id column present)
    canonical_smiles
    p_binder_ensemble_mean
    p_binder_fold_{0..5}        # per-booster probabilities
    rank                        # 1 = highest P(binder)

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

from tbxt_hackathon.deployment import DeploymentModel


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="Input CSV or SDF.")
    ap.add_argument(
        "--sdf", action="store_true",
        help="Treat --input as SDF. Otherwise, CSV is assumed."
    )
    ap.add_argument(
        "--smiles-col", default="smiles",
        help="Column name containing SMILES (CSV mode; default 'smiles')."
    )
    ap.add_argument(
        "--id-col", default=None,
        help="Column name containing compound IDs (CSV mode; default: use row index)."
    )
    ap.add_argument(
        "--output", type=Path, required=True,
        help="Output CSV path."
    )
    ap.add_argument(
        "--model-dir", type=Path, default=None,
        help="Override model directory (default: data/deployment-model/)."
    )
    ap.add_argument(
        "--top-n", type=int, default=None,
        help="Only write the top-N rows ranked by ensemble mean."
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
            f"Got columns: {df.columns}"
        )
    if args.id_col is None:
        df = df.with_row_index("compound_id").with_columns(
            pl.col("compound_id").cast(pl.Utf8).str.zfill(8)
        )
        id_col = "compound_id"
    elif args.id_col not in df.columns:
        raise ValueError(
            f"id column {args.id_col!r} not in input. Got: {df.columns}"
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

    # filter invalid SMILES silently (mark with NaN score)
    valid_mask = np.array(
        [Chem.MolFromSmiles(s) is not None for s in smiles], dtype=bool
    )
    n_invalid = int((~valid_mask).sum())
    if n_invalid:
        logger.warning(f"{n_invalid} SMILES failed to parse; they will get NaN scores")

    valid_smiles = [s for s, ok in zip(smiles, valid_mask) if ok]
    model = DeploymentModel.load(model_dir=args.model_dir)

    per_model = model.predict(valid_smiles, return_per_model=True)
    mean_p = per_model.mean(axis=1)

    # Scatter back to full length (np.nan for invalid)
    n = len(smiles)
    full_per_model = np.full((n, per_model.shape[1]), np.nan)
    full_mean = np.full(n, np.nan)
    valid_idx = np.where(valid_mask)[0]
    full_per_model[valid_idx] = per_model
    full_mean[valid_idx] = mean_p

    # Attach predictions
    pred_cols = {
        f"p_binder_fold_{i}": full_per_model[:, i] for i in range(per_model.shape[1])
    }
    out = frame.with_columns(
        [pl.Series("p_binder_ensemble_mean", full_mean)]
        + [pl.Series(k, v) for k, v in pred_cols.items()]
    )
    # Rank: 1 = highest ensemble mean. NaNs go last (rank = n_valid + i).
    out = out.with_columns(
        pl.col("p_binder_ensemble_mean")
        .rank(method="ordinal", descending=True)
        .alias("rank")
    )

    out = out.sort("rank")
    if args.top_n is not None:
        out = out.head(args.top_n)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(args.output)
    logger.info(
        f"wrote {out.shape[0]} rows to {args.output} "
        f"(from {n_in} input; {n_invalid} invalid dropped to NaN scores)"
    )

    # Quick summary
    top5 = out.head(5).select(["compound_id", "p_binder_ensemble_mean", "rank"])
    logger.info(f"top 5 by ensemble mean:\n{top5}")


if __name__ == "__main__":
    main()
