"""Filter a candidate compound set by Tanimoto similarity to the Zenodo SPR set.

Any input compound whose maximum Tanimoto similarity (Morgan r=2, 2048 bits)
against the 1,545 unique Zenodo SMILES exceeds ``--threshold`` (default 0.85)
is dropped. The script writes the surviving rows plus two annotation columns:

    max_tanimoto_zenodo        highest Tanimoto to any Zenodo compound
    nearest_zenodo_smiles      the Zenodo SMILES achieving that max

Invalid / unparseable SMILES in the input are dropped with a warning.

Usage
-----
    uv run python scripts/filter_zenodo_similarity.py \\
        --input candidates.csv \\
        --smiles-col smiles \\
        --output /tmp/candidates.filtered.csv

    # with a non-default threshold and a reference other than Zenodo:
    uv run python scripts/filter_zenodo_similarity.py \\
        --input candidates.csv --smiles-col smiles \\
        --reference data/zenodo/tbxt_spr_merged.csv --ref-smiles-col smiles \\
        --threshold 0.80 \\
        --output /tmp/candidates.filtered.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from rdkit import Chem
from rdkit.DataStructs import BulkTanimotoSimilarity
from typeguard import typechecked

from tbxt_hackathon.fingerprints import morgan_bitvects

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE = REPO_ROOT / "data" / "zenodo" / "tbxt_spr_merged.csv"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, type=Path, help="Input CSV of candidates.")
    ap.add_argument("--output", required=True, type=Path, help="Output CSV (kept compounds).")
    ap.add_argument(
        "--smiles-col",
        default="smiles",
        help="SMILES column name in --input (default 'smiles').",
    )
    ap.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help=f"Reference CSV to compare against (default {DEFAULT_REFERENCE}).",
    )
    ap.add_argument(
        "--ref-smiles-col",
        default="smiles",
        help="SMILES column name in --reference (default 'smiles').",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Drop candidates with max Tanimoto to reference > this (default 0.85).",
    )
    ap.add_argument(
        "--n-bits",
        type=int,
        default=2048,
        help="Morgan fingerprint size in bits (default 2048).",
    )
    ap.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius (default 2, i.e. ECFP4).",
    )
    return ap.parse_args()


@typechecked
def _load_unique_smiles(path: Path, col: str) -> list[str]:
    """Read a CSV, dedupe non-null SMILES, and verify each parses with RDKit."""
    df = pl.read_csv(path)
    if col not in df.columns:
        raise ValueError(f"column {col!r} not in {path}. Got: {df.columns}")
    raw = (
        df.select(pl.col(col).cast(pl.Utf8))
        .filter(pl.col(col).is_not_null())
        .unique()
        .to_series()
        .to_list()
    )
    kept, dropped = [], 0
    for s in raw:
        if Chem.MolFromSmiles(s) is not None:
            kept.append(s)
        else:
            dropped += 1
    if dropped:
        logger.warning(f"{dropped} unparseable SMILES dropped from {path.name}")
    logger.info(f"{path.name}: {len(kept)} unique valid SMILES")
    return kept


def main() -> None:
    args = parse_args()

    if not args.reference.exists():
        raise FileNotFoundError(f"reference file not found: {args.reference}")
    if not args.input.exists():
        raise FileNotFoundError(f"input file not found: {args.input}")

    # Reference fingerprints (dedup'd).
    logger.info(f"loading reference from {args.reference}")
    ref_smiles = _load_unique_smiles(args.reference, args.ref_smiles_col)
    ref_fps = morgan_bitvects(ref_smiles, n_bits=args.n_bits, radius=args.radius)

    # Load input, keep all original columns; track validity of SMILES.
    logger.info(f"loading input from {args.input}")
    df = pl.read_csv(args.input)
    if args.smiles_col not in df.columns:
        raise ValueError(
            f"SMILES column {args.smiles_col!r} not in input. Got: {df.columns}",
        )
    n_in = df.shape[0]
    df = df.filter(pl.col(args.smiles_col).is_not_null())
    n_null_dropped = n_in - df.shape[0]
    if n_null_dropped:
        logger.warning(f"dropped {n_null_dropped} rows with null SMILES")

    smiles = df[args.smiles_col].cast(pl.Utf8).to_list()
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    valid_mask = np.array([m is not None for m in mols], dtype=bool)
    n_invalid = int((~valid_mask).sum())
    if n_invalid:
        logger.warning(f"dropped {n_invalid} rows with unparseable SMILES")

    df_valid = df.filter(pl.Series(valid_mask))
    valid_smiles = [s for s, ok in zip(smiles, valid_mask) if ok]

    if not valid_smiles:
        logger.warning("no valid candidate SMILES after cleaning; writing empty output")
        df_valid.write_csv(args.output)
        return

    cand_fps = morgan_bitvects(
        valid_smiles, n_bits=args.n_bits, radius=args.radius,
    )
    assert len(cand_fps) == df_valid.shape[0], (
        f"fingerprint count {len(cand_fps)} != row count {df_valid.shape[0]}"
    )

    logger.info(
        f"computing {len(cand_fps):,} x {len(ref_fps):,} Tanimoto similarities",
    )
    max_sim = np.empty(len(cand_fps), dtype=np.float64)
    argmax_idx = np.empty(len(cand_fps), dtype=np.int64)
    for i, fp in enumerate(cand_fps):
        sims = np.asarray(BulkTanimotoSimilarity(fp, ref_fps), dtype=np.float64)
        j = int(sims.argmax())
        max_sim[i] = float(sims[j])
        argmax_idx[i] = j

    nearest_smiles = [ref_smiles[j] for j in argmax_idx]

    annotated = df_valid.with_columns([
        pl.Series("max_tanimoto_zenodo", max_sim),
        pl.Series("nearest_zenodo_smiles", nearest_smiles),
    ])

    kept = annotated.filter(pl.col("max_tanimoto_zenodo") <= args.threshold)
    n_dropped_sim = annotated.shape[0] - kept.shape[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept.write_csv(args.output)

    logger.info(
        f"input rows: {n_in}  |  null-SMILES dropped: {n_null_dropped}  |  "
        f"invalid-SMILES dropped: {n_invalid}  |  "
        f"similarity-dropped (>{args.threshold:.2f}): {n_dropped_sim}  |  "
        f"kept: {kept.shape[0]}",
    )
    logger.info(f"wrote {kept.shape[0]} rows to {args.output}")


if __name__ == "__main__":
    main()
