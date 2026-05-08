"""Score compounds against TBXT pockets using the pocket-assigner model.

Takes an input CSV/parquet with a SMILES column and outputs per-pocket
scores (max Tanimoto, substructure match, combined score, assignment).

Usage:
    uv run python scripts/score_pockets.py --input compounds.csv --output scored.parquet
    uv run python scripts/score_pockets.py --input compounds.csv --smiles-col canonical_smiles
    uv run python scripts/score_pockets.py --input compounds.csv --threshold 0.30
"""

import argparse
from pathlib import Path

import polars as pl
from loguru import logger

from tbxt_hackathon.pocket_assigner import PocketAssigner


DEFAULT_FRAGMENT_CSV = Path("data/structures/sgc_fragments.csv")
DEFAULT_THRESHOLD = 0.35


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Score compounds for TBXT pocket assignment."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input file (CSV or Parquet) with SMILES column.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file path (CSV or Parquet, inferred from extension). "
             "Defaults to input stem + '_pocket_scored.parquet'.",
    )
    parser.add_argument(
        "--fragments", "-f",
        type=Path,
        default=DEFAULT_FRAGMENT_CSV,
        help=f"Path to fragment CSV. Default: {DEFAULT_FRAGMENT_CSV}",
    )
    parser.add_argument(
        "--smiles-col",
        type=str,
        default="smiles",
        help="Name of the SMILES column in the input. Default: 'smiles'.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Min Tanimoto threshold for assignment. Default: {DEFAULT_THRESHOLD}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Processing batch size for progress logging. Default: 10000.",
    )
    return parser.parse_args()


def load_input(path: Path, smiles_col: str) -> pl.DataFrame:
    """Load input data from CSV or Parquet.

    Args:
        path: File path (extension determines format).
        smiles_col: Name of the SMILES column.

    Returns:
        Polars DataFrame with at minimum the SMILES column.
    """
    if path.suffix == ".parquet":
        df = pl.read_parquet(path)
    elif path.suffix in (".csv", ".tsv"):
        separator = "\t" if path.suffix == ".tsv" else ","
        df = pl.read_csv(path, separator=separator)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if smiles_col not in df.columns:
        raise ValueError(
            f"Column '{smiles_col}' not found in {path}. "
            f"Available: {df.columns}"
        )

    n_rows = len(df)
    n_null = df[smiles_col].null_count()
    logger.info(f"Loaded {n_rows} rows from {path} ({n_null} null SMILES)")
    return df


def score_dataframe(
    df: pl.DataFrame,
    assigner: PocketAssigner,
    smiles_col: str,
    batch_size: int,
) -> pl.DataFrame:
    """Score all compounds and add pocket columns.

    Adds columns:
      - pocket_A_prime_tc: max Tanimoto to A_prime fragments
      - pocket_A_prime_sub: substructure match (bool)
      - pocket_F_tc: max Tanimoto to F fragments
      - pocket_F_sub: substructure match (bool)
      - pocket_G_tc: max Tanimoto to G fragments
      - pocket_G_sub: substructure match (bool)
      - pocket_best: assigned pocket label (or null)
      - pocket_best_combined: combined score of best pocket

    Args:
        df: Input DataFrame.
        assigner: Loaded PocketAssigner.
        smiles_col: Name of the SMILES column.
        batch_size: Batch size for progress logging.

    Returns:
        DataFrame with pocket score columns appended.
    """
    smiles_list = df[smiles_col].to_list()
    n_total = len(smiles_list)
    pockets = sorted(assigner.pocket_fps.keys())

    # Pre-allocate result arrays
    tc_arrays: dict[str, list[float]] = {p: [] for p in pockets}
    sub_arrays: dict[str, list[bool]] = {p: [] for p in pockets}
    best_pocket_list: list[str | None] = []
    best_combined_list: list[float] = []

    for i, smi in enumerate(smiles_list):
        if i > 0 and i % batch_size == 0:
            logger.info(f"  Scored {i}/{n_total} compounds...")

        if smi is None:
            for p in pockets:
                tc_arrays[p].append(0.0)
                sub_arrays[p].append(False)
            best_pocket_list.append(None)
            best_combined_list.append(0.0)
            continue

        scores = assigner.score(smi)
        if not scores:
            for p in pockets:
                tc_arrays[p].append(0.0)
                sub_arrays[p].append(False)
            best_pocket_list.append(None)
            best_combined_list.append(0.0)
            continue

        for p in pockets:
            if p in scores:
                tc_arrays[p].append(scores[p].tanimoto)
                sub_arrays[p].append(scores[p].substruct)
            else:
                tc_arrays[p].append(0.0)
                sub_arrays[p].append(False)

        best_p = max(scores, key=lambda p: scores[p].combined)
        best_score = scores[best_p]

        # Apply assignment logic
        if best_score.substruct or best_score.tanimoto >= assigner.threshold:
            best_pocket_list.append(best_p)
        else:
            best_pocket_list.append(None)
        best_combined_list.append(best_score.combined)

    logger.info(f"  Scored {n_total}/{n_total} compounds.")

    # Build result columns
    new_cols: dict[str, pl.Series] = {}
    for p in pockets:
        new_cols[f"pocket_{p}_tc"] = pl.Series(f"pocket_{p}_tc", tc_arrays[p])
        new_cols[f"pocket_{p}_sub"] = pl.Series(f"pocket_{p}_sub", sub_arrays[p])

    new_cols["pocket_best"] = pl.Series("pocket_best", best_pocket_list)
    new_cols["pocket_best_combined"] = pl.Series(
        "pocket_best_combined", best_combined_list
    )

    result = df.with_columns(list(new_cols.values()))

    # Log assignment summary
    assigned = result.filter(pl.col("pocket_best").is_not_null())
    logger.info(
        f"Assignment summary: {len(assigned)}/{n_total} compounds assigned "
        f"({100*len(assigned)/n_total:.1f}%)"
    )
    if len(assigned) > 0:
        counts = assigned.group_by("pocket_best").agg(pl.len().alias("count"))
        for row in counts.sort("pocket_best").iter_rows(named=True):
            pocket_name = row["pocket_best"]
            sub_count = assigned.filter(
                (pl.col("pocket_best") == pocket_name)
                & pl.col(f"pocket_{pocket_name}_sub")
            ).height
            logger.info(
                f"  {pocket_name}: {row['count']} assigned "
                f"({sub_count} substructure matches)"
            )

    return result


def save_output(df: pl.DataFrame, path: Path) -> None:
    """Save scored DataFrame to CSV or Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.write_parquet(path)
    elif path.suffix == ".csv":
        df.write_csv(path)
    else:
        # Default to parquet
        path = path.with_suffix(".parquet")
        df.write_parquet(path)
    logger.info(f"Saved scored output to {path}")


def main() -> None:
    """Load model, score input compounds, save results."""
    args = parse_args()

    # Determine output path
    if args.output is None:
        args.output = args.input.with_name(
            args.input.stem + "_pocket_scored.parquet"
        )

    # Load pocket assigner from fragment CSV
    logger.info(f"Loading pocket assigner from {args.fragments}")
    assigner = PocketAssigner.from_csv(args.fragments, threshold=args.threshold)

    # Load input
    df = load_input(args.input, args.smiles_col)

    # Score
    logger.info("Scoring compounds...")
    scored = score_dataframe(df, assigner, args.smiles_col, args.batch_size)

    # Save
    save_output(scored, args.output)


if __name__ == "__main__":
    main()
