"""Step 01 (try2): build a filtered fold-assignments file for binary training.

Motivation
----------
Per-compound SPR measurement std in this dataset is ~1 pKD unit. The try1
labeling scheme (top-quartile pKD >= 3.837 -> binder) places the decision
boundary right inside the measurement noise, producing a label where a
substantial fraction of rows near the cut are effectively random. Models
learning that label cannot separate real binders from moderate-affinity
noise, which matches the try1 observation that all four ensembles tied on
holdout.

Here we define a cleaner label by carving out the ambiguous middle:

    pKD < ``LO``   -> is_binder = False   (non-binder)
    pKD > ``HI``   -> is_binder = True    (binder)
    LO <= pKD <= HI -> dropped (ambiguous, label noise dominates)

Defaults (LO=3, HI=5) leave 708 / 1,599 compounds with a ~12 % positive
rate and preserve the existing chemical-space fold assignments so we can
reuse all downstream CV and TukeyHSD machinery from try1.

Outputs
-------
    data/classification-models-try2-rjg/fold_assignments_filtered.csv
    data/classification-models-try2-rjg/label_diagnostic.json

Usage
-----
    uv run python scripts/classification-models-try2-rjg/01_make_filtered_labels.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
FOLDS_CSV = ROOT / "data" / "folds-pacmap-kmeans6" / "fold_assignments.csv"
OUT_DIR = ROOT / "data" / "classification-models-try3-rjg"
OUT_CSV = OUT_DIR / "fold_assignments_filtered.csv"
OUT_JSON = OUT_DIR / "label_diagnostic.json"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--lo",
        type=float,
        default=3.0,
        help="pKD strictly below this -> non-binder (default 3.0)",
    )
    ap.add_argument(
        "--hi",
        type=float,
        default=5.0,
        help="pKD strictly above this -> binder (default 5.0)",
    )
    ap.add_argument(
        "--holdout-fold",
        type=int,
        default=3,
        help="Fold ID to use as holdout (default 3, more balanced than try2's fold 4)",
    )
    return ap.parse_args()


def _breakdown_by_fold(df: pl.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for fold in sorted(df["fold"].unique().to_list()):
        sub = df.filter(pl.col("fold") == fold)
        rows.append(
            {
                "fold": int(fold),
                "n": int(sub.shape[0]),
                "n_positives": int(sub["is_binder"].sum()),
                "prevalence": float(sub["is_binder"].mean()),  # type: ignore[arg-type]
                "is_holdout": bool(sub["is_holdout"].any()),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    assert args.lo < args.hi, f"expected lo < hi, got lo={args.lo} hi={args.hi}"

    df = pl.read_csv(FOLDS_CSV)
    n_original = df.shape[0]
    assert {"fold", "is_holdout", "pKD_global_mean", "canonical_smiles"}.issubset(
        df.columns
    ), f"missing expected columns in {FOLDS_CSV}: got {df.columns}"

    # Override the baked-in holdout (fold 4) with the user's chosen fold.
    # Every downstream script reads `is_holdout` / `holdout_fold`, so this is
    # the single point of change.
    available_folds = sorted(df["fold"].unique().to_list())
    if args.holdout_fold not in available_folds:
        raise ValueError(
            f"holdout_fold={args.holdout_fold} not in available folds {available_folds}"
        )
    df = df.with_columns(
        pl.lit(args.holdout_fold).alias("holdout_fold"),
        (pl.col("fold") == args.holdout_fold).alias("is_holdout"),
    )
    logger.info(f"holdout fold set to {args.holdout_fold}")

    gray_mask = (pl.col("pKD_global_mean") >= args.lo) & (
        pl.col("pKD_global_mean") <= args.hi
    )
    n_gray = int(df.filter(gray_mask).shape[0])

    kept = df.filter(~gray_mask).with_columns(
        (pl.col("pKD_global_mean") > args.hi).alias("is_binder"),
    )
    kept = kept.with_columns(
        pl.lit(args.lo).alias("filter_lo_pkd"),
        pl.lit(args.hi).alias("filter_hi_pkd"),
    )

    n_kept = kept.shape[0]
    n_pos = int(kept["is_binder"].sum())
    n_neg = n_kept - n_pos
    logger.info(
        f"filter pKD < {args.lo} (neg) or > {args.hi} (pos): "
        f"kept {n_kept}/{n_original} compounds "
        f"({n_pos} positives, {n_neg} negatives, "
        f"dropped {n_gray} ambiguous in [{args.lo}, {args.hi}])"
    )

    # Sanity-check: every fold must retain at least one positive and one
    # negative, otherwise CV scoring breaks down on that fold.
    fold_breakdown = _breakdown_by_fold(kept)
    for row in fold_breakdown:
        if row["n_positives"] == 0:
            logger.warning(
                f"fold {row['fold']} retains 0 positives after filter "
                f"(n={row['n']}, holdout={row['is_holdout']})"
            )
        if row["n_positives"] == row["n"]:
            logger.warning(
                f"fold {row['fold']} retains 0 negatives after filter "
                f"(n={row['n']}, holdout={row['is_holdout']})"
            )

    kept.write_csv(OUT_CSV)
    logger.info(f"wrote filtered fold assignments to {OUT_CSV}")

    diagnostic = {
        "source_csv": str(FOLDS_CSV.relative_to(ROOT)),
        "holdout_fold": args.holdout_fold,
        "filter_lo_pkd": args.lo,
        "filter_hi_pkd": args.hi,
        "n_original": n_original,
        "n_kept": n_kept,
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "n_dropped_gray": n_gray,
        "prevalence": n_pos / n_kept if n_kept else 0.0,
        "fold_breakdown": fold_breakdown,
    }
    OUT_JSON.write_text(json.dumps(diagnostic, indent=2))
    logger.info(f"wrote label diagnostic to {OUT_JSON}")


if __name__ == "__main__":
    main()
