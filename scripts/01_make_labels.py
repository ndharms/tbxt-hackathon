"""Step 01: load compounds, diagnose pKD, label binders by top-quartile.

Usage:
    uv run python scripts/01_make_labels.py
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from loguru import logger

from tbxt_hackathon.data import label_binders, load_compounds, pkd_diagnostic

ROOT = Path(__file__).resolve().parents[1]
COMPOUNDS_CSV = ROOT / "data" / "processed" / "tbxt_compounds_clean.csv"
OUT_CSV = ROOT / "data" / "processed" / "tbxt_compounds_labeled.csv"


def main() -> None:
    compounds = load_compounds(COMPOUNDS_CSV)
    stats = pkd_diagnostic(compounds)
    logger.info("pKD distribution: " + ", ".join(f"{k}={v:.3f}" for k, v in stats.items()))
    labeled = label_binders(compounds, quantile=0.75)
    # write minimal modeling columns first, then keep extras
    out = labeled.frame.select(
        [
            "compound_id",
            "canonical_smiles",
            "supplier",
            "pKD_global_mean",
            "is_binder",
        ]
    ).with_columns(pl.lit(labeled.threshold).alias("binder_threshold_pkd"))
    out.write_csv(OUT_CSV)
    logger.info(
        f"wrote labeled file to {OUT_CSV} "
        f"({labeled.n_positives}/{labeled.n_total} positives, "
        f"threshold pKD >= {labeled.threshold:.3f})"
    )


if __name__ == "__main__":
    main()
