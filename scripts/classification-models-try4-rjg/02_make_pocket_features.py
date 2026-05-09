"""Step 02 (try4): precompute pocket features for all 708 filtered compounds.

Writes:
    data/classification-models-try4-rjg/pocket_features.csv

Columns (in order):
    compound_id
    pocket_A_tanimoto, pocket_A_substruct
    pocket_B_tanimoto, pocket_B_substruct
    pocket_C_tanimoto, pocket_C_substruct
    pocket_D_tanimoto, pocket_D_substruct

``*_substruct`` is encoded as float 0.0 / 1.0 so the whole block is
homogeneous float32 and can be concatenated onto the fingerprint
block without further type handling.

Usage:
    uv run python scripts/classification-models-try4-rjg/02_make_pocket_features.py
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from loguru import logger

from tbxt_hackathon.pocket_assigner import PocketAssigner

ROOT = Path(__file__).resolve().parents[2]
FOLDS_CSV = ROOT / "data" / "classification-models-try4-rjg" / "fold_assignments_filtered.csv"
FRAGMENTS_CSV = ROOT / "data" / "structures" / "sgc_fragments.csv"
OUT_CSV = ROOT / "data" / "classification-models-try4-rjg" / "pocket_features.csv"

POCKETS = ("A", "B", "C", "D")


def main() -> None:
    df = pl.read_csv(FOLDS_CSV)
    assert {"compound_id", "canonical_smiles"}.issubset(df.columns)

    assigner = PocketAssigner.from_csv(FRAGMENTS_CSV)
    # sanity-check that all four pockets are represented
    for p in POCKETS:
        if p not in assigner.pocket_fps:
            raise RuntimeError(
                f"expected pocket {p!r} in fragment CSV but got {list(assigner.pocket_fps)}"
            )

    logger.info(f"scoring {df.shape[0]} compounds against {len(POCKETS)} pockets")
    rows = []
    smiles_list = df["canonical_smiles"].to_list()
    scores_list = assigner.score_batch(smiles_list)

    for cid, scores in zip(df["compound_id"].to_list(), scores_list):
        row = {"compound_id": cid}
        for p in POCKETS:
            if p in scores:
                row[f"pocket_{p}_tanimoto"] = float(scores[p].tanimoto)
                row[f"pocket_{p}_substruct"] = 1.0 if scores[p].substruct else 0.0
            else:
                row[f"pocket_{p}_tanimoto"] = 0.0
                row[f"pocket_{p}_substruct"] = 0.0
        rows.append(row)

    out = pl.DataFrame(rows)
    assert out.shape[0] == df.shape[0]
    out.write_csv(OUT_CSV)
    logger.info(f"wrote pocket features {out.shape} to {OUT_CSV}")
    # quick summary
    for p in POCKETS:
        tc = out[f"pocket_{p}_tanimoto"].to_numpy()
        sub = out[f"pocket_{p}_substruct"].to_numpy()
        logger.info(
            f"pocket {p}: tanimoto mean={tc.mean():.3f} max={tc.max():.3f}, "
            f"substruct_hits={int(sub.sum())}/{sub.size}"
        )


if __name__ == "__main__":
    main()
