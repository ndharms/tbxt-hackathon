"""Quick stats on the focused-VL parts (or final concat).

Reads ``data/processed/onepot_focused_vl/part_*.parquet`` (and any
straggler shards) directly via DuckDB without re-running concat. Reports
row count, file size, schema, fragment-hit / pocket distributions,
n_fragment_hits histogram, and physchem summary.

Run:
    uv run python scripts/downfilter_onepot/stats.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARD_DIR = REPO_ROOT / "data" / "processed" / "onepot_focused_vl"
FINAL_FILE = REPO_ROOT / "data" / "processed" / "onepot_focused_vl.parquet"


def heading(s: str) -> None:
    print("\n" + "=" * 78)
    print(f" {s}")
    print("=" * 78)


def main() -> int:
    parts = sorted(SHARD_DIR.glob("part_*.parquet"))
    shards = sorted(SHARD_DIR.glob("shard_*.parquet"))
    if not parts:
        raise SystemExit(f"No part files found in {SHARD_DIR}")

    total_part_bytes = sum(p.stat().st_size for p in parts)
    total_shard_bytes = sum(p.stat().st_size for p in shards)

    print(f"Reading parts from : {SHARD_DIR}")
    print(f"  part files       : {len(parts)}  ({total_part_bytes / 1e9:.2f} GB on disk)")
    print(f"  straggler shards : {len(shards)} ({total_shard_bytes / 1e6:.1f} MB on disk)")

    con = duckdb.connect()
    parts_glob = str(SHARD_DIR / "part_*.parquet").replace("'", "''")

    # Source pattern: parts only (avoid stale shards with possibly different
    # schemas). Use union_by_name to tolerate per-part column drift.
    src = f"read_parquet('{parts_glob}', union_by_name=true)"

    heading("1. Row count")
    n = con.execute(f"SELECT count(*) FROM {src}").fetchone()[0]
    print(f"  Total rows: {n:,}")

    heading("2. Schema (sample)")
    cols = con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()
    print(f"  Total columns: {len(cols)}")
    print("  First 10:")
    for col_name, col_type, *_ in cols[:10]:
        print(f"    {col_name:<35} {col_type}")
    n_frag_cols = sum(1 for c in cols if str(c[0]).startswith("frag_"))
    print(f"  ... fragment booleans (frag_*): {n_frag_cols}")

    heading("3. Pocket / matched_sites distribution")
    pocket_rows = con.execute(f"""
        SELECT matched_sites, count(*) AS n
        FROM {src}
        WHERE matched_sites IS NOT NULL
        GROUP BY matched_sites
        ORDER BY n DESC
        LIMIT 20
    """).fetchall()
    for sites, count in pocket_rows:
        pct = 100 * count / n if n else 0
        print(f"  {str(sites):<25} {count:>10,}  ({pct:5.2f}%)")

    heading("4. n_fragment_hits histogram")
    hist = con.execute(f"""
        SELECT n_fragment_hits, count(*) AS n
        FROM {src}
        GROUP BY n_fragment_hits
        ORDER BY n_fragment_hits
    """).fetchall()
    for nh, count in hist:
        bar = "#" * int(60 * count / max(c for _, c in hist))
        print(f"  hits={nh:>2}  {count:>10,}  {bar}")

    heading("5. Per-fragment hit counts (top 20)")
    frag_cols = [c[0] for c in cols if str(c[0]).startswith("frag_")]
    if frag_cols:
        sums = ", ".join(f"sum(CAST({c} AS INT)) AS {c}" for c in frag_cols)
        row = con.execute(f"SELECT {sums} FROM {src}").fetchone()
        ranked = sorted(
            ((c, v) for c, v in zip(frag_cols, row) if v),
            key=lambda x: -x[1],
        )
        for col, count in ranked[:20]:
            pct = 100 * count / n if n else 0
            print(f"  {col:<35} {count:>10,}  ({pct:5.2f}%)")

    heading("6. Physchem summary")
    physchem_cols = ["mw", "logp", "hbd", "hba", "heavy_atoms",
                     "num_rings", "num_aromatic_rings", "num_fused_rings",
                     "tpsa", "rotatable_bonds"]
    selects = []
    for c in physchem_cols:
        selects.append(
            f"min({c}) AS {c}_min, "
            f"avg({c}) AS {c}_mean, "
            f"max({c}) AS {c}_max"
        )
    summary = con.execute(
        f"SELECT {', '.join(selects)} FROM {src}"
    ).fetchone()
    cols_sum = [c[0] for c in con.description]
    print(f"  {'col':<22} {'min':>10} {'mean':>10} {'max':>10}")
    for c in physchem_cols:
        i_min = cols_sum.index(f"{c}_min")
        i_mean = cols_sum.index(f"{c}_mean")
        i_max = cols_sum.index(f"{c}_max")
        print(f"  {c:<22} {summary[i_min]:>10.2f} {summary[i_mean]:>10.2f} {summary[i_max]:>10.2f}")

    heading("7. Final-file status")
    if FINAL_FILE.exists():
        fsize = FINAL_FILE.stat().st_size / 1e6
        f_n = con.execute(
            f"SELECT count(*) FROM read_parquet('{FINAL_FILE}')"
        ).fetchone()[0]
        print(f"  {FINAL_FILE.name}: {fsize:.1f} MB, {f_n:,} rows")
        if f_n != n:
            print(f"  WARNING: final file has {f_n:,} rows but parts have {n:,}.")
            print("  Run --concat-only to refresh the final file.")
    else:
        print(f"  {FINAL_FILE.name} does not exist yet.")
        print("  Run --concat-only to generate it.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
