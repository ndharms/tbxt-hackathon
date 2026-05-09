"""Convert the OnePot CORE catalog (plain CSV or .csv.gz) to Parquet so
that --skip-rows resumes become essentially free.

Why this exists:
    Both gzip and CSV are inherently sequential — DuckDB has to read every
    byte from offset 0 to know where row N begins, so OFFSET on a 75 GB
    .csv.gz or a 750 GB plain CSV pays a per-resume cost proportional to
    the skip target.

    Parquet, in contrast, stores rows in row groups with metadata
    (row counts, per-column statistics). DuckDB can skip past entire row
    groups using the metadata, so OFFSET against a Parquet file is
    constant-time (in practice: milliseconds) regardless of how far you
    skip.

Run:
    uv run python scripts/downfilter_onepot/csv_to_parquet.py \\
        data/external/onepot_core.csv \\
        data/external/onepot_core.parquet

After conversion, point downfilter at the parquet:
    uv run python scripts/downfilter_onepot/downfilter_onepot.py \\
        --source data/external/onepot_core.parquet \\
        --skip-rows 930000000 \\
        --start-part 186

Disk note:
    Parquet with default Snappy compression typically lands at ~10-15% of
    the plain CSV size for SMILES + integer columns. A 750 GB CSV → likely
    ~50-100 GB Parquet. Cheaper than both your CSV and your .csv.gz.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


def fmt_duration(seconds: float) -> str:
    s = int(max(0, seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="Input CSV path (plain or .csv.gz)")
    ap.add_argument("output", help="Output Parquet path")
    ap.add_argument(
        "--row-group-size",
        type=int,
        default=1_000_000,
        help="Rows per Parquet row group (default 1,000,000). Smaller = "
             "finer-grained OFFSET skipping, more metadata overhead.",
    )
    ap.add_argument(
        "--compression",
        default="snappy",
        choices=("snappy", "zstd", "gzip", "uncompressed"),
        help="Parquet compression codec (default snappy: fast, good ratio).",
    )
    ap.add_argument(
        "--memory-limit",
        default=None,
        help="Optional DuckDB memory cap (e.g. '32GB'). Default: DuckDB's "
             "automatic ~80%% of system RAM.",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=None,
        help="DuckDB thread count for the conversion. Default: DuckDB auto. "
             "More threads = faster CSV parsing but higher peak memory.",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.is_file():
        raise SystemExit(f"Input not found: {in_path}")
    if out_path.exists():
        raise SystemExit(
            f"Output already exists: {out_path}\nDelete or move it first."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    in_size = in_path.stat().st_size
    print(f"Input            : {in_path}  ({fmt_bytes(in_size)})")
    print(f"Output           : {out_path}")
    print(f"Row group size   : {args.row_group_size:,} rows")
    print(f"Compression      : {args.compression}")
    if args.memory_limit:
        print(f"DuckDB mem limit : {args.memory_limit}")
    if args.threads:
        print(f"DuckDB threads   : {args.threads}")
    print()

    con = duckdb.connect()
    if args.memory_limit:
        con.execute(f"SET memory_limit='{args.memory_limit}';")
    if args.threads:
        con.execute(f"SET threads={args.threads};")
    con.execute("SET preserve_insertion_order=false;")
    con.execute("SET enable_progress_bar=true;")  # prints % completed to stderr

    in_sql = str(in_path).replace("'", "''")
    out_sql = str(out_path).replace("'", "''")
    sql = (
        f"COPY (SELECT * FROM read_csv_auto('{in_sql}', sample_size=10000)) "
        f"TO '{out_sql}' "
        f"(FORMAT PARQUET, "
        f" ROW_GROUP_SIZE {args.row_group_size}, "
        f" COMPRESSION '{args.compression}')"
    )
    print("Submitting COPY ... TO ... (FORMAT PARQUET). DuckDB will print a")
    print("progress bar to stderr while running.")
    print()
    print(sql)
    print()

    t0 = time.time()
    con.execute(sql)
    elapsed = time.time() - t0

    out_size = out_path.stat().st_size
    n_rows = con.execute(
        f"SELECT count(*) FROM read_parquet('{out_sql}')"
    ).fetchone()[0]
    n_groups = con.execute(
        f"SELECT count(DISTINCT row_group_id) FROM parquet_metadata('{out_sql}')"
    ).fetchone()[0]

    print()
    print(f"Done in {fmt_duration(elapsed)}.")
    print(f"  rows         : {n_rows:,}")
    print(f"  row groups   : {n_groups:,}")
    print(f"  in size      : {fmt_bytes(in_size)}")
    print(f"  out size     : {fmt_bytes(out_size)}  "
          f"({100 * out_size / max(in_size, 1):.1f}% of input)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
