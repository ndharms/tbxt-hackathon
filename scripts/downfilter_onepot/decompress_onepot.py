"""Decompress a local .csv.gz with progress reporting.

Run:
    # default output: same path with .gz stripped
    uv run python scripts/downfilter_onepot/decompress_onepot.py \\
        data/external/onepot_core.csv.gz

    # explicit output path
    uv run python scripts/downfilter_onepot/decompress_onepot.py \\
        data/external/onepot_core.csv.gz \\
        --output data/external/onepot_core.csv

Disk note: a 75 GB gz of CSV typically expands to ~375-750 GB. Check
free space with `df -h .` before running.

Throughput: this is single-threaded gzip. On an M-series Mac you should
see ~150-300 MB/s of compressed input → ~30-60 minutes for a 75 GB input.
If you have `pigz` installed you can do the same job ~3x faster:
    pigz -dk data/external/onepot_core.csv.gz
"""
from __future__ import annotations

import argparse
import gzip
import sys
import time
from pathlib import Path

CHUNK_SIZE = 16 * 1024 * 1024  # 16 MiB; large chunks reduce Python overhead


def fmt_bytes(n: float) -> str:
    """Format a byte count as a human-readable string."""
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
    ap.add_argument("input", help="Path to .csv.gz file")
    ap.add_argument(
        "--output",
        default=None,
        help="Output path. Default: input path with .gz stripped.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    ap.add_argument(
        "--log-every-seconds",
        type=int,
        default=10,
        help="Progress log interval (default: 10s).",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        raise SystemExit(f"Input file not found: {in_path}")

    if args.output:
        out_path = Path(args.output)
    elif in_path.suffix == ".gz":
        out_path = in_path.with_suffix("")
    else:
        raise SystemExit(
            f"Cannot infer output path: input does not end in .gz ({in_path}). "
            "Pass --output explicitly."
        )

    if out_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Output already exists: {out_path}\n"
            "Pass --overwrite or delete it first."
        )

    in_size = in_path.stat().st_size
    print(f"Input  : {in_path}  ({fmt_bytes(in_size)} compressed)")
    print(f"Output : {out_path}")
    print(f"Chunk  : {fmt_bytes(CHUNK_SIZE)}")
    print()

    t0 = time.time()
    bytes_out = 0
    last_log = t0
    last_bytes_in = 0
    last_bytes_out = 0

    # Open the raw file ourselves so we can use raw.tell() to track
    # compressed-input progress; gzip.open(raw, "rb") layers the decoder.
    with (
        in_path.open("rb") as raw_in,
        gzip.open(raw_in, "rb") as gz_in,
        out_path.open("wb") as out_f,
    ):
        while True:
            chunk = gz_in.read(CHUNK_SIZE)
            if not chunk:
                break
            out_f.write(chunk)
            bytes_out += len(chunk)

            now = time.time()
            if now - last_log >= args.log_every_seconds:
                bytes_in = raw_in.tell()
                pct = 100.0 * bytes_in / in_size if in_size else 0.0
                elapsed = now - t0
                # Instantaneous rates (over the last log interval)
                inst_rate_in = (bytes_in - last_bytes_in) / max(now - last_log, 1)
                inst_rate_out = (bytes_out - last_bytes_out) / max(now - last_log, 1)
                # ETA based on compressed bytes remaining
                remaining_in = in_size - bytes_in
                eta = (
                    remaining_in / inst_rate_in
                    if inst_rate_in > 0 and remaining_in > 0
                    else 0.0
                )
                print(
                    f"  {pct:5.1f}%  "
                    f"in:{fmt_bytes(bytes_in)}/{fmt_bytes(in_size)}  "
                    f"out:{fmt_bytes(bytes_out)}  "
                    f"rate_in:{fmt_bytes(inst_rate_in)}/s  "
                    f"rate_out:{fmt_bytes(inst_rate_out)}/s  "
                    f"elapsed:{fmt_duration(elapsed)}  "
                    f"eta:{fmt_duration(eta)}"
                )
                last_log = now
                last_bytes_in = bytes_in
                last_bytes_out = bytes_out

    elapsed = time.time() - t0
    out_size = out_path.stat().st_size
    print()
    print(f"Done in {fmt_duration(elapsed)}.")
    print(f"  compressed   : {fmt_bytes(in_size)}")
    print(f"  decompressed : {fmt_bytes(out_size)}")
    if in_size:
        print(f"  ratio        : {out_size / in_size:.2f}x expansion")
    return 0


if __name__ == "__main__":
    sys.exit(main())
