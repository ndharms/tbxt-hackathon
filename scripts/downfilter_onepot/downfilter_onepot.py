"""Streaming downfilter of the OnePot CORE catalog (~3.4B rows).

Pipeline:
  1. DuckDB streams rows from the OnePot CSV.gz on S3, applying a coarse
     SMILES `LIKE` prefilter so RDKit only sees plausible fragment-bearing rows.
  2. A multiprocessing pool runs the per-row RDKit work: physchem, PAINS,
     custom problematic-motif SMARTS, and per-fragment substructure matching.
  3. Survivors of the hard filters are written as sharded Parquet (one shard
     per chunk; resumable, parallel-safe).
  4. At the end of the run shards are concatenated into a single Parquet file.

Run:
    uv add duckdb pyarrow
    uv run python scripts/downfilter_onepot/downfilter_onepot.py \
        --fragments-csv data/structures/sgc_fragments.csv

The fragments CSV must have columns: frag_id, site, smiles, pdb_id, vendor_code.
The presigned OnePot URL is the default; override with --source.

See scripts/downfilter_onepot/README.md for full documentation.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")  # silence RDKit warnings inside workers

# parents[0] = scripts/downfilter_onepot/, [1] = scripts/, [2] = repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAGMENTS_CSV = REPO_ROOT / "data" / "structures" / "sgc_fragments.csv"
DEFAULT_SHARD_DIR = REPO_ROOT / "data" / "processed" / "onepot_focused_vl"
DEFAULT_FINAL_FILE = REPO_ROOT / "data" / "processed" / "onepot_focused_vl.parquet"

DEFAULT_URL = (
    "https://onepot-clients.s3.amazonaws.com/shared/v1p1/core_v1p1.csv.gz"
    "?AWSAccessKeyId=AKIA2XKLIHG76EKRIH7S"
    "&Signature=s%2Boj0hJH%2ByKeqeIvf6kPZ2x1uqU%3D"
    "&Expires=1778693935"
)

# ---------- Hard-filter cutoffs (per HACKATHON_PLAN.md §3.4) ----------
MW_MAX = 600
LOGP_MAX = 6.0
HBD_MAX = 6
HBA_MAX = 12
HBD_HBA_SUM_MAX = 11
HEAVY_MIN = 10
HEAVY_MAX = 30
NUM_RINGS_MAX = 5
NUM_FUSED_RINGS_MAX = 2

# ---------- Custom problematic motifs (annotated, then used to drop) ----------
# Names match the brief's "avoid" list. PAINS_A/B/C are added separately via
# RDKit's FilterCatalog.
CUSTOM_BAD_SMARTS: dict[str, str] = {
    "acid_halide": "[CX3](=O)[F,Cl,Br,I]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "diazo": "[#6]=[N+]=[N-]",
    "azide": "[N-]=[N+]=[N-]",
    "imine": "[CX3]=[NX2;!$(N=*);!$(N-O)][#6]",
    # 6 contiguous sp3 CH2 = a "long alkyl chain"
    "long_alkyl_chain": "[CH2;X4]~[CH2;X4]~[CH2;X4]~[CH2;X4]~[CH2;X4]~[CH2;X4]",
    "isocyanate": "[NX2]=[CX2]=[OX1]",
    "isothiocyanate": "[NX2]=[CX2]=[SX1]",
    "sulfonyl_halide": "[SX4](=O)(=O)[F,Cl,Br,I]",
    "epoxide": "C1OC1",
    "anhydride": "[CX3](=O)O[CX3](=O)",
    # 3+ fused benzene rings (anthracene / phenanthrene-like)
    "fused_3plus_benzene": "c1ccc2cc3ccccc3cc2c1",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("downfilter")


# ============================================================
# Fragment loading + LIKE-pattern generation (main process)
# ============================================================

@dataclass
class Fragment:
    frag_id: str
    site: str
    pdb_id: str
    smiles: str  # original
    mol: Chem.Mol


def load_fragments(csv_path: Path) -> list[Fragment]:
    df = pd.read_csv(csv_path)
    required = {"frag_id", "site", "smiles"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{csv_path} missing required columns: {missing}")
    out: list[Fragment] = []
    for _, row in df.iterrows():
        smi = str(row["smiles"]).strip()
        if not smi:
            print(f"  [skip] {row['frag_id']}: empty SMILES", file=sys.stderr)
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise SystemExit(f"Cannot parse fragment SMILES for {row['frag_id']}: {smi}")
        out.append(
            Fragment(
                frag_id=str(row["frag_id"]),
                site=str(row["site"]),
                pdb_id=str(row.get("pdb_id", "") or ""),
                smiles=smi,
                mol=mol,
            )
        )
    assert len(out) > 0, "No fragments loaded"
    return out


def fragment_like_patterns(fragments: list[Fragment]) -> list[str]:
    """Generate canonical SMILES substring patterns for the DuckDB LIKE prefilter.

    For each fragment we emit:
      - canonical aromatic SMILES,
      - canonical Kekulized SMILES (uppercase aromatic atoms),
      - same two for the Bemis-Murcko scaffold.

    Deduplicated. Patterns shorter than 4 chars are dropped (too noisy).
    """
    seen: set[str] = set()
    out: list[str] = []

    def emit(mol: Chem.Mol) -> None:
        if mol is None or mol.GetNumAtoms() == 0:
            return
        # aromatic form
        try:
            s = Chem.MolToSmiles(mol, canonical=True)
            if s and len(s) >= 4 and s not in seen:
                seen.add(s)
                out.append(s)
        except Exception:
            pass
        # kekulized form
        try:
            m2 = Chem.Mol(mol)
            Chem.Kekulize(m2, clearAromaticFlags=True)
            s = Chem.MolToSmiles(m2, canonical=True, kekuleSmiles=True)
            if s and len(s) >= 4 and s not in seen:
                seen.add(s)
                out.append(s)
        except Exception:
            pass

    for f in fragments:
        emit(f.mol)
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(f.mol)
        except Exception:
            scaffold = None
        if scaffold and scaffold.GetNumAtoms() > 0:
            emit(scaffold)
    return out


def build_prefilter_sql(smiles_col: str, like_patterns: list[str]) -> str:
    """SQL form of the LIKE filter (kept for --probe-only display only).

    The streaming pipeline applies the equivalent regex in Python (see
    `compile_prefilter_re`) so that raw OnePot rows can be counted as they
    pass through DuckDB.
    """
    if not like_patterns:
        return "TRUE"
    or_clauses = []
    for p in like_patterns:
        p_sql = p.replace("'", "''")
        or_clauses.append(f"\"{smiles_col}\" LIKE '%{p_sql}%'")
    return "(" + " OR ".join(or_clauses) + ")"


def compile_prefilter_re(like_patterns: list[str]) -> re.Pattern | None:
    """Compile the LIKE patterns into a single OR'd regex for vectorized
    pandas .str.contains().

    Returns None if no patterns (i.e. --no-prefilter)."""
    if not like_patterns:
        return None
    return re.compile("|".join(re.escape(p) for p in like_patterns))


def detect_smiles_column(con: duckdb.DuckDBPyConnection, source: str) -> str:
    desc = con.sql(
        f"DESCRIBE SELECT * FROM read_csv_auto('{source}', sample_size=10000)"
    ).fetchall()
    cols = [r[0] for r in desc]
    candidates = [c for c in cols if "smi" in c.lower()]
    if not candidates:
        raise SystemExit(f"No SMILES column found. Columns present: {cols}")
    if len(candidates) > 1:
        log.warning("Multiple SMILES candidates %s — using %s", candidates, candidates[0])
    return candidates[0]


# ============================================================
# Worker side
# ============================================================

# Globals populated by _init_worker (one set per worker process).
_W_FRAGMENTS: list[Fragment] | None = None
_W_PAINS: FilterCatalog | None = None
_W_CUSTOM: dict[str, Chem.Mol] | None = None
_W_SMILES_COL: str | None = None


def _init_worker(fragments_csv: str, smiles_col: str) -> None:
    global _W_FRAGMENTS, _W_PAINS, _W_CUSTOM, _W_SMILES_COL
    _W_FRAGMENTS = load_fragments(Path(fragments_csv))
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)  # PAINS A/B/C
    _W_PAINS = FilterCatalog(params)
    _W_CUSTOM = {name: Chem.MolFromSmarts(s) for name, s in CUSTOM_BAD_SMARTS.items()}
    _W_SMILES_COL = smiles_col


def _physchem(mol: Chem.Mol) -> dict:
    return {
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Descriptors.MolLogP(mol)),
        "hbd": int(rdMolDescriptors.CalcNumHBD(mol)),
        "hba": int(rdMolDescriptors.CalcNumHBA(mol)),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "num_rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "num_aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "num_fused_rings": _max_fused_ring_system(mol),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
    }


def _max_fused_ring_system(mol: Chem.Mol) -> int:
    """Largest fused-ring component size. Two rings are fused if they share >=2 atoms."""
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    if not rings:
        return 0
    n = len(rings)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if len(rings[i] & rings[j]) >= 2:
                union(i, j)
    counts: dict[int, int] = {}
    for i in range(n):
        r = find(i)
        counts[r] = counts.get(r, 0) + 1
    return max(counts.values())


def _passes_hard_filters(pc: dict) -> bool:
    return (
            pc["mw"] <= MW_MAX
            and pc["logp"] <= LOGP_MAX
            and pc["hbd"] <= HBD_MAX
            and pc["hba"] <= HBA_MAX
            and pc["hbd"] + pc["hba"] <= HBD_HBA_SUM_MAX
            and HEAVY_MIN <= pc["heavy_atoms"] <= HEAVY_MAX
            and pc["num_rings"] <= NUM_RINGS_MAX
            and pc["num_fused_rings"] <= NUM_FUSED_RINGS_MAX
    )


def _process_chunk(payload: tuple[int, list[dict]]) -> tuple[int, pd.DataFrame, dict]:
    """Per-batch worker: filter + annotate. Returns (batch_id, survivors_df, stats)."""
    batch_id, rows = payload
    smiles_col = _W_SMILES_COL
    fragments = _W_FRAGMENTS
    pains_cat = _W_PAINS
    custom = _W_CUSTOM

    survivors: list[dict] = []
    stats = {"in": len(rows), "no_parse": 0, "fail_physchem": 0, "fail_pains": 0,
             "fail_custom": 0, "no_fragment_match": 0, "out": 0}

    for row in rows:
        smi = row.get(smiles_col)
        if not isinstance(smi, str) or not smi:
            stats["no_parse"] += 1
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            stats["no_parse"] += 1
            continue

        pc = _physchem(mol)
        if not _passes_hard_filters(pc):
            stats["fail_physchem"] += 1
            continue

        if pains_cat.HasMatch(mol):
            stats["fail_pains"] += 1
            continue

        # custom motifs — drop if any hit
        custom_hits = {f"bad_{name}": bool(mol.HasSubstructMatch(patt))
                       for name, patt in custom.items()}
        if any(custom_hits.values()):
            stats["fail_custom"] += 1
            continue

        # fragment substructure annotation
        any_hit = False
        sites_hit: set[str] = set()
        frag_cols: dict[str, bool] = {}
        for f in fragments:
            h = mol.HasSubstructMatch(f.mol)
            frag_cols[f"frag_{f.frag_id}"] = h
            if h:
                any_hit = True
                # split on '|' so a dual-located fragment annotates both sites
                for s in f.site.split("|"):
                    s = s.strip()
                    if s:
                        sites_hit.add(s)

        if not any_hit:
            stats["no_fragment_match"] += 1
            continue

        out_row = dict(row)  # original cols
        out_row.update(pc)
        # bad_* columns are dropped: by construction they are always False
        # for survivors (any True hit causes the row to fail upstream).
        out_row.update(frag_cols)
        out_row["matched_sites"] = ", ".join(sorted(sites_hit))
        out_row["n_fragment_hits"] = sum(frag_cols.values())
        survivors.append(out_row)
        stats["out"] += 1

    df = pd.DataFrame(survivors) if survivors else pd.DataFrame()
    return batch_id, df, stats


# ============================================================
# Main streaming driver
# ============================================================

def write_shard(shard_dir: Path, batch_id: int, df: pd.DataFrame) -> Path:
    path = shard_dir / f"shard_{batch_id:08d}.parquet"
    df.to_parquet(path, index=False)
    return path


def rollup_to_part(shard_dir: Path, part_idx: int) -> int:
    """Concat all current shard_*.parquet into part_<idx:03d>.parquet, delete shards.

    Returns the number of rows written to the new part file (0 if no shards).
    """
    shard_paths = sorted(shard_dir.glob("shard_*.parquet"))
    if not shard_paths:
        return 0
    part_path = shard_dir / f"part_{part_idx:03d}.parquet"
    log.info("Rolling up %d shards -> %s", len(shard_paths), part_path.name)
    cat_con = duckdb.connect()
    glob_path = str(shard_dir / "shard_*.parquet").replace("'", "''")
    out_path = str(part_path).replace("'", "''")
    cat_con.execute(
        f"COPY (SELECT * FROM read_parquet('{glob_path}')) "
        f"TO '{out_path}' (FORMAT PARQUET);"
    )
    n = cat_con.execute(
        f"SELECT count(*) FROM read_parquet('{out_path}')"
    ).fetchone()[0]
    log.info("  part_%03d.parquet: %d rows", part_idx, n)
    for p in shard_paths:
        p.unlink()
    return int(n)


def concat_to_final(shard_dir: Path, final_file: Path) -> int | None:
    """Concat all part_*.parquet plus any straggler shard_*.parquet into final_file."""
    parts = sorted(shard_dir.glob("part_*.parquet"))
    stragglers = sorted(shard_dir.glob("shard_*.parquet"))
    if not parts and not stragglers:
        log.warning("No parts or shards in %s; nothing to concat.", shard_dir)
        return None
    log.info(
        "Final concat: %d part files + %d straggler shards -> %s",
        len(parts), len(stragglers), final_file,
    )
    con = duckdb.connect()
    sources: list[str] = []
    if parts:
        glob = str(shard_dir / "part_*.parquet").replace("'", "''")
        sources.append(f"SELECT * FROM read_parquet('{glob}')")
    if stragglers:
        glob = str(shard_dir / "shard_*.parquet").replace("'", "''")
        sources.append(f"SELECT * FROM read_parquet('{glob}')")
    union = " UNION ALL ".join(sources)
    out_path = str(final_file).replace("'", "''")
    con.execute(f"COPY ({union}) TO '{out_path}' (FORMAT PARQUET);")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out_path}')").fetchone()[0]
    log.info("Final file: %s (%d rows)", final_file, n)
    return int(n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=DEFAULT_URL,
                    help="OnePot CSV.gz URL or local path. Default = presigned URL.")
    ap.add_argument("--fragments-csv", default=str(DEFAULT_FRAGMENTS_CSV),
                    help="CSV with columns frag_id,site,smiles,pdb_id,vendor_code.")
    ap.add_argument("--shard-dir", default=str(DEFAULT_SHARD_DIR),
                    help="Directory for sharded + part Parquet output.")
    ap.add_argument("--final-file", default=str(DEFAULT_FINAL_FILE),
                    help="Final concatenated Parquet path.")
    ap.add_argument("--batch-size", type=int, default=5000,
                    help="Rows per worker batch.")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 1),
                    help="Worker process count.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after N RAW OnePot rows scanned (for smoke tests).")
    ap.add_argument("--rollup-every-raw", type=int, default=5_000_000,
                    help="Roll up shards into a part file every N raw rows scanned.")
    ap.add_argument("--no-prefilter", action="store_true",
                    help="Disable LIKE prefilter (sends every row through RDKit).")
    ap.add_argument("--probe-only", action="store_true",
                    help="Print schema + LIKE pattern set, then exit.")
    ap.add_argument("--concat-only", action="store_true",
                    help="Skip streaming; only concat existing parts + shards.")
    args = ap.parse_args()

    shard_dir = Path(args.shard_dir)
    final_file = Path(args.final_file)
    shard_dir.mkdir(parents=True, exist_ok=True)
    final_file.parent.mkdir(parents=True, exist_ok=True)

    if args.concat_only:
        concat_to_final(shard_dir, final_file)
        return 0

    fragments_csv = Path(args.fragments_csv)
    if not fragments_csv.exists():
        raise SystemExit(
            f"Fragments CSV not found: {fragments_csv}\n"
            "Create it with columns: frag_id, site, smiles, pdb_id, vendor_code"
        )

    fragments = load_fragments(fragments_csv)
    sites_present = sorted({f.site for f in fragments})
    log.info("Loaded %d fragments across sites %s", len(fragments), sites_present)

    like_patterns = fragment_like_patterns(fragments) if not args.no_prefilter else []
    log.info("Generated %d LIKE prefilter patterns", len(like_patterns))
    like_re = compile_prefilter_re(like_patterns)

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET preserve_insertion_order=false;")

    smiles_col = detect_smiles_column(con, args.source)
    log.info("Detected SMILES column: %s", smiles_col)

    if args.probe_only:
        print(f"\nSMILES column: {smiles_col}")
        print(f"\nLIKE patterns ({len(like_patterns)}):")
        for p in like_patterns:
            print(f"  {p}")
        sql_form = build_prefilter_sql(smiles_col, like_patterns)
        print(f"\nSQL-form WHERE clause length: {len(sql_form)} chars (display only)")
        return 0

    # No WHERE in SQL: we filter in Python so raw rows are visible to the counter.
    # parallel=false keeps source-order streaming — useful for predictable rollups.
    sql = (
        f"SELECT * FROM read_csv_auto('{args.source}', sample_size=10000, "
        f"parallel=false)"
    )
    if args.limit:
        sql += f" LIMIT {args.limit}"
    rel = con.execute(sql)

    log.info(
        "Starting pool: workers=%d, batch=%d, rollup_every_raw=%s",
        args.workers, args.batch_size, f"{args.rollup_every_raw:,}",
    )
    t0 = time.time()
    raw_rows_scanned = 0
    prefilter_passing = 0
    survivors_total = 0
    shards_written = 0
    parts_written = 0
    cumulative = {k: 0 for k in
                  ("in", "no_parse", "fail_physchem", "fail_pains",
                   "fail_custom", "no_fragment_match", "out")}
    next_rollup = args.rollup_every_raw
    last_log_raw = 0

    bid = 0
    in_flight: set = set()
    max_in_flight = args.workers * 4

    def drain_done(blocking: bool) -> None:
        nonlocal in_flight, shards_written, survivors_total
        if not in_flight:
            return
        done, pending = wait(
            in_flight,
            timeout=None if blocking else 0,
            return_when=FIRST_COMPLETED,
        )
        in_flight = pending
        for fut in done:
            bid_done, sdf, st = fut.result()
            for k, v in st.items():
                cumulative[k] += v
            if not sdf.empty:
                write_shard(shard_dir, bid_done, sdf)
                survivors_total += len(sdf)
                shards_written += 1

    def do_rollup() -> None:
        nonlocal parts_written
        # drain any in-flight workers so all shards for this window are on disk
        while in_flight:
            drain_done(blocking=True)
        n = rollup_to_part(shard_dir, parts_written)
        if n > 0:
            parts_written += 1

    interrupted = False
    try:
        with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_init_worker,
                initargs=(str(fragments_csv), smiles_col),
        ) as ex:
            while True:
                df = rel.fetch_df_chunk()
                if df is None or len(df) == 0:
                    break

                raw_rows_scanned += len(df)

                # Python-side LIKE filter (vectorized regex on the SMILES column)
                if like_re is not None:
                    smi = df[smiles_col].astype(str)
                    keep_mask = smi.str.contains(like_re, regex=True, na=False)
                    df_keep = df[keep_mask]
                else:
                    df_keep = df

                prefilter_passing += len(df_keep)

                # Dispatch survivors to workers in batch_size chunks
                if not df_keep.empty:
                    for start in range(0, len(df_keep), args.batch_size):
                        while len(in_flight) >= max_in_flight:
                            drain_done(blocking=True)
                        sub = df_keep.iloc[start: start + args.batch_size]
                        in_flight.add(ex.submit(
                            _process_chunk, (bid, sub.to_dict("records"))
                        ))
                        bid += 1

                # Periodic non-blocking drain + stats every ~1M raw rows
                if raw_rows_scanned - last_log_raw >= 1_000_000:
                    drain_done(blocking=False)
                    elapsed = time.time() - t0
                    log.info(
                        "raw=%s prefiltered=%s survivors=%s shards=%d parts=%d "
                        "rate=%.0f raw/s",
                        f"{raw_rows_scanned:,}", f"{prefilter_passing:,}",
                        f"{survivors_total:,}", shards_written, parts_written,
                        raw_rows_scanned / max(elapsed, 1),
                    )
                    last_log_raw = raw_rows_scanned

                # Rollup trigger: handle backlog with a while loop in case a
                # single chunk vaulted past multiple thresholds (won't happen in
                # practice but cheap to guard).
                while raw_rows_scanned >= next_rollup:
                    do_rollup()
                    next_rollup += args.rollup_every_raw

            # source exhausted; drain remaining workers
            while in_flight:
                drain_done(blocking=True)

    except KeyboardInterrupt:
        interrupted = True
        log.warning("Interrupted by user; rolling up remaining shards before exit.")
        try:
            while in_flight:
                drain_done(blocking=True)
        except Exception as e:  # noqa: BLE001
            log.warning("drain after interrupt failed: %s", e)

    # Final straggler rollup (whatever didn't trigger a rollup boundary)
    do_rollup()

    elapsed = time.time() - t0
    log.info(
        "Streaming done in %.1fs. raw=%s prefiltered=%s survivors=%s "
        "shards=%d parts=%d",
        elapsed, f"{raw_rows_scanned:,}", f"{prefilter_passing:,}",
        f"{survivors_total:,}", shards_written, parts_written,
    )
    log.info("Counters: %s", cumulative)

    # Concat all parts (+ any straggler shards) into the final file.
    concat_to_final(shard_dir, final_file)

    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
