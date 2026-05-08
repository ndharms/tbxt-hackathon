# downfilter_onepot

Streaming downfilter of the OnePot CORE catalog (~3.4B small molecules) into a focused virtual library of compounds that
contain at least one SGC TBXT TEP fragment substructure and pass the Chordoma Foundation chemistry guidance.

## What this does

For every row in the OnePot CSV.gz, in source-file order:

1. **Coarse SMILES `LIKE` prefilter** — a vectorized regex match against canonical aromatic and Kekulized SMILES (and
   Bemis–Murcko scaffolds) of the 37 SGC fragments. Rows that don't contain any fragment-derived substring are dropped
   before RDKit ever runs.
2. **Per-row RDKit pass** in a worker pool:
    - Parse SMILES.
    - Compute physchem descriptors.
    - PAINS check via RDKit's `FilterCatalog` (PAINS A/B/C).
    - Custom problematic-motif SMARTS (acid halide, aldehyde, diazo, azide, imine, long alkyl chain, isocyanate,
      isothiocyanate, sulfonyl halide, epoxide, anhydride, ≥3 fused benzene).
    - For every fragment in `data/structures/sgc_fragments.csv`, RDKit `HasSubstructMatch`.
3. **Hard filters** (a row must satisfy all of these to survive):
    - `MW ≤ 600`, `LogP ≤ 6`, `HBD ≤ 6`, `HBA ≤ 12`, `HBD + HBA ≤ 11`
    - `10 ≤ heavy_atoms ≤ 30`
    - `num_rings ≤ 5`, max fused-ring component size ≤ 2
    - No PAINS hits, no custom problematic-motif hits
    - At least one fragment substructure match
4. **Sharded Parquet writes** — each completed worker batch writes a survivor shard to the shard directory.
5. **Periodic rollup** — every N raw OnePot rows scanned (default 5,000,000), the script blocks until in-flight workers
   finish, concats all current shards into `part_<NNN>.parquet`, and deletes the source shards.
6. **Final concat** — at script exit (whether natural EOF, `Ctrl-C`, or `kill -INT`), remaining shards are rolled up
   into the next part file and all part files are concatenated into a single output Parquet.

## Why the LIKE prefilter is in Python rather than DuckDB

DuckDB can apply `LIKE` patterns server-side via a `WHERE` clause, which is faster — but then only matching rows reach
Python and there is no way to count *raw* OnePot rows scanned. Because the rollup cadence is specified in raw rows, the
equivalent regex is compiled in Python and applied to the streamed SMILES column with `pandas.Series.str.contains`. The
`read_csv_auto` call sets `parallel=false` to keep source-file order so that a "5M raw rows" rollup window corresponds
to a contiguous slice of the source.

## Inputs

- **Source file**: presigned URL to OnePot CORE CSV.gz on S3 (default in the script). Override with `--source`. Local
  paths also work.
- **Fragments CSV**: `data/structures/sgc_fragments.csv` (overridable). Required columns: `frag_id, site, smiles`.
  Optional: `pdb_id, vendor_code`. The downfilter splits multi-site labels on `|` (e.g. the dual-located `5QSC_F9000560`
  row is `site = "F|G"`), so a candidate hitting it gets both `F` and `G` annotated.

## Outputs

Default layout under `data/processed/onepot_focused_vl/`:

| Path                                       | Lifecycle                                              |
|--------------------------------------------|--------------------------------------------------------|
| `shard_<bid>.parquet`                      | Transient — written by workers, deleted at each rollup |
| `part_<NNN>.parquet`                       | Persistent — one per ~5M raw-row rollup window         |
| `data/processed/onepot_focused_vl.parquet` | Final concat of all parts + any straggler shards       |

Each survivor row contains: original OnePot columns, computed physchem (
`mw, logp, hbd, hba, heavy_atoms, num_rings, num_aromatic_rings, num_fused_rings, tpsa, rotatable_bonds`), per-fragment
booleans (`frag_<frag_id>`), and aggregate site annotation (`matched_sites` as a `", "`-separated unique list,
`n_fragment_hits` count).

## Usage

### Smoke test on a small slice

```bash
uv run python scripts/downfilter_onepot/downfilter_onepot.py \
    --fragments-csv data/structures/sgc_fragments.csv \
    --limit 1000000
```

Useful for measuring rate, verifying the LIKE prefilter recall, and seeing the cumulative counters. `--limit` is in
*raw* OnePot rows, so an OnePot file with selectivity ~0.1% will produce ~1k survivors at this limit.

### Schema probe

```bash
uv run python scripts/downfilter_onepot/downfilter_onepot.py --probe-only
```

Prints the detected SMILES column and the full LIKE pattern set, then exits without streaming. Use this first if the
OnePot file's schema is unknown.

### Long run (recommended invocation)

```bash
mkdir -p logs

nohup uv run python scripts/downfilter_onepot/downfilter_onepot.py \
    --fragments-csv data/structures/sgc_fragments.csv \
    > logs/downfilter_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo $! > logs/downfilter.pid
tail -f logs/downfilter_*.log
```

Stop it gracefully (so the rollup-on-exit path runs):

```bash
kill -INT $(cat logs/downfilter.pid)
```

Do not `kill -9` — that skips the final rollup and concat.

### Re-run only the merge

If the streaming half exited but the final concat didn't run (or you want to refresh the final file from existing
parts):

```bash
uv run python scripts/downfilter_onepot/downfilter_onepot.py --concat-only
```

This reads `part_*.parquet` plus any `shard_*.parquet` in the shard directory and writes the final file.

## Flags

| Flag                 | Default                                    | Notes                                                       |
|----------------------|--------------------------------------------|-------------------------------------------------------------|
| `--source`           | OnePot presigned URL                       | Any URL or local path that DuckDB's `read_csv_auto` accepts |
| `--fragments-csv`    | `data/structures/sgc_fragments.csv`        | Required columns: `frag_id, site, smiles`                   |
| `--shard-dir`        | `data/processed/onepot_focused_vl/`        | Holds shards during run, parts after rollups                |
| `--final-file`       | `data/processed/onepot_focused_vl.parquet` | Single concatenated output                                  |
| `--batch-size`       | `5000`                                     | Rows per worker batch                                       |
| `--workers`          | `cpu_count - 1`                            | Worker process count                                        |
| `--limit`            | (none)                                     | Stop after N **raw** rows scanned                           |
| `--rollup-every-raw` | `5000000`                                  | Rollup cadence in raw rows                                  |
| `--no-prefilter`     | off                                        | Sends every raw row through RDKit (very slow)               |
| `--probe-only`       | off                                        | Print schema + LIKE patterns and exit                       |
| `--concat-only`      | off                                        | Skip streaming; just merge existing parts + shards          |

## Hard-filter cutoffs

Defined as module-level constants in `downfilter_onepot.py`. Edit there if you need to tighten or loosen:

```python
MW_MAX = 600
LOGP_MAX = 6.0
HBD_MAX = 6
HBA_MAX = 12
HBD_HBA_SUM_MAX = 11
HEAVY_MIN = 10
HEAVY_MAX = 30
NUM_RINGS_MAX = 5
NUM_FUSED_RINGS_MAX = 2
```

`CUSTOM_BAD_SMARTS` is the dict of problematic-motif SMARTS patterns; any match drops the row.

## Counters and how to read them

At the end of every run the script logs a cumulative dict like:

```
Counters: {'in': 12345678, 'no_parse': 12, 'fail_physchem': 9876543, 'fail_pains': 12345, 'fail_custom': 6789, 'no_fragment_match': 234567, 'out': 12345}
```

- **`in`** — rows that passed the LIKE prefilter and reached an RDKit worker.
- **`no_parse`** — RDKit `MolFromSmiles` returned None.
- **`fail_physchem`** — failed the hard physchem cutoffs.
- **`fail_pains`** — matched at least one PAINS pattern.
- **`fail_custom`** — matched at least one custom problematic-motif SMARTS.
- **`no_fragment_match`** — passed all filters but didn't actually contain any fragment substructure (LIKE false
  positive).
- **`out`** — wrote to a survivor shard.

If `fail_physchem` dominates: cutoffs are too tight or the LIKE prefilter is admitting too many compounds outside the
lead-like envelope (typical with broad scaffold patterns).

If `no_fragment_match` is large: the LIKE prefilter is doing what it's supposed to do — yielding candidates that share
substring patterns with fragments but don't actually contain the fragment as a substructure.

## Restart and resumption

The script is not transactional. If interrupted ungracefully (`kill -9`, OOM, machine sleep), shards from in-flight
workers may be missing or incomplete, and the rollup boundary will not have advanced. To recover:

1. Inspect `data/processed/onepot_focused_vl/` — count `shard_*.parquet` and `part_*.parquet`.
2. Decide whether to keep what's there. If yes, run with `--concat-only` to merge.
3. If you want to continue scanning, you'll need to start over (the script does not currently checkpoint the source-file
   offset). For a partial restart, use `--limit` plus a fresh `--shard-dir`.

## Known limitations

The Python regex prefilter is a coarse substring filter; it does not validate ring closure, aromaticity, or
stereochemistry. It can produce false positives that RDKit then rejects. False *negatives* are also possible if OnePot's
canonical SMILES traversal differs from RDKit's (e.g. different ring-opening atom choice). To estimate recall, run a
smoke test (`--limit 100000`) and compare the regex pass rate to a sample-and-verify against the unfiltered RDKit
substructure check.

`HasSubstructMatch` against a Mol parsed from SMILES treats the fragment as a literal substructure with full atom and
bond constraints. If you want pharmacophore-style flexibility, replace the loaded fragment Mols with SMARTS queries and
update `load_fragments` accordingly.

The `ROW_NUMBER`-style raw-row counter in this script is just `len(df)` per fetched chunk. With `parallel=false` on
`read_csv_auto`, that count maps onto a contiguous source-file slice. If you ever switch to `parallel=true`, the rollup
windows lose their source-order meaning.

## Related files

- `data/structures/sgc_fragments.csv` — the 37 SGC TEP fragments at sites A, A', D, F, G (one row is dual-site `F|G`).
- `scripts/extract_sgc_fragments.py` — RCSB-driven extractor for the fragments CSV (alternative to manual fill).
- `scripts/probe_onepot.py` — schema probe / sanity check on the OnePot file.
- `docs/HACKATHON_PLAN.md` §3.3–§3.4 — the design rationale that this script implements.
