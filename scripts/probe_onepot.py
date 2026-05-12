"""Probe the OnePot CORE catalog schema via DuckDB.

Run:
    uv add duckdb              # one-time
    uv run python scripts/probe_onepot.py <presigned-url>

Writes findings to docs/onepot_schema_notes.md so they can be referenced
when the focused-VL pipeline is built. Does not download the file.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTES = REPO_ROOT / "docs" / "onepot_schema_notes.md"


def heading(s: str) -> None:
    print("\n" + "=" * 78 + f"\n {s}\n" + "=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe OnePot CORE catalog schema via DuckDB")
    parser.add_argument("url", help="Presigned S3 URL for core_v1p1.csv.gz")
    args = parser.parse_args()
    URL = args.url

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET enable_progress_bar=true;")

    out: list[str] = []
    out.append("# OnePot CORE catalog — schema probe\n")
    out.append(f"_Probed: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}_\n")
    out.append(f"_Source: `{URL.split('?')[0]}` (presigned)_\n")

    # ------------------------------------------------------------------
    # 1. Schema (column names + types)
    # ------------------------------------------------------------------
    heading("1. DESCRIBE (column names + inferred types)")
    desc_sql = f"""
        DESCRIBE
        SELECT * FROM read_csv_auto('{URL}', sample_size=10000)
    """
    desc = con.sql(desc_sql).fetchall()
    cols = [r[0] for r in desc]
    print(f"{len(cols)} columns:")
    for r in desc:
        print(f"  {r[0]:<30} {r[1]}")

    out.append("## Schema\n")
    out.append("| column | type |\n|---|---|")
    for r in desc:
        out.append(f"| `{r[0]}` | {r[1]} |")
    out.append("")

    # ------------------------------------------------------------------
    # 2. SMILES column auto-detect
    # ------------------------------------------------------------------
    smiles_candidates = [c for c in cols if "smi" in c.lower()]
    id_candidates = [c for c in cols if c.lower() in ("id", "compound_id", "onepot_id", "cpd_id")]
    physchem_present = sorted(
        c for c in cols
        if c.lower() in {"mw", "molwt", "molecular_weight", "logp", "clogp",
                         "hbd", "hba", "h_donors", "h_acceptors",
                         "heavy_atoms", "num_heavy_atoms", "tpsa", "rotbonds"}
    )

    heading("2. Auto-detected key columns")
    print(f"  SMILES candidates : {smiles_candidates}")
    print(f"  ID candidates     : {id_candidates}")
    print(f"  Physchem columns  : {physchem_present}")

    out.append("## Key columns\n")
    out.append(f"- SMILES candidate(s): {', '.join(f'`{c}`' for c in smiles_candidates) or '_none found_'}")
    out.append(f"- ID candidate(s): {', '.join(f'`{c}`' for c in id_candidates) or '_none found_'}")
    out.append(
        f"- Precomputed physchem: {', '.join(f'`{c}`' for c in physchem_present) or '_none — must compute with RDKit_'}")
    out.append("")

    # ------------------------------------------------------------------
    # 3. Sample rows
    # ------------------------------------------------------------------
    heading("3. First 5 rows")
    sample_sql = f"SELECT * FROM read_csv_auto('{URL}', sample_size=10000) LIMIT 5"
    sample = con.sql(sample_sql).df()
    print(sample.to_string(max_colwidth=80))

    out.append("## Sample rows (first 5)\n")
    out.append("```")
    out.append(sample.to_string(max_colwidth=80))
    out.append("```\n")

    # ------------------------------------------------------------------
    # 4. Row count (may take a while; bail on timeout)
    # ------------------------------------------------------------------
    heading("4. Row count")
    print("(this streams the whole gzip; may take several minutes — Ctrl-C to skip)")
    t0 = time.time()
    try:
        n = con.sql(f"SELECT count(*) AS n FROM read_csv_auto('{URL}')").fetchone()[0]
        elapsed = time.time() - t0
        print(f"  rows = {n:,}  (took {elapsed:.1f}s)")
        out.append(f"## Row count\n\n- Total rows: **{n:,}**\n- Probe wall time: {elapsed:.1f}s\n")
    except KeyboardInterrupt:
        out.append("## Row count\n\n_Skipped (probe interrupted)._\n")
        print("  skipped")

    # ------------------------------------------------------------------
    # 5. Quick sanity: do all sampled SMILES parse?
    # ------------------------------------------------------------------
    if smiles_candidates:
        smiles_col = smiles_candidates[0]
        heading(f"5. RDKit parse check on 1,000 sampled SMILES from `{smiles_col}`")
        try:
            from rdkit import Chem
            samp = con.sql(
                f"SELECT \"{smiles_col}\" AS smi FROM read_csv_auto('{URL}', sample_size=10000) "
                f"USING SAMPLE 1000 ROWS"
            ).df()
            mols = [Chem.MolFromSmiles(s) if isinstance(s, str) else None for s in samp["smi"]]
            ok = sum(m is not None for m in mols)
            pct = 100.0 * ok / len(mols)
            print(f"  parsed {ok}/{len(mols)} ({pct:.1f}%)")
            out.append(
                f"## RDKit parse check\n\n"
                f"- Parsed {ok}/{len(mols)} ({pct:.1f}%) of a 1,000-row sample from `{smiles_col}`.\n"
            )
        except Exception as e:  # noqa: BLE001
            print(f"  RDKit check skipped: {e}")
            out.append(f"## RDKit parse check\n\n_Skipped: {e}_\n")

    # ------------------------------------------------------------------
    # Persist notes
    # ------------------------------------------------------------------
    NOTES.parent.mkdir(parents=True, exist_ok=True)
    NOTES.write_text("\n".join(out))
    print(f"\nNotes written to {NOTES.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
