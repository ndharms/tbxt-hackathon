"""Extract SGC TBXT fragment SMILES from RCSB and write data/structures/sgc_fragments.csv.

Source: SGC TBXT TEP datasheet v1.0, Table 2.
Scope: pockets A, A', D, F, G (37 rows; F9000560 is dual-located F|G).

For each PDB entry the script:
  1. Fetches the entry record from data.rcsb.org REST.
  2. Iterates non-polymer entities, skipping crystallographic additives
     (water, Cd2+, EDO, GOL, DMS, PEG, etc.).
  3. Picks the candidate ligand with the most heavy atoms (the bound fragment).
  4. Reads SMILES from the chem-comp record (SMILES_CANONICAL preferred,
     falling back to other SMILES descriptors).

Run:
    uv run python scripts/extract_sgc_fragments.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "data" / "structures" / "sgc_fragments.csv"

# (PDB ID, vendor ligand code from Table 2, site)
ENTRIES: list[tuple[str, str, str]] = [
    # Pocket A (21)
    ("5QRF", "F9000532", "A"),
    ("5QRG", "XS115742", "A"),
    ("5QRH", "FM001763", "A"),
    ("5QRI", "F9000380", "A"),
    ("5QRJ", "F9000536", "A"),
    ("5QRK", "FM010104", "A"),
    ("5QRL", "F9000949", "A"),
    ("5QRN", "F9000950", "A"),
    ("5QRO", "F9000403", "A"),
    ("5QRP", "F9000414", "A"),
    ("5QRQ", "FM002138", "A"),
    ("5QRS", "F9000441", "A"),
    ("5QRT", "FM010020", "A"),
    ("5QRV", "FM002272", "A"),
    ("5QRX", "F9000951", "A"),
    ("5QRY", "FM010072", "A"),
    ("5QRZ", "F9000591", "A"),
    ("5QS2", "FM001886", "A"),
    ("5QS3", "FM002333", "A"),
    ("5QS4", "UB000200", "A"),
    ("5QS5", "FM002032", "A"),
    # Pocket A' (2)
    ("5QS9", "FM002076", "A'"),
    ("5QSD", "FM002038", "A'"),
    # Pocket D (1)
    ("5QS0", "FM010026", "D"),
    # Pocket F (3 — F9000560 listed separately as F|G below)
    ("5QSA", "FM001580", "F"),
    ("5QSI", "FM001452", "F"),
    ("5QSK", "FM002150", "F"),
    # Pocket G (9 — F9000560 listed separately as F|G below)
    ("5QS6", "FM010013", "G"),
    ("5QS8", "F9000511", "G"),
    ("5QSB", "XS022802", "G"),
    ("5QSE", "F9000674", "G"),
    ("5QSF", "DA000167", "G"),
    ("5QSG", "F9000710", "G"),
    ("5QT0", "XS092188", "G"),
    ("5QSH", "F9000416", "G"),
    ("5QSJ", "FM002214", "G"),
    # Dual location in 5QSC: G (loc 1) + F (loc 2). One row per chemistry.
    ("5QSC", "F9000560", "F|G"),
]

SKIP_CCD = {
    # waters / monoatomic ions
    "HOH", "DOD",
    "CD", "ZN", "MG", "CA", "NA", "K", "MN", "FE", "NI", "CU", "CO",
    "CL", "BR", "IOD", "IODIDE", "F",
    # common crystallization & cryo additives
    "EDO", "GOL", "PEG", "PG4", "PGE", "1PE", "P6G", "MPD", "IPA",
    "DMS", "DMSO",
    "ACT", "SO4", "PO4", "NO3", "PO3", "FMT", "EOH", "MOH",
    "TRS", "BME", "DTT", "IMD", "TMA",
}

USER_AGENT = "TBXT-hackathon-fragment-extractor/1.0"
RCSB_BASE = "https://data.rcsb.org/rest/v1/core"


def fetch_json(url: str, retries: int = 3) -> dict | None:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urlopen(req, timeout=20) as resp:
                return json.load(resp)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(1.2 ** attempt)
    print(f"  [warn] fetch failed: {url} -- {last_err}", file=sys.stderr)
    return None


def smiles_for_pdb(pdb_id: str) -> tuple[str | None, str | None]:
    pdb = pdb_id.lower()
    entry = fetch_json(f"{RCSB_BASE}/entry/{pdb}")
    if not entry:
        return None, None

    nonpoly_ids = (
        entry.get("rcsb_entry_container_identifiers", {})
             .get("non_polymer_entity_ids", [])
    )
    if not nonpoly_ids:
        return None, None

    candidates: list[tuple[str, int, dict]] = []
    for eid in nonpoly_ids:
        np = fetch_json(f"{RCSB_BASE}/nonpolymer_entity/{pdb}/{eid}")
        if not np:
            continue
        ccd = (
            np.get("pdbx_entity_nonpoly", {}).get("comp_id")
            or np.get("rcsb_nonpolymer_entity_container_identifiers", {})
                 .get("nonpolymer_comp_id")
        )
        if not ccd or ccd.upper() in SKIP_CCD:
            continue

        cc = fetch_json(f"{RCSB_BASE}/chemcomp/{ccd}")
        if not cc:
            continue
        n_heavy = (
            cc.get("rcsb_chem_comp_info", {}).get("atom_count_heavy")
            or cc.get("chem_comp", {}).get("formula_weight", 0) // 12  # rough fallback
            or 0
        )
        if n_heavy and n_heavy < 4:
            continue
        candidates.append((ccd, int(n_heavy or 0), cc))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[1], reverse=True)
    ccd, _, cc = candidates[0]

    # extract SMILES — preferred order: SMILES_CANONICAL, then SMILES, then rcsb fields
    descriptors = cc.get("pdbx_chem_comp_descriptor", []) or []
    smiles = None
    for d in descriptors:
        if d.get("type", "").upper() == "SMILES_CANONICAL":
            smiles = d.get("descriptor")
            break
    if not smiles:
        for d in descriptors:
            if d.get("type", "").upper().startswith("SMILES"):
                smiles = d.get("descriptor")
                break
    if not smiles:
        rcsb_desc = cc.get("rcsb_chem_comp_descriptor", {}) or {}
        smiles = rcsb_desc.get("smiles_stereo") or rcsb_desc.get("smiles")
    return smiles, ccd


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    failed: list[tuple[str, str]] = []

    for i, (pdb, vendor, site) in enumerate(ENTRIES, 1):
        print(f"[{i:2d}/{len(ENTRIES)}] {pdb} {vendor} (site {site}) ...", end=" ", flush=True)
        smi, ccd = smiles_for_pdb(pdb)
        if not smi:
            failed.append((pdb, vendor))
            print("BLANK")
        else:
            print(f"CCD={ccd}  SMILES={smi[:80]}{'...' if len(smi) > 80 else ''}")
        rows.append({
            "frag_id": f"{pdb}_{vendor}",
            "site": site,
            "smiles": smi or "",
            "pdb_id": pdb,
            "vendor_code": vendor,
            "ccd_code": ccd or "",
        })

    with OUTPUT.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["frag_id", "site", "smiles", "pdb_id", "vendor_code", "ccd_code"],
        )
        w.writeheader()
        w.writerows(rows)

    n_ok = len(rows) - len(failed)
    print(f"\nWrote {OUTPUT}  ({len(rows)} rows; {n_ok} with SMILES, {len(failed)} blank)")
    if failed:
        print("Blank SMILES for:")
        for pdb, vendor in failed:
            print(f"  {pdb}  {vendor}  -- visit https://www.rcsb.org/structure/{pdb} and fill in manually")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
