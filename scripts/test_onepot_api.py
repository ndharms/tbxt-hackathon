"""Test OnePot API: single-fragment nearest-neighbor query.

Validates connectivity, response structure, and credit usage before
running a full batch of all fragments.

Usage:
    export ONEPOT_API_KEY="your-key-here"
    uv run python scripts/test_onepot_api.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

from onepot import Client

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAGMENTS_CSV = REPO_ROOT / "data" / "structures" / "sgc_fragments.csv"


def load_fragments(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0, f"No rows in {csv_path}"
    return rows


def main() -> int:
    api_key = os.environ.get("ONEPOT_API_KEY")
    if not api_key:
        print("ERROR: Set ONEPOT_API_KEY environment variable first.")
        print("  export ONEPOT_API_KEY='your-key-here'")
        return 1

    client = Client(api_key=api_key)
    fragments = load_fragments(FRAGMENTS_CSV)
    print(f"Loaded {len(fragments)} fragments from {FRAGMENTS_CSV.name}")

    test_frag = fragments[0]
    test_smiles = test_frag["smiles"]
    print(f"\nTest query: pocket={test_frag['pocket']}  pdb={test_frag['pdb_id']}  "
          f"ligand={test_frag['ligand_id']}")
    print(f"  SMILES: {test_smiles}")

    # --- Single-molecule search: small max_results first ---
    print("\n--- Test 1: max_results=5 (sanity check) ---")
    resp_small = client.search(
        smiles_list=[test_smiles],
        max_results=5,
    )
    print(f"Credits used: {resp_small.get('credits_used')}")
    print(f"Credits remaining: {resp_small.get('credits_remaining')}")

    queries = resp_small.get("queries", [])
    assert len(queries) == 1, f"Expected 1 query result, got {len(queries)}"

    results = queries[0].get("results", [])
    print(f"Results returned: {len(results)}")

    if results:
        print("\nTop 5 neighbors:")
        print(f"  {'Rank':<5} {'Similarity':<12} {'Price($)':<10} {'SMILES'}")
        for i, r in enumerate(results, 1):
            print(f"  {i:<5} {r.get('similarity', 'N/A'):<12} "
                  f"{r.get('price_usd', 'N/A'):<10} {r.get('smiles', 'N/A')[:80]}")

    # --- Test with max_results=1000 on a single fragment ---
    print("\n--- Test 2: max_results=1000 (target scale) ---")
    resp_large = client.search(
        smiles_list=[test_smiles],
        max_results=1000,
    )
    large_results = resp_large.get("queries", [{}])[0].get("results", [])
    print(f"Results returned: {len(large_results)}")
    print(f"Credits used: {resp_large.get('credits_used')}")
    print(f"Credits remaining: {resp_large.get('credits_remaining')}")

    if large_results:
        sims = [r["similarity"] for r in large_results if "similarity" in r]
        print(f"Similarity range: {min(sims):.4f} – {max(sims):.4f}")

    # --- Save test output for inspection ---
    out_path = REPO_ROOT / "data" / "onepot_test_response.json"
    with open(out_path, "w") as f:
        json.dump(resp_large, f, indent=2)
    print(f"\nFull response saved to {out_path.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
