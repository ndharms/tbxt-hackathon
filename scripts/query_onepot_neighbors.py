"""Query OnePot for nearest neighbors per SGC fragment, compute ECFP4, and visualize.

Pocket A fragments: 1000 neighbors each.
Pocket B/C/D fragments: 2000 neighbors each.

Outputs:
  - data/onepot_neighbors_all_matches.csv   (all pocket matches preserved)
  - data/onepot_neighbors_assigned.csv      (one row per compound, best-pocket assigned)
  - figures/onepot_pacmap_by_pocket.png     (PaCMAP colored by assigned pocket)

Usage:
    export ONEPOT_API_KEY="your-key-here"
    uv run python scripts/query_onepot_neighbors.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
import pacmap
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAGMENTS_CSV = REPO_ROOT / "data" / "structures" / "sgc_fragments.csv"
OUTPUT_ALL = REPO_ROOT / "data" / "onepot_neighbors_all_matches.csv"
OUTPUT_ASSIGNED = REPO_ROOT / "data" / "onepot_neighbors_assigned.csv"
FIGURE_DIR = REPO_ROOT / "figures"

MAX_RESULTS_BY_POCKET = {"A": 1000, "B": 2000, "C": 2000, "D": 2000}
BATCH_SIZE = 10
FP_RADIUS = 2
FP_NBITS = 2048

POCKET_COLORS = {
    "A": "#E63946",
    "B": "#457B9D",
    "C": "#2A9D8F",
    "D": "#E9C46A",
}


def load_fragments(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    assert df.shape[0] > 0, f"No rows in {csv_path}"
    assert {"pocket", "pdb_id", "ligand_id", "smiles"}.issubset(df.columns)
    df = df.drop_duplicates(subset=["smiles"])
    print(f"Loaded {len(df)} unique fragments across pockets: "
          f"{dict(df['pocket'].value_counts().sort_index())}")
    return df


def query_onepot(fragments: pd.DataFrame) -> pd.DataFrame:
    from onepot import Client

    api_key = os.environ.get("ONEPOT_API_KEY")
    if not api_key:
        print("ERROR: Set ONEPOT_API_KEY environment variable.")
        sys.exit(1)

    client = Client(api_key=api_key)
    all_rows: list[dict] = []
    total_credits = 0

    for pocket, group in fragments.groupby("pocket"):
        max_results = MAX_RESULTS_BY_POCKET.get(pocket, 1000)
        smiles_list = group["smiles"].tolist()
        print(f"\nPocket {pocket}: {len(smiles_list)} fragments, max_results={max_results}")

        for batch_start in range(0, len(smiles_list), BATCH_SIZE):
            batch = smiles_list[batch_start : batch_start + BATCH_SIZE]
            batch_num = batch_start // BATCH_SIZE + 1
            total_batches = (len(smiles_list) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"  Batch {batch_num}/{total_batches}: {len(batch)} queries...")

            t0 = time.time()
            resp = client.search(smiles_list=batch, max_results=max_results)
            elapsed = time.time() - t0

            credits_used = resp.get("credits_used", 0)
            credits_remaining = resp.get("credits_remaining", "?")
            total_credits += credits_used
            print(f"    {elapsed:.1f}s | credits: {credits_used} | remaining: {credits_remaining}")

            for query_obj in resp.get("queries", []):
                query_smi = query_obj.get("query_smiles", "")
                frag_row = group[group["smiles"] == query_smi]
                if frag_row.empty:
                    print(f"    WARNING: no match for {query_smi[:60]}")
                    continue
                frag_info = frag_row.iloc[0]
                for r in query_obj.get("results", []):
                    all_rows.append({
                        "pocket": frag_info["pocket"],
                        "query_pdb": frag_info["pdb_id"],
                        "query_ligand": frag_info["ligand_id"],
                        "query_smiles": query_smi,
                        "hit_smiles": r.get("smiles", ""),
                        "hit_inchikey": r.get("inchikey", ""),
                        "similarity": r.get("similarity"),
                        "price_usd": r.get("price_usd"),
                        "supplier_risk": r.get("supplier_risk", ""),
                        "chemistry_risk": r.get("chemistry_risk", ""),
                    })

    print(f"\nTotal credits used: {total_credits}")
    df = pd.DataFrame(all_rows)
    print(f"Raw results: {len(df):,} rows")
    return df


def compute_ecfp4(smiles_series: pd.Series) -> np.ndarray:
    fps = []
    for smi in smiles_series:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_NBITS)
            fps.append(np.array(fp))
        else:
            fps.append(np.zeros(FP_NBITS, dtype=np.int8))
    return np.array(fps, dtype=np.float32)


def assign_best_pocket(df_all: pd.DataFrame) -> pd.DataFrame:
    return (
        df_all
        .sort_values("similarity", ascending=False)
        .drop_duplicates(subset=["hit_smiles"], keep="first")
        .sort_values(["pocket", "similarity"], ascending=[True, False])
        .reset_index(drop=True)
    )


def collect_all_pocket_labels(df_all: pd.DataFrame) -> pd.Series:
    """For each hit_smiles, collect all pockets it appeared in."""
    return (
        df_all
        .groupby("hit_smiles")["pocket"]
        .apply(lambda x: ",".join(sorted(x.unique())))
        .rename("all_pockets")
    )


def smi_to_image(smi: str, size: tuple[int, int] = (200, 200)) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.ones((*size, 3), dtype=np.uint8) * 255
    img = Draw.MolToImage(mol, size=size)
    return np.array(img)


def place_callouts(
    ax: plt.Axes,
    frag_df: pd.DataFrame,
    data_xlim: tuple[float, float],
    data_ylim: tuple[float, float],
) -> None:
    """Place structure callouts radially around each fragment point."""
    x_range = data_xlim[1] - data_xlim[0]
    y_range = data_ylim[1] - data_ylim[0]
    offset_dist = max(x_range, y_range) * 0.12

    n = len(frag_df)
    for i, (_, row) in enumerate(frag_df.iterrows()):
        angle = 2 * np.pi * i / n
        ox = offset_dist * np.cos(angle)
        oy = offset_dist * np.sin(angle)

        img = smi_to_image(row["hit_smiles"], size=(200, 200))
        imagebox = OffsetImage(img, zoom=0.35)

        ab = AnnotationBbox(
            imagebox,
            (row["pacmap_1"], row["pacmap_2"]),
            xybox=(row["pacmap_1"] + ox, row["pacmap_2"] + oy),
            boxcoords="data",
            arrowprops=dict(arrowstyle="-", color="0.3", lw=0.8),
            bboxprops=dict(
                edgecolor=POCKET_COLORS.get(row["pocket"], "black"),
                linewidth=1.5,
                facecolor="white",
                boxstyle="round,pad=0.1",
            ),
            zorder=15,
        )
        ax.add_artist(ab)


def smiles_to_inchikey(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.inchi.MolToInchiKey(Chem.inchi.MolFromSmiles(smi))


def build_fragment_inchikeys(fragment_smiles: set[str]) -> set[str]:
    keys = set()
    for smi in fragment_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            ik = Chem.inchi.MolToInchiKey(mol)
            if ik:
                keys.add(ik)
    return keys


def make_pacmap_figure(
    df_assigned: pd.DataFrame,
    fragment_smiles: set[str],
    output_path: Path,
) -> None:
    print("\nComputing ECFP4 fingerprints...")
    fps = compute_ecfp4(df_assigned["hit_smiles"])
    valid_mask = fps.sum(axis=1) > 0
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"  WARNING: {n_invalid} SMILES failed to parse, excluded from plot")

    df_plot = df_assigned[valid_mask].copy()
    fps_valid = fps[valid_mask]
    assert fps_valid.shape[0] == len(df_plot)

    print(f"Running PaCMAP on {fps_valid.shape[0]:,} compounds ({FP_NBITS}-bit ECFP4)...")
    embedding = pacmap.PaCMAP(n_components=2, random_state=42)
    coords = embedding.fit_transform(fps_valid)
    df_plot["pacmap_1"] = coords[:, 0]
    df_plot["pacmap_2"] = coords[:, 1]

    frag_inchikeys = build_fragment_inchikeys(fragment_smiles)
    print(f"Built {len(frag_inchikeys)} fragment InChIKeys for matching")
    df_plot["is_fragment"] = df_plot["hit_inchikey"].isin(frag_inchikeys)

    fig, ax = plt.subplots(figsize=(24, 20))

    neighbors = df_plot[~df_plot["is_fragment"]]
    for pocket, color in POCKET_COLORS.items():
        mask = neighbors["pocket"] == pocket
        subset = neighbors[mask]
        if len(subset) == 0:
            continue
        ax.scatter(
            subset["pacmap_1"], subset["pacmap_2"],
            c=color, s=10, alpha=0.35, label=f"Pocket {pocket} ({len(subset):,})",
            rasterized=True,
        )

    frags = df_plot[df_plot["is_fragment"]]
    for pocket, color in POCKET_COLORS.items():
        mask = frags["pocket"] == pocket
        subset = frags[mask]
        if len(subset) == 0:
            continue
        ax.scatter(
            subset["pacmap_1"], subset["pacmap_2"],
            c=color, s=250, alpha=1.0, edgecolors="black", linewidths=1.5,
            zorder=12, label=f"Fragment {pocket} ({len(subset)})",
            marker="*",
        )

    data_xlim = (df_plot["pacmap_1"].min(), df_plot["pacmap_1"].max())
    data_ylim = (df_plot["pacmap_2"].min(), df_plot["pacmap_2"].max())

    print(f"Rendering {len(frags)} structure callouts...")
    place_callouts(ax, frags, data_xlim, data_ylim)

    ax.set_xlabel("PaCMAP 1", fontsize=14)
    ax.set_ylabel("PaCMAP 2", fontsize=14)
    ax.set_title(
        "OnePot Nearest Neighbors by TBXT Binding Pocket\n(ECFP4 2048-bit, PaCMAP)",
        fontsize=16,
    )
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9, markerscale=2)

    x_pad = (data_xlim[1] - data_xlim[0]) * 0.2
    y_pad = (data_ylim[1] - data_ylim[0]) * 0.2
    ax.set_xlim(data_xlim[0] - x_pad, data_xlim[1] + x_pad)
    ax.set_ylim(data_ylim[0] - y_pad, data_ylim[1] + y_pad)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {output_path.relative_to(REPO_ROOT)}")


def main() -> int:
    fragments = load_fragments(FRAGMENTS_CSV)
    replot_only = "--replot" in sys.argv

    if replot_only and OUTPUT_ASSIGNED.exists():
        print("Re-plotting from existing data (--replot mode)")
        df_assigned = pd.read_csv(OUTPUT_ASSIGNED)
    else:
        # --- Query OnePot ---
        df_all = query_onepot(fragments)

        # --- Collect all pocket labels per compound ---
        all_pocket_labels = collect_all_pocket_labels(df_all)

        # --- Save full results (all pocket matches) ---
        df_all.to_csv(OUTPUT_ALL, index=False)
        print(f"All matches saved to {OUTPUT_ALL.relative_to(REPO_ROOT)}")

        # --- Assign best pocket ---
        df_assigned = assign_best_pocket(df_all)
        df_assigned = df_assigned.merge(all_pocket_labels, on="hit_smiles", how="left")
        print(f"After best-pocket assignment: {len(df_assigned):,} unique compounds")
        print("\nPer-pocket counts (assigned):")
        for pocket, group in df_assigned.groupby("pocket"):
            sims = group["similarity"]
            print(f"  {pocket}: {len(group):,} compounds  "
                  f"(sim: {sims.min():.3f}–{sims.max():.3f})")

        df_assigned.to_csv(OUTPUT_ASSIGNED, index=False)
        print(f"Assigned results saved to {OUTPUT_ASSIGNED.relative_to(REPO_ROOT)}")

    # --- PaCMAP visualization ---
    fragment_smiles = set(fragments["smiles"].tolist())
    fig_path = FIGURE_DIR / "onepot_pacmap_by_pocket.png"
    make_pacmap_figure(df_assigned, fragment_smiles, fig_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
