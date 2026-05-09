---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-size: 22px;
    padding: 40px 60px;
  }
  section.title {
    text-align: center;
    justify-content: center;
  }
  section.appendix-divider {
    text-align: center;
    justify-content: center;
    background: #1a3a5c;
    color: white;
  }
  section.appendix-divider h1 { color: white; }
  h1 { color: #1a3a5c; }
  h2 { color: #1a3a5c; border-bottom: 2px solid #457B9D; padding-bottom: 4px; }
  table { font-size: 0.8em; }
  code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
  .small { font-size: 0.75em; }
  .tiny { font-size: 0.65em; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }
---

<!-- _class: title -->

# TBXT Hit Identification
## End-to-end virtual screening pipeline

**Nate Harms, Raymond Gasper**
TBXT Hackathon — May 9, 2026 — Boston

Target: human TBXT (Brachyury) DBD, residues 42–219
Source: onepot 3.4B CORE catalog
Submission: 4 ranked SMILES across pockets A, F, G

---

## Strategy

**"Traditional" modern virtual screening approach**

**Every signal on this target is weak. Stack them; filter aggressively.**

<div class="two-col">

<div>

**Weak, independent signals:**

- Zenodo SPR — ~0.65 AUROC ceiling
- Boltz — pose OK at 3/4 sites
- Vina — classical, false positives
- Fragment similarity — uneven
- Physchem + risk — necessary only

</div>

<div>

**Stack them:**

1. Zenodo → ranker, not predictor
2. Boltz → pose, not raw ΔG
3. Vina → orthogonal + anchor check
4. Physchem + PAINS + synthesis risk throughout

</div>
</div>

---

## Target and pocket plan

<div class="two-col">

<div>

**TBXT / Brachyury** — T-box TF, key dependency in chordoma
DNA-binding domain (res 42–219), PDB 6F59
SGC TEP identifies 5 fragment pockets; we target **A, F, G**

| TEP pocket | Anchor residues | Frags | Slots |
|---|---|---|---|
| **A / A'** | R180, V123, L91, I125, V173, I182 | 26 | 2 |
| **G** | R54, E48, E50, L51, K76 | 10 | 1 |
| **F** | Y88, D177, V173, I182 | 4 | 1 (spec) |
| ~~D~~ | G112, H100, P115 | 5 | 0 (bad Boltz pose) |

</div>

<div class="small">

**Design constraints (Chordoma Fdn):**
- MW ≤ 600, LogP ≤ 6, HBD ≤ 6, HBA ≤ 12
- Non-covalent
- No acid halides / aldehydes / diazo / imines
- ≤ 2 fused rings

**Boltz oracle reliability** (vs Newman co-crystals):
- A: high (OK vs 5QS2)
- G: medium (OK vs 5QS6)
- F: medium, induced pocket (OK vs 5QSA)
- D: low, tilted pose → **dropped**

<span style="color:#666">Pocket IDs follow the SGC TEP datasheet.</span>

</div>
</div>

---

## Modeling the Zenodo SPR data

**Data:** 2,143 SPR records, 14 batches (Oct 2020 – Jan 2023), 1,545 unique compounds after cleaning.

<div class="two-col">

<div>

**The dataset was hard to model.**

- Batch date alone explains R² ≈ 0.15 of pKD variance
- ~30% activity cliffs at Tanimoto ≥ 0.7; replicate std ≈ 0.92 pKD (~100× KD)
- Regression: all attempts negative R² — data ceiling, not model ceiling
- Tried **CheMeleon chemistry foundation model**; competitive signal but kicked out under time pressure

**What worked: reframe as a binary ranker.**

- **Excluded mid-range KD** (pKD 3–5 gray zone) — unlearnable with ~0.9 pKD replicate noise; exclusion gave +0.06 to +0.12 OOF AUROC across variants
- 12-booster XGBoost ensemble (2 fingerprints × 6 leave-one-fold-out splits) on the cleaned label
- OOF AUROC 0.764 · AUPRC 0.337 (random 0.123) · top-30 precision 17/30 on try4 holdout

</div>

<div>

![Per-positive fractional rank by model variant](classification-models-try4-rjg/holdout_rank_box.png)

<div class="small">

**Per-positive rank of 29 holdout positives**, across 9 fingerprint × feature-set variants on the filtered dataset. Lower is better; random = 0.5 (grey line). `maccs_fp_pocket_phys` is the tightest distribution — the recipe baked into deployment.

</div>
</div>
</div>

---

## Pipeline — end-to-end with numbers

```
 onepot CORE (3.4B)                  SGC TEP fragments (45)
        │                                      │
        ▼                                      │
 [1] Downfilter: regex → RDKit                 │
     substructure + physchem + PAINS + risk    │
        │                                      │
        ▼                                      ▼
   897,431 survivors     [1b] onepot API neighbor queries
        │                     + physchem + PAINS + risk
        │                     51,560 unique compounds
        └──────────────┬─────────────────────────┘
                       ▼
         Combined screening library: 178,332
                       ▼
        [2] XGBoost 12-booster ensemble → p_binder rank
                       ▼
        [3] Pocket-assign → split A / F / G (top 5000 each)
                       ▼
        [4] Diversity cluster (Morgan, Butina-style)
                       ▼
        [5] Boltz pose prediction + pose QC
              (site-localization rejects ~90–95%)
              survivors: <100 per pocket
                       ▼
        [5b] Exclude Zenodo-similar compounds
              (ECFP4 Tanimoto > 0.85 to any SPR compound)
                       ▼
        [6] AutoDock Vina + anchor-residue check
              R180 (A) · E48/R54 (G) · Y88 (F)
                       ▼
                Final 4 (2·A, 1·G, 1·F)
```

---

## Recommended compounds *(mock — to be populated)*

| Rank | Pocket | SMILES | Scaffold | Rationale |
|:---:|:---:|:---|:---|:---|
| **1** | A | `TBD` | thiazole-acetamide | Top XGBoost rank in A; Boltz pose recapitulates 8A7N R180 H-bond; Vina −8.9 kcal/mol. |
| **2** | A | `TBD` | morpholino-thiazole | Scaffold-diverse A-pocket hit; strong R180 + L91/V123 hydrophobic cluster engagement. |
| **3** | G | `TBD` | aryl sulfonamide | Polar head engages E48 + R54; highest-ranked G-pocket survivor after pose QC. |
| **4** | F | `TBD` | compact bicyclic | Speculative novel-MoA slot; Y88 contact + D177 proximity; fragment-Tanimoto-led. |

<div class="small">

**Each row is backed by convergent evidence:** XGBoost `p_binder`, fragment Tanimoto ≥ 0.35 in assigned pocket, Boltz pose matching Newman co-crystal, Vina score within 1 kcal/mol of per-pocket best, anchor-residue contact, Chordoma physchem satisfied, Max Tanimoto to Naar set < 0.6.

</div>

---

<!-- _class: appendix-divider -->

# Appendices

---

## Appendix A — Full submission evidence table *(mock)*

<div class="tiny">

| Rank | onepot ID | SMILES | Pocket | Scaffold | Boltz ΔG (kcal/mol) | Vina (kcal/mol) | `p_binder` | Anchor contact | Rationale |
|:---:|:---|:---|:---:|:---|:---:|:---:|:---:|:---|:---|
| **1** | `OP-000000001` | `TBD` | A | thiazole-acetamide | −8.4 | −8.9 | 0.87 | R180 H-bond (2.8 Å) | Top XGBoost rank A; reproduces 8A7N pose; L91/V123/I125/V173 hydrophobic contact cluster. |
| **2** | `OP-000000002` | `TBD` | A | morpholino-thiazole | −8.1 | −8.5 | 0.81 | R180 H-bond (3.0 Å) | Scaffold-distinct A-pocket hit; morpholine O accepts from R180 guanidinium; clean Boltz pose vs 5QSD. |
| **3** | `OP-000000003` | `TBD` | G | aryl sulfonamide | −7.6 | −7.9 | 0.74 | E48 (salt bridge) · R54 (3.1 Å) | Polar head reaches E48/R54 hotspot; pose QC passes vs 5QS6; DNA-interface-adjacent vector. |
| **4** | `OP-000000004` | `TBD` | F | compact bicyclic | −8.8* | −7.3 | 0.63 | Y88 (3.2 Å) · D177 proximity | Novel-MoA speculative slot; high fragment Tanimoto to K2P (5QSA); MW < 400 fits buried cavity. |

</div>

<div class="small">

*Pocket F Boltz ΔG is inflated by burial + induced-fit and is used as pose-only signal, not absolute score.

All 4 satisfy: Chordoma physchem (LogP ≤ 6, HBD ≤ 6, HBA ≤ 12, MW ≤ 600), max Tanimoto to Naar set < 0.6, and 4 distinct Bemis–Murcko scaffolds.

</div>

---

## Appendix B — Stage 1 + 1b details (library build)

<div class="two-col">

<div>

**Downfilter 3.4B catalog**
`scripts/downfilter_onepot/`

1. Coarse SMILES regex against canonical + Kekulized fragment SMILES + Bemis–Murcko scaffolds
2. RDKit worker pool: PAINS + bad-SMARTS + `HasSubstructMatch` vs 45 fragments
3. Hard filters: MW/LogP/HBD/HBA, 10–30 HAC, ≤ 5 rings, ≤ 2 fused, ≥ 1 fragment hit
4. Sharded parquet, rollup every 5M rows

**Output: 897,431 compounds**

</div>

<div>

**onepot API nearest-neighbor queries**
`scripts/query_onepot_neighbors.py`

- Per query pocket (A / F / G): 1 API call per SGC fragment
- Pocket A: 1000 neighbors/frag
- Pocket D/F/G: 2000 neighbors/frag
- ECFP4 (r=2, 2048) → route to best pocket
- Physchem + PAINS + synthesis-risk filters applied
- Fragment-substructure filter skipped (already similarity-selected)

**Output: 51,560 unique compounds**

---

**Combined:** `data/screening_library_combined.csv` — **178,332 compounds**

</div>
</div>

---

## Appendix C — Stage 2 + 3 (model + pocket assignment)

<div class="two-col">

<div>

**XGBoost deployment ensemble**
`src/tbxt_hackathon/deployment.py`

- **12 boosters** = 2 fingerprint types × 6 leave-one-fold-out splits (PaCMAP-KMeans6)
- Fingerprints: **MACCS** (167) + **Morgan ECFP4** (2048), each with pocket (8) + physchem (8)
- 708 training compounds (12.3% positives)
- Task: binary P(pKD > 5), used as **ranker**

**Metrics (OOF):** AUROC 0.764 · AUPRC 0.337 (random 0.123)
Selected from a fingerprint × feature-set ablation.
On MACCS booster, importance: 86% MACCS, 6% pocket, 8% physchem.

</div>

<div>

**Pocket assignment**
`src/tbxt_hackathon/pocket_assigner.py`

- Per pocket: max ECFP4 Tanimoto + substructure boolean
- Combined = `1.0 + Tc` if substruct hit, else `Tc`
- Assign to highest combined score

**Soft spot:** O0P (5QSC) crystallizes at both G and F. Single-assignment forces a choice. 44/45 fragments self-route correctly; O0P is the miss. See `docs/pocket-mapper-soundness.md`.

**CheMeleon foundation model:** tried as embedding + probe and embedding + ensemble; competitive but dropped under time pressure.

</div>
</div>

---

## Appendix D — Stage 4 + 5 (cluster + Boltz)

<div class="two-col">

<div>

**Top 5000 per pocket → diversity cluster**

- Rank by `p_binder_ensemble_mean`
- Butina-style clustering on Morgan FP
- Kills redundant scaffolds *before* the expensive oracle runs

Model score is cheap on 178k. Boltz is the bottleneck — wasting it on scaffold twins shrinks effective pocket coverage.

</div>

<div>

**Boltz pose prediction** (Boltz Lab cloud)

Ran Boltz on the clustered top-5000-per-pocket. Pose QC is strict — the site-localization filter alone rejects **~90–95%** of Boltz outputs.

Post-QC survivors docked with Vina:
- **Pocket A: <100**
- **Pocket G: <100**
- **Pocket F: <100**

Pocket D not run (pose tilted vs crystal).

**Site-localization filter:** reject any compound whose Boltz pose places the ligand outside the intended pocket, even when induced-fit displacement is allowed. This is the dominant filter; catches off-site binders the affinity score alone would not flag.

</div>
</div>

**Per-site cautions carried into ranking:**
- Rank *within* a pocket, not across — different pockets give different score distributions
- Pocket F scores inflated by burial + induced-fit; use pose, not absolute ΔG
- Pocket A rewards hydrophobic burial; watch logP

**Post-Boltz novelty filter:** drop any survivor with ECFP4 Tanimoto > 0.85 to any compound in the Zenodo SPR set. Keeps the submission novel vs prior screening; avoids paying Vina time on near-duplicates.

---

## Appendix E — Stage 6 + 7 (Vina + final selection)

<div class="two-col">

<div>

**AutoDock Vina** · `docs/docking.md`

- Per-pocket receptor PDBs, box centered on reference fragment, 10 Å padding
- Multi-conformer, interaction analysis (H-bonds, π-stacking, hydrophobic, salt bridges)

**Anchor-residue hard filter:**

| Pocket | Required contact |
|---|---|
| A | H-bond to **R180** |
| G | Contact to **E48 or R54** |
| F | Contact to **Y88** (±D177) |

Fail the anchor check → drop regardless of Vina score.

</div>

<div>

**Final 4 — convergent evidence required:**

- ✅ XGBoost `p_binder_mean` ranks high
- ✅ Fragment Tanimoto ≥ 0.35 in assigned pocket
- ✅ Boltz pose resembles crystal fragment
- ✅ Vina score within 1 kcal/mol of per-pocket best
- ✅ Anchor contact present
- ✅ Chordoma physchem satisfied
- ✅ 4 distinct Bemis–Murcko scaffolds
- ✅ Max Tanimoto to Naar set < 0.6

**Slot allocation:** 2× A (most validated) · 1× G · 1× F (novel MoA)

</div>
</div>

---

## Appendix F — Caveats and references

<div class="two-col">

<div>

**Acknowledged limits**

- SPR data has a ~0.65 AUROC ceiling for absolute prediction (batch effects, activity cliffs) → classifier is a ranker, not truth
- Pocket F scores inflated (induced pocket) → pose-only signal
- O0P dual-binder falsifies single-pocket routing; documented fix is multi-assignment
- CheMeleon foundation model tried, dropped for time
- Not chasing sub-µM; target is **credible, fragment-informed, pose-sensible** binders with convergent support

</div>

<div>

**Key numbers**

| Stage | Output |
|---|---|
| Downfilter | 897,431 |
| API neighbors (unique) | 51,560 |
| **Combined library** | **178,332** |
| Top per pocket (pre-cluster) | 5000 |
| Boltz survivors (site-QC rejects ~90–95%) | <100 per pocket |
| After Vina + anchor | final per-pocket shortlist |
| **Submission** | **4** |

</div>
</div>

<div class="small">

**References:** Newman et al. *Nat Commun* 2025 · SGC TBXT TEP · PDB 6F59 · onepot CORE
**In-repo:** `docs/HACKATHON_PLAN.md` · `docs/deployment-model.md` · `docs/docking.md` · `docs/brachyury-site-summary.md` · `docs/pocket-mapper-soundness.md` · `docs/sar-diagnostics-rjg/README.md`

</div>
