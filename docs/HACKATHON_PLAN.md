# TBXT Hit Identification Hackathon — Execution Plan

**Event:** May 9, 2026, Boston, Pillar VC (6 hours)  
**Team:** Nate Harms, Raymond Gasper  
**Submission target:** 4 ranked non-covalent small-molecule SMILES from onepot's 3.4B CORE catalog, each with a
predicted binding pocket, ranking rationale, key computed evidence, and physchem properties.

---

## Status Summary (Updated May 8, 2026)

### COMPLETED
- **SAR diagnostics**: Zenodo SPR dataset unsuitable for regression (batch explains 15% variance, 30% activity cliffs at Tanimoto ≥0.7, kNN pKD std = 67% global)
- **Classification model**: Zenodo binder/non-binder classifier achieves OOF AUROC 0.65 (ceiling for this data)
- **Pocket mapping**: Four TBXT binding sites mapped to Newman pockets:
  - A site = pocket A' (most validated, 22 fragments, good Boltz pose)
  - D site = pocket B (dropped: bad Boltz pose)
  - F site = pocket D (4 fragments, Y88/D177, P300/KDM6 interface, induced pocket)
  - G site = pocket C (10 fragments, crystallographic 2-fold, truncated construct)
- **Fragment catalog**: 29 fragment binders extracted from TEP PDF with per-pocket assignments

### REVISED STRATEGY
**Original plan**: ligand-based VL → multi-model ML → Boltz → Vina → final 4

**New plan** (due to SAR diagnostics showing Zenodo regression ceiling):
- **Abandon Zenodo as regression target**, use only as soft prior (binder/non-binder AUROC 0.65)
- **Per-pocket active learning approach**:
  - Build separate Boltz surrogates for A and G sites (reliable Boltz poses)
  - Manual Boltz for F site (induced pocket, ~20 compounds only)
  - Drop B site (bad Boltz pose)
- **Fragment-based pocket assignment**: 29 TEP fragments as pocket-similarity scorer (independent of Boltz)
- **Slot allocation**: 2× A site (most validated), 1× G site, 1× F site (speculative, novel MoA)

---

## 1. Strategic frame

### Original brief requirements
Per the brief, judging weights "scientific rationale and computational support," "compound quality and tractability,"
and "hit identification judgment" (good prioritization, not volume). The four-compound list should reflect *multiple,
agreeing lines of evidence* and *scaffold/site diversity* so that one synthesis failure does not eliminate the team's
chances at the experimental prizes (1 µM and 300 nM tiers).

The 29 SGC TEP fragments (across 6 clusters) are the explicitly-named structural starting points. The Naar SMILES set (
135 prior-screened compounds) is the explicit "avoid duplication" reference. The known-active SPR set (2153 SPR records,
1913 non-null pKD 2.0–5.7, 1680 unique compounds, 14 batches) is available but has severe limitations (see SAR diagnostics).

### Evidence-based strategy revision
After comprehensive SAR diagnostics (`docs/sar-diagnostics-rjg/README.md`), we've established:
1. Zenodo SPR data has 0.65 AUROC ceiling for classification (batch effects, 100-fold replicate variation, activity cliffs)
2. Newman 2025 fragment structures provide reliable pocket mapping
3. Boltz pose reliability varies by site (A: high, B: low/drop, F: medium/induced, G: medium)
4. Fragment-based pocket assignment can guide compound selection independent of noisy SPR data

---

## 2. Revised execution plan

### Step 1: Fragment-based catalog filtering (onepot 3.4B → ~10K per pocket)
**Owner:** Nate  
**Time:** Pre-hackathon complete; catalog pre-filtered to compounds containing Newman fragment substructures

**Inputs:**
- onepot 3.4B CORE catalog (pre-filtered to fragment-containing compounds)
- 29 TEP fragment SMILES with pocket assignments (A/A', B, F=Newman D, G=Newman C)

**Outputs:**
- Per-pocket compound pools filtered by:
  - Fragment substructure match or ECFP4 Tanimoto ≥ 0.35 to pocket fragments
  - Chordoma physchem filters: LogP ≤ 6, HBD ≤ 6, HBA ≤ 12, MW ≤ 600
  - PAINS filters
  - Naar deduplication (Tanimoto < 0.6)
- Zenodo binder classifier applied as soft prior (AUROC 0.65)

### Step 2: Per-pocket active learning with Boltz oracle (A and G sites only)
**Owner:** Ray  
**Budget:** ~1000 Boltz predictions per pocket (~2000 total, ~50 GPU-hours)

**A site (pocket A') — 2 submission slots:**
- Most validated site (22 fragments, known thiazole binders 14–20 µM)
- Boltz pose reliable (R180 anchor contact)
- Prefer thiazole-acetamide/morpholino-thiazole chemotypes (5QS9/5QSD/7ZK2/8A7N)

**G site (pocket C) — 1 submission slot:**
- 10 fragments, crystallographic 2-fold symmetry
- Boltz pose medium reliability (E48/E50/R54 polar contacts)
- Prefer polar/basic head groups for E48/E50/R54 interactions

**Process per pocket:**
1. **Initial diversity sample**: ~1000 compounds (diversity picker on Morgan FP, fragment-substructure-filtered)
2. **Boltz prediction batch**: predict ΔG for all ~1000 compounds
3. **Train surrogate**: CheMeleon + MLP (SMILES → Boltz ΔG)
4. **Validate surrogate**: Spearman >0.6 on holdout (if fails, use Boltz predictions directly)
5. **Score full catalog**: apply surrogate to full filtered catalog (~10K compounds per pocket)
6. **Re-Boltz top candidates**: re-predict top ~200 per pocket with Boltz for final validation
7. **Pose QC**: filter by anchor residue contacts (R180 for A, E48/R54 for G)

### Step 3: F site manual Boltz (pocket D, induced) — 1 submission slot
**Owner:** Ray  
**Budget:** ~20 Boltz predictions (no surrogate, speculative)

**Rationale:**
- Induced pocket (buried, Y88/D177, P300/KDM6 interface)
- 4 TEP fragments (low count, novel MoA potential)
- Boltz over-estimates affinity (buried cavity), but pose geometry still informative
- MW < 400 preferred (small molecule for buried cavity)

**Process:**
1. Rank full F-site catalog (~10K) by max ECFP4 Tanimoto to 4 pocket-D fragments
2. Select top ~20 by fragment similarity + diversity + MW < 400
3. Boltz predict all ~20
4. Pose QC: Y88 contact, cavity burial check
5. Select best 1 for submission

### Step 4: Secondary docking validation (Vina)
**Owner:** Nate  
**Candidates:** Top ~50 compounds across A/G/F (post-Boltz QC)

**Process:**
- AutoDock Vina on PDB 6F59 (DNA-stripped)
- Per-site search boxes (A: R180 region, G: E48/R54 region, F: Y88/D177 region)
- Vina score + RMSD vs Boltz pose
- Newman fragment retrodocking as pose validation

### Step 5: Final selection (4 compounds)
**Criteria:**
1. **Site diversity**: 2× A site, 1× G site, 1× F site
2. **Scaffold diversity**: 4 distinct Bemis–Murcko scaffolds
3. **Boltz evidence**: A/G ΔG better than Newman thiazole series (−7 to −8 kcal/mol), F ΔG used for pose only
4. **Pose QC**: anchor contacts (R180, E48/R54, Y88), Newman retrodock agreement
5. **Vina confirmation**: score within 1 kcal/mol of best per-site
6. **Zenodo classifier**: prefer p_active > 0.5 (weak prior, AUROC 0.65)
7. **Fragment similarity**: max Tanimoto to pocket fragments ≥ 0.35

---

## 3. Critical technical notes

### Zenodo SPR data limitations (from `docs/sar-diagnostics-rjg/README.md`)
- **Batch effects**: date alone explains R²=0.155 of pKD variance
- **Replicate noise**: mean std = 0.92 pKD (100-fold KD range), median |ΔpKD| = 1.03 for Tanimoto 0.95–1.0
- **Activity cliffs**: 30% of near-identical pairs (Tanimoto ≥ 0.7) diverge >1 pKD unit
- **Model ceiling**: CheMeleon embedding + Ridge probe underperforms train-mean baseline
- **Classification AUROC**: 0.655 (chemeleon_no_val model, OOF)
- **Conclusion**: use as soft prior only, not regression target

### Newman pocket fragment counts (from `brachyury-site-summary.md`)
- A/A' (~22 fragments): most validated, thiazole series 14–20 µM SPR
- B (5 fragments): dropped (bad Boltz pose, tilted)
- D/F (4 fragments): induced pocket, Y88/D177, P300/KDM6 interface, buried
- C/G (10 fragments): crystallographic 2-fold, E48/E50/R54 polar head
- Cd-mediated artifacts: exclude from similarity calculations

### Boltz oracle reliability per site (from `brachyury-site-summary.md`)
- **A (high)**: good pose, R180 anchor, known binders validate
- **B (low)**: tilted pose vs crystallographic, DROP THIS SITE
- **F (medium)**: induced pocket, over-estimates affinity, pose geometry still useful
- **G (medium)**: truncated construct in PDB, 2-fold symmetry, E48/R54 contacts

### Known binder SPR benchmarks
- Newman thiazole series: 14–20 µM (A/A' pocket, 5QS9/5QSD/7ZK2/8A7N)
- TEP progressed compound: 80–104 µM on full-length G177D
- Naar compounds: Z979336988, Z795991852, D203-0031 (sites F or F/G)

### Per-site chemistry filters
- **A site**: prefer thiazole-acetamide/morpholino-thiazole, avoid >2 fused benzene rings
- **G site**: prefer polar/basic head for E48/E50/R54, HBD ≥ 2, avoid lipophilic cores
- **F site**: MW < 400 (buried cavity), avoid long alkyl chains, prefer compact aromatics

---

## 4. Roles and timeline (hackathon day)

| Time | Nate | Ray |
|------|------|-----|
| Hour 0:00–1:00 | Prepare Vina receptor (6F59 DNA-stripped, per-site boxes) | Launch A-site Boltz batch (~1000 compounds) |
| Hour 1:00–2:00 | Monitor A-site Boltz, prepare pose QC scripts | Train A-site CheMeleon+MLP surrogate |
| Hour 2:00–3:00 | Score A-site catalog with surrogate, select top 200 | Launch G-site Boltz batch (~1000 compounds) |
| Hour 3:00–4:00 | Re-Boltz A-site top 200, pose QC, Vina validation | Train G-site surrogate, F-site manual Boltz (~20) |
| Hour 4:00–5:00 | Re-Boltz G-site top 200, pose QC, Vina validation | Pose QC all survivors, Newman retrodocking |
| Hour 5:00–5:45 | Final selection (2× A, 1× G, 1× F), scaffold/site diversity check | Rationale writeup, evidence tables |
| Hour 5:45–6:00 | Pre-submission checklist, submit | Archive outputs, commit to git |

---

## 5. Budget and feasibility

### Boltz predictions
- A site: ~1000 initial + 200 re-Boltz = 1200 predictions
- G site: ~1000 initial + 200 re-Boltz = 1200 predictions  
- F site: ~20 predictions
- **Total: ~2420 predictions at ~2 min each on A100 = ~80 GPU-hours** (feasible with 2× A100 for 6 hours = 12 GPU-hours available? **CHECK THIS ASSUMPTION**)

### Surrogate model viability
- CheMeleon (pretrained) + MLP on ~1000 Boltz ΔG predictions per pocket
- Target: Spearman >0.6 on holdout (if fails, use Boltz predictions directly, no catalog expansion)
- Fallback: if surrogate fails, reduce re-Boltz budget to top 50 per pocket (prioritize pose diversity)

---

## 6. Submission deliverables

**File:** `submission/final_4.csv`

**Columns:**
- `rank` (1–4)
- `smiles` (canonical RDKit)
- `compound_id` (onepot CORE ID)
- `predicted_site` (A, G, or F with Newman pocket mapping)
- `rationale` (3–5 sentences: evidence convergence, novelty, site rationale)
- `evidence` (Boltz ΔG, iPTM, Vina kcal/mol, max Tanimoto to pocket fragments, Zenodo p_active, CheMeleon surrogate score)
- `properties` (MW, LogP, HBD, HBA, heavy atoms, ring systems, TPSA)
- `anchor_contacts` (R180 for A, E48/R54 for G, Y88 for F)
- `scaffold` (Bemis–Murcko SMILES)

**Validation checklist:**
- [ ] All 4 SMILES parse with RDKit, round-trip to canonical
- [ ] All 4 satisfy Chordoma filters (LogP ≤ 6, HBD ≤ 6, HBA ≤ 12, MW ≤ 600)
- [ ] No PAINS hits (RDKit FilterCatalog)
- [ ] All 4 present in onepot CORE (verify by ID)
- [ ] 4 distinct Bemis–Murcko scaffolds
- [ ] Site distribution: 2× A, 1× G, 1× F
- [ ] Boltz pose QC passed (anchor contacts present)
- [ ] Vina score within 1 kcal/mol of best per-site (A/G sites minimum)
- [ ] Max Tanimoto to Naar < 0.6 (no prior-art duplication)

---

## 7. Key decisions and assumptions

### Decisions
1. **Drop Zenodo regression**: use as soft binder/non-binder prior only (AUROC 0.65)
2. **Drop B site**: bad Boltz pose, not worth budget
3. **Per-pocket surrogates**: separate CheMeleon+MLP for A and G (distinct chemistry/pharmacophores)
4. **F site speculative**: 1 slot for novel MoA, fragment similarity only (no surrogate)
5. **Slot allocation**: 2× A (most validated), 1× G (10 fragments), 1× F (speculative)

### Assumptions to validate
1. **Boltz budget**: 80 GPU-hours feasible? (need 2× A100 for 6 hours or equivalent)
2. **Surrogate Spearman >0.6**: achievable with ~1000 training points per pocket?
3. **onepot catalog delivery**: pre-filtered to fragment substructures, ~10K per pocket?
4. **Newman fragment SMILES**: already extracted and pocket-assigned?

---

## 8. Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Boltz GPU time exceeds budget | High | Reduce re-Boltz to top 50 per pocket (prioritize diversity), skip G-site surrogate if needed |
| Surrogate Spearman < 0.6 | Medium | Use Boltz predictions directly, reduce catalog scoring to top 200 by fragment similarity |
| All top A-site candidates are one scaffold | Medium | Force scaffold diversity in top-200 re-Boltz selection (≥10 Murcko clusters) |
| F-site Boltz poses all bury poorly | Medium | Accept best pose by Y88 contact, deprioritize ΔG (use for ranking only) |
| Time slip past hour 5 | High | Cut G-site re-Boltz to 50, cut Vina to A-site only, prioritize submission over validation |

---

## 9. References

- Hackathon brief: https://docs.google.com/document/d/1K2r_7HopkH1_4jsGTrZZX8zR1FKgMTxMe-43dG4xJuA
- TBXT TEP datasheet (SGC, 2024): https://www.thesgc.org/sites/default/files/2024-05/TBXT_TEP_datasheet_v1_0.pdf
- Newman et al., *Nat Commun* 2025: https://doi.org/10.1038/s41467-025-56213-1
- PDB 6F59 (TBXT DBD bound to DNA): https://www.rcsb.org/structure/6F59
- UniProt O15178 (TBXT): https://www.uniprot.org/uniprotkb/O15178/entry
- Naar SMILES (prior screen, deduplicate against): https://docs.google.com/spreadsheets/d/1k-vcM_jVd1s_6W6u-ag2YQabGov_40oB
- onepot CORE: https://www.onepot.ai/
- Experimental prize tiers: https://tbxtchallenge.org/#prizes
- SAR diagnostics (internal): `docs/sar-diagnostics-rjg/README.md`
- Pocket mapping (internal): `brachyury-site-summary.md`
- Zenodo SPR data: `data/zenodo/tbxt_spr_merged.csv` (2153 records, 1913 non-null pKD, 14 batches)
