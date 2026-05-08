# TBXT Hit Identification Hackathon — Execution Plan

**Event:** May 9, 2026, Boston, Pillar VC (6 hours)
**Team:** Nate Harms, Raymond Gasper
**Submission target:** 4 ranked non-covalent small-molecule SMILES from onepot's 3.4B CORE catalog, each with a
predicted binding pocket, ranking rationale, key computed evidence, and physchem properties.

---

## 1. Strategic frame

Per the brief, judging weights "scientific rationale and computational support," "compound quality and tractability,"
and "hit identification judgment" (good prioritization, not volume). The four-compound list should reflect *multiple,
agreeing lines of evidence* and *scaffold/site diversity* so that one synthesis failure does not eliminate the team's
chances at the experimental prizes (1 µM and 300 nM tiers).

The 29 SGC TEP fragments (across 6 clusters) are the explicitly-named structural starting points. The Naar SMILES set (
135 prior-screened compounds) is the explicit "avoid duplication" reference. The known-active SPR set (1,545 cleaned
compounds, pKD 2.0–5.7) is our supervised-learning anchor.

We will pursue a ligand-based virtual screen with structure-based confirmation, layered as: focused VL → multi-model ML
triage → Naar-similarity prune → Boltz pose triage → Vina pose confirmation → diversified final 4.

---

## 2. Roles and shared infrastructure

| Owner | Pre-hackathon                                                                | Hackathon-day primary                                         |
|-------|------------------------------------------------------------------------------|---------------------------------------------------------------|
| Nate  | Step 1 (focused VL via OnePot CORE), XGB scaffolding, pair-model scaffolding | XGB + pair model inference, Naar dedup, Vina, final selection |
| Ray   | Boltz environment, transfer-learning model scaffolding                       | ChemProp + transfer model training/inference, Boltz triage    |
| Both  | Confirm shared git remote, agreed parquet schema, GPU access                 | 30-min syncs at hours 0, 2, 4, 5.5                            |

**Shared artifacts** (commit to repo or shared drive at the named paths):

- `data/processed/onepot_focused_vl.parquet` — focused VL after physchem and PAINS filters
- `data/processed/predictions_<modelname>.parquet` — `(compound_id, pred_pkD, p_active, model_version)`
- `data/processed/ensemble_survivors.parquet` — post-ensemble shortlist
- `data/processed/boltz_top50/` — Boltz outputs (poses, scores, predicted site)
- `submission/final_4.csv` — final SMILES with all submission fields

---

## 3. Pre-hackathon (Today, May 8) — Step 1 only

### 3.1 OnePot file intake (Nate, ~30 min)

Inspect the delivered `csv.gz` *before* writing query code. The schema is unknown; the duckdb plan must adapt.

```python
import duckdb

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
# probe columns and row count
con.sql("DESCRIBE SELECT * FROM read_csv_auto('s3://<bucket>/<key>.csv.gz', sample_size=10000)")
con.sql("SELECT count(*) FROM read_csv_auto('s3://<bucket>/<key>.csv.gz')")
con.sql("SELECT * FROM read_csv_auto('s3://<bucket>/<key>.csv.gz', sample_size=10000) LIMIT 5")
```

Decisions to record in `docs/onepot_schema_notes.md`:

1. SMILES column name and canonicalization status (canonical vs. as-supplied).
2. Whether MW / LogP / HBD / HBA / heavy atoms are precomputed. If yes, trust them for the cheap initial pass and
   recompute only on the survivors. If no, plan an RDKit physchem pass.
3. Compound ID column.
4. Total row count (sanity-check vs. 3.4B claim).
5. File size on disk if downloaded; otherwise plan to keep it on S3 and stream.

Convert to Parquet partitioned by a SMILES-hash bucket if the flat csv.gz is unwieldy:

```python
con.sql("""
COPY (SELECT * FROM read_csv_auto('s3://<bucket>/<key>.csv.gz'))
TO 'data/external/onepot_core.parquet'
(FORMAT PARQUET, PARTITION_BY (mod(hash(smiles), 64)), OVERWRITE_OR_IGNORE)
""")
```

### 3.2 Fragment SMILES retrieval (Nate, ~20 min)

The 29 SGC TEP fragments live in:

- TEP datasheet PDF (linked from the
  brief): https://www.thesgc.org/sites/default/files/2024-05/TBXT_TEP_datasheet_v1_0.pdf
- The accompanying SGC structural data link (PDB ligand entries): https://tinyurl.com/bddybesb
- Newman et al. 2025 *Nature Communications* supplementary tables: https://www.nature.com/articles/s41467-025-56213-1

Save to `data/structures/sgc_fragments.csv` with columns: `frag_id, cluster, site, smiles, pdb_id`. Sites observed in
the brief are F and G; the full clustering (6 clusters) and site assignments must come from the TEP/paper.

### 3.3 Substructure cascade (Nate, ~1.5–2 hr)

The plan is intentionally **iterative**: start strict, fall back if the yield is empty; widen further if still empty;
tighten if the yield is unmanageable.

| Tier           | Method                                                                                   | Yield target                                        | Action if outside target                                             |
|----------------|------------------------------------------------------------------------------------------|-----------------------------------------------------|----------------------------------------------------------------------|
| T1             | RDKit `HasSubstructMatch` against full SMILES of all 29 fragments                        | 10⁵–10⁶                                             | If <10⁴: go to T2. If >10⁷: go to T1b (require ≥2 fragments matched) |
| T2             | Same match against Bemis–Murcko scaffold of each fragment (or 6 cluster representatives) | 10⁵–10⁶                                             | If <10⁴: go to T3                                                    |
| T3             | ECFP4 Tanimoto ≥ 0.35 to any fragment                                                    | 10⁵–10⁷                                             | If <10⁴: lower threshold to 0.30; if >10⁷: raise to 0.40             |
| T4 (intersect) | T3 ∩ (Tanimoto ≥ 0.35 to any pKD ≥ 3.5 SPR active)                                       | If T1–T3 produce ≥10⁷ candidates, this narrows them | Always also save as a "high-confidence" cohort                       |

Engineering details:

1. Pure-SQL substructure isn't possible. Use a two-pass approach:
    - **Coarse pass (DuckDB, in S3 or local Parquet):** for each fragment, generate a few canonical-SMILES *substring*
      heuristics (e.g., its longest aromatic substring) and prefilter with
      `WHERE smiles LIKE '%...%' OR smiles LIKE '%...%'`. This is sloppy but cheap and reduces 3.4B → ≤10⁸.
    - **Confirmation pass (RDKit, parallelized):** run `HasSubstructMatch` on the prefiltered set with
      `multiprocessing.Pool`. Tag each hit with the matched fragment ID(s) and cluster(s). Persist to Parquet.
2. Track which fragment(s) and cluster(s) each candidate matches — this becomes the "predicted binding pocket"
   annotation downstream.
3. If T1 has to escalate to T3/T4, switch from RDKit substructure to RDKit fingerprint Tanimoto with the same parallel
   pattern (compute fingerprints once per shard).

### 3.4 Physchem and chemistry filters (Nate, ~30–60 min)

Apply in this order, with row counts logged after each filter (assertion-driven, per Harms Informatics conventions):

1. Hard Chordoma filters: `LogP ≤ 6`, `HBD ≤ 6`, `HBA ≤ 12`, `MW ≤ 600`.
2. Lead-like soft filters: `10 ≤ heavy_atoms ≤ 30`, `HBD + HBA ≤ 11`, `cLogP < 5`, `num_ring_systems < 5`,
   `max_fused_rings ≤ 2`.
3. PAINS / problematic motifs filter:
    - RDKit `FilterCatalog` with PAINS_A, PAINS_B, PAINS_C catalogs.
    - Custom SMARTS for: acid halides, aldehydes, diazo, imines, > 2 fused benzene rings, long alkyl chains (≥ 6
      contiguous sp³ CH₂), reactive Michael acceptors not justified by fragment match.
4. Optional: drop compounds with Tanimoto ≥ 0.85 to the existing 1,545-compound SPR training set's *inactives* (
   defensible: don't waste budget re-screening near-duplicates of known weak binders).

Output `data/processed/onepot_focused_vl.parquet` with columns:
`compound_id, smiles, mw, logp, hbd, hba, heavy_atoms, num_rings, fused_ring_max, tpsa, rotatable_bonds, matched_fragment_ids, matched_cluster_ids, similarity_to_fragments_max, passes_hard_filters, passes_softer_filters`.

**Target final size:** 10⁵–10⁷. If significantly outside this range, go back to §3.3 and adjust the cascade. Do not
enter the hackathon with an unfiltered VL.

### 3.5 Pre-hackathon code scaffolding (both, parallel, ~1 hr each)

Write training/inference scripts so the hackathon-day work is just `uv run python train_<x>.py`:

- `notebooks/02-xgb-baseline.ipynb` (Nate): Morgan FP (radius 2, 2048 bits) → `XGBClassifier` at pKD ≥ 5.0,
  scaffold-stratified 5-fold CV, save model + inference function.
- `scripts/train_chemprop.py` (Ray): D-MPNN, same label, same scaffold split, save checkpoint.
- `scripts/infer_pair_model.py` (Nate, stretch): Siamese / pair-ranking model that takes (query, reference) → ΔpKD
  prediction. Reference panel = the top-50 pKD compounds in the SPR set. Inference produces a query-vs-reference rank
  score.
- `scripts/train_transfer.py` (Ray): start from a pretrained chemical foundation model (e.g., MoLFormer or ChemBERTa)
  and fine-tune on TBXT pKD.

Lock down: featurization function (shared `src/tbxt_hackathon/featurize.py`), train/val split (shared seed), label
threshold (pKD ≥ 5.0).

---

## 4. Hackathon day (May 9, 6 hours)

Times below are anchors, not deadlines. The 30-min sync points are non-negotiable.

### Hour 0:00 — Sync and kickoff (15 min)

- Confirm both laptops can read `data/processed/onepot_focused_vl.parquet`.
- Confirm Ray's GPU is reachable; Boltz weights are downloaded; Vina is installed locally.
- Re-confirm the active threshold (pKD ≥ 5.0) and the agreement criterion (see §4.3).
- Each person opens their model script and starts training in the background.

### Hour 0:15–2:00 — Step 2: Multi-model ML training and inference

Parallel work:

- **Nate:** retrain XGB on full SPR set (no held-out test now; all 1,545 compounds are training); generate `p_active`
  for every row of the focused VL. Run pair-model inference if scaffolded successfully — score each VL row against the
  top-50 reference panel, take median rank.
- **Ray:** retrain ChemProp on full SPR set; run inference on focused VL. Same for the transfer model.

Expected runtimes (rough, depends on VL size):

- XGB inference on 10⁷ Morgan FPs: <10 min on CPU
- ChemProp inference on 10⁷ SMILES: 30–90 min on a single GPU; consider a **first-stage 10x downsample** (predict on a
  random 10% subset) if total VL > 5×10⁶ to ensure hour-2 sync hits on time
- Transfer model inference: similar to ChemProp

If any model fails to train, fall back to: XGB only is acceptable; ChemProp is highly desirable; transfer + pair models
are stretch.

### Hour 2:00 — Sync 1

Tally model coverage. Compute pairwise Spearman rank correlation between models (sanity: should be 0.3–0.7 — too low
means broken model, too high means redundant).

### Hour 2:00–2:30 — Step 2 cont.: Ensemble and agreement filter

- Rank each compound within each model (lower rank = more active).
- Compute **rank product** across available models. Optionally weight ChemProp higher (typically the strongest single
  learner on small chemical sets).
- Select top compounds by rank product, requiring each surviving model to have placed the compound in its top X% (e.g.,
  top 5%). This implements "highly predicted by all models" cleanly.
- Target: 1,000–5,000 compounds out.

### Hour 2:30–3:00 — Step 3: Naar-similarity dedup and diversity selection

- Compute ECFP4 fingerprints of the 135 Naar compounds.
- For each surviving compound, max Tanimoto to any Naar compound.
- **Drop** compounds with max Tanimoto ≥ 0.6 (avoids prior-art duplication, which the brief explicitly rewards
  avoiding).
- **Drop** compounds whose Bemis–Murcko scaffold is identical to any Naar compound with pKD ≥ 4.0 (more conservative
  duplication check).
- From survivors, select ~50 for Boltz triage by:
    - Murcko-scaffold cluster the survivors (sklearn AgglomerativeClustering on Tanimoto distance, or use Butina). Aim
      for ≥ 10 clusters.
    - Pick the top 5 per cluster by ensemble rank.
    - Force balance across the 6 SGC fragment clusters (i.e., each cluster represented by ≥ 5 candidates if possible).

Output: `data/processed/boltz_input.parquet` (≤ 50 rows).

### Hour 3:00–4:30 — Step 4: Boltz pose triage (Ray)

Boltz inputs: TBXT DBD sequence (residues 42–219, from UniProt O15178) + each ligand SMILES. ~50 compounds at ~1.5 min
each on a single modest GPU = ~75 min of pure compute; budget 90 min including I/O.

Reference set to also run through Boltz (so we can compare candidate scores to a known-active baseline):

- **Z979336988**, **Z795991852**, **D203-0031** (the three Naar compounds with prior CF SPR data, sites F or F/G).
- 3–6 SGC fragments from the most populated clusters as positive controls.

Capture per compound: predicted complex structure (CIF), iPTM, predicted aligned error, Boltz binding-affinity score if
available, predicted pocket residues.

**Survivor criterion:** candidate must (a) score better than the median Naar reference and (b) localize to a site that
overlaps with at least one SGC fragment cluster's residues.

Parallel work for Nate during this window: prepare receptor for Vina (PDBQT from PDB 6F59 chain A; one search box per
known site F, G, plus any other site that the Boltz reference set occupies).

### Hour 4:30 — Sync 2

Review Boltz survivors with Ray. Pick the top 10–15 for Vina.

### Hour 4:30–5:00 — Step 5: Vina confirmation (Nate)

AutoDock Vina on top 10–15 against 6F59 receptor.

- Use site-specific search boxes per the Boltz-predicted site (cover sites F and G at minimum).
- 3D-conformer prep with Meeko or RDKit (`AllChem.EmbedMolecule` + `MMFFOptimizeMolecule`).
- Run with `exhaustiveness=16`, `num_modes=9`.
- For each compound, record: top-pose Vina score, RMSD vs. Boltz pose (sanity check), key contact residues (visual
  review or PLIP).

Survivor criterion: Vina score within 1 kcal/mol of best per-site, and pose qualitatively consistent with Boltz pose.

### Hour 5:00 — Sync 3

Spread the Vina/Boltz/ML evidence across a small grid in a notebook. Pull up structures in py3Dmol or PyMOL for each
remaining candidate.

### Hour 5:00–5:45 — Step 6: Final 4 selection and rationale

Pick the final 4 to maximize:

1. Multi-model ML rank-product (top decile of survivors).
2. Boltz iPTM and predicted affinity strictly better than the median Naar reference.
3. Vina score (consistent with Boltz pose).
4. **Scaffold diversity** — 4 different Murcko scaffolds. This is a hard constraint; do not put all four eggs in one
   basket.
5. **Site diversity** — at least 2 distinct SGC fragment-cluster sites covered.
6. **Tractability** — already in OnePot CORE (synthesis-on-demand by definition).

Document each choice in `submission/final_4.csv` with required fields:

- `rank` (1–4)
- `smiles`
- `compound_id` (OnePot)
- `predicted_site` (e.g., "F", "G", "fragment cluster 3")
- `rationale` (~3 sentences: what makes this compound the rank-N pick, what evidence agreed)
- `evidence` (XGB rank, ChemProp rank, transfer rank, pair-model rank, Boltz iPTM, Boltz affinity, Vina kcal/mol, max
  Tanimoto to fragments, max Tanimoto to Naar)
- `properties` (MW, LogP, HBD, HBA, heavy atoms, ring systems, TPSA)

### Hour 5:45–6:00 — Submission and sanity check

Pre-submission checklist:

- [ ] All 4 SMILES parse cleanly with RDKit and round-trip to canonical form.
- [ ] All 4 satisfy hard Chordoma filters (LogP ≤ 6, HBD ≤ 6, HBA ≤ 12, MW ≤ 600).
- [ ] None contain explicit PAINS hits (re-run RDKit `FilterCatalog`).
- [ ] All 4 are present in OnePot CORE (verify by ID lookup).
- [ ] Rank order is intentional and defensible; rationale aligns with rank.
- [ ] Site assignment is supported by at least Boltz, ideally Boltz + Vina.
- [ ] No two of the 4 share a Bemis–Murcko scaffold.
- [ ] At least 2 SGC fragment-cluster sites are represented.

Submit. Save final state, push to git, archive `data/processed/` and `submission/`.

---

## 5. Risks and contingencies

| Risk                                                                   | Likelihood | Mitigation                                                                                                                 |
|------------------------------------------------------------------------|------------|----------------------------------------------------------------------------------------------------------------------------|
| OnePot file format unexpected (e.g., InChI not SMILES, malformed gzip) | Medium     | §3.1 schema probe surfaces this today, not Day 0                                                                           |
| Substructure cascade yields zero or 10⁸+                               | Medium     | §3.3 explicitly defines the fallback ladder                                                                                |
| ChemProp install/CUDA issue on Ray's machine                           | Medium     | Resolve before Day 0; XGB-only is an acceptable fallback                                                                   |
| Boltz fails on some ligands (hydrogen handling, exotic atoms)          | Medium     | Carry redundancy: 50 → expect ~40 successes                                                                                |
| ML poorly calibrated due to small/narrow training set                  | High       | Use rank-product, not absolute probability; require multi-model agreement                                                  |
| Top-ranked set is one scaffold                                         | Medium     | Diversity selection step in §4 forces ≥10 clusters                                                                         |
| Vina box placement wrong                                               | Medium     | Use Boltz pose as the box centroid; 6F59 has DNA in the active region — verify the receptor is DNA-stripped before docking |
| Time slip past hour 5                                                  | High       | Submission is hard-deadlined; cut Vina before submission, not the writeup                                                  |

---

## 6. What changes if things go right

If §3.3 yields a clean 10⁶-compound focused VL today, we're well-positioned. If it yields 10⁸+, we'll add a stricter
filter (e.g., require ≥2 fragment hits, or intersect with similarity to top SPR actives) and rerun before Day 0.

If Step 2's models all converge with reasonable scaffold-CV AUC (say, ≥ 0.7), the ensemble will narrow nicely and the
Naar dedup step will leave us with hundreds, not thousands. If CV AUC is poor (< 0.6 across all models), we lean harder
on the structural evidence (Boltz + fragment-match strength) and treat ML purely as a coarse prefilter.

---

## 7. Open items before Day 0

1. **OnePot delivery confirmation.** Confirm S3 URL, expiration, credentials.
2. **SGC fragment SMILES extraction.** Pull the 29 fragments + cluster + site from the TEP datasheet/Newman 2025
   supplementary.
3. **Site definitions.** Compile a list of pocket residues for sites F, G (and any others in the TEP) so Boltz outputs
   can be assigned to a site automatically.
4. **Pair model architecture decision.** Specify exact architecture before scaffolding: Siamese GNN with margin loss, or
   pairwise concatenated-FP regressor for ΔpKD. Default to the latter unless Nate has a stronger preference.
5. **GPU access for Ray confirmed.** Boltz throughput estimate (50 compounds in ~90 min) assumes a single A100-class
   GPU. Confirm.

---

## 8. References

- Hackathon brief: https://docs.google.com/document/d/1K2r_7HopkH1_4jsGTrZZX8zR1FKgMTxMe-43dG4xJuA
- TBXT TEP datasheet (SGC, 2024): https://www.thesgc.org/sites/default/files/2024-05/TBXT_TEP_datasheet_v1_0.pdf
- Newman et al., *Nat Commun* 2025: https://doi.org/10.1038/s41467-025-56213-1
- PDB 6F59 (TBXT DBD bound to DNA): https://www.rcsb.org/structure/6F59
- UniProt O15178 (TBXT): https://www.uniprot.org/uniprotkb/O15178/entry
- Naar SMILES (prior screen, deduplicate
  against): https://docs.google.com/spreadsheets/d/1k-vcM_jVd1s_6W6u-ag2YQabGov_40oB
- onepot CORE: https://www.onepot.ai/
- Experimental prize tiers: https://tbxtchallenge.org/#prizes