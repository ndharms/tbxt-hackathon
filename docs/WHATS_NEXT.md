# What's Next — Hackathon Day Status Report

**Date:** May 9, 2026 (hackathon day)
**Team:** Nate Harms, Raymond Gasper

---

## 1. Inventory of completed work

### Data preparation (main, both contributors)
- SPR dataset cleaned: 2,153 records, 1,913 non-null pKD, 1,680 unique compounds
  across 14 batches (`data/zenodo/tbxt_spr_merged.csv`, `notebooks/01-data-prep.ipynb`)
- Binder labels derived: top-quartile (pKD >= 3.837) and strict active (pKD >= 4.5)
  thresholds both available (`data/processed/tbxt_compounds_labeled.csv`)
- Chemical-space folds: PaCMAP + KMeans6, 6 folds with fold 4 as structural holdout
  (`data/folds-pacmap-kmeans6/`)

### Pocket mapping and fragment catalog (main, Nate)
- 45 SGC TEP fragments extracted with per-pocket assignments: A=26, B=5, C=10, D=4
  (`data/structures/sgc_fragments.csv`)
- Newman pocket nomenclature adopted (A/C/D active, B dropped)
- Pocket-assigner built: substructure + max ECFP4 Tanimoto scorer
  (`src/tbxt_hackathon/pocket_assigner.py`)
- Multi-pocket assignment analysis: O0P dual-binder falsifies single-assignment;
  recommendations for multi-pocket routing documented (`docs/pocket-mapper-soundness.md`)

### SAR diagnostics (chemeleon branch, Ray)
- Batch effects explain 15% of pKD variance (R^2 = 0.155 from date alone)
- 30% activity-cliff rate at Tanimoto >= 0.7
- kNN-5 pKD std = 67% of global std (local SAR is flat)
- CheMeleon embedding probes lose to train-mean baseline (data ceiling, not model ceiling)
- Folds and batches are confounded (holdout fold dominated by recent batches)
- Pocket assignment does NOT explain activity cliffs (same-pocket cliff rate 37% > cross 27%)

### Boltz pose validation (prot branch, Ray)
- Pocket A: good pose, R180 anchor contact
- Pocket B: tilted vs crystal, DROPPED
- Pocket D (Newman): looks okay
- Pocket C: looks okay
- Nomenclature confusion between TEP/Newman/Ray resolved in `brachyury-site-summary.md`

---

## 2. Models built

### Ray's models (on main via chemeleon merge)

**Classification (binder vs non-binder, pKD >= 3.837 threshold):**

| Ensemble | OOF AUROC | Holdout AUROC | OOF Brier |
|---|---|---|---|
| chemeleon_no_val | **0.655** | 0.475 | **0.171** |
| chemeleon_with_val | 0.648 | 0.428 | 0.185 |
| xgb_no_val (Morgan 2048 + physchem 8) | 0.632 | 0.456 | 0.216 |
| xgb_with_val | 0.632 | 0.463 | 0.201 |

- No statistically significant differences between ensembles (TukeyHSD p > 0.22)
- All holdout AUROCs below 0.5 (batch-confounded holdout fold)
- Best use: soft binder prior (AUROC 0.655), not a reliable screener

**Regression (pKD prediction):**

All four ensembles have negative R^2 (worse than predicting the mean). Best Spearman
0.11. Conclusion: regression is not recoverable on this dataset. Confirmed by
embedding probes — this is a data ceiling, not a model ceiling.

### Nate's models (on `nate/xgboost-poc`, NOT merged)

**XGBoost binary classifier:**

| Config | AUC-ROC | AUC-PR | F1 |
|---|---|---|---|
| MACCS+Desc, SPR-only, 50 trials | **0.791** | **0.431** | **0.492** |
| MACCS+Desc, Enamine decoys, 50 trials | 0.843 | 0.400 | 0.430 |

- Stricter labels: pKD >= 4.5 active, pKD <= 3.5 inactive, ambiguous excluded
- Butina-clustered 5-fold CV (no separate holdout fold)
- SPR-only negatives outperform Enamine decoys on AUC-PR and calibration
- Saved models: `models/xgb_classifier/` (5 CV + 1 final)

**Pairwise comparison classifier:**
- Predicts relative potency (A more potent than B by >= 1.0 pKD)
- Diff-only features f(A)-f(B): AUC-PR 0.401, ranking Spearman ~0.32
- Complementary signal for ranking, not a replacement for the binary classifier
- Saved models: `models/pairwise_classifier/`

### XGBoost retrained on PaCMAP-KMeans6 folds (`models/xgb_classifier_pacmap_folds/`)

To get an apples-to-apples comparison with Ray's models, we retrained Nate's XGBoost
classifier (same MACCS+Desc features, same 4.5/3.5 labels) on Ray's PaCMAP-KMeans6
folds (5-fold CV on folds 0,1,2,3,5; fold 4 as structural holdout).

| Fold | AUC-ROC | AUC-PR | F1 | n_val | n_pos |
|---|---|---|---|---|---|
| 0 | 0.434 | 0.070 | 0.000 | 184 | 9 |
| 1 | 0.728 | 0.415 | 0.463 | 233 | 53 |
| 2 | 0.556 | 0.126 | 0.000 | 129 | 12 |
| 3 | 0.824 | 0.471 | 0.225 | 332 | 64 |
| 5 | 0.580 | 0.297 | 0.250 | 178 | 14 |
| **Mean** | **0.625** | **0.276** | **0.188** | — | — |

**Holdout (fold 4):** AUROC 0.567, AUPRC 0.055, Brier 0.063 (baseline 0.038).
Only 4 positives in holdout — metrics are extremely noisy.

**Optimal OOF threshold:** 0.254 (Precision 0.391, Recall 0.599, F1 0.473)

### How the models compare

| Model | Folds | Label threshold | OOF AUROC | OOF AUC-PR | Holdout AUROC |
|---|---|---|---|---|---|
| XGBoost (Butina folds) | 5-fold Butina | 4.5 / 3.5 | **0.791** | **0.431** | n/a |
| XGBoost (PaCMAP folds) | 5+1 PaCMAP-KMeans6 | 4.5 / 3.5 | 0.625 | 0.276 | 0.567 |
| CheMeleon no_val (Ray) | 5+1 PaCMAP-KMeans6 | 3.837 (Q3) | 0.655 | n/a | 0.475 |

The PaCMAP folds are substantially harder than Butina folds: OOF AUC-PR drops from
0.431 to 0.276. This is expected — PaCMAP-KMeans6 creates structurally distinct folds
that break scaffold-level correlations, while Butina clusters keep similar scaffolds
together (easier leakage). The per-fold variance is also extreme: fold 3 (AUC-PR 0.471)
vs fold 0 (AUC-PR 0.070), driven by class imbalance across folds (fold 0 has only 9
positives vs fold 3 with 64).

The holdout fold 4 is particularly challenging: only 4 positives with Nate's stricter
labels. Neither model can meaningfully discriminate at this sample size.

### How we should use them

1. **Primary screener**: Nate's XGBoost binary classifier with Butina folds
   (AUC-PR 0.431) for triaging onepot compounds. Its folds are less demanding but
   the model is better calibrated at the decision boundary.
2. **Conservative estimate**: The PaCMAP-fold model (AUC-PR 0.276) gives a more
   honest upper bound on how well this classifier will generalize to truly novel
   chemistry. Expect ~28% precision-recall on structurally distant compounds.
3. **Soft prior**: Ray's chemeleon_no_val (OOF AUROC 0.655) as a second opinion.
4. **Ranking signal**: Nate's pairwise classifier for ordering within a shortlist.
5. **NOT for regression**: no model should be used to predict absolute pKD.

---

## 3. Pipelines built but not yet run

### OnePot CORE downfilter (`nate/downfilter-pipeline`, NOT merged)
- Streams 3.4B compounds from S3 CSV.gz
- Coarse SMILES regex prefilter -> RDKit substructure match against 45 fragments
- Hard filters: MW <= 600, LogP <= 6, HBD <= 6, HBA <= 12, HBD+HBA <= 11,
  10-30 heavy atoms, <= 5 rings, <= 2 fused, PAINS, custom bad SMARTS
- Output: `data/processed/onepot_focused_vl.parquet`
- Status: **CODED AND TESTED (smoke test), NOT RUN ON FULL CATALOG**
- This is the critical-path blocker for everything downstream

### AutoDock Vina docking (`nate/vina-docking`, NOT merged)
- Single-compound mode with 3D visualization and interaction analysis
- Batch mode for scoring hundreds/thousands of SMILES
- Pocket A PDB prepared (`data/structures/TGT_TBXT_A_pocket.pdb`)
- Requires conda env (Vina doesn't pip-install on Apple Silicon)
- Status: **PIPELINE READY, NEEDS CANDIDATES**

---

## 4. Current plan vs. reality

### Step 1: Fragment-based catalog filtering (onepot 3.4B -> ~10K per pocket)
- **Plan**: pre-hackathon complete
- **Reality**: downfilter pipeline coded but NOT run. No filtered catalog exists.
  Running the full 3.4B rows will take hours. This is the single biggest gap.

### Step 2: Per-pocket active learning with Boltz (~1000 compounds/pocket)
- **Plan**: Ray runs Boltz batches during hackathon, trains surrogates
- **Reality**: Boltz Lab cloud validated for poses (prot branch), but no
  compounds ready to submit. Blocked on Step 1.

### Step 3: Pocket D manual Boltz (~20 compounds)
- **Plan**: Ray selects top ~20 by fragment similarity
- **Reality**: Not started. Also blocked on Step 1 (need pocket-D filtered pool).

### Step 4: Vina docking validation (top ~50 across pockets)
- **Plan**: Nate runs Vina on post-Boltz candidates
- **Reality**: Pipeline ready (nate/vina-docking). Blocked on Steps 2-3.

### Step 5: Final selection (4 compounds)
- **Reality**: Not started. Depends on everything above.

---

## 5. What needs to happen now

### Immediate (merge and run)

- [x] **Merge `nate/downfilter-pipeline` to main** — DONE
- [x] **Merge `nate/xgboost-poc` to main** — DONE
- [x] **Merge `nate/vina-docking` to main** — DONE
- [ ] **Reconcile `origin/prot` with main.** The prot branch diverges heavily
      (deletes most of main's content). The useful content (Boltz pose validation,
      nomenclature notes) should be extracted, not merged wholesale.
- [ ] **Start the full OnePot downfilter run.** This is the critical path.
- [x] **Retrain XGBoost classifier with PaCMAP-KMeans6 folds** for direct
      comparison with Ray's models on the same splits (`scripts/build_xgb_classifier_pacmap_folds.py`).
      Output: `models/xgb_classifier_pacmap_folds/`

### Parallel work while downfilter runs

- [ ] **Prepare pocket C and D PDBs for Vina** (only pocket A PDB exists).
- [ ] **Implement multi-pocket assignment** per the soundness analysis
      recommendations (allow 1-2 pocket assignments per compound for O0P-like cases).
- [ ] **Set up Boltz Lab batch submission** — does the cloud platform support
      programmatic submission, or is it manual? This determines whether the active
      learning loop is feasible in 6 hours.

### Decisions needed

- [ ] **Should we run the downfilter on all 3.4B rows or a subset?** A 100M-row
      run would finish in ~30 min and give us a sample to start working with. The
      full run could take 6+ hours, which is the entire hackathon.
- [ ] **Should we drop the Boltz surrogate plan?** If the downfilter produces
      a manageable number of hits (< 5K), we could score all of them with Boltz
      directly instead of training a surrogate. This removes a risky step (surrogate
      Spearman > 0.6 is uncertain).
- [ ] **Should we use the XGBoost classifier as the primary screener instead of
      Boltz?** The classifier can score the entire downfiltered catalog instantly.
      Boltz is expensive and the active learning loop may not converge in 6 hours.
      Alternative plan: downfilter -> XGBoost score -> top 50 per pocket -> Boltz
      pose validation -> Vina confirmation -> pick 4.
- [ ] **What to do about the `prot` branch?** It rewrites the plan and deletes
      Ray's model artifacts. The Boltz pose validation notes are valuable. Should
      we cherry-pick the useful parts?

---

## 6. Plan revisions to consider

### The original plan assumed the downfilter was already done
The entire timeline (Step 2 onward) depends on having a filtered catalog. Since we
don't have one, the plan needs to compress or restructure. Options:

**Option A: Run downfilter with --limit, proceed with partial results**
- Run with `--limit 500000000` (~500M rows, maybe 1-2 hours)
- Score survivors with XGBoost + pocket-assigner
- Pick top candidates per pocket for Boltz and Vina
- Accept that we may miss some good compounds in the unscanned tail

**Option B: Skip downfilter, use fragment similarity directly**
- Query a small sample of onepot via API (if available) or use the fragment
  SMILES as seed queries for onepot's own search tool
- Score returned compounds with our classifiers
- Fastest path to candidates, but limited chemical diversity

**Option C: Full downfilter + simplified downstream**
- Start the full downfilter run immediately
- While it runs, prepare everything else (Boltz targets, Vina boxes, scoring scripts)
- When results start appearing (sharded parquet), score partial results as they come
- Skip the surrogate training entirely; use XGBoost + direct Boltz for top hits only

### The Boltz active learning loop may be too ambitious for 6 hours
The plan calls for ~2400 Boltz predictions, surrogate training, and re-scoring.
Each Boltz prediction takes unknown time on the cloud platform. If we can't do
programmatic submission, manual submission of 1000 compounds is infeasible.

Simpler alternative: use the classifiers + pocket-assigner to rank compounds,
then Boltz-validate only the top ~20-50 per pocket. This eliminates the surrogate
step entirely and is feasible even with manual Boltz submission.

---

## 7. Unmerged branch summary

| Branch | Owner | Key content | Merge strategy |
|---|---|---|---|
| `nate/xgboost-poc` | Nate | XGBoost binary + pairwise classifiers, saved models, modeling notes | Merge to main |
| `nate/vina-docking` | Nate | Vina docking pipeline, pocket A PDB, demo notebook | Merge to main |
| `nate/downfilter-pipeline` | Nate | OnePot CORE streaming downfilter, fragment extractor | Merge to main, then run |
| `origin/prot` | Ray | Boltz pose validation notes, nomenclature mapping, rewritten plan | Cherry-pick Boltz notes only |
| `origin/chemeleon` | Ray | Classification/regression models, SAR diagnostics, folds | Already merged via PR #2 |
