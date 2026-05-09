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

All four ensembles trained on PaCMAP-KMeans6 folds (5-fold CV on folds 0,1,2,3,5;
fold 4 as structural holdout). Label: pKD >= 3.837 (top quartile). N=1,468 CV
compounds, 131 holdout. 25.9% prevalence.

**Classification (binder vs non-binder):**

| Ensemble | OOF AUROC | OOF AUPRC | Precision | Recall | F1 | MCC | Holdout AUROC |
|---|---|---|---|---|---|---|---|
| chemeleon_no_val | **0.655** | **0.404** | 0.312 | **0.842** | 0.455 | 0.184 | 0.475 |
| chemeleon_with_val | 0.608 | 0.357 | 0.299 | 0.739 | 0.426 | 0.123 | 0.428 |
| xgb_no_val (Morgan 2048 + physchem 8) | 0.632 | 0.364 | 0.291 | 0.911 | 0.441 | 0.152 | 0.455 |
| xgb_with_val | 0.632 | 0.351 | 0.326 | 0.734 | 0.452 | 0.182 | 0.463 |

- No statistically significant differences between ensembles (TukeyHSD p > 0.22)
- All holdout AUROCs below 0.5 (batch-confounded holdout fold)
- All models have precision below 33%: ~7 in 10 predicted positives are false
- MCC 0.12–0.18 across models — barely above random after class-imbalance correction
- Best use: high-recall filter (catches 84–91% of actives) when false positives are cheap

**Regression (pKD prediction):**

All four ensembles have negative R^2 (worse than predicting the mean). Best Spearman
0.11. Regression is not recoverable on this dataset. Confirmed by embedding probes —
this is a data ceiling, not a model ceiling.

### Nate's models (on main)

**XGBoost binary classifier (Butina folds):**

| Config | AUC-ROC | AUC-PR | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| MACCS+Desc, SPR-only, 50 trials | **0.791** | **0.431** | **0.479** | 0.506 | 0.492 | — |
| MACCS+Desc, Enamine decoys, 50 trials | 0.843 | 0.400 | — | — | 0.430 | — |

- Stricter labels: pKD >= 4.5 active, pKD <= 3.5 inactive, ambiguous excluded
- Butina-clustered 5-fold CV (no separate holdout fold)
- SPR-only negatives outperform Enamine decoys on AUC-PR and calibration
- Saved models: `models/xgb_classifier/` (5 CV + 1 final)
- **Caveat:** Butina folds are easier than PaCMAP folds (see §3 below)

**XGBoost binary classifier (PaCMAP-KMeans6 folds):**

Retrained on Ray's folds for direct comparison. Same features (MACCS 167 + RDKit
descriptors 217 = 384), same 4.5/3.5 label thresholds.

| Fold | AUC-ROC | AUC-PR | Precision | Recall | F1 | N | Pos |
|---|---|---|---|---|---|---|---|
| 0 | 0.434 | 0.097 | 0.143 | 0.222 | 0.174 | 184 | 9 |
| 1 | 0.728 | 0.429 | 0.377 | 0.755 | 0.503 | 233 | 53 |
| 2 | 0.556 | 0.159 | 0.333 | 0.083 | 0.133 | 129 | 12 |
| 3 | 0.824 | 0.482 | 0.494 | 0.672 | 0.570 | 332 | 64 |
| 5 | 0.580 | 0.308 | 0.217 | 0.357 | 0.270 | 178 | 14 |
| **Mean** | **0.625** | **0.276** | — | — | — | — | — |

**OOF aggregate (N=1056, 14.4% prevalence, threshold 0.254):**

| Metric | Value |
|---|---|
| AUC-ROC | 0.747 |
| AUC-PR | 0.366 |
| **Precision** | **0.391** |
| Recall | 0.599 |
| F1 | 0.473 |
| MCC | 0.374 |

Confusion matrix: TN=762, FP=142, FN=61, TP=91

**Holdout (fold 4):** AUROC 0.567, AUPRC 0.084, Precision 0.063. Only 4 positives
in holdout — metrics are extremely noisy and not interpretable.

**Pairwise comparison classifier:**
- Predicts relative potency (A more potent than B by >= 1.0 pKD)
- Diff-only features f(A)-f(B): AUC-PR 0.401, ranking Spearman ~0.32
- Complementary signal for ranking, not a replacement for the binary classifier
- Saved models: `models/pairwise_classifier/`

---

## 3. Model comparison and analysis

### Direct comparison (all on PaCMAP-KMeans6 folds, OOF)

| Model | Label | N | Prev | AUROC | AUPRC | Prec | Rec | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|
| Ray chemeleon_no_val | >=3.837 | 1468 | 25.9% | 0.655 | 0.404 | 0.312 | 0.842 | 0.455 | 0.184 |
| Ray xgb_with_val | >=3.837 | 1468 | 25.9% | 0.632 | 0.351 | 0.326 | 0.734 | 0.452 | 0.182 |
| **Nate XGB (PaCMAP)** | **4.5/3.5** | **1056** | **14.4%** | **0.747** | **0.366** | **0.391** | **0.599** | **0.473** | **0.374** |

### Why Nate's model has higher precision despite simpler architecture

The label threshold is the primary driver, not the model. Three effects compound:

1. **De-noised labels.** The strict 4.5/3.5 thresholds with ambiguous exclusion remove
   ~412 compounds near the decision boundary where within-compound replicate std (~0.92
   pKD) makes the true label unknowable. Ray's top-quartile threshold at 3.837 forces
   the model to discriminate pKD 3.5 from 4.0 — a distinction the assay cannot
   reliably make.

2. **Lower prevalence.** 14.4% vs 25.9% positive rate means fewer positives to find,
   but the remaining positives are higher-confidence. The model's discriminative power
   (AUC-ROC 0.747 vs 0.655) benefits from cleaner signal.

3. **Conservative threshold.** At F1-optimal threshold 0.254, the model produces 142
   false positives vs Ray's 706 (chemeleon_no_val). For screening, this 5x reduction
   in false positives matters more than the recall loss (60% vs 84%).

### Butina vs PaCMAP folds

| Folds | OOF AUC-ROC | OOF AUC-PR | OOF Precision | OOF F1 |
|---|---|---|---|---|
| Butina 5-fold | 0.969 | 0.885 | 0.889 | 0.880 |
| PaCMAP-KMeans6 | 0.747 | 0.366 | 0.391 | 0.473 |

The Butina model's 89% precision is inflated by scaffold leakage. Butina clusters at
Tanimoto 0.4 keep chemically similar compounds in the same cluster, so validation
folds contain scaffolds seen during training. The PaCMAP folds create structurally
distinct splits that break scaffold-level correlations. The PaCMAP numbers are the
honest estimate of generalization to novel chemistry.

Per-fold variance is extreme on PaCMAP folds: fold 3 (AUC-PR 0.482, 64 positives) vs
fold 0 (AUC-PR 0.097, 9 positives). This is driven by class imbalance across folds
and the batch-fold confounding Ray identified.

### How we should use them

1. **Primary screener**: Nate's XGBoost (PaCMAP folds) for triaging onepot compounds.
   Best precision (39%) and MCC (0.374) at reasonable recall (60%). Expect ~4 in 10
   predicted hits to be truly active on structurally novel chemistry.
2. **High-recall sweep**: Ray's chemeleon_no_val as a second pass to catch compounds
   the primary screener misses (84% recall). Accept the higher false-positive rate
   (31% precision) and filter downstream with docking.
3. **Ranking signal**: Nate's pairwise classifier for ordering within a shortlist.
4. **NOT for regression**: no model should be used to predict absolute pKD.
5. **NOT the Butina model for novel chemistry**: its 89% precision does not generalize;
   use only as a sanity check on known scaffolds.

---

## 4. Pipelines built but not yet run

### OnePot CORE downfilter (`scripts/downfilter_onepot/`)
- Streams 3.4B compounds from S3 CSV.gz
- Coarse SMILES regex prefilter -> RDKit substructure match against 45 fragments
- Hard filters: MW <= 600, LogP <= 6, HBD <= 6, HBA <= 12, HBD+HBA <= 11,
  10-30 heavy atoms, <= 5 rings, <= 2 fused, PAINS, custom bad SMARTS
- Output: `data/processed/onepot_focused_vl.parquet`
- Status: **CODED AND TESTED (smoke test), NOT RUN ON FULL CATALOG**
- This is the critical-path blocker for everything downstream

### AutoDock Vina docking (`scripts/vina_docking/`)
- Single-compound mode with 3D visualization and interaction analysis
- Batch mode for scoring hundreds/thousands of SMILES
- Pocket A PDB prepared (`data/structures/TGT_TBXT_A_pocket.pdb`)
- Requires conda env (Vina doesn't pip-install on Apple Silicon)
- Status: **PIPELINE READY, NEEDS CANDIDATES**

---

## 5. Current plan vs. reality

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
- **Reality**: Pipeline ready. Blocked on Steps 2-3.

### Step 5: Final selection (4 compounds)
- **Reality**: Not started. Depends on everything above.

---

## 6. What needs to happen now

### Immediate (merge and run)

- [x] **Merge `nate/downfilter-pipeline` to main** — DONE
- [x] **Merge `nate/xgboost-poc` to main** — DONE
- [x] **Merge `nate/vina-docking` to main** — DONE
- [x] **Retrain XGBoost with PaCMAP-KMeans6 folds** — DONE
      (`scripts/build_xgb_classifier_pacmap_folds.py`,
      output: `models/xgb_classifier_pacmap_folds/`)
- [x] **Compute traditional classification metrics** for all models — DONE (see §3)
- [ ] **Reconcile `origin/prot` with main.** The prot branch diverges heavily
      (deletes most of main's content). The useful content (Boltz pose validation,
      nomenclature notes) should be extracted, not merged wholesale.
- [ ] **Start the full OnePot downfilter run.** This is the critical path.

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

## 7. Plan revisions to consider

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

### Revised recommended pipeline
Given the model analysis, the most pragmatic pipeline is:

1. **Downfilter** (partial or full) -> fragment-matched focused library
2. **XGBoost score** (Nate's PaCMAP model, instant) -> rank by P(active)
3. **Pocket-assign** (pocket-assigner) -> route to pockets A, C, D
4. **Top 20-50 per pocket** -> Boltz pose validation (manual submission OK)
5. **Vina confirmation** on Boltz-validated poses -> final ranking
6. **Pick 4** across pockets, prioritizing pocket A (most fragment data)

This avoids the surrogate training step entirely and can work even with a
partial downfilter run. The XGBoost classifier replaces the Boltz surrogate
as the primary scoring function; Boltz becomes a pose-validation step rather
than a scoring oracle.

---

## 8. Unmerged branch summary

| Branch | Owner | Key content | Merge strategy |
|---|---|---|---|
| `origin/prot` | Ray | Boltz pose validation notes, nomenclature mapping, rewritten plan | Cherry-pick Boltz notes only |
| `origin/chemeleon` | Ray | Classification/regression models, SAR diagnostics, folds | Already merged via PR #2 |
