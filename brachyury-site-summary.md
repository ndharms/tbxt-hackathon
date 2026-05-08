# TBXT Binding Site / Pocket Mapping

Source of truth: Newman et al., *Nat Commun* 16:1596 (2025) — "Structural insights into human brachyury DNA recognition and discovery of progressible binders for cancer therapy." https://www.nature.com/articles/s41467-025-56213-1

The Newman paper defines **five pockets**: A, A', B, C, D. Verified by superposing all four PDBs on 5QS2 Cα and computing contact distances to Newman's signature residues.

## Verified mapping: our site labels → Newman pocket labels

All four sites are genuinely distinct (ligand centroids 16–40 Å apart after Cα alignment).

| Our label | PDB | Ligand | Newman pocket | Evidence (closest approach, Å) |
|---|---|---|---|---|
| **A site** | 5QS2 | LV4 | **Pocket A'** | R180: 2.64, V123: 3.86, L91: 4.04, I125: 4.19, V173: 5.95, I182: 6.67. Matches Newman's A' pharmacophore exactly. S89 is 7.8 Å (close but not the anchor — pockets A and A' are on the same β-sheet face). |
| **D site** | 5QS0 | NZ4 | **Pocket B** | Direct ligand contacts to G112/H100/G113/P115/P111/K114 loop; E116 at 7.96 Å (Newman: "loop 116–120, M159, E116"). This is the only site within reach of strand *e'* residue E116. |
| **F site** | 5QSA | K2P | **Pocket D** | Y88: 2.79, D177: 4.43, V173: 4.33, I182: 4.82, M181: 4.50, T183: 4.48. Y88 is the P300-interaction residue Newman specifically calls out for pocket D; D177 is the G177D variant residue itself. |
| **G site** | 5QS6 | K0G | **Pocket C** | R54: 3.36, E48: 3.81, E50: 3.93, L51: 4.50, K76: 4.80. Newman: "R54, E48, K76, near C-terminal end of final helix." Exact match. |

**Previous mapping in this file was wrong.** The AI summary had the "G site" labeled as pocket D — in reality the G site is pocket C, and our F site is pocket D.

## Background on the crystal systems

Fragment screen used two different apo crystal forms:

| Crystal form | Construct | Space group | Contains Cd²⁺? | Pockets Newman identified |
|---|---|---|---|---|
| WT apo (5QS0, 5QS2 etc.) | res 41–211 WT | P 4₁ 2 2 | yes (5 surface sites) | A (24 hits), B (4 hits) |
| G177D apo (5QSA, 5QS6 etc.) | res 41–211 G177D | H 3 2 | no | A' (3 hits), C (9 hits), D (4 hits) |

Note the paradox: 5QS2 is in a WT crystal yet the ligand binds where Newman describes pocket A'. The two pockets are on the same hydrophobic β-sheet face (both involve R180, V123, I125, V173, I182), separated essentially by which fragment happened to bind and which face was accessible in that crystal packing. For our purposes, treat A and A' as **the same region of the protein with slightly different sub-site occupancy**. "A site" (5QS2 LV4) and Newman's optimized thiazole series (5QS9 → 8A7N) are in equivalent chemical space.

## What's known about each pocket as a drug target

### Our A site = Pocket A' / A region (5QS2 and related WT/G177D structures)
- **Druggability**: borderline — ICM Drug-Like-Density score comparable to Bcl-2, lower than a typical kinase. Largest pocket identified on the protein.
- **Evidence**: this is the **most-validated site on brachyury**. 24 WT fragment hits in pocket A, 3 G177D fragment hits in pocket A', and the only fragment series Newman progressed to low-µM (14–20 µM SPR K_D) with the thiazole → morpholino-thiazole → cyclopropylacetamide series (5QS9 → 5QSD → 7ZK2 → 8A7N).
- **Pharmacophore**: R180 H-bond + hydrophobic cluster (L91/V123/I125/V173/I182 — conserved across T-box family). S89 H-bond also seen for a subset.
- **Therapeutic rationale**: within ~3 Å of the palindromic dimer interface. Hits here could disrupt cooperative DNA binding without directly competing with DNA, or serve as PROTAC warhead anchor (degradation MoA is validated for brachyury via dTAG).
- **Caveats**: mostly lipophilic surface patch, not a deep pocket — ligand efficiency will be hard. Two binding modes seen for parent thiazole; Newman had to engineer selectivity for one mode. WT crystals have CdCl₂; one fragment bound primarily via Cd²⁺ (keep in mind for 5QS2).

### Our D site = Pocket B (5QS0)
- **Druggability**: moderate. Only 4 fragment hits in Newman's WT screen.
- **Pharmacophore**: polar contacts to main chain of loop 116–120 (partially disordered in DNA-bound structures), side chains M159, E116. Our NZ4 ligand contacts the adjacent 111–115 loop and H100.
- **Therapeutic rationale**: ~8 Å from the DNA interface. Fragments could in principle be grown toward DNA-contact residues — a conventional **DNA-competitive** strategy rather than allosteric.
- **Caveats**: low hit rate in the fragment screen; the contacting loop is partly disordered, so pocket definition is soft. No published potency progression.

### Our F site = Pocket D (5QSA)
- **Druggability**: Newman gives this the **highest druggability score** among all pockets (per the AI summary your original doc quoted). But note: the pocket is partly *induced* by ligand binding — MPD (cryoprotectant) can also occupy it, so ground-state druggability is lower than the static score suggests.
- **Pharmacophore**: buried cavity near the N-terminus, lined by Y88 and including residues around D177 (the G177D variant). 4 fragment hits including a 2-methyl-2,4-pentanediol (from cryo).
- **Therapeutic rationale** — this is the **most biologically interesting pocket**:
  - Y88 is required for brachyury's interaction with **P300** (histone acetyltransferase; Beisaw et al., *EMBO Rep* 2018).
  - Equivalent pocket in murine T-bet mediates interactions with **KDM6A/KDM6B** (H3K27 demethylases) — i.e. this is the **chromatin co-activator recruitment surface**.
  - Disease mutations from multiple human T-box disorders cluster at equivalent positions in family members — functionally important *in vivo*.
  - Distant from both DNA and dimer interfaces → genuinely **allosteric**. Mechanism of inhibition would be disrupting PPIs with co-regulators, not competing with DNA.
  - Includes the G177D variant residue itself — a ligand here would naturally engage both WT and variant forms.
- **Caveats**: ligand-induced / cryo-competed → pocket may be transient. Only 4 hits, no SAR progression published. Highest upside / highest risk.

### Our G site = Pocket C (5QS6)
- **Druggability**: moderate-to-good. 9 G177D fragment hits — the largest G177D hotspot.
- **Pharmacophore**: polar hotspot (R54, E48, K76) near the C-terminal end of the final DNA-binding α-helix.
- **Therapeutic rationale**: in the G177D apo crystal it sits on a crystallographic 2-fold, but Newman notes the pocket is "significantly larger and extends down to the DNA interface" in the full-length DNA-bound structure. Within ~8 Å of the DNA interface → potential fragment growth toward DNA-contact residues.
- **Caveats**: the construct used for the fragment screen was truncated (41–211), removing the C-terminal helix that inserts into the DNA minor groove. The pocket as modelled is partly a consequence of that truncation plus crystal packing, so part of it may not exist in the physiologically relevant full-length form. Interpret with care.

## Screening strategy

**Constraints we are working under:**
- Commercial catalog (onepot, ~3.4B compounds) pre-filtered to molecules that contain at least one substructure from a Newman fragment-screen hit. This bakes in strong pharmacophore priors and dramatically reduces the search space before Boltz sees anything.
- Boltz is our affinity oracle — moderately effective, but with known pose-fidelity problems on at least one of our sites.
- We only submit a handful of compounds (4 ranked SMILES). Opportunity cost per slot is high.
- Newman's best published binders top out at ~14–20 µM SPR K_D. Anything predicting sub-µM on this target is probably wrong, not genius.

### Boltz oracle reliability per site

Key concern: **Boltz's predicted pose at our D site (Newman pocket B, 5QS0) is visibly tilted w.r.t. the crystal pose.** That means the model is not correctly locating the ligand even when given a known binder, so the affinity score it outputs at that site is not anchored to the real geometry. The other three sites reproduce the crystal pose reasonably.

| Our site | Newman pocket | Boltz pose fidelity | Oracle trust |
|---|---|---|---|
| A | A' | OK vs 5QS2 | High — trust the ranking |
| D | B | **Tilted vs 5QS0** | **Low — treat scores as noisy** |
| F | D | OK vs 5QSA | Medium — but see pocket-induced caveat below |
| G | C | OK vs 5QS6 | Medium |

Additional oracle-physics risks beyond pose:
- **F site (pocket D)**: the pocket is partly ligand-induced and normally occupied by MPD cryoprotectant. Boltz assumes a rigid receptor from the input structure, so it will under-estimate the entropic cost of opening the cavity and over-estimate affinity. Also the cavity is buried, which inflates scores for anything that fits geometrically regardless of chemistry.
- **A site (pocket A')**: lipophilic surface patch. Boltz (like most scoring functions) rewards burial of hydrophobic surface, so expect it to favor greasy, high-logP catalog compounds that will probably be insoluble or promiscuous. Watch logP and aromatic ring counts.
- **G site (pocket C)**: sits on a crystallographic 2-fold, and the construct used for screening was truncated. The pocket in our input structure is a blend of real and crystal-artifact features. Scores will be internally consistent but may not reflect in-solution physics.

### Per-site recommended slot allocation (4 total)

Given that each submission needs to credibly bind TBXT, and our oracle is differentially reliable:

- **2 slots → A site (pocket A')**. Most de-risked target. Published µM binders, clear pharmacophore (R180 H-bond + L91/V123/I125/V173/I182 hydrophobic cluster), Boltz pose looks right. Spend the majority of our budget here.
- **1 slot → G site (pocket C)**. Second-most fragment evidence (9 hits). Boltz pose is reasonable. DNA-interface-adjacent angle is differentiated from the A-site story.
- **1 slot → F site (pocket D)** *as a speculative swing*. Novel MoA (disrupts P300/KDM6 recruitment, engages D177 variant). Accept that the oracle is unreliable here; compensate by leaning harder on fragment-substructure similarity and manual pose inspection, not on Boltz scores.
- **0 slots → D site (pocket B)**. The Boltz pose gap makes this the worst-informed site. Only 4 fragment hits, shallow pocket, disordered loop, and the DNA-competitive angle is already covered by G site. **Drop unless something extraordinary comes up.**

### Chemistry / library pre-filters per site

Every catalog compound already contains at least one Newman fragment substructure (baseline filter). On top of that:

- **A site**: prefer compounds containing the **thiazole-acetamide / morpholino-thiazole** Newman substructure (5QS9, 5QSD, 7ZK2, 8A7N). Want an H-bond donor/acceptor reachable to R180 + extension into a hydrophobic vector. Reject logP > 5, aromatic ring count > 3, or obvious PAINS. This is the site where Newman actually progressed SAR, so mimicking that series is the highest-prior-probability play.
- **G site**: prefer compounds containing the Newman pocket-C fragments (9 hits available in the PDB group) with a basic/polar head that can engage E48/E50/R54 — essentially a small cation or H-bond donor array. Avoid purely lipophilic catalog hits here; the pocket is polar.
- **F site**: prefer compounds containing Newman pocket-D fragments (the 5QSA K2P benzoic acid, and the other 3 fragment hits in that pocket). Keep MW modest (<400) and avoid compounds that require the pocket to open substantially beyond what K2P occupies. The pocket is small and buried — bigger scaffolds will not fit regardless of what Boltz says.
- **D site** (if we keep it for any reason): we don't have a reliable oracle here, so any compound we submit must have independently strong justification (e.g. high 3D similarity to NZ4 or another pocket-B fragment hit).

Apply Chordoma Foundation constraints across all sites: LogP ≤ 6, HBD ≤ 6, HBA ≤ 12, MW ≤ 600; ideally 10–30 heavy atoms, HBD+HBA ≤ 11, cLogP < 5, <5 ring systems, ≤2 fused rings; reject acid halides, aldehydes, diazo, imines, >2 fused benzenes.

### Interpreting Boltz scores

- **Rank within a site, not across sites.** Different pockets give different score distributions; cross-site comparison is meaningless. Pick the top 1–2 per site.
- **Always inspect the predicted pose.** If the pose doesn't engage the Newman-fragment-derived anchor residue(s) (R180 for A, E48/R54 for G, Y88 for F), down-weight the score regardless of number.
- **Sanity-check with Newman's compounds.** Dock Newman's actual µM binders (thiazole series for A; K2P for F) through the same Boltz pipeline. If Boltz scores a known 15 µM binder at e.g. -8 kcal/mol, anything we submit scoring tighter than that without a good reason is probably a pose artifact.
- **Expect low-µM at best.** Predictions of sub-µM binding on TBXT should raise suspicion, not excitement.
- **F site specifically**: do not trust absolute Boltz scores. Use Boltz only to triage among compounds that already have strong pocket-D-fragment similarity and a pose that engages Y88 and/or D177.

### Orthogonal validation plans

Cheap checks that don't require more compute but catch oracle errors:

1. **Pose QC against the crystal fragment.** For each top candidate, superpose the Boltz pose on the relevant Newman co-crystal (5QS9/8A7N for A, 5QS6 for G, 5QSA for F). The Newman fragment substructure inside our compound should occupy roughly the same volume as the crystal fragment. If it doesn't, reject.
2. **Anchor-residue contact check.** A-site pose must H-bond R180; G-site pose must contact E48/R54; F-site pose must contact Y88. Enforced as a hard filter, not a soft one.
3. **2D/3D similarity to Newman fragments as an orthogonal scorer.** Tanimoto on ECFP4 plus ROCS-style shape similarity against the relevant Newman fragment give a score that is independent of Boltz physics. If Boltz and similarity disagree strongly, trust the similarity signal more than Boltz.
4. **Second docking tool for final candidates only.** Before committing submission, dock the final 4 with something classical (AutoDock Vina or equivalent) in the same pocket. Not to replace Boltz, just to catch cases where Boltz has produced a geometric nonsense pose we missed.
5. **Known-binder retrodock.** Put Newman's published µM binders (K2P, the thiazole series, NZ4) through our full pipeline. If they don't rank highly and pose-QC correctly, the pipeline has a problem we should fix before trusting novel predictions.

### What we are NOT trying to do

- Not trying to beat Newman's 14 µM thiazole. Realistic goal is credible, fragment-informed, pose-sensible binders — not potency breakthroughs.
- Not trying to cover all 4 sites equally. Oracle reliability and fragment-evidence weight strongly favor A and G.
- Not trusting a single Boltz number in isolation. Every submission should survive pose QC + anchor-contact + similarity cross-check.

## Original notes preserved below

### A site (our original)

looks okay
https://lab.boltz.bio/app/raymond-gasper-tbxt-challenge-ray-nate-9iaH/p/tbxt-5Pyh/targets/90bbcf9c-b9c4-471c-a868-0af327c108e7/structure
https://www.rcsb.org/3d-view/5QS2

Verified: = Newman **pocket A'** (R180/L91/V123/I125/V173/I182 cluster).

### D site (our original)
is tilted w.r.t. crystal structure

https://www.rcsb.org/3d-view/5QS0?preset=ligandInteraction&label_asym_id=G
https://lab.boltz.bio/app/raymond-gasper-tbxt-challenge-ray-nate-9iaH/p/tbxt-5Pyh/targets/3ca29498-8214-4f8a-b9b6-886d5e5e3ae3/structure

Verified: = Newman **pocket B** (loop 111–120 / E116 / H100). This is a genuinely different site from A, ~36 Å away. The "tilt" is real — it's a different pocket on the opposite side of the β-sandwich.

### F site (our original)

looks okay
https://lab.boltz.bio/app/raymond-gasper-tbxt-challenge-ray-nate-9iaH/p/tbxt-5Pyh/targets/6afe49c7-a645-4499-8c15-0655177bd52f/structure
https://www.rcsb.org/3d-view/5QSA

Verified: = Newman **pocket D** (Y88 / D177 N-terminal buried cavity — the P300 / KDM6 co-activator interface).

### G site (our original)

https://www.rcsb.org/structure/5QS6
https://lab.boltz.bio/app/raymond-gasper-tbxt-challenge-ray-nate-9iaH/p/tbxt-5Pyh/targets/223afac9-fafb-4fc1-88d0-42f315f0feee/structure

Verified: = Newman **pocket C** (R54 / E48 / K76 near C-terminal helix). The AI summary in the original doc misidentified this as pocket D — it is pocket C. Everything the AI summary said about "pocket D near N-terminus / G177D / druggable / 5QS6 with K0G" conflated two different sites: K0G in 5QS6 sits at Newman's pocket C, while Newman's pocket D is the Y88/D177 cavity captured in 5QSA.
