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

## Recommendations for our screening effort

1. **All four of our sites are distinct and legitimate** — no redundancy. My earlier claim that A and D were the same pocket was wrong; they're 36 Å apart.
2. **Our A site (Newman A') is the most de-risked** — only brachyury site with published µM binders and a validated pharmacophore. Best target for fast-to-credible submissions.
3. **Our F site (Newman D) is the highest-upside / highest-risk** — genuinely allosteric, disrupts co-activator recruitment, engages the disease variant residue. Novel-MoA angle.
4. **Our G site (Newman C) and D site (Newman B)** are the candidates for DNA-competitive strategies — both approach the DNA interface. G site has more fragment evidence; D site is less characterized.
5. Newman's best published binders are low µM. Expect our onepot hits to land in the same ballpark at best — don't overweight single-digit-µM predicted affinities from virtual screening.

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
