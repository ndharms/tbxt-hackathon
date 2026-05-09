# Pocket-Mapper Soundness and Multi-Pocket Assignment

## Question

Is the pocket-mapper (substructure match + ECFP4 Tanimoto against Newman fragment
co-crystals) a sound way to route catalog compounds to TBXT binding pockets, given
that at least one fragment (O0P, 5QSC) was crystallized at two different pockets?

## Summary

The mapper is sound as a triage filter for pockets with strong fragment evidence
(A, C), and low-confidence for pockets with n ≤ 5 (D, B). The single-assignment
design is falsified by the O0P dual-binder and should be relaxed to allow
multi-pocket assignment, with Boltz acting as the per-pocket tiebreaker.

## Evidence gathered

Fragment CSV: `data/structures/sgc_fragments.csv` (45 entries).

Pocket sizes:

| Pocket | Fragments |
|---|---|
| A  | 26 |
| B  |  5 (dropped downstream) |
| C  | 10 |
| D  |  4 |

Inter-pocket ECFP4 Tanimoto (mean / max, excluding identical SMILES):

|   | A | B | C | D |
|---|---|---|---|---|
| **A** | 0.17/0.65 | 0.14/0.47 | 0.15/0.46 | 0.12/0.42 |
| **B** | 0.14/0.47 | 0.13/0.19 | 0.13/0.31 | 0.13/0.20 |
| **C** | 0.15/0.46 | 0.13/0.31 | 0.16/0.42 | 0.16/**1.00** |
| **D** | 0.12/0.42 | 0.13/0.20 | 0.16/**1.00** | 0.21/0.33 |

Cross-pocket substructure overlaps at ≥6 heavy atoms: **0**. The substructure-match
rule (the scorer's dominant signal, `combined = 1.0 + Tc`) is internally clean.

Self-assignment validation: **44/45 correct**. The single failure is
**5QSC/O0P** (`CN(Cc1cccc(c1)F)S(=O)(=O)N`), crystallized at both C and D.
The scorer returns `combined = 2.00` for both and the tie is broken by
pocket-size (C has more fragments than D), routing O0P and any O0P-like
query to C regardless of which pocket is more biologically appropriate.

## Soundness by pocket

### Where confidence is warranted

- **Pocket A (n=26, progressed SAR)**: substructure-match rule rests on real
  Newman SAR (thiazole → morpholino-thiazole → cyclopropylacetamide series,
  5QS9 → 8A7N). Any catalog compound containing a bona-fide subgraph of one of
  these fragments has a genuine prior on occupying the A pocket.
- **Pocket C (n=10)**: enough fragments for a meaningful max-Tanimoto floor;
  distinct pharmacophore (E48/E50/R54/K76 polar hotspot) means compounds
  mis-routed from C to A tend to flag on chemistry alone (C wants polar heads,
  A wants lipophilic extension).
- **Pockets are physically distinct**: ligand centroids 16–40 Å apart after Cα
  alignment (per `brachyury-site-summary.md`). The biological question being
  asked ("which pocket does this compound resemble?") is well-posed.

### Where confidence is NOT warranted

1. **Small-n pockets (D: 4, B: 5).** With n=4, a single unusual fragment dominates
   the per-pocket similarity distribution. Any compound assigned to D on Tanimoto
   alone (no substructure match) is resting on very thin evidence.

2. **The O0P dual-binder falsifies the single-pocket assumption.** It crystallized
   at both C and D. The scorer cannot represent this — it must pick one. The
   current tiebreaker (pocket-size) silently biases the D submission pool against
   O0P-like chemistry, because anything resembling O0P gets routed to C.

3. **Max inter-pocket Tanimoto between C and D is 1.00** (the O0P pair). A–C
   is 0.46, A–D is 0.42. Non-trivially-similar compounds exist across pockets
   even setting O0P aside. Borderline catalog hits (Tc 0.4–0.6, no substructure)
   get forced into a single pocket when crystallography would permit two.

4. **Fragment-count asymmetry biases assignment.** Pocket A has 26 fragments;
   any query gets 26 shots at a high max-Tanimoto vs. 4 shots for D. Near-threshold
   compounds drift toward A on statistics alone. "Compound X assigned to A" is
   weaker evidence than "compound X assigned to D" — the opposite of what the
   fragment-count-as-evidence framing suggests.

5. **Labels are structural, not mechanistic.** The mapper has no information
   about pocket geometry, druggability, or induced fit. Pocket D is partly
   ligand-induced; resemblance to its 4 fragments is weaker evidence than it
   looks for a static-pocket assigner.

## Proposed change: multi-pocket assignment

Allow the mapper to return 1–2 pocket candidates per compound and let Boltz
decide per-pocket whether the pose actually works. This is more faithful to the
crystallography (O0P is literal proof) and shifts the work to the stage (Boltz
+ pose QC) where geometry is actually evaluated.

### Rules for when to assign multiple pockets

1. **Substructure match in >1 pocket** → assign all such pockets. This is the
   O0P case. `combined ≥ 1.0` across multiple pockets is unambiguous evidence
   of multi-pocket plausibility.
2. **Top combined scores within ~0.1–0.15** → assign both. Captures near-ties
   where the size-based tiebreaker is doing work it should not.
3. **Otherwise**: single assignment as today.

Expected expansion: most compounds remain single-assigned; only genuinely
ambiguous ones fan out. Budget ~5–10% Boltz-run inflation.

### API change (sketch)

```python
def assign_all(smiles: str) -> list[str]:
    """Return all pockets meeting the multi-assign criteria (1 or more)."""
    ...

def assign(smiles: str) -> str | None:
    """Existing single-best assignment, unchanged."""
    ...
```

Downstream Boltz submission iterates over `assign_all` output.

### Guardrails when interpreting multi-pocket Boltz results

- **Never rank across pockets with raw Boltz scores.** Different pockets produce
  different score distributions. Use per-site z-scores or percentile rank against
  the other compounds assigned to that same pocket.
- **Pose QC is per-pocket and mandatory.** Anchor-residue hard filters from
  `brachyury-site-summary.md`: A must H-bond R180, C must contact E48/R54,
  D must contact Y88/D177. A compound tested at A and D with a clean R180 pose
  but a Y88-miss is one hit at A, not two options.
- **Picking the final pocket when both pass QC**: prefer the pocket with more
  fragment evidence (A > C > D) and the cleaner pose. If only one pocket
  passes, that is the answer.
- **Pocket D specifically**: multi-assign *into* D increases the D candidate
  pool (helps the n=4 problem) but does not fix the induced-pocket caveat.
  Still manually inspect poses there and weight fragment-substructure similarity
  over Boltz absolute scores.

## Summary table: current vs. proposed behavior

| Case | Current behavior | Proposed behavior |
|---|---|---|
| Substructure in 1 pocket | Assign that pocket | Same |
| Substructure in >1 pocket (O0P-like) | Pick larger pocket by size | Assign all, test each in Boltz |
| Top-1 Tc ≫ top-2 | Assign top-1 | Same |
| Top-1 and top-2 Tc within 0.1–0.15 | Assign top-1 (size tiebreak) | Assign both |
| No match ≥ threshold | None | Same |

## Framing

Multi-pocket assignment is not hedging. It is refusing to discard information
the crystallography already provided. The single-assignment design was a
convenience that did not survive contact with O0P.
