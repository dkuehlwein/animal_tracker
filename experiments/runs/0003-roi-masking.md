---
id: 3
slug: roi-masking
status: concluded         # proposed | running | concluded | rolled_back | parked
validation: live          # live | replay | parked
hypothesis: "Restrict motion ROI to suppress edge vegetation FP; incremental FP reduction without masking real animals"
param_delta: null         # ROI is NOT an env lever (not in guardrails.BOUNDS) — would have been a code change; never warranted
predicted_effect: { fp_rate: "potentially -", fn_risk: "raises FN if animals use the masked band (structurally unmeasured)" }
created: 2026-06-13       # re-opened after camera moved to new location
started: 2026-06-17       # spatial diagnostic run
concluded: 2026-06-17
decision: inconclusive    # keep | rollback | inconclusive — diagnosis succeeded; no viable ROI, no deploy
baseline: { fp_rate: 0.762, fp_ci: [0.615, 0.865], fn: unmeasured }   # 06-16 reference night
result:   { fp_rate: null, fp_ci: null, fn: null }                    # diagnosis-only, no deploy
confidence: medium        # robust direction (every FP band also catches animals); n=18 animals is thin
---

## Hypothesis & method

Backlog #3: a spatial ROI mask over edge vegetation should cut FP without masking
real animals — IF there is an edge band where FPs cluster but animals never appear.
The FN-veto (FN structurally unmeasured) has HELD a deploy for the last three ticks
pending exactly this evidence. ROI is a **code change** (not in `guardrails.BOUNDS`),
so the bar is: prove a zero-observed-FN edge band exists before writing code.

**Method (in-tick, throwaway — `scripts/_tmp_spatial_fp_diag.py`, since deleted):**
For every labeled detection whose burst frames are still on disk, diff consecutive
burst frames (high-res 1920×1080 captures), accumulate the abs-diff, threshold, take
the largest blob's centroid, normalize to [0,1]. Bucket by reconciled label:
`animal` ∪ `wrong_species` = real animal present (n=18); `false_positive` = FP (n=52).
274 labeled rows total; 204 had no frames on disk (retention ~100 bursts) or no
recoverable motion blob → analysis is the recent retention window only.

## Result — no viable ROI; FP and animals are spatially entangled

Centroid distributions overlap heavily; both classes are **center-weighted**
(median ~0.5/0.5), which is expected — the detector's central-region weighting
amplifies center motion for FP and animals alike.

Candidate edge-band masks (fraction of each class falling INSIDE the band):

| band            | animals in | FP in   | verdict                       |
|-----------------|------------|---------|-------------------------------|
| top 15% (y<.15) | 2/18 (.11) | 5/52(.10)| wash — equal hit              |
| left 15% (x<.15)| 3/18 (.17) | 4/52(.08)| **hurts animals MORE than FP**|
| right 15%(x>.85)| 2/18 (.11) | 6/52(.12)| wash                          |
| bottom 15%(y>.85)| 0/18 (.00)| 1/52(.02)| clean but negligible (1 FP)   |

There is **no edge band that removes a meaningful share of FP without removing a
comparable-or-greater share of real animals.** The only zero-animal band (bottom
15%) catches a single FP — not worth a code change + restart. The FN-veto that has
held this experiment is now backed by data, not assumption.

## Decision & cross-experiment significance

**Conclude exp #3: not viable in the current scene. No deploy, no code change.**

This is the third independent axis on which FP and real-animal triggers refuse to
separate in this scene:
- exp #4: motion **magnitude** knobs (threshold/contour/consecutive) don't separate them.
- exp #3 (here): motion **location** doesn't separate them.
- exp #1: therefore the **notification layer** (🔍 REVIEW prefix on no_animal/
  unclassifiable, same channel) is the correct and live mitigation — it sorts FP
  from real *post-trigger* using SpeciesNet, which is the only stage that does separate
  them, at zero motion-FN cost.

Trigger-side FP suppression appears to be a genuine plateau for this camera placement.
Remaining open lever is post-trigger (SpeciesNet-stage) — see exp #2 (parked on replay).

**Caveat / re-open condition:** n=18 animals is thin and time-boxed by retention. If
the camera is repositioned, or if a future night yields a markedly larger animal
sample showing a true animal-free edge band, re-open with the same diagnostic.
