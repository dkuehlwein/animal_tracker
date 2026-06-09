---
id: 4
slug: mog2-recurrent-frames
status: running           # proposed | running | concluded | rolled_back | parked
validation: live          # live | replay | parked
hypothesis: "Many FP are recurrent near-identical static scenes MOG2 should have absorbed; diagnose why before tuning"
param_delta: null         # none yet — diagnosis phase, no deploy
predicted_effect: { fp_rate: "potentially large -", fn_risk: "unknown (motion-sensitivity changes risk FN)" }
created: 2026-06-09
started: 2026-06-09
concluded: null
decision: null            # keep | rollback | inconclusive
baseline: { fp_rate: 0.798, fp_ci: [0.700, 0.870], fn: unmeasured }   # 06-08; see trust caveat below
result:   { fp_rate: null, fp_ci: null, fn: null }
confidence: null
---

## Hypothesis & method

Daniel's 2026-06-09 observation (backlog #4): a large share of FP triggers are
recurrent, near-identical static scenes that MOG2 background subtraction should
have absorbed into the background model. Candidate causes: central-region
weighting re-amplifying the same edge motion; `motion_threshold` (500px) low
enough that residual MOG2 noise clears it; MOG2 learning rate / `history=500`
vs trigger cadence; shadow/lighting drift. **Diagnose before tuning.**

## Daily observations

(Append-only. Never rewrite prior observations — anti-self-poisoning.)

### 2026-06-09 — first new data day; diagnosis advanced; deploy HELD

Ingested 185 new detections (watermark 84 → 269), all 06-09 daytime (hours 7–18).
47 carry human labels. Measured **FP 0.616** (114/185), 95% CI [0.544, 0.683]
vs the 06-08 baseline 0.798 — but see the trust caveat: this drop is **not a
validated win**.

**Self-audit (CRITICAL — auto-labels untrustworthy).** Tier-1 auto-labels agree
with humans only **17/47 (36%)**, and the disagreements are systematically
biased: 24/30 are tier1="animal" where the human said "false_positive" (the
auto-labeler calls FPs "animal"), plus 4 tier1="false_positive" vs human
"wrong_species". Human-label distribution on the 47: 43 false_positive, 4
wrong_species, **0 confirmed true animals**. Implications:
- The reconciled FP rate (0.616) is an **underestimate** — auto-labels under-count
  FP on the ~138 unlabeled rows. True FP is higher.
- The 0.798 → 0.616 "improvement" is **not trustworthy**; both days lean on the
  same biased auto-labeler. Do not treat it as a win.
- The "animal" group below is contaminated with real FPs, weakening every
  FP-vs-animal comparison.

**#4 diagnosis — motion features do NOT separate FP from animal.** Per-trigger
aggregates (medians, reconciled labels): motion_area FP 1143 vs AN 1142;
contour_count 122 vs 145; foreground_pixel_count 6036 vs 6343; largest_contour
669 vs 752 — all overlapping. **0/114 FP** sit anywhere near the 500px threshold
(min FP motion_area 806). So a `MOTION_THRESHOLD`/area bump cannot separate FP
from animals — it would cut both and **raise FN** → FN-veto HOLD (FN unmeasured).

The recurrence hypothesis (Daniel's) is about *temporal/spatial* repetition of
near-identical scenes, which these per-trigger aggregates cannot test. We log
`contour_count`, `largest_contour_area`, `foreground_pixel_count`, `motion_area`
— but nothing capturing **scene recurrence** (trigger ROI location or a
frame-similarity signal). That is the missing instrumentation.

**Decision: HOLD (no deploy).** No env knob reaches the root cause (threshold
tuning is FN-vetoed and futile per above). The concrete next step is
**observability-only** instrumentation (zero FN risk, takes effect on restart),
then re-diagnose on the next night's data:
- *Option A — ROI centroid:* log the largest-contour centroid (x,y) per trigger.
  Cheap; tests "same edge re-triggers" (spatial clustering). Weak for full-scene
  near-duplicates.
- *Option B — perceptual/aHash of the motion ROI (or downscaled frame):* log an
  8×8 average-hash. Directly tests "near-identical recurrent scene" via Hamming
  distance between triggers. Stronger fit to Daniel's wording; slightly more code.
- Either adds one nullable column (schema migration), a field in `MotionResult`,
  a compute in `motion_detector`, a pass-through in `wildlife_system`, and an
  ingest projection. Multi-file + migration → best designed/reviewed with Daniel
  rather than shipped blind in an autonomous tick. Flagged in tonight's summary.

Cross-cutting: the **label-trust** problem (auto-labels 36% concordant, biased
low) gates *every* FP experiment's validation, not just #4. Until auto-labeling
is fixed or enough human labels accrue, FP "wins" cannot be trusted. The
shadow-gate (#1) remains the cleanest lever — re-confirmed tonight: of 92
`gate_would_suppress=True`, 88 are FP + 4 wrong_species, **0 true animals**
(100% precision, zero FN on the labelled corpus) — but #1 is still infra-blocked
(no live gate in `notification_service.py`).

## Decision & rationale

(Filled when concluded. Current: running/diagnosis, HOLD on deploy; next action
is observability instrumentation for recurrence, pending design review.)

## 2026-06-09 (late, human-directed) — CORRECTION: recurrence WAS testable on existing data; CONFIRMED

Correction to the diagnosis above. The claim that scene recurrence "needs new
instrumentation (ROI centroid / aHash column), flagged for design review" was wrong —
it was answered immediately on **existing data**, no schema change, no new logging.

**Method.** For each `detections` row whose `image_path` file still exists, load the
saved JPG → grayscale → resize 8×8 → threshold at the frame mean → 64-bit average-hash
(aHash) fingerprint. Order by `timestamp`; greedy-cluster members within Hamming
distance ≤ 6 of a cluster representative.

**Result** (100 of 269 triggers still had frames on disk — cleanup had pruned the
older ~169): **80% of time-adjacent trigger pairs were near-identical**, and the 100
triggers collapsed into **~15 visual scenes**. Two dominate: **62 triggers over ~6 h**
(11:48–17:38) and **23 triggers** (13:51–18:53). Daniel's "all pretty much the same
images" is confirmed.

**Corrected mechanism — these are NOT static scenes MOG2 failed to absorb.** The
cluster frames show a fixed, bright sunlit garden (bird-feeder on a shepherd's hook,
lavender, flowers, wooden bench). The *framing* recurs; each trigger is driven by
genuine small motion — the hanging feeder swinging, vegetation swaying in wind, moving
sun-dapple. MOG2 is an inter-frame **change** detector, not a scene-**novelty**
detector, so it correctly fires. It does not absorb the motion into background because
(a) wind motion is wide-amplitude / non-periodic, so the per-pixel Gaussian mixture
never settles into a stable mode, and (b) after every trigger the motion loop stops
sampling for ~45 s (30 s cooldown + ~17 s burst+species-ID), so the model goes stale
and re-reads the moved vegetation as fresh foreground rather than learning it.
intra-cluster 8×8 Hamming ≤ 7 confirms the moving region is small enough to be
invisible at hash resolution yet still clears the 500 px motion gate. This also
explains the earlier "motion features don't separate FP from animal" finding —
vegetation/feeder motion has animal-scale `motion_area` (cluster medians ~1090).

**Implication (supersedes the "add instrumentation" next step).** "Fix MOG2 to absorb
static scenes" is the wrong frame; the scenes aren't static. Real levers:
1. **Scene-recurrence / dedup gate** — suppress repeat triggers matching a known-FP
   scene fingerprint (the aHash above; computable live on the motion frame at ~no cost,
   no schema change). Most direct answer to the observation. FN risk: a real animal
   entering a previously-FP scene must not be deduped — gate on motion-region change,
   not whole-frame identity.
2. **SpeciesNet no-animal gate (#1)** — these triggers have 0 confirmed animals; the
   gate was 100 % precision on the labelled set but remains infra-blocked.
3. **Vegetation/feeder-motion suppression** — ROI masking is awkward here (the feeder
   is also where real birds land); color/texture or animal-shape contour filters are
   safer. Any sensitivity change is FN-gated.
