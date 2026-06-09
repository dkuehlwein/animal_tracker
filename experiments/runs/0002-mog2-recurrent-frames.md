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
