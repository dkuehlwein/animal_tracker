---
id: 24
slug: dusk-blind-divergence
status: running
validation: live   # code change + paired env delta, restart-gated; verified end-to-end on the real leak frames
occupies_active_slot: false  # defect repair of exp #21's shipped mechanism, not a new tuning lever — exp #21 keeps the slot
hypothesis: "Exp #21's burst human sweep cannot fire in low light. Its trigger counts pixels differing by >40 RAW grey levels, so it measures the scene's dynamic range as much as its content: at a frame mean of 11/255 almost no pixel pair can clear 40 levels, however different the pictures are. Dusk is precisely when motion blur makes the selected frame likeliest to misread a person, so the sweep is blind where it matters most. Normalising each frame to a canonical mean/contrast before differencing restores the measure to content."
created: 2026-09-12
promoted_from: "found during night-3 tier-2 adjudication of exp #21: burst 5169 held a recognisable person, two of its five frames return `human` at >=0.93, and the sweep never ran."
confidence: high   # reproduced on the real frames with the real model; the fix re-ranks the three known person bursts to 1/2/3 corpus-wide
commit: c5fe171
env_delta: {"PERFORMANCE_HUMAN_SWEEP_MAX_FRAMES": 4}
restart_at: 2026-09-13T03:25:00+02:00
---

## The miss

**Burst 5169, 2026-09-12 19:17:14, `unclassifiable`, `person_confidence=0.034`,
review-class.** Its five frames show a person in a red shirt walking across the
garden carrying a large patterned cloth — the composition changes completely
from frame to frame (the cloth fills the right half of frame 1 and is gone by
frame 5). Re-run frame by frame with the real model:

| frame | status | raw top-1 | score |
|---|---|---|---|
| frame1 | unclassifiable | sentinel | 0.608 |
| frame2 ← selected | unclassifiable | vehicle | 0.514 |
| frame3 | **human** | homo sapiens | 0.934 |
| frame4 | no_animal | — | — |
| frame5 | **human** | homo sapiens | 0.970 |

This is exactly exp #21's leak class — a review-class burst whose selected
frame does not represent it, with a person sitting in the siblings — and exp
#21 did not fire. `_frame_divergence` returned **0.0005** against every
sibling, 60x under `T=0.03`, so `candidates` was empty and no sweep ran.

Nothing about the burst was subtle. The frames are unrelated pictures. The
measure is what failed: it counts pixels differing by >40 **raw** grey levels,
and this burst has a mean of 11.1 and a std of 4.8 out of 255. At that
exposure no pixel pair can differ by 40 levels no matter what is in front of
the lens. The same magnitude of content change in daylight (burst 5119) scored
0.2146.

Only the Human-Proximity window saved it — a HUMAN-status burst 96 s earlier
armed the anchor. Had this been the **leading** burst of the visit, the frames
of a recognisable person would have gone to REVIEW and stayed on disk for the
full rotation. That is the precise scenario exp #21 exists to prevent.

## Why this is structural, not bad luck

Low light is the regime where the sweep matters most and where it was
guaranteed to be inert:

- Dynamic range collapses at dusk (tonight: std 4.8 at 19:17 vs ~40 at midday),
  and a fixed 40-level test scales with dynamic range, not with content.
- The per-frame verdict is *least* stable there — one burst returned
  unclassifiable / unclassifiable / human 0.93 / no_animal / human 0.97 — so
  the divergence between the selected frame's verdict and its siblings' is at
  its largest exactly where the trigger reads zero.
- Motion blur is worst at dusk (all four evening bursts scored below the
  sharpness floor), which is what makes the selected frame misread a person in
  the first place.

Same failure shape as exp #23: a shipped guard that is inert on the case it
was built for, hidden behind a plausible number. Exp #21's threshold was
validated on two daylight leaks and every night since has been daylight-quiet,
so tonight is its first positive test — and it is a miss.

## The fix

`_frame_divergence` now normalises both downsampled frames to zero mean and a
canonical contrast (std 50) before the >40-level test, i.e. it asks whether
the frames disagree by more than **0.8 standard deviations of frame contrast**
rather than by 40 absolute levels. A floor of 1.0 on the divisor keeps a
near-constant frame from amplifying sensor noise without bound; real dusk
frames sit at std 3–6, well above it.

Measured over **all 274 on-disk review-class bursts**:

| rank | burst | raw | normalised | contents |
|---|---|---|---|---|
| 1 | 5169 | 0.0005 | **0.5838** | **person (tonight's miss)** |
| 2 | 5119 | 0.2146 | **0.2060** | **person (face, exp #21 leak 1)** |
| 3 | 5096 | 0.1694 | **0.1915** | **person (leg, exp #21 leak 2)** |
| 4 | 5172 | 0.0000 | 0.0893 | empty, dusk |
| 5 | 4987 | 0.0819 | 0.0731 | empty, sun/shade |

All three known person bursts now rank above every empty burst in the corpus,
with a >2x margin under the lowest of them, so `T=0.03` stands unchanged and
keeps its pre-registered meaning. The two original leaks score essentially what
they scored before (0.21→0.206, 0.169→0.192): **the daylight validation exp #21
did carries over intact.** Sweep rate goes from 10/274 (3.6%) to 25/274 (9.1%)
— inside the ~10% envelope exp #21 pre-registered and costed. Per night over
09-01..09-12 that is 0–9 sweeps, median 2.

Pure exposure shifts now score ~0, since normalisation removes brightness and
contrast. Those were the main source of pointless sweeps before (4939 at 0.067,
4959 at 0.055 — both sun/shade transitions, now 0.027/0.030).

## Paired env delta: `PERFORMANCE_HUMAN_SWEEP_MAX_FRAMES` 2 → 4

Fixing the trigger alone would still have missed 5169. All four siblings score
0.49–0.58 — when every frame of a burst is equally unlike the selected one, the
*ranking* among them carries no information about which holds the person, and
the two human frames rank 3rd and 4th. With the cap at 2 the sweep would check
frame4 (no_animal) and frame1 (unclassifiable) and stop.

At 4 the sweep checks every sibling of a 5-frame burst and stops at the first
human. In-bounds (`BOUNDS` = (0, 4)), and the pre-registered lever for exactly
this. Cost: a SpeciesNet pass measured at 11.5 s tonight, so ≤46 s per swept
burst with no person present (a real leak stops early), ~2 swept bursts/night →
~90 s of extra blind time per night against a ~10 h active window; the worst
day in the corpus (09-09, 9 sweeps) would cost ~7 min.

## Phantom-escalation cost, measured not assumed

The 15 bursts that newly sweep are mostly daylight empties plus 2 dark ones.
Ran the real model over the top-2 siblings of four of them, chosen to include
both the newly-swept dark burst and the purely normalisation-induced daylight
ones (5172, 5089, 4996, 4999 — 8 frames): **every one came back review-class,
zero HUMAN.** No new phantom anchors observed in this sample. Exp #14's phantom
cost is the thing to watch here and it did not appear.

## Gates

- **FN (animals):** the sweep only converts review-class → HUMAN, and
  review-class already means no animal was found on the selected frame. No
  animal alert that fires today stops firing. Escalation needs the model's own
  `human` verdict on a real frame, not a heuristic.
- **FN (blind time):** ~90 s/night, see above.
- **Volume:** ~1 fewer review send/night at most; tonight it would have changed
  nothing that was actually sent (5169 was already proximity-muted, 5172 was
  sampled out). No collapse.
- Feedback-starved: no (human labels arrived 2026-09-11). Paused: no.
- Slot: exp #21 keeps it; this is a defect repair of its shipped mechanism,
  same precedent as exp #23 against exp #9. The two are separable in the record
  — a `[HUMAN-SWEEP]` escalation of a **dusk** burst is this fix's evidence.

## Verification

- 597/597 deterministic tests pass. Two new: the same content change rendered
  at noon and at dusk amplitude must both clear the threshold (the regression
  test for tonight), and a pure exposure shift must not. The existing helper
  had to stop writing flat grey frames — under a contrast-normalised measure a
  flat frame carries no content, which is the point.
- End-to-end on the real frames with the real model and the shipped method:
  5169 → 0.584 (sweeps; frame3 returns `human` 0.934 within the new cap of 4,
  so the burst is escalated, suppressed and purged at 48 h), 5119 → 0.206,
  5096 → 0.192 (both still sweep, unchanged behaviour).

## Prediction

The next dusk burst holding a person in a sibling frame produces a
`[HUMAN-SWEEP]` escalation instead of a REVIEW send. Falsified if a swept dusk
burst escalates to HUMAN on frames containing no person (phantom anchor), or if
sweep volume exceeds ~4/night sustained. Rollback: `git revert c5fe171`, or
`PERFORMANCE_HUMAN_SWEEP_MAX_FRAMES=0` / `_DIVERGENCE_THRESHOLD=0` to disable
the sweep entirely.

## Night 1 (2026-09-13) — the repair fires in the regime it was built for

Live since the 03:30 restart (commit `c5fe171` + `PERFORMANCE_HUMAN_SWEEP_MAX_FRAMES=4`).

Only two review-class bursts existed all day, but one of them is precisely the
test case: **burst 5210, 19:00:59, `no_animal`, frame mean 14.4 / std 10.9** —
dusk, and its selected frame1 holds a dark out-of-focus limb of someone walking
past the lens while frames 2–5 are the empty pond.

| measure | frame2 | frame3 | frame4 | frame5 | verdict at T=0.03 |
|---|---|---|---|---|---|
| old, raw 40-level | 0.0027 | 0.0029 | 0.0029 | 0.0029 | **inert** (10x under) |
| new, contrast-normalised | 0.1782 | 0.1353 | 0.1379 | 0.1387 | **4 candidates** (4.5–6x over) |

This is the predicted failure and the predicted repair measured on a *new*
burst, not on the one the fix was written from. The old measure would have swept
nothing; the new one swept all four siblings (the raised cap is what let it
reach all of them).

No escalation followed, and that is correct: the person is in the **selected**
frame, not a sibling. The gate scored it `person_confidence=0.303`, just under
the deployed 0.5 (exp #14's measured operating point — not to be reverted on this
single instance, which sits in the band where 32/33 adjudicated bursts were
empty). Privacy held anyway: the Human-Proximity **density** condition muted it
(`>= 8 human detections in the last 1800s`), and the ±240 s retention-proximity
window will purge its frames at 48 h alongside HUMAN burst 5211, 19 s later.
Adjudicated: a dark blurred limb, no face, not recognisable.

Sweep cost this night: 4 model runs (~46 s) on one burst, zero phantoms — inside
the costed envelope.

Exp #21 (the sweep itself) is now 4 nights live and has **never** escalated a
burst. Not evidence of failure yet — the burst supply is tiny (2 review-class
bursts today, 6–12 on recent days) — but it is the second consecutive night
where the mechanism's own trigger, not its judgment, is all that could be
verified.
