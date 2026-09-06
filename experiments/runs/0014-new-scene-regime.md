---
id: 18
slug: new-scene-regime
status: running
validation: observational   # no code or env change shipped this tick; every trigger-side lever is FN-vetoed for lack of in-scene animal evidence
occupies_active_slot: true  # exp #15 (runs/0013) concluded KEEP this tick; this takes the slot
hypothesis: "The camera was physically re-aimed during the 2026-09-01 outage, from a wide garden view to a tight close-up of the pond and its running fountain. Every scene-derived conclusion in this notebook — the ROI entanglement measurement (#3), the scene-gate similarity distribution (#17), and the volume baseline — was measured in the OLD scene and does not transfer. The new scene's dominant false-positive source is continuously moving water, and it contains zero animal bursts so far, so no trigger-side threshold can be validated in it yet."
created: 2026-09-05
promoted_from: "opened by the 2026-09-05 tick after two consecutive 100%-false-positive days (48 + 33 triggers, 0 animals) forced the question of what changed."
confidence: high   # the scene change is directly observed in saved frames on both sides of a known outage window; the vetoes below are measured, not assumed
---

## The finding: the input distribution changed on 2026-09-02, and the loop did not notice

Sampling one saved frame per day across the whole retention window (22 days,
2026-08-10 → 09-05) shows a single, sharp discontinuity:

- **Through 2026-09-01**: a wide garden view — lawn, planting border, roses, the
  pond area small in the lower-left, heavy circular vignetting.
- **From 2026-09-02 15:51 onward**: a tight close-up of the pond — rocks, water
  surface, bamboo, and the **fountain nozzle jetting water** in the upper right.

The changeover sits exactly inside a known gap in the trigger stream: last
pre-outage trigger **4894 @ 2026-09-01 18:35:33**, first post-restart trigger
**4895 @ 2026-09-02 15:51:29**. That is the ~20 h manual-stop outage already
recorded in runs/0013 (`wildlife-camera.service` stopped 09-01 19:13, restarted
09-02 15:14). So the re-aim and the outage are **one event**: the camera was
stopped, worked on, moved, and restarted. The 09-04 frames independently
corroborate someone working at the pond — burst 4904's frame 1 is a person at
arm's length from the lens.

**The loop ran a full tick on 2026-09-02 and did not see this.** It concluded
exp #14 and shipped exp #15 that night against a scene that no longer existed.
Nothing in the pipeline compares today's framing to yesterday's; `systemctl
is-active` was `active` the whole time, so exp #15's liveness check was correct
and silent. **Service liveness is not scene liveness.**

## What the new scene does

| date | triggers | identified | human | adjudicated verdict |
|---|---|---|---|---|
| 2026-09-04 | 48 | 0 | 2 | 46 review-class, all empty pond |
| 2026-09-05 | 33 | 0 | 0 | 33 review-class, all empty pond |

81 triggers, two days, **zero animals**, fp_rate 1.00 both days. Every burst
adjudicated tonight shows the same static pond with the fountain running. The
water jet and the wind-moved bamboo now occupy the frame's motion-sensitive
centre, so the camera is essentially pointed at a permanent motion source.

Inter-frame motion centroids over tonight's 33 bursts cluster hard in the
upper-right quadrant where the jet and bamboo sit: cx median 0.68 (0.35–0.83),
cy median 0.26 (0.13–0.68), 27/32 above the horizontal midline. Suggestive, but
**not** used as evidence for a mask below — see the veto section.

## Every trigger-side lever is FN-vetoed, by measurement

Three independent discriminators were tested tonight against the animal-evidence
corpus (307 rows: every `identified` burst plus every human `animal` /
`animal_wrong_id` label). All three are entangled.

**1. Motion area — vetoed.** Animals sit *inside* the false-positive band, not
above it:

| corpus | min | p10 | median | max |
|---|---|---|---|---|
| animal evidence (n=307) | 800 | 829 | **1047** | 186728 |
| animal evidence since 08-01 (n=9) | 835 | — | 1287 | 2970 |
| tonight's FP (n=33) | 835 | — | ~1100 | 4917 |

201/307 animal rows are below 1200 px. Raising `MOTION_THRESHOLD` from 800 to
1200 — the smallest step that would dent tonight's volume — would cost roughly
two thirds of all animal detections on record. Rejected.

**2. Contour fragmentation — vetoed.** This one looked genuinely promising:
tonight's FP bursts have a median `contour_count` of **47** (water ripple
scatter), while animal bursts since 07-01 have a median of **3** (one compact
blob). Both fields are in the DB for both classes, so the FN cost is directly
measurable rather than guessed. It does not survive the measurement:

| rule (`contour_count >=` / `largest_contour_area <=`) | FP suppressed (n=81) | animals lost (n=307) | animals lost since 07-01 (n=174) |
|---|---|---|---|
| 10 / 1500 | 44% | 31% | 20% |
| 20 / 1500 | 41% | 25% | 13% |
| 30 / 2000 | 42% | 25% | 11% |
| 40 / 1000 | **26%** | 15% | **5%** |

The best corner of the grid still trades 8 confirmed real animals for ~21 fewer
false alarms, and the animals it discards are ordinary `identified` bursts
(2211 cc=177, 3802 cc=67, 4175 cc=34 …). The reason is physical, not statistical:
**a bird landing at the pond splashes**, so it produces the same fragmented
multi-contour signature as the fountain. Rejected.

**3. Spatial masking — held, cannot be validated.** The centroid clustering above
is the only lever the new scene actually suggests. It cannot be validated: there
are **zero animal bursts in the new scene** to measure a mask against, and
backlog #3's "FP and animals are spatially entangled" result was measured in the
*old* framing and no longer applies in either direction. FN unmeasured + a change
that could plausibly raise FN → the guardrail contract says HOLD.

This is now the **third** independent trigger-side discriminator to fail (area,
space, fragmentation), consistent with #3 and #4. Recorded so future ticks stop
re-deriving it: **in this camera, trigger-side suppression does not separate
false positives from animals.** The architecture answer remains notification-layer
routing, which is where every shipped win in this notebook actually lives.

## What is NOT claimed

- **Not** that the re-aim caused the animal drought. `identified` bursts had
  already decayed to zero *before* the move — 68/week (W27) → 8 (W30) → 5 (W31)
  → 1 (W32) → **0 for W33, W34, W35** — with the last one on **2026-08-16**, in
  the old framing. Weeks 34–35 are also human-dominated (100/136 and 114/238
  triggers are HUMAN-status), i.e. a busy garden. Season, garden use and framing
  are confounded and this notebook cannot separate them.
- **Not** that the framing is wrong. That is Daniel's call about what he wants
  photographed; the loop's job is to report that the current aim points at a
  permanent motion source and has produced no animals in two days.

## Actions taken this tick

1. **No code change, no env delta.** All three levers vetoed above; the honest
   move is to hold rather than ship a threshold that trades animals for quiet.
2. **`baselines.volume_per_night` 192 → 27.** The 192 baseline predates the move
   by weeks (flagged as stale on 09-03) and is the comparison point for the
   volume-collapse guardrail: at 192, `check_volume` would demand a rollback on
   any night under 20 triggers — i.e. on a *normal* night in the new scene
   (09-02: 18, 09-03: 8). 27 is the mean of the four post-move nights (18, 8, 48,
   33); collapse now trips below 3, explosion above 135. Maintenance, not tuning.
3. **Reported to Daniel in plain English** that the camera now looks at the
   running fountain and that this is what generates the false alarms.

## Exit criteria

- **Promote to a real tuning experiment** once ≥5 animal-containing bursts exist
  in the new scene (any of: `identified`, human `animal`/`animal_wrong_id`).
  Then, and only then, re-derive in-scene: the spatial mask above, and backlog
  #17's scene-gate threshold `T = max(animal similarity) + 0.02` (the same rows
  serve both, since `scene_similarity` is now recorded for every status as of
  commit f14ed0d).
- **Conclude as a scene-change record** if the framing changes again or if the
  fountain stops, in which case this regime ends and its numbers should not be
  carried forward.
- **Escalate** if the zero-animal streak reaches ~2 weeks in the new scene: at
  that point "the camera no longer sees animals" stops being a plausible seasonal
  effect and becomes a finding in its own right.

## Standing duties discharged tonight

- All 33 review-class bursts adjudicated: **0 concealed animals, 0 person frames**.
- 5 bursts in the `person_confidence` [0.30, 0.50) watch band (4976, 4983, 4984,
  4986, 4988) — all five inspected frame-by-frame, all empty pond. Exp #14's
  standing duty is clean.
- 0 scene-gate mutes, 0 proximity mutes, 0 HUMAN-status bursts (nothing to leak).
- 6 below-sharpness-floor bursts (4996–5001, dusk) — all empty, none muted
  incorrectly.
- 1 fresh human label today (4987 → `false_positive`), matching my adjudication.
  Not feedback-starved.

---

## Night 2 — 2026-09-06: the FP storm did not recur, and the loop's blind spot got an instrument

22 triggers (12:07–18:54), and the day's shape is completely different from the
two that opened this experiment:

| date | triggers | HUMAN | review-class | identified | review-class sent |
|---|---|---|---|---|---|
| 2026-09-04 | 48 | 2 | 46 | 0 | ~23 |
| 2026-09-05 | 33 | 0 | 33 | 0 | ~16 |
| **2026-09-06** | **22** | **19** | **3** | **0** | **0** |

Water-driven false alarms fell 33 → 3. The day was instead a three-hour human
work session at the pond (14:14–17:02, 19 HUMAN-status bursts), plus one earlier
visitor at 12:07.

### The fountain is still running — the drop is not "the FP source went away"

Measured rather than assumed. For every burst with ≥3 frames on disk, mean
inter-frame absolute difference over the water/jet region (rows 15–60%, cols
40–95% of the frame):

| date | review-class bursts | median water-region inter-frame diff |
|---|---|---|
| 2026-09-04 | 46 | 1.83 |
| 2026-09-05 | 33 | 2.56 |
| 2026-09-06 | 3 | 1.14 (values 0.96, 1.14, 8.38) |

Today's empty-pond bursts sit at the low end of, but inside, the storm days'
distribution — the water is still moving, at comparable magnitude. Burst 5013's
frames show the jet plainly. **The fountain was not turned off.**

What *did* change is unmeasurable from this corpus: there were zero triggers
before 12:07 today, whereas 09-04 and 09-05 fired steadily from 10:00. Frames
only exist where a trigger fired, so light/wind conditions on the water surface
cannot be reconstructed for the hours that stayed quiet. **Recorded as
unexplained.** The operational conclusion is narrower and safe: review-class
volume in this scene is highly variable (4, 4, 46, 33, 3 over 09-02…09-06), so a
single day is not a trend and no lever should be sized off one.

### Framing check: no second re-aim

Eight review-class frames sampled across 09-02 → 09-06 show one continuous scene.
The re-aim was a single event on 09-02, not ongoing drift.

## The real gap this experiment named, now closed: a scene-change instrument

Night 1's central finding was not about water. It was: *"Nothing in the pipeline
compares today's framing to yesterday's. Service liveness is not scene
liveness."* The loop burned three nights analysing a scene that no longer
existed. That is now instrumented.

**Why the existing `scene_similarity` column cannot do this job.** It compares
against a rolling reference set bounded at `scene_gate_ref_max_age_hours` = 6 h,
seeded from the DB at startup. The 09-02 re-aim coincided with a **20 h** outage,
so at restart every candidate reference was stale, the set was empty, and the
gate failed open — by construction it could not have seen the move. A
cross-*day* comparison is a different measurement, not a threshold tweak.

**Comparator selection was measured, not chosen.** `scene_gate.py`'s comparator
(normalized-intensity mean-abs-diff) does **not** separate the classes here:

| statistic | same-scene days | across the re-aim boundary |
|---|---|---|
| intensity mean-abs-diff (scene_gate's) | 0.7089 – 0.9389 | 0.5581 – 0.6851 |
| **edge-structure NCC** | **0.6154 – 0.9607** | **0.0389 – 0.2604** |

The intensity metric's bands nearly touch (0.7089 vs 0.6851) because the new
close-up scene is dappled-sunlight-dominated — exactly backlog #17's finding that
the score measures illumination drift more than subject presence. Comparing
**edge structure** (Sobel magnitude of a blurred 128×128 grayscale, zero-mean /
unit-std, compared by normalized cross-correlation) inverts the situation: a
re-aim changes structure, sun does not.

**The shipped statistic is a match *fraction*, not a max.** A max over pairs is
defeated by a low-volume day whose "recent" frames straddle the move; a median is
defeated by a person filling the frame. Instead: of the last 12 saved frames, how
many match *any* baseline frame from 3–7 days ago at edge-NCC ≥ 0.45? Replayed
over the whole retention window:

| day | match fraction | scene |
|---|---|---|
| 08-14, 08-17, 08-20, 08-23, 08-28, 08-29, 09-01 | 0.92 – 1.00 | unchanged |
| 08-16 | 0.58 | unchanged |
| **08-31** (26 HUMAN bursts, heaviest human traffic in the corpus) | **0.75** | unchanged |
| 09-02, 09-03 (frames straddle the move; 6 and 10 files on disk) | 0.92, 0.75 | changed |
| **09-04, 09-05, 09-06** | **0.00, 0.00, 0.00** | **changed** |

Eleven same-scene days span 0.58–1.00; all three days with a clean post-move
sample score 0.00. The alert threshold is **0.25**, sitting in a gap more than
twice as wide as either margin. The 08-31 data point is the one that matters for
false alarms: a day where people occlude the frame in 26 bursts still scores
0.75, because a body blocks the scene's edges but does not replace them.

Detection latency is trigger-volume-bound: 09-04 was the first day with enough
post-move frames to fill the 12-frame window, so the alert would have fired on
the 09-04 tick — **one night earlier than the human-driven discovery on 09-05**,
and without spending a tick's analysis budget on it.

### Why this is shippable tonight, under the guardrail contract

- **FN-veto: N/A by construction.** The check lives in `loop/nightgate.py`
  alongside the exp #15 dead-man's switches. It reads saved frames and may send
  one Telegram message. It cannot change what is captured, classified, muted or
  notified — same standing as commit `f14ed0d`, which shipped observability under
  backlog #17 while exp #15 held the active slot.
- **Volume guardrail:** at most one message per event (7-day cooldown), on a
  detector that fired 0 times across 11 same-scene days in replay.
- **Fails silent:** insufficient frames (<8 recent or <8 baseline) returns None
  and never alerts; any exception is caught and never changes the gate's exit
  code, matching the two existing checks.
- **One experiment at a time:** this is not a new experiment. It is the
  instrument for *this* experiment's own finding, and it is monitoring-only.
- Rollback: `git revert <sha>`. No camera restart needed — `loop.nightgate` is
  loop-side code, live on the next tick.

## Standing duties discharged tonight

- All 3 review-class bursts adjudicated: 5006 and 5013 empty pond
  (`false_positive`), **5020 contains a person** (`person`). 0 concealed animals.
- **Exp #14's watch band** (`person_confidence` ∈ [0.30, 0.50), review-class):
  one burst, 5020 at pc 0.314 — a torso and legs at arm's length, motion-blurred,
  no face. It was **not sent**: `human_proximity_muted=1`, muted by the 240 s
  window (burst 5019, HUMAN, 113 s earlier). Pre-registered rollback criterion is
  a recognizable person *reaching REVIEW*; this is the 4774 near-miss class, so
  `SPECIES_HUMAN_DETECTION_CONFIDENCE` stays at 0.50. Second consecutive test the
  layered gates have passed. Note for the record that in this close-up framing a
  person can fill the frame and still score 0.31 — the demoted band is not
  hypothetical here, and the proximity gate is doing the real work.
- **Zero review-class messages were sent today** (5006 proximity-muted + sampled
  out, 5013 sampled out, 5020 proximity-muted). No privacy exposure.
- 0 scene-gate mutes (`scene_gate_muted=0` on all 3 rows); backlog #17 gains no
  new evidence — still 0 animal rows carrying a `scene_similarity`, so its
  promotion criterion (≥5) is untouched.
- 2 below-sharpness-floor bursts (5005, 5023) — both HUMAN-status, correctly
  suppressed by the privacy gate, not the blur mute.
- Last human label 2026-09-05 (4987). Not feedback-starved.

### One gap worth recording (no action)

Burst 5013 (15:12:13) logged `motion_area` 57694 with `contour_count` 2 — one
large, compact moving object — on frames showing an empty pond. That is the
signature of a person passing very close to the lens and gone before the
high-res capture. No human-proximity condition covers it: 976 s after the last
HUMAN burst (window is 240 s), density 6 in the trailing 1800 s (threshold 8),
and no HUMAN burst inside the 240 s deferral. It leaked nothing, because the
saved frames contain no person. Recorded as the nearest miss in the proximity
stack, not as a defect — widening any of those three parameters to cover it
would mute far more on speculation than it protects.

## Exit criteria — unchanged

Zero-animal streak is now **5 days** in the new scene (21 days since the last
`identified` burst, 2026-08-16, in the old framing). Escalation point is ~2 weeks
in-scene; not reached.
