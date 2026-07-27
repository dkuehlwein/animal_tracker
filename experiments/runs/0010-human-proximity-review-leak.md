---
id: 11
slug: human-proximity-review-leak
status: running
validation: live          # shipped 2026-07-27, live at the 2026-07-28T03:25 restart
occupies_active_slot: true   # exp #9 (runs/0008) concluded KEEP this tick; this takes the slot
hypothesis: "The human/privacy gate has a fourth leak path none of its three triggers cover: MegaDetector scores extreme close-up / motion-blurred PARTIAL human bodies (legs, an arm, a torso filling the frame) at ~0.02-0.15 person confidence, and SpeciesNet calls them no_animal, so they reach Telegram as REVIEW notifications showing a recognisable person. Such bursts are, however, temporally clustered with correctly-gated human bursts. Muting review-class bursts that land within 120s of a HUMAN-status detection closes the leak at zero measured false-negative cost."
created: 2026-07-27
promoted_from: "Loop tick 2026-07-27 — three confirmed person-in-REVIEW leaks found while adjudicating the night's sent review-class bursts (ids 3544, 3553, 3554)."
confidence: high          # FN cost measured directly from timestamps, not inferred from images
---

## Origin — what the adjudication actually found

Tonight's standing duties (scene-gate mutes, sampled-out bursts) both came back
clean. The leak was found by a third, unplanned check: 3554 carried a human
`person` label on a `no_animal` row, i.e. Daniel told us a REVIEW notification
had shown him a person. Pulling its frames confirmed it — a person's legs and
torso at ~1m from the lens, filling the left third of the frame.

`person_confidence` on that row is **0.041**. MegaDetector did not merely score
it below the 0.30 gate; it found no person box at all (`detection_count=0`).
The ensemble said `no_animal`, `top_species_raw` is NULL. All three of the
human gate's triggers (person box >= 0.30, `homo` in the ensemble taxonomy,
`homo` in the raw top-1 — exp #9's addition) are structurally blind to this
frame class. **This is not an exp #9 regression; it is a fourth path.**

Checking every review-class burst that was *sent* tonight and landed within
120s of a human-status burst (5 rows, all frames still on disk):

| id | time | pc | what the frames show | Daniel's label |
|---|---|---|---|---|
| 3544 | 18:21:21 | 0.15 | blurred arm/limb at the left edge | `false_positive` |
| 3553 | 18:38:56 | 0.04 | torso + bare arm, white shirt, close-up | `false_positive` |
| 3554 | 18:39:28 | 0.04 | legs walking through frame, close-up | `person` |
| 3580 | 20:20:57 | 0.02 | empty dusk garden | — |
| 3607 | 20:57:07 | 0.12 | empty dark garden | — |

**3 of 5 are people.** Note that two of the three were labelled
`false_positive` — from Daniel's side a person photo and a leaf photo are both
"nothing there", so the label stream *understates* this leak. The DB is the only
place it is visible.

## The FN evidence (the part that clears the FN-veto)

Unlike the scene gate — whose threshold could not be validated because the
animal-labelled review rows all predate image retention — this gate's cost is
measurable from **timestamps alone**, no frames required.

All human-labelled `animal`/`animal_wrong_id` rows on review-class statuses
since the human gate went live (2026-07-08), by distance to the *preceding*
human-status burst:

| id | date | label | prev human burst |
|---|---|---|---|
| 1718 | 07-08 | animal | none |
| 2011 | 07-14 | animal_wrong_id | 747 s |
| 1838 | 07-10 | animal_wrong_id | **329 s** ← closest |
| 1840 | 07-10 | animal_wrong_id | 1 704 s |
| 1818 | 07-10 | animal_wrong_id | 4 381 s |
| 1777 | 07-09 | animal | 2 759 s |
| 1733 | 07-08 | animal | 6 721 s |
| 2025, 2028, 3116, 1861, 1862 | 07-14 .. 07-24 | animal_wrong_id | 28 755 – 88 915 s |

A **120 s** look-back window costs **0 of 12** known animal-labelled review-class
rows, with a 2.7x margin to the nearest case (329 s). The three leaks tonight sit
at 79 s, 76 s and 108 s — comfortably inside it. A 300 s window would also cost
0/12 but with only a 10% margin, so 120 s is the evidence-supported choice, not
the maximal one.

Volume cost: **118 / 902 = 13%** of review-class bursts since 07-08 would have
been muted. Tonight: 20/93 review-class rows (14 of them ones that were sent).

## Change shipped (commit `50aa451`, live at the 2026-07-28T03:25 restart)

`PerformanceConfig.human_proximity_window_seconds = 120.0`
(`PERFORMANCE_HUMAN_PROXIMITY_WINDOW_SECONDS`, BOUNDS `[0, 600]`, `0` = off).
A review-class burst whose timestamp is within the window of the last
HUMAN-status detection is still species-ID'd and DB-logged (new
`human_proximity_muted` BOOLEAN column, `NULL` for non-review-class rows) but
not sent to Telegram — a `[HUMAN-PROXIMITY]` log line, same shape as the blur
and scene mutes. The last-human timestamp is held in memory and seeded from the
DB at startup so a restart doesn't reopen the window. Fails open on any error.

**Precedence is now: Human > Human-Proximity > Blur > Scene > Sampling** —
exactly one suppression log per burst. 474 tests pass.

Rollback lever: `PERFORMANCE_HUMAN_PROXIMITY_WINDOW_SECONDS=0` + restart.

### Gates cleared
- **FN-veto**: cleared *by direct measurement* (0/12, 2.7x margin) — this is the
  evidence the scene gate never had, not an assumption.
- **Volume**: notification-layer only; capture volume untouched.
- **Feedback-starved freeze**: 29 human labels tonight, not starved.
- **One experiment at a time**: exp #9 concluded KEEP this tick (see runs/0008).
- Not paused.

### Pre-registered exit criteria
- **FN-veto event** = a human `animal`/`animal_wrong_id` label on a row with
  `human_proximity_muted=1`, or a concealed animal found while adjudicating one.
  Response: shorten the window strictly below that row's gap to the previous
  human burst (in-bounds), or `0` if no in-bounds window would have spared it.
- **Nightly duty (starts the 07-28 tick)**: adjudicate every
  `human_proximity_muted=1` burst for a concealed animal, same as the
  scene-gate and sampled-out duties.
- **Conclude KEEP** after 5 nights with >= 3 muted bursts adjudicated clean and
  no new person-in-REVIEW leak.

## Second change this tick (infrastructure, not an experiment)

`PERFORMANCE_MAX_IMAGES` **100 → 300** (added to BOUNDS, deployed as an env
delta). Cause: on tonight's 217-trigger day, the ~100-burst retention window is
about five hours long, and **17 of 36 sampled-out bursts had already lost their
frames** before the tick ran — the loop cannot perform its own FN-adjudication
duty on a busy day. 300 bursts ≈ 850 MB against 8.6 GB free. This changes no
detection or notification behaviour; it only keeps the evidence the loop is
required to inspect.

## Also this tick: the volume baseline was stale

`baselines.volume_per_night` was **42**, set when nights were quiet. Tonight's
217 triggers therefore tripped `check_volume`'s 5x explosion guardrail
(217 / 42 = 5.17x). **Rollback is not indicated and was not performed**, on
cause: the deployed set contains only notification-layer levers (scene gate,
sampling rate, and now the proximity window) — none of them can change capture
volume — and 122 of tonight's 217 triggers are confirmed `human`-status bursts
from a full day of gardening, the same scene cause as 07-25 (192) and 07-23
(258). Rolling back to `best_known_good` (`{}`) would not have removed a single
trigger.

The real defect was the baseline: the trailing 7-day trigger counts are
221 / 106 / 258 / 41 / 192 / 43 / 217, median **192**. `baselines.volume_per_night`
is updated 42 → 192 so the guardrail measures against the current season instead
of firing on every ordinary busy day.

## Observations — 2026-07-27 (night 1 of exp #11 is the *next* tick; this is the origin night)

**Volume & labels.** 217 triggers (122 human, 89 no_animal, 4 unclassifiable,
2 identified). 95 labelled, `fp_rate` 0.979 [0.926, 0.994]; human tier n=29
(27 FP + 1 animal + 1 person), MD-auto n=66, 36 sampled out. Label supply is
healthy at the 0.50 sample rate — 29 human labels vs 14 the night before.

**Scene-gate duty — vacuously clean.** **Zero** `scene_gate_muted=1` bursts in
93 review-class rows. Tonight's `scene_similarity` distribution tops out at
0.948 (median ~0.90), entirely under T=0.97. Two nights in, the scene gate has
muted 1 burst total; essentially all of the REVIEW-volume reduction is coming
from the sampling gate. Recorded, not acted on: per PROTOCOL's 2026-07-26
override the threshold is not to be re-derived, and lowering T is the unsafe
direction with the animal bucket still empty.

**Sampled-out duty — clean, but coverage-limited.** 36 sampled-out bursts; 19
still had frames and all 19 were inspected frame-by-frame (motion-boxed contact
sheets): wind on bamboo, the hose/water stream at the pond edge, a static yellow
object at the pond rim in the dusk frames. No animals, no people. **The other 17
had already been deleted by retention** — hence the `max_images` change above.
The pre-registered escalation (a 2nd real animal in a sampled-out burst → rate
1.0) did **not** fire; rate stays 0.50.

**MAIN channel.** 2 `identified` rows. 3558 (18:55, `aves;;;;;bird` @ 0.84) →
MAIN, human-labelled `animal` — correct catch, and it was below the sharpness
floor (8.3), i.e. the exp #6/#8 dusk path working as designed. 3483 (15:44,
generic `;;;;;;animal` @ 0.75, raw top-1 `blank` @ 0.84, sharpness 5.4) → MAIN,
human-labelled `false_positive`; frames already purged by retention, so whether
it was a person could not be checked — another data point for the retention bump.

**Human gate.** 122 bursts correctly routed to HUMAN, 0 MAIN leaks. 12 of them
fired with `person_confidence < 0.30` via the ensemble `homo` path. Exp #9's
raw-top-1 `homo` trigger was again not exercised (0 rows) — 6 nights live, still
unexercised, which matches its measured base rate of 3 rows corpus-wide.
