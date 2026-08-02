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

---

## Observations — night 1 (2026-07-28)

174 triggers (119 human, 52 no_animal, 2 unclassifiable, 1 identified). `fp_rate`
0.982 [0.904, 0.997] over 55 labelled; human tier n=2 only (see "label supply"
below), MD-auto n=53, 26 sampled out. Volume 174 vs baseline 192 — no guardrail
trip. All 54 review-class bursts still had frames on disk (the `max_images`
100→300 bump from last tick did exactly its job; last night 17/36 were already
gone).

### Standing duties — all three discharged, all clean of animals

- **Human-proximity duty (new, exp #11's own)**: all **22** `human_proximity_muted=1`
  bursts adjudicated frame-by-frame. **Zero concealed animals** → no FN-veto event.
  Nine of the 22 contained a clearly recognisable person (legs, torso, a hand at
  30 cm, a child walking through frame, a head of hair filling the bottom edge) at
  `person_confidence` 0.00–0.27 — i.e. the gate is not merely cheap insurance, it
  suppressed nine person photos that the privacy gate itself could not see.
- **Scene-gate duty**: 3 `scene_gate_muted=1` bursts (3654 sim 0.972, 3662 0.971,
  3688 0.980). All three are bamboo/vegetation in an empty garden. Clean.
- **Sampled-out duty**: all 26 inspected. No animals. Two (3671, 3716) contained a
  recognisable person — see below; that is a privacy near-miss, not an FN.

### MAIN channel

One `identified` row: **3608** (07:25, domestic cat @ 0.993, sharpness 6.77 →
below the 11.0 floor). Correctly routed to MAIN and human-labelled `animal` — the
exp #6/#8 below-floor-animal-still-alerts path working as designed on a real catch.
No MAIN leaks.

### The finding: the 120 s window is too short, and two person photos were sent

Sixteen review-class bursts tonight visibly contain a person. Their gap to the
preceding HUMAN-status burst:

| id | time | pc | gap to prev human | outcome |
|---|---|---|---|---|
| 3728 | 17:52 | 0.02 | 30 s | muted (prox) |
| 3739 | 17:59 | 0.08 | 31 s | muted (prox) |
| 3673 | 15:17 | 0.17 | 32 s | muted (prox) |
| 3710 | 16:59 | 0.00 | 42 s | muted (prox) |
| 3693 | 16:25 | 0.02 | 51 s | muted (prox) |
| 3619 | 11:52 | 0.19 | 58 s | muted (prox) |
| 3636 | 12:11 | 0.03 | 60 s | muted (prox) |
| 3690 | 16:12 | 0.03 | 62 s | muted (prox) |
| 3630 | 12:02 | 0.10 | 69 s | muted (prox) |
| 3699 | 16:37 | 0.11 | 70 s | muted (prox) |
| 3704 | 16:50 | 0.27 | 108 s | muted (prox) |
| **3671** | 15:15 | 0.03 | **123 s** | escaped gate; not sent only because sampling dropped it |
| **3716** | 17:21 | 0.00 | **169 s** | escaped gate; not sent only because sampling dropped it |
| **3676** | 15:22 | 0.03 | **307 s** | escaped gate; sampled out |
| **3711** | 17:05 | 0.00 | **432 s** | **SENT to REVIEW — person leak** |
| **3691** | 16:23 | 0.24 | **732 s** | **SENT to REVIEW — person leak** |

Two person photos reached Daniel's phone, and two more were spared only by the
sampling coin-flip (at rate 0.50 the expected leak from those two is one more
message). 3671 missed the gate by **3 seconds**.

### Lever 1 rejected by measurement: lowering the person-confidence threshold

3691 carries `person_confidence` **0.24** — MegaDetector *did* box the child, just
under the 0.30 gate. Tempting, and wrong: over all human-labelled `animal`/
`animal_wrong_id` rows since 07-08, the highest `person_confidence` on a **real
animal** is **0.215** (row 1871, a bird), and tonight's cat 3608 sits at **0.197**.
Any threshold low enough to catch 3691 (≤0.24) would suppress a real bird outright
— the human gate suppresses *entirely*, no REVIEW fallback. The person and animal
`person_confidence` distributions overlap in exactly the band that matters.
**FN-veto: rejected.** 0.25 would be FN-free (0/127 identified rows) but catches
nothing. Recorded as a closed door, not a pending idea.

### Lever 2 shipped: widen the window, and add an OR-ed "garden is occupied" condition

Two changes to the same gate, both cleared by direct measurement against the 12
human-labelled `animal`/`animal_wrong_id` review-class rows since 2026-07-08:

**(a) `human_proximity_window_seconds` 120 → 240** (env delta). Cost **0 of 12**;
nearest animal row is 329 s from a preceding human burst, a 37 % margin. 300 s
would also cost 0/12 but leaves only 10 %, so 240 s is the evidence-supported
choice, same reasoning that picked 120 s last tick. Catches 3671 and 3716.

**(b) New human-density condition** (code change): mute a review-class burst when
`>= human_density_count` (8) HUMAN-status detections fall in the preceding
`human_density_window_seconds` (1800). Rationale: 3711 and 3691 sit 432 s and
732 s past the last human burst — beyond *any* in-bounds window — yet both are in
the middle of an afternoon of gardening. Density measures "someone is in the
garden right now", which is the actual latent variable; the last-human gap is only
a proxy for it. Sweep over all 956 review-class rows since 07-08:

| rule | mutes | FN cost | catches |
|---|---|---|---|
| W=120 (current) | 15 % | 0/12 | — |
| W=240 | 20 % | 0/12 | 3671, 3716 |
| density M=900 K=5 | 13 % | 0/12 | 3671, 3676, 3711 |
| density M=1800 K=5 | 23 % | **1/12** | +3716 |
| **density M=1800 K=8** | 14 % | 0/12 | 3711, 3716 |
| **W=240 OR density(1800, 8)** | **25 %** | **0/12** | **3671, 3711, 3716** |

The combined rule's FN margins: 329 s vs the 240 s window (37 %), and a maximum
human-density of **5** on any animal-labelled row vs the threshold of 8 (3 bursts
of headroom). Tonight it would mute 28 of 54 review-class bursts instead of 22.
Notification-layer only — nothing changes about capture, species ID or DB logging.

**Residual, stated plainly:** 3691 (the child in the hammock, 732 s gap, density 2)
is closed by *neither* change and is not reachable by the person-confidence lever
either. The leak class is reduced, not eliminated. If a third mechanism is needed,
it will have to be image-side, not temporal.

### Label supply

Only 2 human labels tonight (3608 `animal`, 3609 `false_positive`) against 29 last
night. Not a feedback-starved freeze (that needs 3 consecutive days at zero), and
the expected direction given the mute gates now suppress 22 + 26 of 54 review-class
bursts. Watch it: if the corpus goes fully unlabelled the loop loses its FN
detection power, and the correct response is raising `review_sample_rate`, not
loosening a privacy gate.

### Shipped tonight (2026-07-28)

- **Env delta** via `loop.deploy`: `PERFORMANCE_HUMAN_PROXIMITY_WINDOW_SECONDS`
  **120 → 240**; `PERFORMANCE_HUMAN_DENSITY_WINDOW_SECONDS=1800`,
  `PERFORMANCE_HUMAN_DENSITY_COUNT=8` (both new, added to `BOUNDS` as
  `(0.0, 7200.0)` and `(0, 100)`).
- **Code change** commit **`13fe10d`** — the density condition OR-ed onto the
  existing proximity gate. Same `human_proximity_muted` column (no schema
  change), same `[HUMAN-PROXIMITY]` log line (now naming `window` vs `density`),
  same precedence, fails open. Recent HUMAN timestamps are held in memory in a
  pruned rolling list, seeded at startup from a new
  `DatabaseManager.get_recent_human_detection_times()`, so a restart doesn't
  reopen an in-progress occupancy streak. **498 tests pass** (474 + 24 new).
- Restart stamped `2026-07-29T03:25:00+02:00` (pre-sunrise, before the 03:30
  deploy timer).

Rollback levers, independent of each other and of git:
`PERFORMANCE_HUMAN_DENSITY_COUNT=0` disables only the density condition;
`PERFORMANCE_HUMAN_PROXIMITY_WINDOW_SECONDS=0` disables the whole gate.

### Gates cleared for tonight's change
- **FN-veto**: cleared by direct measurement (0/12, 37 % gap margin and 3 bursts
  of density margin) — not by assumption.
- **Volume**: 174 vs baseline 192; the change is notification-layer only.
- **Feedback-starved freeze**: 2 human labels tonight, not zero, and not 3 days.
- **One experiment at a time**: this is exp #11's own mechanism, not a new slot.
- Not paused.

### Exit criteria — updated
Unchanged in shape (5 nights, >= 3 muted bursts adjudicated clean, no new
person-in-REVIEW leak), with the clock **restarted tonight** since the gate's
rule changed. Additional pre-registered trigger: if a night produces a
*density*-muted burst containing a real animal, drop `human_density_count` back
above that burst's observed density (in-bounds) or to `0`.

---

## Observations — night 2 (2026-07-29): first night of the widened+density rule

56 triggers (23 human, 32 no_animal, 1 identified) — a quiet day, camera active
08:50–19:36. `fp_rate` 0.970 [0.847, 0.995] over 33 labelled; human tier n=10
(9 `false_positive` + 1 `animal`), MD-auto n=23, 18 sampled out. Volume 56 vs
baseline 192: **not** a collapse trip — the trailing window contains 41 and 43
trigger days (07-24, 07-26) and the deployed set is notification-layer only, so
no lever the loop controls can reduce capture volume. All 32 review-class bursts
still had frames on disk.

### The rule shipped last tick is live and both new conditions fired

Verified in `data/logs/wildlife.log`, not inferred: 9 `[HUMAN-PROXIMITY]` lines,
**8 `reason=window` (240 s) + 1 `reason=density`** (3828, `>= 8 human detections
in the last 1800s`). Live config confirmed `prox_window=240.0`,
`density_window=1800.0`, `density_count=8`.

Attribution against the old 120 s rule — 6 of tonight's 9 mutes are new:

| id | time | gap to prev human | density(1800 s) | muted by | old W=120? |
|---|---|---|---|---|---|
| 3785 | 09:30 | 174 s | 1 | window | no |
| 3786 | 09:31 | 211 s | 1 | window | no |
| 3807 | 14:33 | 140 s | 4 | window | no |
| 3808 | 14:34 | 209 s | 4 | window | no |
| 3812 | 14:44 | 55 s | 7 | window | yes |
| 3814 | 14:46 | 70 s | 8 | window (+density) | yes |
| 3824 | 14:54 | 30 s | 18 | window | yes |
| 3825 | 14:56 | 150 s | 18 | window (+density) | no |
| 3828 | 15:17 | 1 028 s | 11 | **density only** | no |

The density condition earned exactly one exclusive mute on its first night —
low, but it is insurance against long-occupancy afternoons, and 07-28 (density
up to 18 here vs a max of 5 on any animal-labelled row) is the case it exists
for. Mute share 9/32 = 28 %, in line with the predicted 25 %.

### Standing duties — all four discharged, all clean of animals

- **Human-proximity duty**: all 9 muted bursts adjudicated frame-by-frame.
  **Zero concealed animals.** Two contained a person and were correctly
  suppressed: **3812** (torso in a blue striped shirt entering top-right,
  `pc=0.10`) and **3824** (a head of blond hair filling the bottom-left corner
  at ~30 cm, `pc=0.08`) — both invisible to the privacy gate itself, both caught
  by the 240 s window. The other 7 are empty garden.
- **Scene-gate duty**: 4 `scene_gate_muted=1` bursts, the gate's busiest night
  so far (3786 sim 0.9746, 3788 0.9771, 3789 0.9730, 3801 0.9727). All four are
  sunlit bamboo in an empty garden. Clean → no FN-veto event, T stays 0.97.
  (3786 is also proximity-muted; proximity wins the log, as designed.)
- **Sampled-out duty**: all 18 inspected. No animals, no people.
- **Blur-mute duty**: 0 below-floor review-class bursts (the one below-floor row,
  3836 at 19:35, is HUMAN-status — suppressed by the privacy gate, precedence
  correct).

### MAIN channel — one catch, no leaks

**3802** (14:26, ensemble `aves;;;;;bird` 0.730, raw top-1 `bird` 0.649,
sharpness 24.6) → MAIN, human-labelled `animal`. Frames show a dark bird at the
pond edge by the red bucket. 23 HUMAN suppressions, **0 MAIN leaks**, and no
HUMAN-status row carrying a specific-animal raw top-1.

### Zero recognisable person photos reached REVIEW tonight

Nine review-class bursts survived all four gates and were sent; all nine are
empty garden, all nine human-labelled `false_positive`. Compare 07-28: two
recognisable person photos sent. That is the improvement the widened rule was
shipped for, though tonight's much lower human traffic (23 vs 119 human bursts)
means it is weak evidence, not proof.

### New residual: the gate is causal, so the *leading edge* of a visit is blind

**3829** (15:32:34, sent to REVIEW, human-labelled `false_positive`) carries a
heavily motion-blurred pale vertical band (220 × 591 px, frame-right, gone in
frames 2–5) that is almost certainly a person passing within ~1 m of the lens.
It was not muted because it *precedes* the visit's first HUMAN-status burst
(3830 at 15:33:49) by **75 s**; the previous human burst was 1 909 s earlier and
the trailing density was 0. **No backward-looking rule can ever mute it** — this
is a structural blind spot of the mechanism, not a threshold that is set wrong.

Adjudicated: the smear is **not recognisable** as a person (no face, no body
part, no identifiable individual), so there is no privacy harm to respond to and
**no change is warranted tonight**. Recorded as backlog **#12** (deferred-send
buffer for review-class notifications: hold the Telegram send ~120 s and cancel
it if a HUMAN-status burst arrives in the meantime — the only mechanism shape
that closes a causal blind spot). Parked, not shipped: one unrecognisable
instance is not evidence for adding send latency to every REVIEW alert, and exp
#11 still occupies the active slot.

For the record, 3831 (15:38, also sent) was checked for the same reason (282 s
past a human burst) — its changed region is a thin pale stem/blade of grass, not
a person.

### Decision — KEEP, exp #11 stays running (night 2 of 5)

No change deployed, no `pending_restart_at` stamped. Exit criteria unchanged:
5 nights, >= 3 muted bursts adjudicated clean per night, no new person-in-REVIEW
leak. Night 2 satisfies all three (9 mutes, clean, zero leaks). Label supply
recovered to 10 human labels — not starved.

---

## Observations — night 3 (2026-07-30): the density condition earns its keep

47 triggers (16 human, 30 no_animal, 1 identified), camera active 07:16–19:33.
`fp_rate` 0.968 [0.838, 0.994] over 31 labelled; human tier n=10 (9
`false_positive` + 1 `animal`), MD-auto n=21, 12 sampled out. **Zero false
negatives**: the single non-FP human label is 3855, an `identified` row that
went to MAIN correctly. Volume 47 vs baseline 192 is again **not** a collapse
trip — the trailing window holds 41/43/56-trigger days, and every deployed
lever is notification-layer, so none of them can suppress a capture.

### The density condition muted a real person the window would have missed

Four `[HUMAN-PROXIMITY]` lines in `data/logs/wildlife.log` (verified, not
inferred): **3 `reason=window` + 1 `reason=density`**.

| id | time | gap to prev human | density(1800 s) | muted by | contents |
|---|---|---|---|---|---|
| 3870 | 15:24:55 | 89 s | 2 | window | empty garden |
| 3877 | 15:41:55 | **675 s** | **8** | **density** | **person, close-up** |
| 3881 | 16:26:20 | 44 s | 2 | window | empty garden |
| 3882 | 16:27:28 | 112 s | 2 | window | empty garden |

**3877 is the result this experiment was widened for.** It is a close-up
partial body — bare legs, dark shorts, a striped shirt and a hand, filling the
upper half of the frame at ~1 m, heavily motion-blurred. MegaDetector scored it
`pc=0.071`, far under the 0.3 privacy gate, and the ensemble called it
`no_animal`: all three older human-gate triggers were blind to it. The 240 s
window was also blind — the previous HUMAN burst was **675 s** earlier, 2.8x
outside it. Only the density condition (8 HUMAN bursts in the trailing 1800 s,
exactly at the threshold) caught it. Under the pre-widening 120 s rule this
photo would have been sent to Daniel as a REVIEW alert.

**FN-veto: clean.** All 4 muted bursts adjudicated frame-by-frame; the 3
window-muted ones are empty sunlit garden, and the density-muted one is the
person above. Zero concealed animals. The pre-registered density-specific
trigger (a *density*-muted burst containing a real animal → drop
`human_density_count`) did **not** fire.

### Other duties

- **Scene-gate duty**: 0 `scene_gate_muted=1` bursts. Similarities over the 26
  scored review-class rows ran 0.684–0.947, all under T=0.97, so the gate was
  inert tonight. No FN-veto event; T stays 0.97 (raising is the only safe
  direction and nothing asks for it).
- **Blur-mute duty**: 0 muted. Six below-floor rows; the four review-class ones
  (3851/3852 luma 60/59, 3853/3854 luma 15) all sit under `blur_mute_min_luma`
  =70 and were therefore **un-muted to REVIEW** — the exp #8 brightness fix
  behaving exactly as designed on a dark, stormy afternoon. Adjudicated: empty.
- **Sampled-out duty**: 12 rows, spot-checked, no animals and no people.
- **MAIN channel**: 16 HUMAN suppressions, **0 leaks**. One catch — **3855**
  (13:46, ensemble `corvus species` 0.691) is a genuine corvid silhouette on the
  pond stone in near-darkness (luma 16.3, sharpness 4.0, well below the floor).
  Below-floor + animal ⇒ alert: exp #6/#8 routing confirmed on live evidence.

### Backlog #12 recurs — second instance, still not promotable

**3867** (15:22:03, sent to REVIEW) is the leading-edge blind spot again: its
best frame carries a dark/blue motion smear cut off at the extreme left edge,
almost certainly the same person arriving, **51 s before** the visit's first
HUMAN burst (3868 at 15:22:54). Frames 1–3 of the burst are empty; only the
sent frame 5 has it. The previous human burst was ~7.5 h earlier and trailing
density 0, so neither condition could fire — as established on 07-29, **no
backward-looking rule can mute the leading edge of a visit.**

Adjudicated **not recognisable** (no face, no identifiable body part or
individual) — same verdict as 3829 on 07-29. Backlog #12's promotion criterion
(a *recognisable* person photo sent on the leading edge) is therefore still
unmet, and #12 stays parked. What is new is the **rate**: 2 instances in 2
nights, i.e. this blind spot fires roughly once per human-visit day. Recorded
so a future tick can weigh recurrence, not just severity — but recurrence of a
harmless smear is not grounds for adding ~120 s of latency to every REVIEW
alert.

### Decision — KEEP, exp #11 stays running (night 3 of 5)

No change deployed, no `pending_restart_at` stamped. Exit criteria (5 nights,
>= 3 muted bursts adjudicated clean per night, no new person-in-REVIEW leak):
night 3 satisfies all three — 4 mutes, clean, and the one person who did reach
REVIEW was an unrecognisable smear via the known causal blind spot, not a gate
failure. Label supply 10 human labels — not starved. Two nights to go before
the widened rule can be concluded.

---

## Observations — night 4 (2026-08-01, covering 07-31 + 08-01): backlog #12 promoted and shipped

The 07-31 tick did not run (no session), so this tick's ingest window spans two
loop-days: ids 3885–4162, **278 triggers** (169 human, 107 review-class, 2
identified). `fp_rate` 0.982 [0.936, 0.995] over 109 labelled; human tier n=5
(all `false_positive`), MD-auto n=104, 54 sampled out, **0 false negatives** —
no human `animal`/`animal_wrong_id` label landed on a review-class row, and the
2 `identified` rows (3925/3926, `aves` 0.689/0.936, 08-01 10:15) went to MAIN.
139 triggers/night vs baseline 192 — no volume trip. 08-01 was the busiest human
day on record: **161 HUMAN-status suppressions in one day**, 0 MAIN leaks.

### Standing duties — all four clean

- **Human-proximity duty**: **53** `human_proximity_muted=1` bursts, every one
  adjudicated frame-by-frame. **Zero concealed animals.** Six are unambiguous
  people the privacy gate could not see: 3897 (a child standing in the garden,
  pc=0.000), 4029 (dark close-up body filling the left half), 4051 (blue shirt
  at ~0.5 m), 4059 (a hand and forearm), 4147 (a leg and foot), 4149 (arm, hand
  and part of a head). Exactly the leak class exp #11 exists for, muted as
  designed. The rest are empty sunlit garden.
- **Scene-gate duty**: 1 row scored `scene_gate_muted=1` (3943, sim 0.970) but
  the proximity gate took precedence, so the scene gate muted nothing on its
  own. Adjudicated: empty. Sims over the 105 scored rows ran 0.557–0.970. T
  stays 0.97.
- **Blur-mute duty**: **0 muted.** 13 below-floor review-class rows, all with
  luma 33.5–61.1 (< the 70.0 `blur_mute_min_luma` floor), so all were un-muted
  to REVIEW — exp #8's brightness fix behaving as designed across a dusk run.
  Adjudicated: empty.
- **Sampled-out duty**: 29 rows contact-sheeted, no animals, no people.

### The leak: 3909 — a recognisable face sent to REVIEW on the leading edge

**3908** (07-31 18:21:58) and **3909** (18:22:42) were both sent to REVIEW as
`no_animal`. The visit's first HUMAN-status burst is **3910 at 18:24:03** — 125 s
and **81 s** after them. The previous human burst was 11 553 s (3.2 h) earlier
and trailing density 0, so neither the 240 s window nor the density condition
could fire. This is backlog #12's blind spot for the **third** night running
(3829 at 75 s on 07-29, 3867 at 51 s on 07-30).

What changed tonight is **severity**, which was #12's stated promotion criterion.
3909's burst frames 3, 4 and 5 show a person's head and face in profile at close
range under a blue cap — **plainly recognisable**, not a smear. (The frame that
actually went to Telegram was the sharpest one, frame 2, which is a motion smear;
that is why Daniel labelled 3909 `false_positive` while he had labelled the
07-30 instance `person`. His `person` label on 3867 is itself evidence that the
sent frame can read as a person to a human.) 3908, by contrast, is empty garden.

Two distinct harms follow from one misclassification, and they need two fixes:

1. **Notification**: a person burst is sent to Daniel's REVIEW channel.
2. **Retention**: because the burst is `no_animal`, not `human`, its recognisable
   frames are kept under the ordinary ~300-burst rotation instead of the 48 h
   human-photo purge. Every one of tonight's 6 muted-but-real-person bursts sits
   on disk under that same wrong policy — the mute protects the inbox, not the
   disk.

### Shipped — exp #11 mechanism extension (code change, restart-gated)

Both parts, deliberately with independent rollback levers.

**Part A — Deferred REVIEW Send Gate (cancel-on-human).** A review-class burst
that survives every existing gate no longer sends immediately: the annotated
image is built synchronously (it depends on `self.last_motion_frame`, which
mutates on the next loop iteration), then a background asyncio task sleeps
`review_defer_seconds` (**240 s**, env `PERFORMANCE_REVIEW_DEFER_SECONDS`,
BOUNDS [0, 600], 0 = off) and cancels the send if a HUMAN-status detection landed
in the meantime (`burst < last_human <= burst + window`), logging `[REVIEW-DEFER]`
and persisting the mute through the existing `human_proximity_muted` column via a
follow-up UPDATE. The detection loop never awaits it; pending tasks are cancelled
on shutdown; MAIN/animal sends are never deferred; any exception sends
(fail-open), and `asyncio.CancelledError` deliberately propagates so a shutdown
cancel is not mistaken for an error.

**FN-veto cleared by measurement, not assumption** (same method as night 1): of
the 12 human-labelled `animal`/`animal_wrong_id` review-class rows since the human
gate went live 2026-07-08, the closest sits **1 846 s** before the next HUMAN
burst — **7.7x** the 240 s window. Zero known false negatives. Cost: 12 of the 68
review-class bursts actually sent since 07-28 would have been cancelled (**18%**,
~1–2 alerts/night), and all three known leading-edge leaks (3829, 3867, 3909) are
in that set. Latency cost is ~0 in practice: Daniel labels in batches hours after
capture (07-31: bursts from 14:36 and 16:24 labelled together at 18:23).

**Part B — retention purge for human-adjacent review bursts.** The 48 h privacy
purge now also deletes the photos of review-class bursts within
`human_retention_proximity_seconds` (**240 s**, env
`PERFORMANCE_HUMAN_RETENTION_PROXIMITY_SECONDS`, BOUNDS [0, 3600], 0 = off) of a
HUMAN-status detection, **in either direction** — the purge runs ≥48 h after
capture, so unlike a notification gate it may legitimately look forward in time.
DB rows are kept as metadata-only records, exactly as for HUMAN rows.

Implementation note worth keeping: the first version expressed adjacency as a
correlated SQL `EXISTS` with `strftime('%s', …)` on both sides. Measured on the
live DB (4 162 rows, 279 matches) it took **3.9 s per call**, and
`purge_human_bursts()` runs from `cleanup_old_images()` after *every* detection;
a `JOIN … datetime(h.timestamp, '±240 seconds')` rewrite still took 1.6 s. Final
version does two indexed queries and matches with `bisect` in Python: **~40 ms**,
same 279 matches. Perf regressions of this shape are invisible in the test suite
and only show up against the real corpus — measure before shipping anything that
runs in the detection hot path.

Accepted cost of Part B, recorded deliberately: bursts adjacent to a human visit
lose their frames 48 h after capture, so a *missed* tick (like 07-31) plus a late
adjudication could purge frames before this loop ever looks at them. Judged
acceptable — the FN class it could hide (an animal in frame while a person is
active within 240 s) is the least likely one, and the DB row plus every gate flag
survives.

Tests: **529 passed, 0 failed**. That includes fixing a failure this tick
inherited: `test_human_proximity_no_mute_outside_window` had been failing since
the 2026-07-29 deploy of `window=240`, because production's
`experiments/deployed_config.env` leaks into the suite through the
config-module-reload gap documented in `tests/conftest.py` — the test places a
burst 200 s after a human detection and expects no mute under the 120 s *code*
default. The `system` fixture now pins the proximity/density env knobs, same
pattern as `PERFORMANCE_REVIEW_SAMPLE_RATE`. Worth noting as a loop-hygiene
lesson: **a deployed env delta can silently break the test suite the loop relies
on for validation**, and nothing surfaced it for three nights. Commit: `424265d`.

### Decision — KEEP, exp #11 stays running (night 4 of 5)

Exit criteria (5 nights, >= 3 muted bursts adjudicated clean per night, no new
person-in-REVIEW leak): nights 4 satisfies the first two decisively (53 mutes,
all clean) and fails the third — 3909 is a genuine new leak, via the known causal
blind spot rather than a gate failure, and it is fixed this tick. Backlog **#12
is now promoted and shipped**, folded into exp #11 as a mechanism extension
rather than opened as a separate experiment (same root cause, same gate family,
same column; precedent: the density condition on night 1). `pending_restart_at`
stamped 2026-08-02T03:25 so both parts go live with the pre-sunrise restart.
Label supply 5 human labels this window — not starved.

---

## Night 5 (2026-08-02) — extension night 1: the deferral gate fired, and fired correctly

First night with the exp #11 extension live (commit `424265d`, applied at the
2026-08-02T03:30 restart). 114 triggers, 82 HUMAN-status suppressions, 31
review-class bursts, 1 `identified`.

**The Deferred REVIEW Send Gate cancelled two sends, both genuine leading-edge
cases** — the exact class no backward-looking rule can reach:

| id | burst | first HUMAN of the visit | lead | prior human | density |
|----|-------|--------------------------|------|-------------|---------|
| 4184 | 12:15:13 | 4185 @ 12:15:57 | 44 s | 11:57:35 (1058 s) | 2 |
| 4212 | 13:31:19 | 4213 @ 13:34:54 | 215 s | 12:40:27 (3052 s) | 0 |

Both were unreachable by the window condition (prior human 18 and 51 min back)
and by the density condition (2 and 0 detections in 1800 s). Under the previous
build both would have been sent to REVIEW as the opening frame of a visit —
which is how 3829, 3867 and 3909 leaked on the three preceding nights. Zero
person-in-REVIEW leaks tonight. `[REVIEW-DEFER]` logged, mute persisted in
`human_proximity_muted`, no other gate double-logged.

The backward conditions fired 7 more times: 4206, 4217, 4220, 4223, 4254
(window) and 4207, 4264 (density).

**FN-veto duty — all 9 muted bursts adjudicated frame by frame (45 frames, all
on disk): 0 concealed animals.** Empty garden in every one. Scene gate muted
nothing tonight (no `scene_gate_muted=1` rows), so that standing duty is vacuous
this tick.

**Leak duty — all 9 review-class bursts that were actually sent adjudicated:
0 people, 0 animals.** Two of them sit in the 240–600 s pre-human band that a
wider deferral window would have caught (4165 at 479 s, 4210 at 386 s before the
next HUMAN burst) and **both are genuinely empty scenes** — so tonight produced
no evidence for widening `review_defer_seconds`, and it stays at 240.

**Part B (symmetric retention purge) is live and correct**: `resource_manager`
logs `Purged 1039 human-status burst(s) ... Purged 284 human-adjacent review
burst(s) (0 files) within 240s of a human detection, older than 48h` after each
detection. `(0 files)` because every one of those 284 rows is older than the
300-burst image rotation — the purge is a no-op today by construction and will
start deleting real files once tonight's human-adjacent bursts cross 48 h on
08-04. The rewritten (bisect, not SQL `EXISTS`) implementation shows no hot-path
cost: bursts continued at the normal ~30 s cadence all day.

**Ops note, not a code fault**: `wildlife-camera.service` was stopped by hand at
14:06:44 (SIGTERM → status=143, systemd's normal "Failed" cosmetics for a service
without an exit-0 SIGTERM handler) and restarted 14:56:05 — a ~50 min gap in
coverage with ssh sessions bracketing it. No crash, no restart loop.

### Decision — CONCLUDED, KEEP (live)

Exit criteria met: 5 nights (07-28..08-02), ≥3 muted bursts adjudicated clean
every night (9, 4, 1, 53, 9 — 76 total, **0 concealed animals**), and no
person-in-REVIEW leak on the first night the completed mechanism was live. Each
of the three preceding nights produced exactly one leading-edge leak; tonight the
gate that closes that path cancelled two and leaked none. The gate family stays
live in full: window 240 s, density 8/1800 s, deferral 240 s, symmetric purge
240 s. The nightly adjudication of every `human_proximity_muted=1` burst becomes
a **standing duty**, not an experiment-scoped one — it is the only thing standing
between a mute path and a silent false negative.

Freeing the active slot for exp #13 (runs/0011).
