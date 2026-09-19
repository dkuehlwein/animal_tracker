---
id: 32
slug: unnamed-animal-blank-mute
status: running
validation: live   # code change, commit f4d7730; replayed over the full corpus before shipping
occupies_active_slot: false  # notification-routing fix on a leak path, same shape as exp #26/#29; exp #21 keeps the slot
hypothesis: "An IDENTIFIED burst carrying SpeciesNet's fully-generic ';;;;;;animal' rollup is a false positive when the classifier's raw top-1 over the crop is 'blank' at low confidence — the two models disagree and the classifier isn't even sure the crop is empty. Mute those, and only those."
created: 2026-09-19
promoted_from: "night of 2026-09-19: bursts 5365 and 5374, both MAIN-channel 'animal detected' alerts on a demonstrably empty garden — the only two FPs all night that reached the main channel."
confidence: medium   # FP side n=6, animal side n=1 independent counter-example; the threshold is a carve-out, not a validated separator
delta: {}   # code change; PERFORMANCE_UNNAMED_ANIMAL_BLANK_MUTE_THRESHOLD ships at its 0.90 default
commit: f4d7730
restart_at: 2026-09-20T03:25:00+02:00
opens: backlog #32 (unnamed-animal-blank-mute)
---

## Tonight

31 triggers, the busiest night in weeks. Four of them are one animal.

At 10:29-10:39 a blackbird worked the gravel strip and the far bank and tripped
four bursts (5360-5363). **All four were named and all four alerted** — the
species pipeline got a clean sweep on this visit, in direct contrast to last
night's 5341-5345 where it named only two of five. The bursts immediately
bracketing the visit (5359 at 10:28, 5364 at 10:42) were inspected frame by
frame at full resolution: no bird in any of the ten frames. So exp #31's FN
class did not recur tonight, on a visit of comparable length and distance. One
night is not evidence of a fix — nothing was changed — but it is evidence the
failure is intermittent rather than a hard distance/contrast floor.

One real person (5357, `person_confidence` 0.738, raw `homo;sapiens;human`
0.961) suppressed by the primary human gate, correctly. The remaining 26
triggers are false positives, every one of them the same thing: a gust moving
the bamboo stand in the top-right of the frame. All 24 unlabelled bursts were
adjudicated visually this tick; Daniel labelled the other two.

### The gates, audited

- **Confident-Blank Mute Gate (exp #29, night 2 live):** 6 mutes — 5371, 5382,
  5383, 5384, 5385, 5387. All six adjudicated: empty garden, all six correct.
  Zero animals muted. Running total over two nights: 9 mutes, 9 correct.
- **Scene-Unchanged Gate (exp #30, night 1 at T=0.982):** 0 mutes. Tonight's
  `scene_similarity` maxed at 0.9322, nowhere near the new threshold. This is
  the predicted behaviour and confirms last night's conclusion: *as an FP lever
  the scene gate is finished.* It stays enabled and armed.
- **Human-Proximity Gate:** 0 mutes; the single human burst had no review-class
  neighbours.
- **Review Sampling:** 12 of the surviving review-class bursts sampled out.

## The finding: the generic-animal rollup leaks FPs to MAIN

Two bursts, 5365 (11:19) and 5374 (13:34), were sent to the main channel as
species alerts. Both frames are an empty garden. Both carry SpeciesNet's
fully-generic `<uuid>;;;;;;animal` label — "MegaDetector boxed something, the
classifier could not name it" — whose ensemble confidence is literally the box
confidence, with no classifier verdict behind it.

That label routes to `DetectionStatus.IDENTIFIED`, and **every** review-class
mute path (Blur, Confident-Blank, Scene, Sampling, Deferral) tests
`is_review_detection`. So this shape bypasses all of them by construction. Exp
#26 already widened the human-proximity gate to cover the *privacy* leak of this
shape on 2026-09-14; the plain *false-positive* leak was left unhandled, and it
is the more common one.

These two bursts are the whole of tonight's MAIN-channel damage. Twenty-four
other FPs were either muted or correctly 🔍 REVIEW-prefixed. A false "animal
detected" in the main channel is the single most expensive FP this system can
produce — it is the one that teaches Daniel to stop trusting the alert.

### The discriminator is the classifier's own raw top-1

All 82 unnamed-animal rows corpus-wide, 52 of them labelled:

| raw top-1 over the crop | animal | FP | person | unlabelled |
|---|---|---|---|---|
| **names an animal** (`bird`, `american crow`, …) | **34** | **0** | 1 | 4 |
| **`blank`** (generic, empty-frame) | 2 | **6** | 0 | 1 |
| none recorded (pre-observability, < 2026-07-09) | 0 | 8 | 0 | 24 |

When the classifier looks at the crop MegaDetector drew and names an animal,
the burst is real — 34 for 34. When it says the crop is *empty*, the two models
contradict each other and the rollup is the ensemble splitting the difference
on box confidence alone. Six of eight such labelled rows are false positives.

`confidence_score` (= the box confidence) does not separate these populations at
all — animal 0.5024-0.9020, FP 0.5017-0.7454, heavily overlapping. The raw
top-1 does.

### Why the threshold is 0.90, and what it is not

The two blank-raw rows that *are* animals are ids 2212 and 2213 — **six minutes
apart on 2026-07-17, i.e. one visit, n=1 independent counter-example** — scoring
0.9722 and 0.9795. The FPs score 0.0561, 0.0594, 0.5901, 0.8411, 0.9690, 0.9862.

So the gate mutes only *below* a threshold, and the honest description is:
**this is a safety carve-out around a single known counter-example, not an
independently validated discriminator.** I have no mechanism for why a
*confident* blank would mark a real animal and an *unsure* one a false positive;
with n=1 it may well be coincidence. The run file should not be read as claiming
otherwise.

`T = 0.90` mutes 4 of the 6 measured FPs and leaves a 0.072 margin under the
counter-example. The protocol's own rule, mirrored for a mute-below gate
(`min(animal) - 0.02 = 0.9522`), mutes *exactly the same four rows* — there is
nothing between 0.8411 and 0.9690 — so the wider margin costs nothing measured
and buys real insurance against an n=1 estimate. For a mute-below gate, lowering
is the FN-safe direction, so 0.90 is the conservative choice by the protocol's
own logic. `loop.guardrails.BOUNDS` caps the loop at `(0.0, 0.9522)` so it can
never raise the threshold past the counter-example; only a human editing the env
file can set 0.0 to disable.

### Replayed before shipping, over the whole corpus

At T=0.90 the shipped code would have muted exactly 5 rows in the system's
entire history:

| id | date | blank score | label |
|----|------|-------------|-------|
| 1940 | 2026-07-13 | 0.4730 | unlabelled (frames rolled off disk) |
| 3483 | 2026-07-27 | 0.8411 | false_positive |
| 5222 | 2026-09-14 | 0.5901 | false_positive |
| 5365 | 2026-09-19 | 0.0594 | false_positive |
| 5374 | 2026-09-19 | 0.0561 | false_positive |

Zero animal-labelled rows. Zero person-labelled rows (the one person carrying
this shape, 5270, has a *named* raw top-1 and is untouched — this gate creates
no privacy exposure). Roughly 1.5 mutes/month, all of them MAIN-channel.

## What shipped

Commit `f4d7730`, restart-gated 2026-09-20T03:25.

- `PERFORMANCE_UNNAMED_ANIMAL_BLANK_MUTE_THRESHOLD` (default 0.90, config bounds
  `[0.0, 1.0]`, loop bounds `[0.0, 0.9522]`). `0.0` is a special-cased DISABLE.
- New nullable DB column `unnamed_animal_blank_muted` — True/False when the gate
  applied, NULL when it didn't (same "gate didn't apply" convention as
  `scene_gate_muted`), written on the initial INSERT.
- Precedence: Human/Privacy > Human-Proximity > **Unnamed-Animal-Blank** > Blur >
  Confident-Blank > Scene > Review Sampling > Deferred Send. The shape is never
  review-class, so it cannot collide with the four gates below it; a burst the
  human-proximity gate already mutes produces one log, not two.
- Fails open to False (never mutes) on any error. One `[UNNAMED-BLANK]` log line.
- 17 new tests; full suite 681 passed.

## Monitoring duty (nightly, from 2026-09-20)

Adjudicate **every** `unnamed_animal_blank_muted = 1` burst for a concealed
animal, exactly as the blur-mute and scene-gate paths are adjudicated. Given the
n=1 counter-example this duty carries more weight than usual. A concealed animal
in a muted burst is an **FN-veto event**: respond the same tick by lowering
`PERFORMANCE_UNNAMED_ANIMAL_BLANK_MUTE_THRESHOLD` strictly below that burst's
recorded `top_species_score`, or by disabling the gate at `0.0` if no in-bounds
threshold would have prevented it. Do not defer to the next tick.

Pre-registered kill condition: **one** animal-labelled row in the muted band
retires the threshold; **two** retire the gate.

## Rollback

`PERFORMANCE_UNNAMED_ANIMAL_BLANK_MUTE_THRESHOLD=0` + service restart, or
`git revert f4d7730` + restart.
