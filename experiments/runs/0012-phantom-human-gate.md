---
id: 14
slug: phantom-human-gate
status: running
validation: live          # shipped 2026-08-30 (commit 6d8bcc1 + env delta), live at the 2026-08-31T03:25 restart
occupies_active_slot: true   # exp #13 (runs/0011) concluded KEEP this tick; this takes the slot
hypothesis: "The human/privacy gate fires on MegaDetector person boxes at >=0.30, below MegaDetector's own 0.5 operating threshold. On empty frames that sub-threshold score is noise, so ~a quarter of all HUMAN-status suppressions are phantoms: no person is present. Raising SPECIES_HUMAN_DETECTION_CONFIDENCE to 0.50 aligns the gate with the detector's operating point, at zero measured privacy cost because the layered proximity/deferral/density gates still mute the real people who score in the demoted band."
created: 2026-08-30
promoted_from: "runs/0011 closing note (2026-08-03): id 4278, detection_status='human' at person_confidence=0.330 on frames containing no person. Deliberately not acted on then — one data point, and the loop was frozen. This tick found 32 more."
confidence: high          # 32/32 adjudicated bursts in the demoted band contain no person; the class separation was re-measured, not assumed
---

## Origin — the one-data-point note, 27 nights later

The 2026-08-03 tick closed with a warning it explicitly refused to act on: id
4278 was classified HUMAN at `person_confidence=0.330`, a hair over the 0.3
gate, on an empty garden frame. That tick's own reasoning was that one instance
is not a pattern and that `SPECIES_HUMAN_DETECTION_CONFIDENCE` "trades directly
against the privacy gate this loop spent five nights hardening and must not be
touched on one data point."

The loop then went dark for 27 nights (OAuth failure, see runs/0011 Resolution).
The backlog it woke up to answers the question the note left open.

## The measurement — 483 triggers, 184 of them suppressed as HUMAN

Since the watermark (id 4280): **184 of 483 triggers (38%) were classified
HUMAN** and therefore suppressed entirely — no Telegram, not even a REVIEW
prefix. On 2026-08-30 alone it was 64 of 75 (85%).

`person_confidence` on those 184 rows is bimodal, and the split is visible in
the frames. Every HUMAN burst with frames still on disk (the 48 h human
retention window, i.e. 08-29 and 08-30) was adjudicated:

| person_confidence band | bursts adjudicated | contain a person |
|------------------------|--------------------|------------------|
| 0.17 – 0.43            | 30                 | **0**            |
| 0.435                  | 1 (id 4741)        | **1**            |
| 0.46 – 0.48            | 2                  | **0**            |
| 0.496 – 0.96           | 3 sampled          | **3**            |

32 of 33 bursts below 0.5 are the empty garden. Every burst at 0.5 and above is
a real person. The single exception, 4741 (pc 0.435, 17:08:32), sits four
minutes into a genuine gardening visit whose other bursts score 0.61–0.96.

**Mechanism.** 182 of the 184 HUMAN rows carry `detection_count = 0` —
MegaDetector produced no box above its own 0.5 operating threshold. The gate
reads `max(conf)` over raw person-category boxes and fires at 0.3, so it is
consuming detections the detector itself rejected. On a dark, low-contrast,
low-sharpness frame MegaDetector's person head emits 0.2–0.47 on foliage. That
is not a weak person signal; it is noise.

## Why this is worth fixing — it is not a benign suppression

Suppressing an empty frame is harmless on its own. The cost is downstream:

1. **Phantom mute-arming.** Every HUMAN row seeds `_last_human_detection_at`,
   the 1800 s density counter, and the 240 s deferral cancel. Measured over the
   window since the proximity gate went live (2026-07-28): of 134
   proximity/deferral-muted review-class bursts, **23 (17%) were muted only
   because of phantom HUMAN rows** — no burst scoring ≥0.5 was anywhere near
   them. The privacy machinery built by exps #11/#12 is being triggered by
   people who were never there, and it mutes real review-class bursts when it is.
2. **Structural FN risk.** The human gate is evaluated *before* the animal
   branch, by design, so a burst containing an animal plus a 0.35 noise score on
   foliage is suppressed with no species ID, no notification and no
   observability. No instance was observed (all 32 adjudicated phantoms are
   empty), but 38% of triggers passing through that branch is a large exposure.
3. **Blinded observability.** The HUMAN branch returns
   `metadata={'person_confidence': ...}` only, so `top_species_raw`,
   `top_species_score` are NULL on all 184 rows. Over a third of the corpus is
   unlabellable and invisible to the metrics.

## The fix and its FN-veto / privacy-veto measurement

`SPECIES_HUMAN_DETECTION_CONFIDENCE` 0.30 → **0.50**, aligning the privacy gate
with MegaDetector's own two-stage operating threshold (detection @ 0.5), so the
gate stops consuming boxes the detector rejected. Rows that fire via the
`homo`-taxonomy or raw-classifier-homo triggers (`pc < 0.30`) are untouched —
those paths are independent of this threshold.

The knob had no `BOUNDS` entry, so `loop.deploy` rejected it as "not a tunable
parameter". Commit `6d8bcc1` adds it as `(0.3, 0.7)`: floored at the shipped
default because lowering it only enlarges the phantom class, capped at 0.7 so
the loop cannot gut the gate. 532 tests pass.

**FN direction: strictly improving.** The change can only *reduce* suppression —
bursts move from "never sent, never identified" to review-class, where the
existing mute stack applies. No new suppression path is created.

**Privacy direction — the real risk, measured not assumed.** Over the whole
corpus since the human gate went live (2026-07-08, 1 467 HUMAN rows), simulating
the raise:

| T | rows demoted from HUMAN | still muted by proximity / deferral / density | would reach REVIEW |
|------|------|------|------|
| 0.40 | 206 | 150 | 56 |
| 0.45 | 280 | 195 | 85 |
| **0.50** | **345 (24%)** | **252** | **93** |
| 0.60 | 458 | 352 | 106 |

At T=0.50, 93 bursts over 54 days (~1.7/day) would have reached the REVIEW
channel instead of being suppressed. Of those 93, the 30 with frames still on
disk were adjudicated this tick and **every one is an empty garden scene**. The
one genuine person in the demoted band, 4741, is **not** in the unprotected set —
the 240 s proximity gate mutes it, exactly as exps #11/#12 designed it to. So
the measured privacy regression is **zero known leaks**, and the layered gates
are doing the work the single threshold was doing badly.

Residual risk, recorded honestly: the 63 demoted-and-unprotected rows from
before 08-29 have no frames left (image rotation), so they could not be
adjudicated. The adjudicable subset is 30/93 and is unanimous.

Second-order effect worth watching: a demoted burst that *does* contain a person
is no longer purged under the 48 h human-retention policy unless it falls within
the 240 s symmetric human-adjacent purge window (`human_retention_proximity_seconds`).
The protected/demoted rows do fall in it by construction; an unprotected one
would not.

## Pre-registered predictions (check on the next tick)

- HUMAN-status share of triggers drops from ~38% to ~29% (24% of HUMAN rows demoted).
- Review-class volume rises by ~1.7 bursts/day *before* the mute stack; after the
  scene gate and 0.5 sampling, expect **+0.5 to +1 REVIEW message/day**. A larger
  rise than ~4/day is a volume-explosion signal → roll back.
- Phantom-armed proximity mutes (currently 17% of all proximity mutes) fall to ~0.
- `top_species_raw` becomes non-NULL on the demoted rows, restoring them to the
  labelled corpus.

## Nightly duty while this runs

Adjudicate every review-class burst with `person_confidence` in [0.30, 0.50) —
these are exactly the bursts this change un-suppressed. **A recognizable person
reaching the REVIEW channel is a rollback event**, not a threshold-tuning event:
restore `SPECIES_HUMAN_DETECTION_CONFIDENCE=0.3` the same tick. The standing duty
to adjudicate every `human_proximity_muted=1` burst continues unchanged.

Rollback levers: `loop.deploy --rollback` (restores 0.3), or
`SPECIES_HUMAN_DETECTION_CONFIDENCE=0.3` + restart. The `BOUNDS` entry itself is
revertible via `git revert 6d8bcc1`.
