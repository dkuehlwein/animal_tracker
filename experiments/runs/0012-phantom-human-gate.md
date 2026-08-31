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

---

## Night 1 — 2026-08-31 (T=0.50 live since 03:30)

Restart applied cleanly: `wildlife-deploy` stamped `{"restarted": true, "reason":
"applied deploy stamped 2026-08-31T03:25:00+02:00"}` at 03:30:04, camera up since.
45 triggers (08:53–18:18), a high-human-activity day: 26 HUMAN, 18 no_animal,
1 unclassifiable, 0 identified.

### The gate is doing exactly what was configured

Every person-box-fired HUMAN row today sits at pc **0.507–0.934** — zero below
0.50. Under the old T=0.30, five more bursts would have been suppressed as HUMAN.
Discriminator used throughout this section: a HUMAN row whose `confidence_score`
equals `person_confidence` fired via the MegaDetector person-box branch; otherwise
it fired via the `homo`-taxon (or raw-homo) branch, which stores the ensemble
prediction score instead.

### Nightly duty: the demoted band (pc ∈ [0.30, 0.50)) — 5 bursts, all adjudicated

| id | time | pc | sent? | contents |
|----|------|----|-------|----------|
| 4765 | 09:01 | 0.378 | sent (09:05) | empty garden |
| 4769 | 10:03 | 0.384 | muted (sampling) | empty garden |
| 4774 | 10:55 | 0.361 | muted (sampling) | **a real person** |
| 4782 | 11:34 | 0.340 | sent (11:38) | empty garden |
| 4784 | 11:44 | 0.323 | sent (11:48) | empty garden |

4 of 5 are phantoms correctly demoted — the predicted effect. **4774 is a real
person** and is the first measured instance of the privacy risk this experiment
pre-registered. It is *not* a rollback event, on three independent grounds:

1. **It was never sent.** `review_sampled_out=1` muted it before the send.
2. **Two gates covered it, not one.** The next burst, 4775 at 10:55:56, is a
   HUMAN-status detection 41 s later — well inside `review_defer_seconds=240`,
   so had it survived sampling the deferral gate would have cancelled the send.
   The rollback criterion is a recognizable person *reaching REVIEW*; the defence
   in depth built by exp #11/#12 held on its first real test under T=0.50.
3. **Not recognizable.** Frames 1–3 show a motion-blurred figure mid-stride: hair
   colour, skin tone and a blue top are resolvable, no facial features are. This
   is the 3829/3867 "unrecognisable smear" class, not the 3909 "face in profile"
   class that triggered the exp #12 promotion.

Recorded as a near-miss, not a regression. If a *recognizable* person lands in
REVIEW, roll back that tick as pre-registered.

### Standing duty: proximity mutes — 3 bursts, all clean

4773, 4792, 4798 adjudicated: all empty garden, 0 concealed animals, 0 people.
No scene-gate mutes today (`scene_gate_muted=0` on all 19 review-class rows).

### Volume

8 REVIEW messages sent (verified against `sendPhoto` calls in `wildlife.log`,
which reconcile exactly with 19 review-class − 4 blur − 3 proximity − 4 sampling).
Three of the 8 (4765, 4782, 4784) are demoted-band rows, i.e. **+3 messages
attributable to this change** — above the +0.5–1/day point prediction, below the
>4/day volume-explosion trip. n=1 night with unusually heavy human traffic
(26 HUMAN bursts vs a 483-trigger backlog average of ~7/day); do not conclude
from one night. Prior days for reference: 08-28 → 2 sends, 08-29 → 2, 08-30 → 0.

### Pre-registered predictions: status after night 1

- **HUMAN share 38% → ~29%**: not evaluable. Today is 26/45 = 58%, driven by real
  garden occupancy, not by the gate. The like-for-like number is the demotion
  rate: 5 of a would-be 31 HUMAN rows = 16% demoted (predicted 24%).
- **Phantom-armed proximity mutes 17% → ~0**: *partially* achieved, and the
  shortfall is informative. Of today's 3 proximity mutes, 4773 and 4792 were armed
  by genuine people, but 4798 was armed by 4797 — a phantom. 4797 fired via the
  **homo-taxon** branch (pc 0.216, taxon score 0.53), which this experiment's lever
  does not touch. See below.
- **`top_species_raw` non-NULL on demoted rows**: holds — all 5 demoted rows carry
  observability fields; 4773 even recorded a `blank` raw top-1.
- **FN**: unmeasured, as always. 2 human labels arrived today (on 08-29 rows), both
  `false_positive`/`cant_tell`; no animal label anywhere in the window. Not
  feedback-starved (labels on 08-27, 08-28, 08-31).

### Verdict: KEEP RUNNING

The lever works as specified, the demoted band is 80% phantoms, the one real
person in it was stopped by two downstream gates and is unrecognizable, and no
guardrail tripped. Continue; the volume figure needs 2–3 more nights before it
means anything.

---

## Night 1 finding → new backlog item #16: the human gate has a *second* phantom path

Exp #14 fixed the MegaDetector person-box trigger. Tonight's adjudication shows
the `homo`-taxon trigger has an independent phantom class of the same kind, which
`SPECIES_HUMAN_DETECTION_CONFIDENCE` cannot reach at any value in BOUNDS.

**Mechanism.** `human_gate_fired = max_person_conf >= T or is_homo_taxon or
raw_homo_leak`. The `is_homo_taxon` arm is a pure membership test — any `homo`
segment in the ensemble label fires the gate *regardless of the prediction score*.
There is no threshold on it.

**Measurement.** Of 1493 HUMAN rows since 2026-07-08, 284 fired via the taxon arm.
Their prediction scores are sharply **bimodal**:

```
0.45: 1    0.70: 9
0.50: 29   0.75: 9
0.55: 24   0.80: 7
0.60: 25   0.85: 19
0.65: 12   0.90: 39
           0.95: 110
```

Low mode 0.45–0.66 (91 rows), trough 0.70–0.80 (25 rows), high mode ≥0.85 (168).

**Adjudication, 17/17 of the low mode with frames still on disk — all empty garden,
zero people:** 4701 (0.61), 4704 (0.62), 4713 (0.54), 4719 (0.58), 4721 (0.57),
4725 (0.55), 4726 (0.62), 4727 (0.61), 4728 (0.66), 4730 (0.56), 4731 (0.61),
4764 (0.57), 4780 (0.50), 4785 (0.54), 4786 (0.62), 4795 (0.65), 4797 (0.53).

**Contrast, high mode:** 4772 (0.93) and 4804 (0.99) are unambiguous, fully
recognizable people, correctly suppressed. The separation is clean and wide.

**Harm — the same three costs exp #14 was opened for.** All 91 low-mode rows carry
pc < 0.50, so all are unreachable by tonight's lever. Each one arms the 240 s
proximity mute, the 1800 s density counter and the 240 s deferral cancel (4798
tonight was muted on the strength of phantom 4797); each suppresses the animal
branch before species ID, with no notification and no observability; each writes
NULL `top_species_raw`. 91 rows ≈ 6% of all HUMAN rows since 07-08, ~1.7/day.

**Why it is NOT shipped tonight.** No env knob exists — this is a code change
(score threshold on the taxon arm), and the guardrail contract's "one active
experiment at a time" applies to code changes exactly as to env deltas. Exp #14 is
on night 1 with 45 triggers and not one of its pre-registered predictions is yet
evaluable; landing a second change into the *same gate*, with a demotion path of
comparable size (~1.7/day vs the ~1.7/day already deployed), would make tonight's
+3 REVIEW messages permanently unattributable. This is a guardrail hold with a
concrete release trigger, not a deferral for approval — the protocol's Autonomy
section is explicit that there is no approval step, and none is being waited on.

**Release trigger:** ship as the first act of the tick that concludes exp #14
(2–3 clean nights, or immediately on an exp #14 rollback, since rolling back T to
0.30 leaves the taxon phantoms untouched and the case only strengthens).

**Pre-registered design, for that tick:** add a score floor to the taxon arm
(`is_homo_taxon and prediction_score >= homo_taxon_min_score`), env
`SPECIES_HOMO_TAXON_MIN_SCORE`, default **0.75** — placed in the empty trough,
above the 0.66 top of the adjudicated-empty mode and below the 0.85 bottom of the
confirmed-people mode, so it is a boundary read off the data rather than a fitted
edge. Privacy-veto must be re-measured over the full 284-row taxon set before
shipping, exactly as exp #14 did: count how many demoted rows would survive the
proximity/deferral/density stack and reach REVIEW, and adjudicate every one with
frames on disk. The raw-homo-leak arm (exp #9) stays unthresholded — it is
rare-event insurance with a measured base rate of 3 rows corpus-wide.
