---
id: 33
slug: animal-proximity-review-exemption
status: running
validation: live   # code change, commit ea652bc; replayed over the full corpus before shipping
occupies_active_slot: false  # notification-routing fix on an FN leak path, same shape as exp #26/#29/#32; exp #21 keeps the slot
hypothesis: "When SpeciesNet has just named a real animal, a review-class burst arriving seconds later is far more likely to be the same animal than a fresh false positive — so the Review Sampling Gate, which is blind to that context, should not be allowed to discard it."
created: 2026-09-20
promoted_from: "night of 2026-09-20: burst 5389, a blackbird clearly visible in-frame 25s after the same bird was correctly identified in 5388, came back unclassifiable and was sampled out — it never reached Telegram in any form. Second measured instance of backlog #31's FN class."
confidence: high   # the FN side is 3 independent animal-labelled rows, the volume cost is 6 sends over 8 weeks, and the gate being bypassed is a coin flip with no evidential content
delta: {}   # code change; PERFORMANCE_ANIMAL_PROXIMITY_WINDOW_SECONDS ships at its 180.0 default
commit: ea652bc
restart_at: 2026-09-21T03:25:00+02:00
opens: backlog #33 (animal-proximity-review-exemption)
---

## Tonight

8 triggers — the quietest day since 09-11 (6) and well inside the observed
6-68 range, on a calm day with no wind in the bamboo. The service ran the full
day (03:30 restart, sunset stop at 19:34, zero errors in `wildlife.log`), so
this is weather, not a collapse: no volume guardrail is implicated.

One animal visit, one person visit, two false positives:

- **5388** (09:12:59) — a blackbird on the gravel at the pond edge, correctly
  IDENTIFIED as `aves;;;;;bird` @0.749 and alerted to MAIN. Below the sharpness
  floor (4.5) and alerted anyway, which is the Blur Gate behaving as designed.
- **5389** (09:13:24) — **the same bird, 25 seconds later, plainly visible in
  frame1**, returned `unclassifiable` (`no cv result`), raw top-1 blank @0.512.
  Review-class. It then lost the Review Sampling coin flip and was never sent.
- **5390** (14:49), **5391** (15:32) — empty garden, false positives. 5390's
  63k motion area is an AE brightness step across the burst, not a subject.
- **5392-5395** (16:50-16:55) — one person, four bursts, all HUMAN-status and
  all suppressed. 5392 is a close-pass blur filling the right third of frame;
  `person_confidence` 0.306 there but the raw `homo;sapiens;human` taxonomy
  check carried it. No phantom-human, no leak.

Daniel had already labelled 5388/5390/5391 by hand and agreed with tier-2 on
all three (3/3 concordance).

### The gates, audited

Nothing was muted by anything tonight. Every mute gate is clean by inspection:

- **Confident-Blank (exp #29, night 4):** 3 review-class rows evaluated, 0
  mutes — blank raw scores 0.512 / 0.862 / 0.738, all under T=0.92. Note 5389
  is a **new animal-side data point at 0.512**, comfortably under the corpus
  animal ceiling of 0.8475 the threshold was built on. No change indicated.
- **Scene gate (exp #30, night 2 at T=0.982):** 0 mutes, similarities 0.736 /
  0.829 — nowhere near the band. Nothing to adjudicate.
- **Unnamed-Animal-Blank (exp #32, night 1 live):** no burst of that shape
  occurred, so the gate was never evaluated (`unnamed_animal_blank_muted` NULL
  on all 8 rows). Unexercised is the expected outcome at ~1.5 mutes/month.
- **Human-proximity / demoted-band / density:** 0 mutes; the person visit was
  caught by the primary gate, and no review-class burst followed it.

One note against exp #32 for the record, not a kill-condition event: 5389 is an
animal-bearing burst whose raw top-1 is blank at **0.512** — the low end of the
band exp #32 mutes. It is `unclassifiable`, not the `;;;;;;animal` shape the
gate acts on, so it is outside that gate's population and the pre-registered
kill condition (an animal-labelled row of that shape inside the muted band) has
NOT fired. But it is the first evidence that "low-confidence blank raw top-1"
can co-occur with a real animal, which is exactly the direction exp #32's n=1
carve-out was flagged as vulnerable to. Watch it.

## Why a code change, and why this one

Backlog #31 (SpeciesNet misses a visible bird) recurred tonight for the second
time in three days. That root cause is a model-quality problem with no env
lever — but tonight's instance exposed a *second*, separable failure that does
have one: even when the pipeline fails to name the animal, the burst still
carries the frames, and the only thing standing between those frames and
Daniel's phone was a **context-free coin flip**.

The Review Sampling Gate (`PERFORMANCE_REVIEW_SAMPLE_RATE`, 0.5) is a pure
volume lever: it hashes `detection_id` and discards half of all surviving
review-class bursts. It has no notion of whether an animal was on camera nine
seconds ago. PROTOCOL's own rule for this gate says raising the rate is the
correct response to genuine FN evidence — and we now have that evidence — but
raising it globally would double REVIEW volume to buy back a handful of bursts.
A targeted exemption buys the same bursts for ~1% of the volume.

### Measured before shipping, over the whole corpus

Anchors = IDENTIFIED rows whose label **names** an animal (not `;;;;;;animal`,
not `blank`, not homo): 166 of them. For review-class rows landing within W
seconds after one:

| W | in-window review rows | of those, muted by... | newly sent under the exemption |
|---|---|---|---|
| 120s | 35 | sampling 5, everything else **0** | 5 (2 animal-labelled) |
| 180s | 42 | sampling 6, everything else **0** | 6 (3 animal-labelled) |
| 300s | 49 | sampling 7, everything else **0** | 7 (3 animal, +1 FP) |

The first column of that table is the load-bearing result: **the scene gate,
the confident-blank gate, the blur gate and the human-proximity gate have
never once muted a review-class burst in an animal's wake.** Sampling is the
entire leak. So exempting sampling alone is not a compromise — it is the
complete fix for this class, and it leaves every privacy-relevant gate
untouched.

Threshold selection, on evidence: the three animal-labelled sampled-out rows
sit at **25s, 25s and 124s** (ids 5342, 5389, 5345); the nearest
`false_positive`-labelled row sits at **206s**. W=180 clears max(animal)=124
with a 56s margin and stops 26s short of the nearest known FP — the same
shape of rule the protocol uses elsewhere, and there is a real gap to put it
in. W=120 would have missed 5345 by four seconds.

Cost: 6 extra REVIEW sends across the ~8 weeks sampling has been live
(~0.75/week), at a 50% animal hit rate — against the 1-real-catch-per-155-pings
ratio that motivated sampling in the first place. Zero person-labelled rows
appear anywhere in the newly-sent set at W=180 (the first one appears only at
W≥300, and is not in the newly-sent bucket even there).

### Gates cleared

- **FN-veto**: inapplicable in the blocking direction — the change only ever
  sends more. It *recovers* 3 known FNs and cannot create one.
- **FP rate**: unaffected by construction. `fp_rate` is label-conditioned and
  nothing about capture, classification or the FP gates changes.
- **Volume**: +~1 message/week. No collapse, no explosion.
- **Privacy**: the exemption sits at the Review Sampling position, below every
  human gate. A burst muted by Human/Privacy, Human-Proximity (window, density
  or demoted-band), Unnamed-Animal-Blank, Blur, Confident-Blank or Scene stays
  muted; the deferred cancel-on-human send is untouched. The implementation can
  only flip `review_sampled_out` True→False, and `review_sampled_out` is
  consulted only after all six earlier flags have declined to mute. Two
  regression tests assert exactly this.
- **Not paused**, not feedback-starved (Daniel labelled 3 rows today).

## Shipped

`ea652bc`, restart-gated 2026-09-21T03:25. New
`PERFORMANCE_ANIMAL_PROXIMITY_WINDOW_SECONDS` (180.0 default, bounds
[0.0, 600.0], `0.0` = disable = rollback lever, also in `loop.guardrails`),
new `utils.is_named_animal_label`, new
`DatabaseManager.get_last_animal_detection_time()` seeding
`WildlifeSystem._last_animal_detection_at` at startup, one `[ANIMAL-PROXIMITY]`
log line per exemption. **No new DB column** — the decision is reconstructable
offline because `_review_sample_fraction(detection_id)` is deterministic, so a
schema migration would have recorded something already derivable. Fails open
everywhere. 715 tests pass.

Nightly duty from 09-21: adjudicate every burst carrying an
`[ANIMAL-PROXIMITY]` exemption. Pre-registered kill condition — if the
exemption delivers **five consecutive** review-class bursts with no animal in
any of them, the window is too wide; shrink toward 120s before disabling.

---

## Night 1 live (2026-09-22) — no firing opportunity, experiment continues

Exp #33 shipped 09-20 but only reached the camera at **03:30:03 today**, when
exp #34's `_as_aware()` fix let `wildlife-deploy.service` consume the stamp for
the first time. This is its first measurement night, not its third.

**The exemption did not fire, and could not have.** 7 triggers. The two
review-class bursts sit 4.4h before (5406, 10:55) and 1.4h after (5412, 16:50)
the day's only animal identification — nowhere near the 180s window. Zero
`[ANIMAL-PROXIMITY]` lines in `wildlife.log`. The pre-registered kill condition
(five consecutive exempted bursts with no animal) is untouched at 0. The
experiment is neither confirmed nor refuted; it needs days on which a
review-class burst actually lands inside the window.

### What today did show: the FN class exp #33 exists to catch is real and recurring

A single blackbird worked the pond edge from **15:22:08 to 15:25:11** — five
consecutive bursts, 5407-5411, all adjudicated `animal` tier-2 on visual
inspection. SpeciesNet identified all five and every one reached MAIN:

| id | ensemble label | conf | raw top-1 | sharp | luma |
|---|---|---|---|---|---|
| 5407 | `aves;;;;;bird` | 0.662 | `aves;;;;;bird` 0.441 | 8.14 | 43.1 |
| 5408 | `aves;;;;;bird` | 0.914 | blue whistling-thrush 0.499 | 8.35 | 43.7 |
| 5409 | `aves;;;;;bird` | 0.747 | `aves;;;;;bird` 0.614 | 8.57 | 44.7 |
| 5410 | `;;;;;;animal` | 0.651 | black woodpecker 0.188 | 8.76 | 45.5 |
| 5411 | `;;;;;;animal` | 0.548 | `aves;;;;;bird` 0.317 | 9.03 | 46.6 |

Two of the five degraded to the fully-generic `;;;;;;animal` rollup — the exact
label exp #26 and exp #32 were built around — and the raw top-1 was a Himalayan
congener on one burst and a woodpecker on another. The classifier was holding
on by its fingernails across this visit. Had any burst tipped one step further
to `unclassifiable`, as 5389 did on 09-20 with the same species at the same
spot, it would have been a review-class burst 25-90s after a named-animal
identification — precisely exp #33's case. The window did not open tonight, but
the mechanism that opens it was visibly close all afternoon.

### Monitoring duties, discharged

- **Scene gate**: 0 bursts muted. Similarities 0.68-0.70, far under T=0.982.
  Vacuous, as on 09-21.
- **Blur gate**: 0 bursts muted — see the self-audit below, this is now
  structural rather than incidental.
- **Human/proximity/density/deferral gates**: no HUMAN-status burst today; all
  inert by construction. `person_confidence` was 0.0 on all 7 rows.
- **Confident-blank gate**: muted 5412 (raw top-1 blank @0.952 ≥ 0.92). Frame
  and intra-burst motion blobs inspected — empty pond, bamboo and pond glint
  only. Correct mute.
- **Review sampling**: sampled out 5406. Inspected, empty. No loss.

Net: **zero false positives reached Telegram today, and all five animal bursts
did.** That is the best-separated day in the notebook.

## Self-audit: the blur gate has been inert for three weeks, and that is load-bearing

Prompted by today's numbers — every frame of every bird burst scored 8.1-9.2
against a `min_sharpness_threshold` of 11.0, and the frame selector logged
`fallback (none cleared sharpness floor)` on six of seven bursts.

**`[BLUR]` has not fired since 2026-09-01.** Zero occurrences across
`wildlife.log` and backups `.1`-`.3` (09-01 → now); the last 13 are all in
`.4`/`.5`, dated 08-23 to 09-01. The cause is exp #8's luma guard: the garden's
best-frame luma is now 43-53 through the afternoon, well under
`blur_mute_min_luma=70`, so `luma_supports_blur_mute` is False and the mute
never arms. Today only the 10:55 burst (luma 100.7) was even eligible.

**This is doing real work, not idling.** The sharpness floor is not a valid FP
discriminator on this corpus — measured over every row with a recorded
`sharpness_score`:

| review-class bucket | n | below floor |
|---|---|---|
| animal-labelled | 16 | **4 (25.0%)** |
| false-positive-labelled | 497 | **99 (19.9%)** |

Below-floor is *more* common among real animals than among false positives. A
gate keyed on it alone would discard animals at a higher rate than FPs —
strictly worse than a coin flip, the same "the discriminator does not
discriminate" shape exp #16/#17 found for pixel similarity and exp #29 fixed by
switching discriminators. Today's visit is the concrete case: all five bursts
were below floor, all five held a clearly visible blackbird.

The luma guard is the only thing standing between that statistic and real
false negatives. Recorded as a standing caution: **`blur_mute_min_luma` must
never be lowered, and `min_sharpness_threshold` must never be raised, on
FP-reduction grounds.** Neither is in `loop.guardrails.BOUNDS`, so the loop
cannot do it by accident — this note exists so no future tick argues for adding
them.

## Measured, not shipped: is underexposure what makes SpeciesNet miss? (→ backlog #36)

Backlog #31 (`speciesnet-misses-visible-bird`) has no mechanism attached. The
obvious candidate after today is light: `CAMERA_AE_EXPOSURE_MODE=short`, shipped
for the dusk fix in run 0006, biases AE toward shorter exposures, and the scene
is now dark enough mid-afternoon (luma 43-46 at 15:22) that this may be
starving the classifier.

Tested on every labelled row whose frame is still on disk (286 rows; luma
computed offline from the best frame):

| animal-labelled outcome | n | luma min / median / max |
|---|---|---|
| → `identified` (hit) | 25 | 17.5 / **73.4** / 97.2 |
| → review-class (**miss**) | 6 | 17.4 / **54.3** / 74.7 |

The six misses: 5176 (57.7), 5214 (74.7), 5342 (56.5), 5344 (52.0), 5345
(50.6), 5389 (17.4). Median luma of a miss is 19 points below that of a hit,
and every miss but one sits under 60.

**Not actionable yet, deliberately.** n=6, the ranges overlap (hits reach as low
as 17.5), and the midday-luma-by-day series over 09-01..09-22 is dominated by
cloud cover rather than a clean seasonal slope (medians bounce 45.5 → 102.9 →
49.0 → 45.5). A 6-row bucket cannot carry a change to the camera's exposure
policy, which would alter *every* frame the system captures and confound exp
#33's window outright. Filed as backlog #36 with the numbers attached, to be
re-measured when the miss bucket reaches ~15 rows.

## Status

**Running.** Night 1 of a live window that has produced no test of the
hypothesis. No change deployed, no `pending_restart_at` stamped. Exp #33 keeps
the active slot; nothing new opened tonight for the same reason exp #34 gave
last night — anything touching classification, exposure or review routing would
confound the window before it has yielded a single data point.
