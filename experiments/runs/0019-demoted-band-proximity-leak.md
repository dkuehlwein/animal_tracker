---
id: 27
slug: demoted-band-proximity-leak
status: running
validation: live   # code change, restart-gated; FN cost measured over the full 3103-row review-class+unnamed corpus
occupies_active_slot: false  # scope repair of the Human-Proximity Gate (exp #10/#11/#26), same precedent as exp #23 vs #9, #24 vs #21, #26 vs #21
hypothesis: "Exp #14 raised SPECIES_HUMAN_DETECTION_CONFIDENCE 0.3 -> 0.5 on the explicit promise that 'the layered proximity/deferral/density gates still mute the real people who score in the demoted band'. Tonight that promise failed for the first time: burst 5305 scored person_confidence 0.436 — a real person's clothing filling the lens — and was missed by BOTH layered conditions (480 s > the 240 s window; 5 human bursts < the density count of 8). It reached Telegram's doorstep and was held back only by the 50% review-sampling coin flip. A sub-threshold person score is not noise when a human visit is already in progress: conditioning the proximity window on the demoted band closes the leak at zero measured FN cost, because no human-labelled animal row in the entire corpus scores above person_confidence 0.079."
created: 2026-09-15
promoted_from: "tier-2 adjudication of 2026-09-15: burst 5305, a close-up of clothing during a 21-burst gardening session, survived every privacy gate."
confidence: high   # all 3103 review-class/unnamed rows replayed; the 4 the change mutes contain zero human-labelled animals
commit: 3752a7b
env_delta: {}
restart_at: 2026-09-16T03:25:00+02:00
---

## Tonight — 27 triggers, zero animals, zero notifications sent

| status | n |
|---|---|
| `human` (suppressed) | 21 |
| `unclassifiable` | 5 |
| `no_animal` | 1 |
| `identified` | **0** |

Not one Telegram message left the Pi tonight. The 21 human bursts were
suppressed by the privacy gate; of the 6 review-class bursts, 2 were
proximity-muted (5297, 5304) and 4 were sampled out (5280, 5294, 5305, 5306).
All 27 bursts still have all 5 frames on disk and all 27 were adjudicated.

**All 21 human-status rows are genuine people.** Six carry a
`person_confidence` at or near zero (5301 at 0.000, 5295 at 0.017, 5298 at
0.023, 5299 at 0.072, 5289 at 0.099, 5300 at 0.361) and were caught only by the
burst sweep / raw-homo path — legs, arms and clothing at close range, every one
unambiguous on inspection. No phantom-human suppression, so no silent FN from
that direction.

**None of the 6 review-class bursts contains an animal.** 5280 is the empty
pond under a light shift. 5294, 5297, 5304 and 5306 are out-of-focus objects
sweeping past the lens during the gardening session — 5297 in particular is a
large pale mass entering from the right, and it was correctly proximity-muted
(98 s after a HUMAN burst).

## The leak — 5305, the demoted band's first real person

5305 (18:25:14, `unclassifiable`, `person_confidence` **0.436**) is a smooth,
featureless grey-beige mass filling the left half of frames 1–2: clothing at
touching distance from the lens. It is the same visual class as 5301
(`person_confidence` 0.000), which the raw-homo path classified HUMAN twenty
minutes earlier. 5305's raw top-1 was `blank` @0.54, so no raw path caught it,
and 0.436 sits just under the 0.5 HUMAN gate.

Both layered conditions missed it, neither by much:

| condition | value at 5305 | threshold | verdict |
|---|---|---|---|
| time since last HUMAN burst | 480 s | ≤ 240 s | miss (2.0x over) |
| HUMAN bursts in trailing 1800 s | 5 | ≥ 8 | miss |

What actually stopped it was `review_sampled_out=1` — a deterministic hash of
the detection id landing on the mute side of a 50% gate. A coin flip is not a
privacy control.

This is the residual risk exp #14 (runs/0012) named and accepted. Its
adjudication found exactly one real person in the demoted band (4741 at pc
0.435) against 32 phantoms; 5305 is the second, at pc 0.436 — within 0.001 of
the first. The band is not empty of people, it is just mostly noise.

## Why no env knob reaches it — measured, not assumed

Both existing levers are FN-vetoed at the setting that would have caught 5305.
Replayed over all 20 human-labelled `animal`/`animal_wrong_id` rows in the
review-class + unnamed-animal corpus:

| lever | setting needed for 5305 | animal-labelled rows it mutes |
|---|---|---|
| `PERFORMANCE_HUMAN_PROXIMITY_WINDOW_SECONDS` | ≥ 480 s | **1** — id 1838 at 329 s |
| `PERFORMANCE_HUMAN_DENSITY_COUNT` | ≤ 5 | **1** — id 2011 at density 5 |

The global window costs a known false negative at anything above ~400 s; the
density count costs one at 5 or below. Both are rejected under the FN-veto.
Hence a code change, per the protocol's "no env knob reaches the root cause"
clause.

## The change — a band-conditional look-back window

`person_confidence` separates the two populations by a factor of four, with no
overlap at all:

- max `person_confidence` over **all 20** human-labelled animal rows: **0.0789**
- 5305: **0.436**

So the burst's own person score can safely gate a longer window. When a
review-class (or unnamed-animal IDENTIFIED, per exp #26) burst carries
`person_confidence >= human_demoted_person_floor` (0.3 — exp #14's old HUMAN
threshold, the bottom of the demoted band), the human-proximity look-back
window becomes `human_demoted_window_seconds` (1800 s, the same "garden is
occupied" horizon the density condition already uses) instead of 240 s.
Everything else is unchanged: same `human_proximity_muted` column, same
`[HUMAN-PROXIMITY]` log line, same precedence, same density condition, same
fail-open behaviour.

The conjunction is what makes it safe. "MegaDetector saw something
person-shaped" is noise on its own (exp #14 measured 32 phantoms); "a human
visit happened in the last half hour" is weak on its own (the density condition
needs 8 bursts to act). Together they are specific.

**Measured cost over the whole corpus** (3103 review-class + unnamed-animal
rows, ~3 months): only **19** carry `person_confidence >= 0.3` at all. The
change newly mutes **4** of them — and zero human-labelled animals:

| burst | date | pc | since last human | adjudication |
|---|---|---|---|---|
| 5305 | 09-15 | 0.436 | 480 s | **person, close-up clothing** |
| 5090 | 09-08 | 0.313 | 547 s | empty garden (a demoted-band phantom) |
| 4765 | 08-31 | 0.378 | 490 s | frames rolled off disk |
| 4782 | 08-31 | 0.340 | 1747 s | frames rolled off disk |

~1.3 mutes/month — the same order as exp #26. One is the leak this experiment
exists to close, one is an empty-scene FP whose muting is a small REVIEW-volume
win, two are unrecoverable.

Widening the window to 2400 s would mute 6 (still zero animals); 1800 s was
chosen to reuse the existing density horizon rather than introduce a third
time constant.

## Rollback

`PERFORMANCE_HUMAN_DEMOTED_WINDOW_SECONDS=0` + service restart disables the
band-conditional window and restores the flat 240 s behaviour, leaving the
density condition and every other gate untouched. Full revert: `git revert 3752a7b`.

## Scene gate / sampling gate duties

- Scene gate: **0 mutes** tonight, and 0 in 16 days. Tonight's review-class
  `scene_similarity` ran 0.583–0.860 against T=0.97. Still inert (backlog #17),
  still not re-derived per the 2026-07-26 override.
- Sampling gate: 4 sampled out (5280, 5294, 5305, 5306), all four adjudicated
  above — three empty/near-empty scenes and 5305. Zero animals. Hold at 0.5.
  Note 5305 is the first sampled-out burst whose muting mattered for privacy
  rather than volume; that is an argument for fixing the privacy gate, not for
  trusting the sampler.
- Exp #26 (unnamed-animal gate, live since the 03:30 restart today): **no
  observation** — there were zero IDENTIFIED rows tonight, so the code path was
  never exercised. Carry forward.
