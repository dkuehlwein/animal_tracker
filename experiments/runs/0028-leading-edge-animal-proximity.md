---
id: 39
slug: leading-edge-animal-proximity
status: running
validation: live   # code change, commit 657a30c; replayed over the full corpus before shipping
occupies_active_slot: true   # takes the slot from exp #33, concluded KEEP tonight (see below)
hypothesis: "The Animal-Proximity Review Exemption is backward-looking, so it cannot save the LEADING edge of an animal visit: a review-class burst that the Review Sampling Gate mutes seconds BEFORE SpeciesNet names the same animal is a silent false negative. Deferring the sampled-out burst by animal_proximity_window_seconds and re-checking for a named-animal detection recovers it."
created: 2026-09-25
promoted_from: "night of 2026-09-25: burst 5448 held a cat at close range, was sampled out, and the same cat was IDENTIFIED 37s later on burst 5449."
confidence: high   # single knob, can only add notifications, corpus-replayed
delta: {}   # code change, reuses the existing PERFORMANCE_ANIMAL_PROXIMITY_WINDOW_SECONDS knob
commit: 657a30c
restart_at: 2026-09-26T03:25:00+02:00
opens: none
---

## Tonight

Eight triggers, and the first night in the loop's history with **zero false
positives**. `fp_rate 0.000` over 7 labelled triggers (95% CI 0.000–0.354).
Every single trigger was a real subject.

- **5448** (07:36:43) — `unclassifiable`, `[REVIEW-SAMPLE]` suppressed. Tier-2:
  **animal**. See below; this is the experiment.
- **5449 / 5450** (07:37:20, 07:37:52) — `IDENTIFIED` as
  `mammalia;carnivora;;;;carnivorous mammal`. A calico cat, unambiguous at full
  resolution once the frames are contrast-normalised. Raw top-1 was
  `domestic dog` @0.596 on 5449 and `domestic cat` @0.576 on 5450 — the ensemble
  rolled both up to the correct, if generic, `carnivorous mammal`. Two MAIN
  alerts, both correct.
- **5451–5454** (11:07:36–11:09:29) — `IDENTIFIED` as `aves;;;;;bird` @0.51–0.89.
  A Eurasian blackbird, black plumage and orange bill both plainly visible in
  the crops. Four MAIN alerts, all correct. Daniel human-labelled 5454 `animal`
  at 19:40, before this tick ran — **1/1 agreement with tier-2**, feedback clock
  reset.
- **5455** (15:44:52) — `human` @ `person_confidence` 0.758, legs in dark
  trousers crossing the frame. Correctly suppressed by the Human/Privacy Gate.

Zero FP, one FN (5448). The FN is the whole story of the night.

## The defect: the leading edge of an animal visit

Exp #33 (`animal-proximity-review-exemption`, shipped 2026-09-20) exists for
exactly the class of miss 5448 belongs to — SpeciesNet naming an animal on one
burst of a multi-burst visit and missing it on a neighbouring burst. But it is
**backward-looking by construction**: it exempts a review-class burst landing
*after* a named-animal `IDENTIFIED` detection. 5448 came **37s before** the
naming, so nothing in the exemption could reach it.

This is the precise structural mirror of the human-side gap exp #11 already
solved. The Human-Proximity Mute Gate was likewise backward-looking and likewise
blind to the leading edge of a visit (burst 3909, 2026-07-31, leaked a
recognisable face 81s *before* the visit's first HUMAN burst); the fix was the
Deferred REVIEW Send Gate — hold the send, re-check when the future has arrived.
The same instrument, pointed the other way, recovers 5448.

Corroborating evidence that the backward half was the wrong half: `grep`ing every
rotation of `data/logs/wildlife.log`, **`[ANIMAL-PROXIMITY]` has never fired
once** in the five nights since exp #33 shipped. Its first real firing
opportunity arrived tonight and arrived pointing the wrong way.

### Which gate actually muted 5448 — checked, not assumed

`below_sharpness_floor = 1` on 5448 (sharpness 3.13 vs the 11.0 floor), so the
Blur Gate is the obvious suspect and is wrong. Exp #8's `blur_mute_min_luma=70`
guard did its job: the 07:36 scene has mean luma far under 70, so
`luma_supports_blur_mute` was False and the blur mute never armed. The log is
unambiguous:

```
07:36:43,377 [REVIEW-SAMPLE] Suppressing notification for detection 5448
             (sampled out, rate=0.500)
```

A coin flip, nothing more. `human_proximity_muted=0`, `scene_gate_muted=0`,
`blank_confidence_muted=0` — every other gate abstained.

### Aside: tonight's sharpness collapse is exposure, not focus

All 8 rows scored `below_sharpness_floor=1`, and the daylight ceiling was **6.98**
against a typical 20–28 on other days — the lowest in the series. Checked before
drawing any conclusion from it, because a focus drift would be a hardware
problem and a much bigger deal. It is not: a fixed static region (the bamboo
wall, x 1100–1700 / y 100–500) scores lapvar 33.4 @ mean luma 107.6 on 09-21,
11.9 @ 49.2 on 09-24 and 5.7 @ 32.8 today. The scene is **3.3x darker** than
09-21, and Laplacian variance scales with contrast. CLAHE-normalising today's
frames restores bamboo-leaf edge detail comparable to 09-21. Focus is fine; the
day was dark. This is exp #7's finding (`min_sharpness_threshold` on raw
Laplacian variance is a scene-brightness gate, not a blur gate) reconfirmed, and
`blur_mute_min_luma=70` absorbed it exactly as designed. **No action** — and
specifically, the standing caution from run 0024 holds: `blur_mute_min_luma`
must never be lowered, `min_sharpness_threshold` never raised.

## Measurement before shipping

Replayed over the full corpus: 484 rows with `review_sampled_out = 1`, 175
`IDENTIFIED` rows satisfying `utils.is_named_animal_label`. For each sampled-out
row, the nearest named-animal detection landing *after* it:

| forward window | rows un-muted | animal-labelled | fp-labelled | unlabelled |
|---:|---:|---:|---:|---:|
| 60s  | 1 | 0 | 0 | 1 |
| 120s | 2 | 0 | 1 | 1 |
| **180s** | **2** | **0** | **1** | **1** |
| 240s | 2 | 0 | 1 | 1 |
| 300s | 3 | 0 | 1 | 2 |
| 600s | 6 | 0 | 1 | 5 |

The two rows at 180s are **5448** (37s, tonight's cat — the "unlabelled" row in
the table, since the replay predates tonight's tier-2 write) and **5359** (72s,
tier-2 `false_positive`). So the change costs **one extra REVIEW message in two
months** and recovers a confirmed animal. The window is flat between 120s and
240s, so 180s is not a knife-edge choice — it is simply the value exp #33
already measured and shipped, reused.

## The change

Reuses `PERFORMANCE_ANIMAL_PROXIMITY_WINDOW_SECONDS` (default 180.0, bounded
`[0.0, 600.0]`). **No new config field, no new DB column, no new rollback
lever** — `0` still disables both halves at once.

In `_process_and_notify_detection`, a sampled-out review-class burst is no
longer dropped on the spot when the window is non-zero: it falls through to the
same branch that builds the annotated image and is handed to
`_deferred_review_send` with `require_animal_proximity=True`. That coroutine
sleeps `animal_proximity_window_seconds`, then re-reads
`self._last_animal_detection_at`:

- not within `(timestamp, timestamp + window]` → log `[REVIEW-SAMPLE]` and
  return without sending, leaving `review_sampled_out = True`. The common case,
  and still silent on Telegram.
- within the window → log `[ANIMAL-DEFER]`, persist
  `update_review_sampled_out(detection_id, False)` so the row records what
  actually happened (exp #33's convention), and continue.

Waking exactly at `timestamp + window` is what makes the check trivially
correct: `_last_animal_detection_at` is by definition the most recent named
animal at or before that instant, so any value greater than `timestamp` is
necessarily inside the window. The upper bound is still asserted explicitly
rather than relied upon.

**Privacy precedence is preserved.** A recovered burst then sleeps only the
*remaining* human-defer time and runs the unchanged cancel-on-human test over
its full `review_defer_seconds` window. A person arriving after the burst still
wins, exactly as before.

**Fail-open direction differs by phase, deliberately.** An exception raised
before the animal decision falls back to today's behaviour — **suppress**; a
sampled-out burst is already a deliberate drop, and a bug here must not
manufacture notification volume out of nothing. Once the animal decision has
said "recover", any later failure falls open to **send**, as everywhere else in
this file. `asyncio.CancelledError` keeps propagating untouched so shutdown
never fires a spurious send.

## Gate audit

- **Review Sampling Gate:** 1 mute (5448) — **wrong**, and the subject of this
  experiment. First sampling-gate FN since exp #33.
- **Blur Gate:** 0 mutes despite 8/8 below-floor rows; `blur_mute_min_luma`
  correctly disarmed it on every one.
- **Confident-Blank / Scene / Human-Proximity / Unnamed-Animal-Blank Gates:**
  no firing opportunity. `scene_similarity` NULL on the two non-review rows,
  0.807–0.840 on the IDENTIFIED rows — nowhere near T=0.982.
- **Human/Privacy Gate:** 1 correct suppression (5455).
- **Exp #35/#37 purge duty:** no `human_proximity_muted = 1` row tonight;
  nothing to sweep.

## Exp #33 — CONCLUDED, KEEP

Five nights live, zero firings, zero cost. It remains correct for the trailing
edge (its motivating 5388/5389 case was a 25s trailing miss) and is now one half
of a symmetric pair rather than a lone gate. Keeping it is free; removing it
would reopen the trailing-edge case. The slot passes to exp #39.
