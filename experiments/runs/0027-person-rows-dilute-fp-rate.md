---
id: 38
slug: person-rows-dilute-fp-rate
status: running
validation: live   # code change, commit 568746d; recomputed over the full corpus before shipping
occupies_active_slot: false  # tick-side measurement plumbing, touches no live gate; exp #33 keeps the slot
hypothesis: "A person-labelled trigger is neither a false alarm nor a wildlife detection, so counting it as a denominator success in fp_rate makes a person-heavy night print a flattering false-alarm rate that a later tick can misread as an FP win."
created: 2026-09-24
promoted_from: "night of 2026-09-23: 29 of 32 triggers person-labelled, fp_rate printed 0.094 — the lowest value in the whole of metrics/daily.csv — while every one of the 3 non-person triggers was a false positive. Filed as backlog #38 that night and deliberately not changed then (one change per tick)."
confidence: high   # not a discriminator; an accounting error with a single defensible fix
delta: {}   # code change, no new knob
commit: 568746d
restart_at: null   # tick-side only — loop.metrics/loop.report are not imported by wildlife_system
opens: none
---

## Tonight

Three triggers. The quietest night the loop has recorded.

- **5445** (08:08) — `unclassifiable`, blank raw top-1 @ 0.982, muted by the
  Confident-Blank Gate.
- **5446** (13:56) — `unclassifiable`, sampled out.
- **5447** (14:48) — `unclassifiable`, sampled out.

All three adjudicated at full resolution this tick: empty garden, wind in the
bamboo. 5445 additionally shows a global frame-to-frame translation (motion_area
2555 from only 10 contours, largest 7121 px) — the whole scene shifting, i.e.
wind on the camera mount, not a subject. Zero animals, zero people, and for the
second night running **no Telegram message was sent at all**.

### The gates, audited

- **Confident-Blank Gate (exp #29):** 1 mute (5445, blank @ 0.982 ≥ T=0.92),
  correct. Running total still clean.
- **Review Sampling Gate:** 2 mutes, both correct on inspection.
- **Scene Gate (exp #30, T=0.982):** 0 mutes; similarities 0.77 / 0.79. Still
  vacuous as an FP lever, still armed. No `scene_gate_muted=1` row to adjudicate.
- **Human gates:** wholly inert — `person_confidence` 0.0 on all three rows.
- **Animal-Proximity Exemption (exp #33):** no firing opportunity, a fourth
  night running. It has still never produced a data point.

### Exp #35/#37 purge monitoring duty — first run, clean

The duty opened by run 0026 is: any `human_proximity_muted = 1` row older than
`human_retention_hours` whose `image_path` still exists on disk means the purge
is not doing what the fix claims. Measured tonight, the night after 0bc9ec9 went
live at the 03:30 restart:

- 171 `human_proximity_muted = 1` rows older than 48h — **0 still on disk**.
- 1779 HUMAN-status rows older than 48h — **0 still on disk**.

The five rows run 0026 found stranded (5074/5075/5076 at 15 days, 5154 at 11
days, 5444 at 70s outside the old window) are gone. The fix did what it said.

### Self-audit: human labels arrived and they agree with tier-2

Four human `animal` labels landed at 21:09 on 2026-09-23, after that tick had
already run — on 5407/5408/5409/5410, the 09-22 blackbird visit. Tier-2 had
labelled all four `animal` on 09-22. **5 of 5 agreement** (5411 was labelled by
hand earlier the same day, also matching). No new false negative is implied: all
five were correctly IDENTIFIED and sent to MAIN at the time. The feedback-starved
clock is reset — last human label 2026-09-23.

## The finding: person rows were scored as successes

`compute_metrics` built its `labeled` denominator from every row with a
reconciled label except `cant_tell`, and its numerator from `false_positive`
rows only. A `person` row therefore sat in the denominator contributing nothing
to the numerator — scored, in effect, as a correct wildlife detection.

It is neither. A person trigger is a real subject the privacy gate deliberately
suppresses on its own path; it is not a false alarm, and it is not wildlife.
`cant_tell` was already excluded on exactly this reasoning. `person` was not,
because the label only entered the vocabulary in the 2026-07-09 keyboard
redesign, after the denominator rule was written.

### Measured over the full corpus, before shipping

71 of 3671 labelled rows corpus-wide are `person` — 1.9%, which sounds
negligible. It is not, because the effect is concentrated in exactly the nights
where it does the most damage. Restating every date whose fp_rate moves:

| date | labelled | fp | fp_rate as printed | person | corrected labelled | corrected fp_rate |
|------|---------:|---:|------:|-------:|------:|------:|
| 2026-07-10 | 41 | 23 | 0.561 | 4 | 37 | 0.622 |
| 2026-07-14 | 37 | 25 | 0.676 | 4 | 33 | 0.758 |
| 2026-07-15 | 35 | 31 | 0.886 | 1 | 34 | 0.912 |
| 2026-07-16 | 53 | 44 | 0.830 | 2 | 51 | 0.863 |
| 2026-07-17 | 28 | 10 | 0.357 | 1 | 27 | 0.370 |
| 2026-07-18 | 5 | 3 | 0.600 | 1 | 4 | 0.750 |
| 2026-07-19 | 89 | 88 | 0.989 | 1 | 88 | 1.000 |
| 2026-07-21 | 85 | 75 | 0.882 | 7 | 78 | 0.962 |
| 2026-07-27 | 95 | 93 | 0.979 | 1 | 94 | 0.989 |
| 2026-07-30 | 31 | 29 | 0.935 | 1 | 30 | 0.967 |
| 2026-09-03 | 4 | 3 | 0.750 | 1 | 3 | 1.000 |
| 2026-09-06 | 3 | 2 | 0.667 | 1 | 2 | 1.000 |
| 2026-09-08 | 31 | 29 | 0.935 | 2 | 29 | 1.000 |
| 2026-09-12 | 16 | 11 | 0.688 | 1 | 15 | 0.733 |
| 2026-09-13 | 3 | 0 | 0.000 | 1 | 2 | 0.000 |
| 2026-09-14 | 11 | 6 | 0.545 | 3 | 8 | 0.750 |
| 2026-09-15 | 6 | 3 | 0.500 | 3 | 3 | 1.000 |
| 2026-09-17 | 23 | 22 | 0.957 | 1 | 22 | 1.000 |
| 2026-09-18 | 16 | 10 | 0.625 | 1 | 15 | 0.667 |
| 2026-09-19 | 31 | 26 | 0.839 | 1 | 30 | 0.867 |
| 2026-09-20 | 8 | 2 | 0.250 | 4 | 4 | 0.500 |
| **2026-09-23** | **32** | **3** | **0.094** | **29** | **3** | **1.000** |

The last row is the whole argument. **0.094 is the lowest fp_rate in the entire
recorded history of this system**, and it is an artifact of foot traffic. It was
written into `metrics/daily.csv` the night after exp #35 shipped. A future tick
scanning the trend for "what did we change that worked" would find a 3x FP
improvement immediately following a privacy fix that cannot affect fp_rate at
all — and the loop's own protocol tells it to trust that CSV.

This is a self-poisoning path, and it is the kind the protocol's
anti-self-skepticism section exists to catch: the loop grading its own homework
with a rubric that rewards the wrong thing.

## What shipped

Commit `568746d`. No restart stamped — `loop.metrics` and `loop.report` run
inside the tick and are not imported by `wildlife_system` (verified: `src/` imports
only `loop.guardrails` and `loop.state`/`loop.deploy`, from `config.py` and
`telegram_feedback.py`).

- `person` leaves the `labeled` denominator in `compute_metrics`, alongside
  `cant_tell`.
- `_per_tier_partition` skips `person` winners at all three tiers, preserving the
  documented invariant `n_human + n_claude + n_md == labeled_triggers`
  (re-verified against 2026-09-23's rows: holds).
- New `n_person`, surfaced in the report as
  `• Person (not counted as a false alarm): N` and added to the `daily.csv`
  schema.
- The legacy arithmetic fallback for the "Not yet labelled" remainder subtracts
  `n_person` too, so person rows can't reappear there.
- 734 tests pass (723 before; +11 from this change).

### History deliberately not rewritten

Historical `daily.csv` rows keep a blank `n_person` and their original
`fp_rate`. Recomputing them would mean re-deriving each night's ingest *window*
from timestamps, which is not the same grouping the ticks actually used — that
would change more than the person definition and quietly rewrite a record that
was correct under the rule in force at the time. The restatement table above is
the correction, and it lives in git.

**Read `daily.csv` accordingly: there is a definition change at 2026-09-24.**
Before it, `fp_rate` includes person rows in its denominator; from it, it does
not.

### Gates

FN-veto does not apply — no notification, classification or routing behaviour
changes; no animal can be hidden by an accounting change. Volume guardrail
unaffected. `paused` false. Feedback-starved clock reset (human labels
2026-09-23). Exp #33 keeps the active slot; this occupies none, same as exps
#26/#29/#32/#35, being tick-side and not a live gate.

The honest cost: `fp_rate` will read *higher* from tonight on, and the 09-23→
09-24 step from 0.094 to 1.000 is mostly this change plus an n=3 night, not a
regression. Stated here so no future tick reads it as one.

## Monitoring duty

None ongoing. The invariant `n_human + n_claude + n_md == labeled_triggers` is
asserted in the test suite.

## Rollback

`git revert 568746d`. No restart, no env lever — the change is inert with
respect to the running camera.
