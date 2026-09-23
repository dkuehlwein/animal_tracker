---
id: 35
slug: human-proximity-purge-gap
status: running
validation: live   # code change, commit 0bc9ec9; replayed over the full corpus before shipping
occupies_active_slot: false  # retention/privacy fix on a leak path, same shape as exp #26/#29/#32; exp #33 keeps the slot
hypothesis: "A burst the human-proximity mute gate already muted is human-adjacent by the system's own verdict, so its frames must follow the 48h human-photo policy — not a second, narrower ±240s time test that the gate's density and demoted-band conditions can legitimately fall outside of."
created: 2026-09-23
promoted_from: "night of 2026-09-23: burst 5444, a recognisable person at close range, muted by the DENSITY condition at 310s from the nearest HUMAN burst and therefore left on disk for the full max_images rotation."
confidence: high   # not a discriminator; a consistency repair between two rules that were meant to agree
delta: {}   # code change, no new knob
commit: 0bc9ec9
restart_at: 2026-09-24T03:25:00+02:00
opens: backlog #35 is taken (corner-roi-bamboo, rejected) — this is backlog #37
---

## Tonight

32 triggers, and for the first time since the loop started, **not one of them
produced a Telegram message.** Every single trigger was correctly suppressed.

- **1** generic-animal rollup (5413, 16:17) — wind in the top-right bamboo.
- **28** HUMAN-status bursts, 17:38–18:31, one long gardening session.
- **3** review-class bursts inside that session (5425, 5432, 5444).

Zero animals were in the garden tonight. All 32 were adjudicated visually this
tick.

### The gates, audited

- **Unnamed-Animal-Blank Gate (exp #32): first live mute, and it is correct.**
  5413 carries `;;;;;;animal` with raw top-1 `blank` at 0.760 < T=0.900, so it
  was muted with `[UNNAMED-BLANK]`. The five frames are the bamboo stand moving
  in the upper-right corner; the largest inter-frame blob is 13383 px at
  x=1898-2028, y=0-248, i.e. entirely inside the bamboo. No animal. Without this
  gate that burst would have been a MAIN-channel "animal detected" — precisely
  the failure exp #32 was built for, reproduced four nights after shipping.
  Running total: **1 mute, 1 correct.** Kill condition untouched.
- **Human/Privacy Gate:** 28 suppressions, all correct. Spot-checked 5438 and
  5442 at full resolution (both `person_confidence` 0.0, both routed on the
  `homo` taxonomy segment alone) — patterned clothing and hair filling the
  frame at arm's length. The gate is carrying this night almost single-handed.
- **Human-Proximity Gate:** 3 mutes. 5425 (window, 240s) and 5432 (window) are
  empty garden — harmless. **5444 (density, ≥8 HUMAN in 1800s) is a person**:
  the left third of the frame is green fabric with visible cloth folds at
  extreme close range, `person_confidence` 0.063, far below the 0.5 primary
  gate. This is the exp #11 leak class, caught by the density condition that
  exp #11's extension added. The notification path did its job.
- **Confident-Blank Gate (exp #29):** evaluated 5425/5432 as muted (blank @
  0.984/0.986) but human-proximity took precedence, so one log each, as designed.
- **Scene Gate (exp #30, T=0.982):** 0 mutes; similarities 0.49-0.93. Still
  vacuous as an FP lever, still armed.
- **Review Sampling / Deferral / Animal-Proximity Exemption (exp #33):** no
  firing opportunity. Nothing survived to them.

## The finding: the purge rule and the mute rule disagreed

`human_retention_hours` (48h) is the human-photo policy. Exp #11's extension
widened it to review-class bursts within `human_retention_proximity_seconds`
(±240s) of a HUMAN row, because a burst that really contains a person but was
misclassified `no_animal` would otherwise keep recognisable frames for the full
`max_images` rotation — days.

But the *mute* gate that identifies those bursts does not use a 240s window. It
fires on **window OR density OR demoted-band**: 240s normally, 1800s if the
burst's own `person_confidence` ≥ 0.3, or ≥8 HUMAN detections in the trailing
1800s regardless of gap. By construction a burst muted on density can sit
arbitrarily far from any single HUMAN row.

`get_human_adjacent_review_detections` then re-tested every candidate — including
rows already carrying `human_proximity_muted = 1` — against the narrow ±240s
rule. So the system would decide "this is a person, do not notify" and then
decline to apply its own retention policy to the same burst.

5444 is the measured instance: muted by density at **310s**, 70s outside the
purge window, holding a clearly recognisable person.

### Measured corpus-wide, before shipping

Replaying the ±240s test against every `human_proximity_muted = 1` row in the
system's history: **12 rows** ever fell in this gap, ~1.4/month.

| id | date | status | frames on disk |
|----|------|--------|----------------|
| 3828 | 2026-07-29 | no_animal | no |
| 3877 | 2026-07-30 | no_animal | no |
| 4095 | 2026-08-01 | no_animal | no |
| 4207 | 2026-08-02 | no_animal | no |
| 4264 | 2026-08-02 | unclassifiable | no |
| 4674 | 2026-08-29 | no_animal | no |
| 4724 | 2026-08-30 | no_animal | no |
| 5074 | 2026-09-08 | unclassifiable | **yes** |
| 5075 | 2026-09-08 | no_animal | **yes** |
| 5076 | 2026-09-08 | no_animal | **yes** |
| 5154 | 2026-09-12 | no_animal | **yes** |
| 5444 | 2026-09-23 | no_animal | **yes** |

Five still had frames on disk at fix time — 5074/5075/5076 at **15 days** and
5154 at **11 days**, against a 48h policy. All four of those were inspected and
are empty garden, so the realised exposure is 1 in 5, not 5 in 5. Stating it
plainly: the leak rate inside the gap is modest. The reason to fix it anyway is
that the consequence when it does hit is exactly the thing the 48h policy exists
to prevent, and the fix costs nothing — it removes image files from bursts the
system has *already* decided are people.

## What shipped

Commit `0bc9ec9`, restart-gated 2026-09-24T03:25+02:00.

- A row with `human_proximity_muted = 1` (review-class, or the exp #26 IDENTIFIED
  unnamed-animal shape) is purge-eligible **unconditionally** — no time test.
  Purge eligibility follows the gate's verdict instead of re-deriving it.
- The ±`window_seconds` test is untouched for unmuted review-class rows, which
  is the leading-edge case it was built for: a burst BEFORE a visit's first
  HUMAN burst necessarily has `human_proximity_muted` False, so nothing that
  rule covers is affected.
- Strictly a superset — the change can only ever purge more, never less.
- Image files only. DB rows are kept as metadata-only records, exactly as before.
- **No new knob, no notification routing change, no classification change.**
  `window_seconds <= 0` still short-circuits the whole extension, so
  `PERFORMANCE_HUMAN_RETENTION_PROXIMITY_SECONDS=0` remains a complete rollback.
- 5 new tests; full suite **723 passed**.

### Gates

FN-veto does not apply: no notification behaviour changes at all, so no animal
can be hidden by this. The one real cost is that muted bursts lose their frames
at 48h instead of days, which slightly shortens the window for retroactive
adjudication — but only for bursts the gate already judged human-adjacent, and
the loop adjudicates the same night, well inside 48h. `paused` false; a human
label arrived today (5411 at 10:36), so the feedback-starved clock is reset;
exp #33 keeps the active slot.

## Monitoring duty (nightly, from 2026-09-24)

Cheap and specific: on any tick, a `human_proximity_muted = 1` row older than
`human_retention_hours` whose `image_path` still exists on disk means the purge
is not doing what this fix claims. Check it the same way the 12-row replay above
was run.

## Rollback

`PERFORMANCE_HUMAN_RETENTION_PROXIMITY_SECONDS=0` + service restart, or
`git revert 0bc9ec9` + restart.

## Measurement note recorded, not acted on

Tonight's `fp_rate` is **0.094 (3/32)** against 0.286 yesterday, and that
improvement is not real. `loop.metrics` counts `person`-labelled rows in the
`labeled_triggers` denominator but never in `fp_count`, so 29 person rows
deflated the ratio. The honest statement is: **3 non-person triggers, 3 of them
false positives, 0 animals.** A person-heavy night will always print a
flattering FP rate, and `metrics/daily.csv` now carries a dip that a future tick
could misread as an FP win. Filed as backlog #38 rather than changed here — one
change per tick, and this is measurement plumbing, not a live gate.
