---
id: 26
slug: unnamed-animal-main-leak
status: running
validation: live   # code change, restart-gated; FN cost measured over the full 78-row corpus of this label shape
occupies_active_slot: false  # scope repair of the Human-Proximity Gate (exp #10/#11), same precedent as exp #23 vs #9 and exp #24 vs #21
hypothesis: "SpeciesNet's fully-generic rollup `<uuid>;;;;;;animal` means 'something is there, I cannot name it'. It is routed to IDENTIFIED, so it fires a MAIN-channel species alert that bypasses EVERY review-class mute path — human-proximity, blur, scene, sampling, deferral. A person photographed at extreme close range is exactly the input that produces it: MegaDetector boxes a torso/leg filling the frame, the classifier cannot name it, and the burst is delivered as 'animal detected'. Gating this one label shape (and only this one) on the existing Human-Proximity Gate closes the leak at zero measured FN cost."
created: 2026-09-14
promoted_from: "tier-2 adjudication of 2026-09-14: bursts 5222 and 5270, both close-ups of a person during a 55-burst gardening session, both sent to MAIN as generic `animal`."
confidence: high   # 78/78 corpus rows checked against the proposed gate; the 4 it would mute contain zero human-labelled animals
commit: 80d0c00
env_delta: {}
restart_at: 2026-09-15T03:25:00+02:00
---

## The leak

2026-09-14 was the busiest human day in the corpus: 55 HUMAN-status bursts
between 16:03 and 17:01, every one correctly suppressed, including six with
`person_confidence` at or near 0.0 that only the raw-classifier homo path
caught. The privacy gate worked all afternoon.

Two bursts did not go through it at all.

| burst | time | status | ensemble label | conf | pc | raw top-1 | contents |
|---|---|---|---|---|---|---|---|
| 5222 | 16:13:05 | `identified` | `1f689929-…;;;;;;animal` | 0.695 | 0.017 | `blank` 0.59 | fabric/clothing filling the entire frame |
| 5270 | 16:51:21 | `identified` | `1f689929-…;;;;;;animal` | 0.578 | 0.367 | `chinese monal` 0.19 | a person's trousered leg, left 55% of frame |

Both were delivered to the MAIN channel as a species alert. 5222 came **30 s**
after a HUMAN-status burst; 5270 came 82 s after one, with **34** HUMAN bursts
in the preceding 1800 s. Every mute path that exists for this situation —
human-proximity window, human-proximity density, deferred send, the ±240 s
retention purge — is conditioned on `is_review_detection(status)`, and these
rows are `identified`. The gates were not evaded; they were never consulted.

The ensemble `confidence` on both rows equals MegaDetector's own box
confidence (0.6946 / 0.5785 exactly), which is what `;;;;;;animal` is: a
detector box with no classifier verdict attached. It is the *absence* of an
identification being reported as one.

Same shape as exp #13 (`;;;;;;blank` at 0.99 firing a MAIN alert). That fix
re-routed the label. This one cannot: `animal` frequently IS a real animal.

## Why the fix is a scope repair, not a re-route

78 rows corpus-wide (2026-06-09 → today) carry the `;;;;;;animal` shape.
Human labels on them: **16 `animal` + 2 `animal_wrong_id`**, 10
`false_positive`, 16 legacy `wrong_species`. Routing the label to review-class
wholesale would push 18 known real-animal alerts off the MAIN channel — a large
FN cost, immediately vetoed.

So the label is not the signal. The label *plus human proximity* is. Replaying
the existing Human-Proximity Gate (window 240 s OR ≥8 HUMAN bursts in 1800 s)
over all 78 rows:

| burst | date | would mute because | what it is |
|---|---|---|---|
| 1988 | 07-13 19:50 | window | raw top-1 `human` @ 0.591 — unlabelled, almost certainly a person |
| 3483 | 07-27 15:44 | window + density | human-labelled **`false_positive`** |
| 5222 | 09-14 16:13 | window (30 s) | **person, adjudicated tonight** |
| 5270 | 09-14 16:51 | window + density | **person, adjudicated tonight** |

**4 of 78. Zero of them human-labelled `animal`/`animal_wrong_id`.** All 18
animal-labelled rows sit outside both conditions. The FN cost is not assumed to
be small, it is measured to be zero over every instance of this label shape the
corpus has ever produced — ~1.3 mutes/month, all four of which are non-animals
and two of which are confirmed people.

A named identification is untouched by construction: tonight's domestic cat
(5213, `…;felis;catus;domestic cat` @ 0.95) carries taxonomy, so it would still
alert mid-gardening-session. Only "I can't tell you what this is" alerts are
subject to the garden-is-occupied test.

## The change

1. `utils.is_unnamed_animal_label()` — deliberately narrow, modelled on
   `_is_blank_prediction`: last segment `animal` AND no taxonomy segment names
   anything, with `no cv result`/`blank` counted as empty (exp #23's sentinel
   lesson, so the guard can't be silently disabled by a sentinel-filled
   taxonomy). `aves;;;;;bird` does not match.
2. `process_detection` evaluates the Human-Proximity Gate for review-class
   bursts **or** IDENTIFIED unnamed-animal bursts. Same window/density logic,
   same fail-open `except`, same `human_proximity_muted` column (which now
   carries True/False on these IDENTIFIED rows instead of NULL — a widening of
   the column's convention, documented in code and in CLAUDE.md).
3. The notification layer's `is_human_proximity_review` test widens the same
   way, so a muted burst is suppressed entirely with exactly one
   `[HUMAN-PROXIMITY]` log. Blur/scene/sampling still require review-class, so
   no second suppression log is possible.
4. `get_human_adjacent_review_detections` also returns `identified` rows with
   `human_proximity_muted=1`, so a burst the loop has decided is probably a
   person has its frames purged on the 48 h human-retention policy instead of
   sitting on disk for the full 300-burst rotation.

Scene gate, review sampling and the deferred send are untouched and remain
review-class-only.

## Gates

- **FN-veto:** measured above — 0/18 animal-labelled rows affected, 4/78 rows
  muted, all non-animals. Not "unmeasured, therefore hold": measured, therefore
  ship.
- **Volume:** ~1.3 fewer MAIN alerts per month. No collapse, no explosion.
- **Feedback-starved:** no (human labels arrived 09-12). **Paused:** no.
- **Slot:** exp #21 keeps it. This repairs the *scope* of an already-shipped
  mechanism (exp #10/#11's Human-Proximity Gate), exactly as exp #23 repaired
  exp #9's guard and exp #24 repaired exp #21's trigger.
- **Autonomy:** privacy/notification-routing changes are explicitly the loop's
  call under PROTOCOL's Autonomy section. Shipped this tick, not deferred.

## Verification

- **612/612 deterministic tests pass.** New: 10 for the helper (including
  `aves;;;;;bird` and a sentinel-filled taxonomy), 3 for the gate in
  `process_detection`/notification (muted within window → no send + DB flag 1;
  no recent human → sends + flag 0; named species → sends + flag NULL), 2 for
  the widened purge query.
- **End to end against the real DB rows with the shipped helper**, replaying the
  deployed window/density values (240 s, ≥8 in 1800 s):

  | burst | unnamed? | window | density | muted | expected |
  |---|---|---|---|---|---|
  | 5222 (person) | yes | yes | no | **yes** | yes |
  | 5270 (person) | yes | yes | yes | **yes** | yes |
  | 5213 (domestic cat) | no | — | — | no | no |
  | 5164 (bird, human-labelled `animal`) | yes | no | no | no | no |
  | 5123 (bird, human-labelled `animal`) | yes | no | no | no | no |

  Full-corpus sweep with the shipped code mutes `{1988, 3483, 5222, 5270}` and
  zero human-labelled `animal`/`animal_wrong_id` rows — the pre-registered
  number, reproduced by the implementation rather than by the analysis script.

## Prediction

The next `;;;;;;animal` burst within 240 s of a HUMAN burst (or during a
≥8-in-30-min occupied garden) produces a `[HUMAN-PROXIMITY]` log instead of a
MAIN alert, and its frames are gone at 48 h. Falsified if a named-species alert
is ever muted by this path, or if a human labels a muted `;;;;;;animal` burst
`animal`. Rollback: `git revert 80d0c00` + restart (the existing
`PERFORMANCE_HUMAN_PROXIMITY_WINDOW_SECONDS=0` / `PERFORMANCE_HUMAN_DENSITY_COUNT=0`
levers also disable this path, at the cost of disabling the review-class gate
with it).

## Retention, applied by hand this once

The two leaked bursts were written with `human_proximity_muted = NULL` (the gate
did not run on them), so the widened 48 h purge will never match them — and
rewriting that column would falsify the record of what the deployed system
actually decided. Their frames were therefore deleted by hand at the end of this
tick, after the verification above had used them: 12 files across bursts 5222 and
5270, DB rows kept as metadata-only, exactly the treatment `purge_human_bursts`
gives a HUMAN burst (44 h earlier than policy, in the protective direction).
Rows written from the 09-15 restart onward need no manual step.

## Also tonight

- **The cat.** First non-bird animal in weeks: 5213 (13:45, `domestic cat`
  @ 0.95) is a correct MAIN alert. 5214, 31 s later, is the **same cat** sitting
  behind the fence and bamboo — read `unclassifiable`, sent to REVIEW. That is
  the **second** animal-labelled review-class row with frames still on disk
  (after 5176 yesterday), and the bucket whose emptiness has blocked scene-gate
  threshold validation since 2026-07-26 is now n=2. Its recorded
  `scene_similarity` is **0.8204** against T=0.97 — the gate would not have
  muted it, a second FN-side data point landing 0.15 on the safe side (5176
  scored 0.7384).
- **Scene gate:** 0 mutes tonight, 0 in 15 days. Tonight's review-class
  similarities run 0.727–0.891, still well under T=0.97. Nothing to adjudicate,
  nothing changed.
- **Sampling gate:** 4 bursts sampled out (5215, 5217, 5232, 5279). Screened
  all four: three empty pond scenes and one person's legs (5232, already
  human-proximity-muted). Zero animals sampled out → PROTOCOL's trigger for
  raising the rate did not fire; hold at 0.5.
- **Human gate:** 55/55 HUMAN-status bursts adjudicated as real people, six of
  them at `person_confidence` ≤ 0.05 where only the raw homo path could fire.
  Zero phantoms in the busiest human session on record — exp #14's 0.5
  operating point continues to hold.
