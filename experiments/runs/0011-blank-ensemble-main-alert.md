---
id: 13
slug: blank-ensemble-main-alert
status: running
validation: live          # shipped 2026-08-02 (commit 55234f1), live at the 2026-08-03T03:25 restart
occupies_active_slot: true   # exp #11 (runs/0010) concluded KEEP this tick; this takes the slot
hypothesis: "SpeciesNet's explicit empty-frame verdict — the fully-generic label '<uuid>;;;;;;blank' — scores ~0.99, so it clears unknown_species_threshold and is reported as DetectionStatus.IDENTIFIED. That sends a MAIN-channel species alert on a frame the model itself called empty, and (worse) bypasses the entire review-class mute stack. Routing a blank ensemble prediction to NO_ANIMAL removes a pure-FP MAIN alert class at zero measured false-negative cost."
created: 2026-08-02
promoted_from: "Loop tick 2026-08-02 — id 4175 surfaced while adjudicating the night's non-review-class rows: detection_status='identified', species_name='f1856211-...;;;;;;blank', confidence 0.9985, five frames of empty garden."
confidence: high          # 11/11 labelled instances are false_positive, 0 animal labels corpus-wide
---

## Origin — an alert whose species name is "blank"

Tonight's adjudication swept the review-class rows as usual, then looked at the
one non-review row in the window. Id 4175 (10:45:54) carries
`detection_status='identified'`, `species_name='f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank'`,
`confidence_score=0.9985`. Its five frames are an empty garden bed — no animal,
no person, nothing moving but light.

`blank` is not a species. It is SpeciesNet's own label for "this crop contains
nothing", and the ensemble emits it *confidently* — which is precisely what makes
it dangerous here. `species_identifier._parse_prediction` handles the sibling
sentinel `no cv result` explicitly (→ UNCLASSIFIABLE) but has no branch for
`blank`, so a 0.99-scored blank sails past `unknown_species_threshold` (0.5) into
the success case and is returned as `DetectionStatus.IDENTIFIED`.

## Evidence — the whole corpus, not just tonight

54 bursts carry a `;blank` ensemble label, 2026-06-08 → 2026-08-02. Of the 11
that carry a feedback label:

| label | source | n |
|-------|--------|---|
| false_positive | human | 8 |
| false_positive | tier2 | 3 |
| animal / animal_wrong_id / person | — | **0** |

Zero animal labels of any kind, corpus-wide, over eight weeks. Every one of the
54 also shows `detection_count=1` with `max_detection_confidence` 0.21–0.31 —
i.e. MegaDetector produced one weak box that the classifier then declared empty.
That is the signature of a vegetation/lighting trigger, not a hiding animal.

## The second harm — a privacy hole, not just noise

Every mute path in this system keys on review-class status: human-proximity
(window + density), the deferred send, blur, scene, sampling. An `identified`
burst is evaluated by **none** of them. So a person captured while the ensemble
happens to say `blank` is sent straight to the MAIN channel, in full, with no
proximity check and no deferral — the exact harm exp #11 spent five nights
closing on the REVIEW side. Tonight's 4175 sits 35 min from the nearest human
burst so it is not an instance, but the path is open and 4175 shows how the
class arises. This is the stronger reason to fix it.

## Fix — mirror the sentinel branch that already exists

Commit **`55234f1`**. `_is_blank_prediction()` recognises a label whose last
semicolon segment is `blank` **and** whose taxonomy segments are all empty; such
a prediction returns `NO_ANIMAL` with `animals_detected=False`, `species_name
='Unknown species'`, and the observability metadata preserved so `top_species_raw`
still records the blank label for the loop. Placed directly before the
confidence-threshold check, immediately after the `no cv result` branch it
mirrors. Deliberately narrow: a populated taxonomy that happens to end in
`blank` (`abc;mammalia;...;vulpes;blank`) is still an identification — covered by
a test. 532 tests pass (3 new).

No env knob reaches this — the routing is a hard-coded status mapping — so this
is a code change under the protocol's code-lever clause. Rollback: `git revert
55234f1` + restart.

## Gate check

- **FN-veto — cleared by measurement.** 0 animal labels across 54 instances and
  11 labels; today's instance adjudicated frame-by-frame as an empty scene. The
  residual exposure is not a suppression either: these bursts become review-class,
  so they keep their DB row, keep species ID, and ~50% still reach Telegram with
  the 🔍 REVIEW prefix at the current sample rate. The change moves a class from
  "always MAIN" to "half REVIEW", not to silence.
- **Volume.** 54/4276 = 1.3% of all triggers; ~1 MAIN alert removed on a busy
  day, ≤0.5 REVIEW messages added. No collapse/explosion risk.
- **One experiment at a time.** Exp #11 concluded KEEP this tick (runs/0010
  night 5); this takes the slot.
- **Feedback-starved freeze — not tripped, but 2 of 3 days gone.** Last human
  label 2026-07-31 18:23; none on 08-01 or 08-02. If 08-03 also produces none,
  the loop freezes on the next tick and holds `best_known_good`. Noted here so
  the next tick does not have to rediscover it.

## Prediction (to check on the next tick)

Once live at the 08-03T03:25 restart: no new row should carry
`detection_status='identified'` together with a `;blank` species name. Any
`;blank` burst should instead appear as `no_animal` with the blank label
preserved in `top_species_raw`, flowing through the normal mute stack. Exit
criteria: 3 nights with ≥1 blank-labelled burst correctly routed to review-class
and no animal-labelled blank burst.

---

## Night 1 (2026-08-03) — live but unexercised; the day itself was the story

Code live from the 03:30 restart (`55234f1`, AE=normal, all deployed env
unchanged). **The gate never fired: zero `;blank` bursts, zero `identified`
rows, 4 triggers all day.** Nothing confirms or refutes the fix yet — exit
criteria unchanged, still needs 3 nights with ≥1 blank burst.

**Volume: 4 triggers vs a 192/night baseline — a nominal collapse, dismissed on
positive evidence, not on the assumption that exp #13 is harmless** (it is
post-trigger routing and cannot affect trigger count, but that argument alone
would be exactly the kind of self-serving reasoning the guardrail exists to
catch). Three independent checks:

1. **The detector is alive.** 4277/4279/4280 fired at motion_area 923–1729
   against the 800 threshold; the 06:02 sunrise start, the 1500-frame warmup,
   and the 21:13 sunset stop all logged normally.
2. **The camera is alive.** The timelapse FN-audit stream (`data/timelapse`,
   one grayscale frame/20 s) wrote 179 frames/hour all day with a textbook
   daylight luma curve — 8.3 (06h) → 51 (08h) → 96 (10–12h) → 22 (18h) → 1.0
   (21h) — and a nonzero frame-to-frame diff throughout. A frozen sensor or a
   stuck AE would flatten both; neither is flat.
3. **The scene was genuinely still.** Mean inter-frame diff on 08-03 runs
   1.4–2.2 per hour vs 3.0–6.9 on 08-01/08-02, and peak diff 6–19 vs 28–71.
   08-01 (252 triggers) and 08-02 (114, of which 82 HUMAN) were gardening days;
   08-03 was a Monday with an overcast morning (08h luma 51 vs 99.5 same hour
   on 08-02). Fewer people and less sun means fewer shadow-driven triggers.

**FN audit over the timelapse stream — clean.** Ranked all 1 070 of today's
timelapse frames by inter-frame difference and adjudicated the top 20. Each was
decomposed into a global luma shift vs. residual localized blobs after removing
that shift. Every candidate resolves to illumination, not an object: 08:38:24
(the largest, diff 19.5) is a +19.4 luma AE step plus a sun/shade boundary
brightening the left wall; 10:45:34 (19.2) is a −16.0 shift with 720 residual
pixels, the largest blob a 23×18 shadow edge crossing the grass; 12:53, 11:40,
08:53 leave ≤41 residual pixels and no blob ≥40 px at all. **No animal was
missed today** — the quiet day is real, not a blind detector.

**Adjudication of the day's 4 rows.** All 3 review-class rows were sampled out
(`review_sampled_out=1`), so **zero REVIEW messages and zero MAIN messages were
sent all day**. No proximity mutes, no scene-gate mutes, nothing to adjudicate
under the standing duty. Frames inspected anyway (CLAHE-brightened, the 08:33
burst being near-dark): all four are the empty garden.

**One finding worth recording: 4278 is a false HUMAN.** `detection_status='human'`
at `person_confidence=0.330`, just over the 0.3 gate — but its frames contain no
person. Its neighbour 4277 (27 s earlier) scored 0.277 and 4279/4280 scored
0.252/0.169, all on empty scenes; MegaDetector's person head is simply noisy on
dark, low-contrast frames (08:33 luma ~51, sharpness 3.6). The failure direction
is benign — an empty frame was suppressed instead of sent — but it is not free:
a false HUMAN seeds `_last_human_detection_at`, arming the 240 s proximity mute
and the 240 s deferral cancel for a person who was never there. Today nothing
followed within 240 s (next trigger 09:45), so the cost was zero. **Not acted on
tonight** (freeze, and one instance is not a pattern); noted for the next tick to
check whether false-HUMAN-at-low-luma recurs, since raising
`SPECIES_HUMAN_DETECTION_CONFIDENCE` trades directly against the privacy gate
this loop spent five nights hardening and must not be touched on one data point.
