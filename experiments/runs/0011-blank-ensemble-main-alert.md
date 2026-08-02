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
