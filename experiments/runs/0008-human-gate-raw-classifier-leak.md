---
id: 9
slug: human-gate-raw-classifier-leak
status: running          # proposed | running | concluded | rolled_back | parked
validation: live         # code change, restart-gated; monitoring live post-restart
hypothesis: "The human/privacy gate misses a class of person captures: SpeciesNet's ensemble can roll a confident homo-sapiens RAW classifier top-1 up into a generic label (';;;;;;animal', 'unclassifiable', blank) carrying NO 'homo' segment, while the MegaDetector person box sits below human_detection_confidence (0.30). Both existing gate paths then miss it and the person's photo escapes suppression — reaching MAIN when the generic ensemble label still notifies (id 1988, 07-13) or REVIEW when it is review-class (id 2548, 07-21). Fix: also fire DetectionStatus.HUMAN when the RAW classifier top-1 contains a 'homo' taxonomy segment AND the ensemble did not confidently ID a specific animal (so a real animal ID is never overridden)."
created: 2026-07-21
promoted_from: "backlog id 9 (filed 2026-07-13 at runs/0007 CROSS-CUTTING FINDING); activated 2026-07-21 after exp #8 concluded (keep). Was 'HELD for Daniel's OK' in the backlog text — that hold is stale vs the 2026-07-17 Autonomy reaffirmation (PROTOCOL.md 'Autonomy: the gates below are the ONLY approval mechanism'), which explicitly names privacy-gate changes as in-scope and calls holding #9 for approval a protocol violation. Activated under the gates, not a greenlight."
confidence: high         # specificity measured across the WHOLE detections DB
---

## Why this is now the active experiment

Exp #8 (`runs/0007`) concluded 2026-07-21 (keep, live) after two clean dusk-heavy
nights — the single-experiment slot is free. Backlog #9 is the natural next: it is
the only open item addressing an actual observed leak of Daniel's strongest product
lever (the privacy gate), and tonight (07-21) produced a **third** confirming data
point (id 2548).

## The measured problem

The human gate in `src/species_identifier.py` fires `DetectionStatus.HUMAN` when
EITHER `max_person_conf >= human_detection_confidence` (0.30) OR a `homo` segment
appears in the **ensemble** prediction (`is_homo_taxon`). It misses the case where:
1. the MegaDetector person box is sub-threshold (<0.30), AND
2. the ensemble rolled the classifier's homo-sapiens top-1 up into a generic label
   with no `homo` segment (`;;;;;;animal`, `unclassifiable`, blank).

Observed instances (raw classifier top-1 = `...hominidae;homo;sapiens;human`):
- **1988** (07-13 19:50): ensemble generic `;;;;;;animal` conf 0.72 → **notified to
  MAIN** as a "detection" (the leak that filed this backlog).
- **1852**: earlier homo-raw human.
- **2548** (07-21): ensemble `unclassifiable`, raw score 0.573, person_conf 0.058 →
  reached 🔍 REVIEW (review-class, not MAIN), Daniel-labeled `person`.

## Specificity / FN-safety (measured DB-wide)

A `homo` RAW classifier top-1 (`top_species_raw`/`metadata['top_classifier_prediction']`)
occurs on **exactly 3 rows across the entire detections DB — 1852, 1988, 2548 — all
confirmed humans, ZERO real animals.** So the new trigger catches every observed leak
with negligible false-suppression (FN) risk. The "ensemble did not confidently ID a
specific animal" guard is belt-and-suspenders on top of that: a confident specific
animal ID is never overridden to HUMAN even if a homo raw segment were present.

## The change (code, restart-gated)

Add a THIRD human-gate trigger: fire HUMAN when the raw classifier top-1 label contains
a `homo` segment AND the ensemble prediction is generic/blank/review-class (not a
confident specific animal). Never-crash: malformed/legacy `classifications` degrade to
"no raw-homo trigger". No env knob reaches this → code change, committed separately with
the exp id, restart-gated (`pending_restart_at` stamped ≤ 03:30 per the deploy-timer
convention). Rollback = `git revert` + restart.

**Gates cleared:** FN-veto — homo-raw = 0 real animals DB-wide, guard protects specific
IDs, so FN risk negligible/unmeasured-but-bounded. Volume — moves at most a handful of
rows from REVIEW/MAIN to fully-suppressed; no collapse/explosion. One-experiment-at-a-time
— exp #8 concluded first. Not paused. Feedback not starved (human labels present tonight).

## Implementation & validation — 2026-07-21

Implemented via Sonnet TDD subagent; diff reviewed in the main session and the full
suite independently re-run before commit.

**Commit:** `c366087` (2026-07-21), restart-gated. Files: `src/species_identifier.py`
(+`tests/test_species_identifier.py`), 226 insertions / 39 deletions.

**What changed (`_parse_predictions`):** the raw-classifier top-1 extraction
(`classifications` → `top_predictions`/`top_classifier_prediction`) was moved *ahead of*
the human/privacy gate (verbatim, not rewritten; the later site is now a comment) so the
gate can inspect the raw top-1. A third trigger `raw_homo_leak` was added to
`human_gate_fired`: `is_raw_homo_taxon AND not ensemble_is_specific_animal`, where
`is_raw_homo_taxon` = an exact `homo` segment in the raw top-1 label and
`ensemble_is_specific_animal` = new static `_is_specific_animal_taxon(ensemble_prediction)`
— True only when both the genus (`parts[-3]`) and species (`parts[-2]`) segments of the
semicolon taxonomy are non-empty (mirrors `WildlifeSystem._best_guess_line`). So
`;;;;;;animal`, `aves;;;;;bird`, `unclassifiable`, and blank all count as generic → the
trigger may fire; `...;vulpes;vulpes;Red Fox` counts as specific → the trigger is
unconditionally disabled, guaranteeing a confident specific animal ID is never
overridden. The post-gate confidence/logging was split into three explicit branches so
the raw-homo path reports the raw classifier's own score and logs a distinct message.
Never-crash: None/malformed/legacy-list `classifications` degrade to no raw-homo trigger.

**Validation:** 6 new tests (generic-rollup→HUMAN, unclassifiable→HUMAN, specific-animal
NOT overridden, non-homo raw + generic ensemble unchanged, malformed-list no-crash,
None-classifications no-crash). Two key tests confirmed failing for the right reason
pre-implementation. **Full suite: `426 passed` (re-run independently in the main session
before commit).** Replay (Layer A) STUB → skipped; Layer B = bounds+predicted-live-effect
+ FN-veto, all cleared (see gates above).

**Restart:** `pending_restart_at` stamped 2026-07-22 03:29 (≤ 03:30 deploy-timer fire,
per the convention fix) so `apply_pending_deploy` reloads the new code at the next
restart. Goes live post-restart; then the leak-watch duty continues as verification —
any future `homo` raw top-1 row should now be suppressed as HUMAN (metadata-only DB row,
no Telegram), and any real animal wrongly suppressed would be an FN-veto event (respond
by disabling / narrowing the trigger). None expected on evidence.

**Post-restart monitoring (nightly):** confirm (a) no real animal is suppressed by the new
trigger (FN-veto — check any newly-HUMAN row that carries a specific animal raw top-1),
and (b) the previously-leaking pattern (homo raw + generic ensemble + sub-0.30 person box)
now routes to HUMAN, not MAIN/REVIEW.
