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

## Monitoring log (post-restart verification)

### 2026-07-22 (night 1 live) — CLEAN, new path not exercised
Restart confirmed: `wildlife-camera.service` came up 03:30 today running HEAD `0a39b23`
(contains fix commit c366087), so exp #9 is live from the 07-22 loop-day. Ingest window
ids 2751–2856 (106 triggers, daytime-heavy garden activity; night-window dusk 18–19h).

- **(a) No real animal suppressed by the new trigger** — zero rows this window carry a
  `homo` raw top-1, so `_is_raw_homo_taxon` never fired. No FN-veto event. The two MAIN
  animal IDs (2753 common blackbird raw 0.48; 2754 generic `aves;;;;;bird` raw 0.33)
  correctly routed to MAIN, untouched by the new gate (neither is a homo raw top-1).
- **(b) One HUMAN suppression this window (id 2855, 19:20)** — caught by the *existing*
  person-box path (`person_confidence=0.668 ≥ 0.30`), not the new raw-classifier path.
  Frame confirms a person in foreground. So the new path had no positive to demonstrate
  tonight — expected, since homo-raw-top-1 is a ~3-row-per-corpus rarity. Leak-watch
  continues: the value shows up only on a future sub-0.30 person-box + homo-raw burst.

FN-veto: **CLEAN**. All 9 below-floor no_animal dusk bursts had luma < 70 → un-muted to
REVIEW under exp #8's `blur_mute_min_luma=70` (none blur-muted), so zero blur-mute FN
risk. Adjudicated on-disk dusk frames (2851 luma 60.5, 2856 luma 28.3) + the human 2855:
empty garden/pond scenes, no concealed animals. Scene gate still disabled.

### 2026-07-23 (night 2 live) — CLEAN, new path still not exercised
Ingest window ids 2857–3114 (258 triggers, high-volume daytime gardening day: 125 HUMAN,
126 no_animal, 2 unclassifiable, 5 identified). **No human labels tonight** (n_human=0), so
fp_human/FN are unmeasured; the reported fp_rate 0.962 is entirely MegaDetector auto-label.

- **(a) No real animal suppressed by the new trigger** — zero rows this window carry a
  `homo` raw top-1, so `_is_raw_homo_taxon` never fired (second night the new path had no
  positive — homo-raw-top-1 remains the predicted ~3-row-per-corpus rarity). No FN-veto
  event. All 5 MAIN animal IDs are birds (2930/2931/2964/2965 generic `aves;;;;;bird`
  raw 0.34–0.48; 3111 `corvus species` raw 0.72 @20:02 dusk) — raw top-1 is bird/corvid,
  correctly untouched by the gate; `_is_specific_animal_taxon` / no-override confirmed.
- **(b) 125 HUMAN suppressions** — all via the *existing* person-box / homo-taxon paths,
  none via the new raw-classifier path (0 homo raw top-1 rows). Bulk is daytime yard work
  (08:00–18:39, high person_confidence); late rows 3102 (18:39, pconf 0.72) and 3110
  (20:00, pconf 0.78) are real people. No specific-animal raw top-1 on any HUMAN row.

FN-veto: **CLEAN**. 13 below-floor dusk (h≥18) no_animal bursts, all luma-dark → un-muted
to REVIEW under exp #8; scene_gate NULL throughout (disabled). Adjudicated darkest on-disk
dusk frames (3105 19:17, 3109 19:58, 3114 20:27): all the identical static pond/garden
scene at progressively lower light — empty, no concealed animals. Scene gate stays disabled
(still 0 review-class animal-labeled frame on disk). Volume 258 environmental (gardening +
summer garden), no collapse/explosion → no rollback. **Hold; leak-watch continues.**

### 2026-07-24 (night 3 live) — CLEAN, feedback-rich, new path still not exercised
Ingest window ids 3115–3155 (41 triggers; volume right at baseline 42, down from the
106/221/258 gardening-driven prior nights). **32 human labels tonight** (n_human=32) — the
richest feedback night in the run: fp_human 28/32 = 0.875, FN still unmeasured but now
directly checkable against human truth.

- **(a) No real animal suppressed by the new trigger** — 0 rows this window carry a `homo`
  raw top-1, so `_is_raw_homo_taxon` never fired (third night with no positive; homo-raw-top-1
  stays the predicted ~3-row-per-corpus rarity). No FN-veto event. The 4 human `animal`/
  `animal_wrong_id` labels are all genuine animals correctly handled: 3115 (07:26 bird
  `aves;;;;;bird` raw 0.744 → MAIN) and 3117 (07:49 generic-ensemble, bird raw 0.433 → MAIN)
  and 3155 (17:00 `common blackbird` raw 0.425 → MAIN) — raw top-1 is bird, `_is_specific_animal_taxon`
  / no-override kept them off the gate. 3116 (07:26, below-floor no_animal, animal_wrong_id)
  is the faint companion burst of the 3115 bird event → surfaced to REVIEW and human-corrected
  (the bird itself reached MAIN via 3115); not a silent miss.
- **(b) 9 HUMAN suppressions** — all via the *existing* person-box / ensemble-homo paths
  (daytime yard work 14:36–15:42, pconf 0.03–0.936; 3146 pconf 0.03 fired via ensemble-homo,
  not the new raw path). 0 leaked to MAIN. No specific-animal raw top-1 on any HUMAN row.

FN-veto: **CLEAN**. 0 muted bursts this window — every non-HUMAN trigger was surfaced and
human-labeled (nothing blur-muted or scene-muted to adjudicate); the 2 below-floor no_animal
rows (3116, 3154) both reached REVIEW and were labeled. scene_gate NULL throughout (disabled).

**Scene-gate PROTOCOL trigger fired (first time) — re-validated, stays disabled.** 3116 is
the FIRST and ONLY on-disk review-class (`no_animal`) row carrying a human `animal_wrong_id`
label in the whole corpus (the other 17 such rows all predate retention, on_disk=False —
matches the run-0007 finding). Per the scene-gate enablement procedure I re-ran
`scripts/validate_scene_gate.py`: full-corpus `human_animal` tally is now 18 (was 17, +3116),
but the **scored** animal bucket is still n=0 — 3116 is the first review-class row of the
morning and has no review-class reference frame within the 6h window, so it is unscoreable
(the gate would fail open / never evaluate it anyway). No safe threshold derivable → script
recommends `scene_gate_enabled=False`, unchanged. Precise reason upgraded from "no on-disk
animal frame" to "the one on-disk animal frame is unscoreable (first-of-morning, no 6h ref)".

**FP/FN.** loop.metrics: total 41, labeled 32, fp_rate 0.875 (human truth, fp_human 28/32).
Volume 41 ≈ baseline 42, no collapse/explosion → no rollback. **Hold; leak-watch continues.**
