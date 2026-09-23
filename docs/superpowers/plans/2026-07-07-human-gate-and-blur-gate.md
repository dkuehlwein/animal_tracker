# Human-Gate Suppression + 48h Retention + Blur-Gate Removal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> Per project owner preference: tasks describe intent/structure, not literal code. Implementers write the code TDD-style (failing test → minimal impl → pass → commit) and run the deterministic suite with `uv run pytest tests/ -v`.

**Goal:** Stop family-member captures from reaching Telegram (suppress, purge photos after 48h) and stop the sharpness gate from silently discarding low-light captures (ID everything, alert whenever an animal is found).

**Architecture:** Add a HUMAN detection status driven by MegaDetector person detections (conf ≥ 0.3) or a `homo` final classification inside `SpeciesIdentifier._parse_predictions`. The main loop suppresses notifications for HUMAN, a retention sweep purges human bursts older than 48h, and `_capture_and_select_best_frame` no longer drops below-floor bursts — instead a `below_sharpness_floor` flag routes blurry no-animal results to DB-only (no Telegram).

**Tech Stack:** Python 3.13, UV, pytest (asyncio), SQLite, SpeciesNet v5.0.2 (mocked in tests).

## Global Constraints

- All new parameters live in `Config` dataclasses with env-var overrides and validation, following existing patterns in `src/config.py`.
- Never crash the pipeline: identification must always return a valid `IdentificationResult` (existing invariant).
- DB stores plain-string statuses (see `DetectionStatus` docstring in `src/data_models.py:19`) — new status must be a plain string `"human"`.
- File deletions must only ever touch files inside the configured images directory, and delete bursts as whole units (all `capture_<ts>_frame*.jpg` siblings), matching `resource_manager.py` conventions.
- Do not change motion detection, cooldown, or MOG2 behavior in this plan.
- Evidence base (for docs task): 2026-07-07 forensics — 4 bursts at the pond 19:03–19:09 all dropped by the sharpness gate while the owner watched a bird bathe; 110 silent drops 18:05–21:28; person-gate measurement on 19 REVIEW frames: person ≥ 0.3 catches 7/8 visible humans, 0 false positives on empty frames.

---

### Task 1: HUMAN status + person/homo gate in SpeciesIdentifier

**Files:**
- Modify: `src/data_models.py` (DetectionStatus block at :19-53)
- Modify: `src/config.py` (SpeciesConfig, ~:196; env overrides + validation + as_dict)
- Modify: `src/species_identifier.py` (`_parse_predictions`, :136-240)
- Test: `tests/test_species_identifier.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `DetectionStatus.HUMAN = "human"`; helper `is_human_detection(status) -> bool` in `data_models.py`; config field `config.species.human_detection_confidence: float = 0.3` (env `SPECIES_HUMAN_DETECTION_CONFIDENCE`).
- `IdentificationResult` with `status == DetectionStatus.HUMAN`, `species_name == "human"`, `confidence` = max person-box confidence (or ensemble confidence for homo-taxon path), `detection_result` still populated with the raw detections.

**Intent:**
- In `_parse_predictions`, after extracting raw `detections`, also collect person detections: `d['category'] in (2, '2', 'person', 'human')`. Compute max person confidence.
- Human gate fires when EITHER (a) max person conf ≥ `human_detection_confidence`, OR (b) the ensemble `prediction` string contains the `homo` genus — parse the semicolon taxonomy (`';homo;' in prediction` or a taxonomy segment equal to `homo`), covering the observed `...;hominidae;homo;;homo species` format.
- The human gate takes precedence over the animal branch (privacy first): evaluate it before the `animals_detected` early-return and before classification parsing. A frame with both a person (0.4) and a confident animal returns HUMAN.
- Steps (TDD): failing tests first — synthetic prediction dicts covering: person 0.8/no animal → HUMAN; person 0.25 → unchanged NO_ANIMAL; homo-taxon ensemble with person conf 0 → HUMAN; person 0.4 + animal 0.9 → HUMAN precedence; existing no-detection and identified paths unchanged (regression). Config test: default 0.3, env override, validation rejects values outside [0,1]. Then minimal implementation, suite green, commit.

### Task 2: Suppress Telegram notifications for HUMAN detections

**Files:**
- Modify: `src/config.py` (PerformanceConfig, near `review_prefix_enabled` :155)
- Modify: `src/wildlife_system.py` (main-loop notification block :699-769)
- Test: `tests/test_wildlife_system.py` (or the existing integration test module covering `send_notification` flow)

**Interfaces:**
- Consumes: `DetectionStatus.HUMAN`, `is_human_detection` from Task 1.
- Produces: `config.performance.suppress_human_alerts: bool = True` (env `PERFORMANCE_SUPPRESS_HUMAN_ALERTS`).

**Intent:**
- After `process_detection` returns, if `species_result['detection_status']` is HUMAN and `suppress_human_alerts` is on: skip annotation building and `send_notification` entirely; log one INFO line tagged `[HUMAN-GATE]` with detection id; still run `cleanup_old_images`. DB logging already happened inside `process_detection` — no change there.
- Steps (TDD): failing test using `MockSpeciesIdentifier`/monkeypatched identifier returning a HUMAN result — assert notification service is never called and the DB row exists; a non-human result still notifies; flag off → human notifies (escape hatch). Then implement, suite green, commit.

### Task 3: 48h retention purge for human bursts

**Files:**
- Modify: `src/config.py` (PerformanceConfig or StorageConfig — follow wherever `max_images` lives)
- Modify: `src/wildlife_system.py` (`cleanup_old_images` wrapper invoked at :769)
- Modify: `src/resource_manager.py` (`cleanup_old_images`, :69 — add a burst-deletion helper if none is reusable)
- Modify: `src/database_manager.py` (add a query: image paths + ids of detections with `detection_status='human'` older than a cutoff)
- Test: `tests/test_resource_manager.py`, `tests/test_database_manager.py`

**Interfaces:**
- Consumes: `"human"` status rows written via Task 1.
- Produces: `human_retention_hours: int = 48` (env `PERFORMANCE_HUMAN_RETENTION_HOURS`); DB method returning human detections older than cutoff; purge routine callable from the existing cleanup path.

**Intent:**
- During the existing cleanup pass, additionally: query human-status detections older than `human_retention_hours`; for each, delete ALL sibling frames of the burst (derive `capture_<ts>_frame*.jpg` pattern from the stored `image_path`), constrained to the images directory; keep the DB row (metadata-only record, per owner decision). Idempotent when files already gone.
- Steps (TDD): failing tests with tmp images dir + test DB — human row aged 49h → all 5 frames gone, DB row intact; human row aged 1h → untouched; old non-human row → untouched by this sweep (FIFO still governs it); missing files → no exception. Then implement, suite green, commit.

### Task 4: Blur-gate removal — ID everything, alert on animals

**Files:**
- Modify: `src/wildlife_system.py` (`_capture_and_select_best_frame` :285-305, main-loop notification block :699-769)
- Test: `tests/test_wildlife_system.py`

**Interfaces:**
- Consumes: HUMAN suppression from Task 2 (a blurry human burst is suppressed by that path, not this one).
- Produces: `sharpness_info` dict gains `below_sharpness_floor: bool`; below-floor bursts now return a real `(path, sharpness_info)` instead of `(None, None)`.

**Intent:**
- `_capture_and_select_best_frame`: when `best_score < min_sharpness_threshold`, do NOT return `(None, None)`. Return the best frame path with `below_sharpness_floor=True` in `sharpness_info` (keep the existing INFO log, rephrased — it is no longer "skipping").
- Main loop: after `process_detection`, when `below_sharpness_floor` is true AND the status is review-class (`is_review_detection` — NO_ANIMAL/UNCLASSIFIABLE), skip Telegram (DB row already written); log one INFO line tagged `[BLUR]`. When an animal was found (identified/animal_uncertain), notify as usual — the existing sharpness_info in the caption already communicates blurriness; ensure the caption path tolerates the new flag.
- Net effect: the sharpness floor no longer creates untracked events (every burst gets a DB row) and no longer costs animal alerts; it only mutes Telegram for blurry nothing-found bursts, so REVIEW volume does not increase.
- Steps (TDD): failing tests — below-floor burst + animal result → notification sent; below-floor + no_animal → no notification, DB row written; above-floor behavior unchanged; below-floor + HUMAN → suppressed via Task 2 path (single suppression log, no double-send). Then implement, suite green, commit.

### Task 5: Docs, lab notebook, changelog

**Files:**
- Modify: `CLAUDE.md` (config parameter lists: new env vars `SPECIES_HUMAN_DETECTION_CONFIDENCE`, `PERFORMANCE_SUPPRESS_HUMAN_ALERTS`, `PERFORMANCE_HUMAN_RETENTION_HOURS`; describe HUMAN status, suppression, 48h retention, and the new blur-gate semantics)
- Modify: `experiments/runs/0004-human-main-channel-leak.md` (status → implemented; record owner decision of 2026-07-07: suppress alerts entirely, purge photos after 48h; note the fix landed as a MegaDetector person-gate at 0.3 + homo-taxon check, superseding the REVIEW-tag proposal; leak-watch continues as post-deploy verification)
- Create: next-numbered run doc in `experiments/runs/` for the blur-gate false-negative fix (follow the existing file naming/frontmatter conventions in that directory; include the 2026-07-07 evidence from Global Constraints and the decision "ID everything, alert on animals, DB-log always")
- Modify: the notebook journal (`experiments/JOURNAL.md` or equivalent — follow existing conventions) with a dated entry linking both changes
- Test: `uv run pytest tests/ -v` full suite green

**Intent:** Documentation must let the nightly autonomous loop understand that config/behavior changed on 2026-07-07 (its volume/rollback baselines assume "stock config"). State explicitly: notification volume will drop (humans suppressed, blurry no-animal muted) and DB rows/day will RISE (~2x: formerly-dropped blurry bursts now logged) — the loop must not read either shift as an anomaly. Commit with a message following the repo's `loop(...)`/`tick:` -free conventional style for feature work.
