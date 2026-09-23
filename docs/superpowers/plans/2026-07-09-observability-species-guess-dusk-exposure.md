# Observability Columns + Species Best-Guess + Dusk Exposure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> Per project owner preference: tasks describe intent/structure, not literal code. Implementers write the code TDD-style (failing test → minimal impl → pass → commit) and run the deterministic suite with `uv run pytest tests/ -v`.

**Goal:** Make the tuning loop's attribution fields real (sharpness/person-confidence DB columns), make logs survive reboots, surface the classifier's raw species guess instead of generic "bird", and bias auto-exposure short at dusk so pond captures clear the sharpness floor.

**Architecture:** Extend the existing nullable-column migration in `database_manager.py` and thread already-computed values (sharpness_info, person confidence, top classifier prediction) through `IdentificationResult.metadata` → `process_detection` → `log_detection`. Add a `RotatingFileHandler` beside the existing console logging. Add one new camera control (`AeExposureMode`) behind a config field with env override.

**Tech Stack:** Python 3.13, UV, pytest (asyncio), SQLite, Picamera2/libcamera, SpeciesNet v5.0.2 (mocked in tests).

## Global Constraints

- All new parameters live in `Config` dataclasses with env-var overrides and validation, following existing patterns in `src/config.py`.
- Never crash the pipeline: identification must always return a valid `IdentificationResult`; camera controls that the sensor rejects must degrade gracefully (log a warning, continue with previous behavior).
- New DB columns must be **nullable** and added via the existing `_migrate_detection_columns` mechanism (`src/database_manager.py:113` — column dict at :20-29 is the single source of truth). Old rows stay NULL; no backfill.
- The live systemd service (`wildlife-camera.service`) runs from this checkout. Work on branch `feat/observability-and-dusk`; do NOT restart the service until the final task.
- Tests must pass without camera hardware or SpeciesNet models (use existing Mock implementations).
- Do not change motion detection, cooldown, MOG2, or the human/blur gate semantics shipped 2026-07-08.

## Evidence base (context for implementers and docs)

- The 2026-07-08 journal entry tells the nightly loop to attribute metric shifts using `sharpness_info.below_sharpness_floor` — but no such DB column exists; the loop cannot query it (Task 1 fixes this).
- Journald history before the 2026-07-08 21:14 reboot is gone despite `/var/log/journal` existing (Storage=auto); first-ever `[HUMAN-GATE]`/`[BLUR]` live log lines were lost. `data/logs/` exists and is empty (Task 2 fixes this).
- All 19 bird identifications on 2026-07-08 rolled up to `aves;;;;;bird`; SpeciesNet's classifier is an `always_crop` model (already crops to the detection bbox), so the rollup is an ensemble-confidence artifact. `_parse_predictions` already computes `top_classifier_prediction` (`src/species_identifier.py:257-261`) and discards it (Task 3 fixes this).
- Dusk pond bursts 2026-07-08 19:33–19:35 scored sharpness 10.0–10.4 against the 11.0 floor — barely blurry, caused by auto-exposure choosing long exposures at dusk (Task 4 addresses this).

---

### Task 1: Persist sharpness + person-confidence in the detections table

**Files:**
- Modify: `src/database_manager.py` (migration column dict at :20-29, `log_detection` :126-165)
- Modify: `src/species_identifier.py` (`_parse_predictions` :136-240 — put max person confidence into `IdentificationResult.metadata`)
- Modify: `src/wildlife_system.py` (`process_detection` :148-192 — pass new kwargs to `log_detection`; note `sharpness_info` currently joins the result dict only *after* `process_detection` at :553-554, so the wiring must move sharpness into `process_detection`'s inputs — follow how `motion_area` flows)
- Test: `tests/test_database_manager.py`, `tests/test_species_identifier.py`, `tests/test_wildlife_system.py`

**Interfaces:**
- Consumes: existing `sharpness_info` dict (`sharpness_score: float`, `below_sharpness_floor: bool` — built at `src/wildlife_system.py:325-335`); person detections already extracted in `_parse_predictions`.
- Produces: nullable columns `sharpness_score REAL`, `below_sharpness_floor BOOLEAN`, `person_confidence REAL`; `log_detection(..., sharpness_score=None, below_sharpness_floor=None, person_confidence=None)`; `IdentificationResult.metadata['person_confidence']: float` (0.0 when no person detection; set on every parsed result, not only HUMAN ones — the loop needs sub-threshold person scores for attribution).

**Intent:**
- Add the three columns to the migration dict; extend `log_detection` and the INSERT.
- In `_parse_predictions`, record the max person-category confidence in `metadata` on all code paths that build an `IdentificationResult` (including HUMAN, NO_ANIMAL, IDENTIFIED).
- In `process_detection`, read sharpness values from its (new) `sharpness_info` parameter and `person_confidence` from the identification result's metadata; pass all three to `log_detection`.
- Steps (TDD): failing tests first — migration adds columns to an existing legacy DB file (create table without them, instantiate manager, assert columns exist and old rows read back NULL); `log_detection` round-trips the three values; `_parse_predictions` sets `metadata['person_confidence']` for a person-at-0.25 prediction (below gate threshold — result stays NO_ANIMAL but metadata carries 0.25); end-to-end `process_detection` writes them. Then minimal implementation, suite green, commit.

### Task 2: Rotating file logs that survive reboots

**Files:**
- Modify: `src/wildlife_system.py` (logging setup in `main`, :860-867)
- Modify: `src/config.py` (StorageConfig or PerformanceConfig — follow where paths like the images dir live; add `log_dir` default `data/logs`, env `STORAGE_LOG_DIR`)
- Test: `tests/test_wildlife_system.py` (or a small new `tests/test_logging_setup.py` if the setup is extracted into a testable function — extraction preferred)

**Interfaces:**
- Produces: a `configure_logging(config)` function that installs BOTH the existing stream handler (journald keeps working) AND a `logging.handlers.RotatingFileHandler` at `<log_dir>/wildlife.log`, `maxBytes=5_000_000`, `backupCount=5`, same format string as today; creates the directory if missing.

**Intent:**
- Extract the current `logging.basicConfig` block into `configure_logging(config)` and call it from `main`. Keep the picamera2/noisy-logger level tweaks (:865-867).
- File handler level INFO (DEBUG stays console-only — the `DIAG-MOTION` firehose at :~5s cadence would churn the rotation).
- Steps (TDD): failing test — call `configure_logging` with a tmp log dir, emit an INFO and a DEBUG record through a module logger, assert the file exists, contains the INFO line, not the DEBUG line, and root logger still has a stream handler. Then implement, suite green, commit.

### Task 3: Surface the classifier's raw top-species guess

**Files:**
- Modify: `src/species_identifier.py` (`_parse_predictions` — `top_classifier_prediction` already computed at :244-265; attach to `metadata`)
- Modify: `src/database_manager.py` (migration dict + `log_detection`: two more nullable columns)
- Modify: `src/wildlife_system.py` (notification caption builder around :454-458; `process_detection` → `log_detection` wiring)
- Test: `tests/test_species_identifier.py`, `tests/test_database_manager.py`, `tests/test_wildlife_system.py`

**Interfaces:**
- Consumes: `IdentificationResult.metadata` conventions from Task 1; `log_detection` kwargs pattern from Task 1.
- Produces: nullable columns `top_species_raw TEXT` (full semicolon taxonomy string from the classifier's top class), `top_species_score REAL`; `metadata['top_classifier_prediction']: {'label': str, 'score': float} | None`; a caption line `Best guess: <common name> (NN%)`.
- Common-name extraction: last non-empty segment of the semicolon taxonomy string (e.g. `...;turdus;merula;eurasian blackbird` → `eurasian blackbird`) — implement as a small helper in `src/utils.py` reusable by caption and tests.

**Intent:**
- Attach the already-computed top classifier prediction to `metadata` and persist label/score via `log_detection`.
- Caption rule: append the best-guess line only when it adds information — the ensemble label is a generic rollup (its species/genus taxonomy segments are empty, e.g. `aves;;;;;bird` or `;;;;;;animal`) AND a top classifier prediction exists AND its label is not itself generic (`blank`/`no cv result`/generic `animal`). Show even low scores (that is the point), rounded to whole percent. Never let a caption-formatting error block the notification (wrap defensively, consistent with the never-crash constraint).
- Steps (TDD): failing tests — parse a synthetic prediction with ensemble rollup `aves;;;;;bird` + classifier top-1 `...;turdus;merula;eurasian blackbird` @ 0.34 → metadata carries it, caption contains `Best guess: eurasian blackbird (34%)`; species-level ensemble result → no best-guess line; DB round-trip of the two columns; helper extracts common names correctly incl. trailing-empty-segment taxonomies. Then implement, suite green, commit.

### Task 4: Dusk exposure — short-bias auto-exposure

**Files:**
- Modify: `src/config.py` (CameraConfig: `ae_exposure_mode: str = "short"`, env `CAMERA_AE_EXPOSURE_MODE`, validation: one of `normal|short|long`)
- Modify: `src/camera_manager.py` (control setup where AE/manual exposure is configured, :157-164)
- Test: `tests/test_config.py`, `tests/test_camera_manager.py`

**Interfaces:**
- Consumes: nothing from other tasks (independent).
- Produces: `config.camera.ae_exposure_mode`; PiCameraManager maps it to libcamera `AeExposureMode` via `controls.AeExposureModeEnum` (`Normal`/`Short`/`Long`).

**Intent:**
- Only applies in auto-exposure mode (the existing manual-exposure branch at :158-161 is untouched; when manual exposure is active, skip the AE mode control).
- Wrap the control assignment so an unsupported enum/sensor raises no error: log a warning and continue (Global Constraints). Log the chosen mode at INFO alongside the existing exposure log line.
- Default is `short` — this IS the fix (bias AE toward shorter exposure + higher gain; negligible effect midday when AE already picks short, helps dusk). Rollback lever for Daniel or the nightly loop: `CAMERA_AE_EXPOSURE_MODE=normal` in `.env` + service restart.
- Steps (TDD): failing tests — config default `short`, env override, validation rejects `bogus`; MockCameraManager/PiCameraManager-with-mocked-Picamera2 asserts the control dict contains the mapped enum in auto mode and omits it in manual mode; exception from control setting does not propagate. Then implement, suite green, commit.

### Task 5: Docs, lab notebook, deploy + restart

**Files:**
- Modify: `CLAUDE.md` (new env vars `STORAGE_LOG_DIR`, `CAMERA_AE_EXPOSURE_MODE`; new DB columns; file logging; best-guess caption; ALSO sync the stale claim "motion_threshold current: 500" → actual `.env` value 800, unchanged since April)
- Modify: `experiments/runs/0005-blur-gate-false-negative.md` (append post-deploy evidence: 2026-07-08 19:33–19:35 pond birds, sharpness 10.0–10.4, all four alerted — the exact failure mode of 07-07 now producing alerts)
- Create: `experiments/runs/0006-dusk-short-exposure.md` (follow existing frontmatter/naming conventions; evidence: the four 10.0–10.4 dusk bursts vs 11.0 floor; lever `CAMERA_AE_EXPOSURE_MODE`; success metric: dusk-hour sharpness distribution shifts above floor without midday regression — now measurable via the Task 1 `sharpness_score` column)
- Modify: `experiments/JOURNAL.md` (dated 2026-07-09 entry — see Intent for required loop-facing notes)
- Test: `uv run pytest tests/ -v` full suite green, then deploy

**Intent:**
- The journal entry must tell the nightly loop, explicitly: (a) `sharpness_score`/`below_sharpness_floor`/`person_confidence`/`top_species_raw`/`top_species_score` columns exist from 2026-07-09 onward (NULL before) — attribution instructions in the 07-08 entry are now actionable; (b) AE short-bias shipped as code default — dusk sharpness scores are expected to RISE; not an anomaly; rollback lever documented; (c) **verify the first 48h human purge**: detection id 1725 (2026-07-08 14:49, `capture_20260708_144907_frame*.jpg`) must have its frame files deleted and DB row intact after 2026-07-10 ~14:49 — check on the first tick after that time; (d) `deployed={}` still means "no env-lever override", not "no behavior change" (reaffirming the 07-08 note).
- Deploy: merge `feat/observability-and-dusk` → `main`, run the full suite once more on main, then `sudo systemctl restart wildlife-camera.service`; verify via `systemctl status` (active) and the new `data/logs/wildlife.log` (INFO lines flowing, AE mode line present).
- Commit docs separately from the merge; conventional style, no `loop(...)`/`tick:` prefixes.
