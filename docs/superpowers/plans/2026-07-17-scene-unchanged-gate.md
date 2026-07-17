# Scene-Unchanged Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-07-17-scene-unchanged-gate-design.md` (read it first — it is the contract).

**Goal:** Mute review-class detections (NO_ANIMAL/UNCLASSIFIABLE) whose best frame is near-identical to recent known-empty reference frames — DB-logged, no Telegram — automating Daniel's "compare to earlier images" false-positive check.

**Architecture:** A new `src/scene_gate.py` module (frame comparator + rolling reference set) is invoked from `WildlifeSystem.process_detection`, which records `scene_similarity`/`scene_gate_muted` via `DatabaseManager.log_detection`; `_process_and_notify_detection` adds a third suppression branch after the human gate and blur mute. An offline validation script picks the threshold against labeled frames before the gate goes live.

**Tech Stack:** Python 3.13, OpenCV (cv2, already a dependency), pydantic-settings config, SQLite, pytest.

**Per project convention (CLAUDE.md):** plans describe intent and structure — implementers write the actual code and run `uv run pytest tests/ -v` (from `/home/daniel/animal_tracker`).

## Global Constraints

- Fail-open invariant: every error/missing-data path must result in "notify as today", never "mute". A mute requires an affirmatively computed similarity ≥ threshold.
- Gate applies ONLY to statuses where `data_models.is_review_detection(status)` is true; HUMAN and IDENTIFIED handling unchanged; human gate > blur mute > scene gate precedence.
- HUMAN-status frames must never enter the reference set (they get purged; privacy).
- New DB columns are nullable and added via the existing idempotent `_DETECTION_EXTRA_COLUMNS` migration only.
- Env naming: `PERFORMANCE_SCENE_GATE_*` via `PerformanceConfig` fields (pydantic auto-parsing, snake_case fields).
- Similarity is a float in [0, 1], 1.0 = identical; comparator returns `None` on any failure (unreadable file, size mismatch after resize, cv2 error).
- No new runtime dependencies.

---

### Task 1: `src/scene_gate.py` — comparator + reference set

**Files:**
- Create: `src/scene_gate.py`
- Test: `tests/test_scene_gate.py` (new)

**Interfaces (Produces):**
- `compute_scene_similarity(frame_path: str | Path, reference_path: str | Path) -> Optional[float]`
  - Loads both grayscale via cv2, resizes to 64×64, normalizes each to zero-mean/unit-ish contrast (subtract mean, divide by std with epsilon) so global exposure/AE shifts don't dominate, similarity = `1.0 - clamped mean-abs-diff` mapped to [0,1]. Returns `None` if either image can't be read or any exception occurs (log at WARNING, never raise).
- `class SceneReferenceSet(max_refs: int, max_age_hours: float)`
  - `.add(image_path, timestamp: datetime)` — append; evict oldest beyond `max_refs` and anything older than `max_age_hours` relative to the newest add.
  - `.best_similarity(frame_path, now: datetime) -> Optional[float]` — prunes stale refs vs `now`, returns max `compute_scene_similarity(frame_path, ref)` over remaining refs, skipping `None` comparisons; `None` if no refs or all comparisons failed.
  - `.seed(rows: list[tuple[str, datetime]])` — bulk-load from DB rows (oldest→newest), same eviction rules; skip paths that don't exist on disk.

**Steps (TDD, red→green per behavior):**
- [ ] Tests first in `tests/test_scene_gate.py` using synthetic images written to `tmp_path` via cv2/numpy (no fixtures from hardware): identical frames → similarity > 0.98; +40 uniform brightness shift → still > 0.95 (normalization works); a 20×20 dark blob pasted on one copy (a "bird") → measurably lower than the identical case; unreadable/missing path → `None`; reference-set eviction by count and age; `seed` skips missing files; `best_similarity` with empty set → `None`.
- [ ] Run `uv run pytest tests/test_scene_gate.py -v` → failures for missing module.
- [ ] Implement `src/scene_gate.py` minimally; re-run → all pass.
- [ ] Run full suite `uv run pytest tests/ -v` → no regressions.
- [ ] Commit: `feat(scene-gate): frame comparator + rolling empty-scene reference set`

### Task 2: DB columns + seed query

**Files:**
- Modify: `src/database_manager.py` (`_DETECTION_EXTRA_COLUMNS` ~line 21-44; `log_detection` ~line 138-192; new query method near `get_human_detections_older_than` ~line 360)
- Test: `tests/test_database_manager.py`

**Interfaces:**
- Consumes: existing migration mechanism `_migrate_detection_columns`.
- Produces:
  - `_DETECTION_EXTRA_COLUMNS` gains `"scene_similarity": "REAL"`, `"scene_gate_muted": "BOOLEAN"`.
  - `log_detection(..., scene_similarity=None, scene_gate_muted=None)` — two new trailing keyword args persisted like the sharpness fields.
  - `get_recent_review_detections(limit: int, max_age_hours: float) -> list[tuple[str, datetime]]` — `(image_path, timestamp)` for `detection_status IN ('no_animal','unclassifiable')` within the age window, ordered timestamp DESC, modeled on `get_human_detections_older_than`.

**Steps:**
- [ ] Extend existing test templates: migration test asserts the two new columns appear (the `_DETECTION_EXTRA_COLUMNS`-driven test at ~line 56 may already cover this generically — verify, and add explicit persistence tests modeled on `test_log_detection_persists_observability_fields` (~114) and the `_default_null` test (~137)); new tests for `get_recent_review_detections` (status filtering — human/identified rows excluded; age window; ordering; limit).
- [ ] Red → implement → `uv run pytest tests/test_database_manager.py -v` green → full suite green.
- [ ] Commit: `feat(scene-gate): scene_similarity/scene_gate_muted columns + review-detection seed query`

### Task 3: Config + guardrails

**Files:**
- Modify: `src/config.py` (`PerformanceConfig`, ~lines 143-174; bounds-validator pattern per lines 232-249)
- Modify: `src/loop/guardrails.py` (`BOUNDS`, ~line 19)
- Test: `tests/test_config.py`

**Interfaces (Produces):**
- `PerformanceConfig` fields (env auto-derived with `PERFORMANCE_` prefix):
  - `scene_gate_enabled: bool = True`
  - `scene_gate_similarity_threshold: float = 0.97` (conservative placeholder — Task 5's offline validation sets the real default; 0.97 mutes almost nothing until then)
  - `scene_gate_ref_count: int = 3`
  - `scene_gate_ref_max_age_hours: float = 6.0`
- `BOUNDS["PERFORMANCE_SCENE_GATE_SIMILARITY_THRESHOLD"] = (0.80, 1.0)` + a `field_validator` on the threshold enforcing it (same pattern as existing bounds validators in config.py).

**Steps:**
- [ ] Tests: defaults present; env override via `monkeypatch.setenv` + `_env_file=None`; out-of-bounds threshold raises (both below 0.80 and above 1.0); pattern-match `TestPerformanceConfig` / `test_human_detection_confidence_env_override`.
- [ ] Red → implement → `uv run pytest tests/test_config.py -v` green → full suite green.
- [ ] Commit: `feat(scene-gate): config knobs + guardrail bounds`

### Task 4: Wire the gate into WildlifeSystem

**Files:**
- Modify: `src/wildlife_system.py` — startup seeding (where `DatabaseManager` is ready, e.g. `__init__`/initialize path); observability compute block (~lines 195-222); `log_detection` call (~224-246); suppression chain in `_process_and_notify_detection` (~655-673)
- Test: `tests/test_wildlife_system.py`

**Interfaces:**
- Consumes: `SceneReferenceSet`, `compute_scene_similarity` (Task 1); `log_detection(..., scene_similarity, scene_gate_muted)` + `get_recent_review_detections` (Task 2); config fields (Task 3); `is_review_detection`/`is_human_detection` from `data_models`.
- Produces (behavior contract):
  1. **Startup:** if `scene_gate_enabled`, build `SceneReferenceSet(ref_count, ref_max_age_hours)` and `.seed(db.get_recent_review_detections(ref_count, ref_max_age_hours))`.
  2. **In `process_detection`** (alongside the other observability fields): compute `scene_similarity = reference_set.best_similarity(image_path, now)` ONLY when gate enabled AND `is_review_detection(status)`; `scene_gate_muted = scene_similarity is not None and scene_similarity >= threshold`. Else both `None`/`None` (store `scene_gate_muted=None` when not evaluated — mirrors nullable semantics). Pass both to `log_detection`. **After** the decision, if `is_review_detection(status)`, `.add(image_path, now)` (muted bursts included; HUMAN/IDENTIFIED/ERROR never added). Surface the decision to the caller the same way sharpness flows today (e.g. in the returned result dict).
  3. **In `_process_and_notify_detection`:** new branch after `is_blurry_review`: `elif is_scene_unchanged_review:` → log one line with a `[SCENE-GATE]` marker (include similarity + threshold in the message), do NOT send. Precedence: `is_human` first, `is_blurry_review` second, scene gate third.
- Fail-open assertions: gate disabled, no references, comparator `None`, non-review status, ERROR status → notification behavior identical to today.

**Steps:**
- [ ] Tests modeled exactly on the blur-gate quartet (~lines 545-652): (a) review-class + similarity ≥ threshold → `telegram.send_*` not called, `[SCENE-GATE]` in caplog, DB row has `scene_gate_muted=1` and the similarity; (b) review-class + similarity below threshold → notification sent (with 🔍 REVIEW prefix as today); (c) identified animal frame near-identical to reference → still notifies (gate never touches IDENTIFIED); (d) human + would-match-scene → suppressed via HUMAN-GATE with a single log line (precedence); (e) blurry review-class + would-match-scene → `[BLUR]` branch wins (precedence); (f) no references seeded → notifies (fail-open); (g) gate disabled via env → notifies; (h) reference-set update: a review-class detection becomes a reference for the next call; HUMAN detection does not. Use the existing `system` fixture pattern (monkeypatch env, isolated `DatabaseManager`, mocked `identify_species`, `_mock_telegram`).
- [ ] Red → implement → `uv run pytest tests/test_wildlife_system.py -v` green → full suite green.
- [ ] Commit: `feat(scene-gate): mute review-class detections matching recent empty-scene references`

### Task 5: Offline validation script + threshold selection

**Files:**
- Create: `scripts/validate_scene_gate.py`
- Modify (after running it): `src/config.py` default `scene_gate_similarity_threshold` (and, only if no safe threshold exists, `scene_gate_enabled: bool = False`)

**Interfaces:**
- Consumes: `compute_scene_similarity` (import from `scene_gate` with `PYTHONPATH=src`), sqlite DB at the configured `database_path`, `detection_feedback` table, frames under `data/images/`.
- Produces: a report on stdout — for every review-class detection whose frame is still on disk, replay chronologically: maintain the same reference-set rules (K=3, 6h) from prior review-class frames, compute best similarity, then bucket by ground truth (human labels first: `animal`/`animal_wrong_id` = animal-present; `false_positive` = FP; tier-2/auto labels reported separately per the auto-labels-are-not-truth rule). Output: similarity distributions per bucket, and the **highest threshold T such that zero animal-labeled rows score ≥ T**, with the FP-mute yield at T. Also report yield at a safety margin (T − 0.02).
- **Acceptance rule (from spec):** zero labeled animals muted. If no T ≤ 1.0 with nonzero yield satisfies it, the gate ships `enabled=False` by default.

**Steps:**
- [ ] Write the script (argparse: `--db`, `--images-root`, `--ref-count`, `--ref-max-age-hours`; read-only, never writes the DB).
- [ ] Run: `PYTHONPATH=src uv run python scripts/validate_scene_gate.py` from repo root. Record the full output in the Task 5 report.
- [ ] Set the config default threshold from the result (prefer the margin value T − 0.02 if its yield is close to T's); update the Task 3 config test expectation for the new default.
- [ ] Full suite green. Commit: `feat(scene-gate): offline validation script; set threshold from labeled-corpus replay`

### Task 6: Docs + nightly-loop handoff

**Files:**
- Modify: `CLAUDE.md` (new "Scene-Unchanged Gate" bullet under Key Configuration Parameters, style-matched to the Blur Gate bullet: what it does, env knobs, the two DB columns, fail-open + precedence, rollback lever `PERFORMANCE_SCENE_GATE_ENABLED=false` + restart)
- Modify: `experiments/PROTOCOL.md` (extend the nightly-cycle "Label"/adjudication guidance: the loop OWNS the scene-gate mute path — adjudicate every `scene_gate_muted=1` burst nightly for concealed animals exactly like the blur-mute path; a concealed animal is an FN-veto event → lower threshold within BOUNDS or set enabled=false via env delta; loop may re-run `scripts/validate_scene_gate.py` as labels grow)
- Modify: `experiments/JOURNAL.md` (entry: gate shipped by Daniel+Claude interactive session 2026-07-17, commit SHAs, chosen threshold + validation numbers, monitoring handed to the loop)

**Steps:**
- [ ] Write all three doc updates.
- [ ] Commit: `docs(scene-gate): CLAUDE.md + loop protocol handoff (loop owns scene-gate mute-path adjudication)`

### Task 7: Deploy

- [ ] Full suite: `uv run pytest tests/ -v` → all green.
- [ ] Identify the camera service unit (check `systemctl list-units | grep -i wild`) and restart it with `sudo -n systemctl restart <unit>` (the `apply_pending_deploy` precedent). If `sudo -n` is denied interactively, report to Daniel to restart instead of retrying.
- [ ] Verify from the journal/log (`data/logs/wildlife.log`) that the system starts clean and the reference set seeds without error.

## Self-Review Notes

- Spec coverage: hook point + precedence (T4), reference set (T1/T4), metric + offline validation + FN-veto rule (T1/T5), config/BOUNDS (T3), observability columns (T2/T4), loop integration (T6), risks are design-level (fail-open asserted in T4 tests). No gaps found.
- Types consistent: similarity `Optional[float]` [0,1] everywhere; refs are `(image_path, timestamp)` tuples in T1 `.seed` and T2 query.
