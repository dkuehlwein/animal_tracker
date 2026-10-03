# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Raspberry Pi 5-based wildlife camera system that automatically detects motion, captures photos, identifies species using Google SpeciesNet AI, and sends notifications to a Telegram channel. The system uses OpenCV for motion detection, Picamera2 for camera control, and SpeciesNet for AI-powered species identification.

### Active initiative: autonomous detection-tuning loop

A self-improving nightly loop reduces false positives **and** false negatives: human feedback over Telegram + a git-backed lab notebook (`experiments/`) + an on-Pi Claude Code session. The loop is **armed and running**; its per-tick prompt is `experiments/loop.md`, its SOP is `experiments/PROTOCOL.md`, live state is `experiments/state.json`. Design: `docs/ADR-004-autonomous-tuning-loop.md`. Experiment history: `experiments/runs/NNNN-<slug>.md` (file number ≠ experiment id — check the front matter), `experiments/JOURNAL.md`, `experiments/LEARNINGS.md`.

## Development Workflow (default — no need to restate per session)

- **Subagent-driven development is the default** for any multi-step implementation: the main session orchestrates (superpowers:subagent-driven-development); implementation, test runs, and code review are delegated to subagents. Don't do heavy implementation work in the main session.
- **Model selection**: Sonnet for coding/implementation subagents; Opus for planning/design/architecture agents. Escalate a coder to Opus only if the task is genuinely hard (cross-cutting design, subtle concurrency, repeated Sonnet failures).
- **Plans describe intent and structure, not literal code** — let the coding subagents write the code and run the deterministic test suite (`uv run pytest tests/ -v`).

## Key Commands

The project uses **UV** with Python 3.13. UV lives at `~/.local/bin/uv` (on PATH in interactive shells; for cron/systemd set `PATH=$HOME/.local/bin:/usr/bin:$PATH`).

```bash
uv sync                                         # install/sync dependencies
uv run python src/wildlife_system.py            # run the system (prod: wildlife-camera.service)
uv run pytest tests/ -v                         # full test suite
uv run pytest tests/test_config.py -v           # single file
uv run python scripts/test_classification.py    # capture + full SpeciesNet pipeline (first run downloads ~214MB)
python3 scripts/camera_preview.py               # MJPEG focus/aim preview on http://<pi-ip>:8000
```

Loop CLIs run from the repo root with `PYTHONPATH=src` (`.env` must be found): see `experiments/loop.md` for the exact chain (`loop.ingest` → `loop.metrics` → `loop.report` → `loop.endtick`, `loop.deploy`, `loop.checkpoint`).

Use VIM, not nano, for console edits.

## Architecture

- **`wildlife_system.py`**: main orchestrator and event loop; owns `process_detection` (species ID, burst human sweep, per-gate flags, DB write) and `_process_and_notify_detection` (executes the `decide()` result, deferred REVIEW sends)
- **`notification_gate.py`**: `decide(ctx) -> Decision` — the single ordered source of notification-gate precedence; `RecentHumanEvents` store of HUMAN timestamps; the human-proximity / blank-confidence evaluators
- **`config.py`**: pydantic-settings dataclasses (`CameraConfig`, `MotionConfig`, `PerformanceConfig`, `StorageConfig`, `SpeciesConfig`, `LocationConfig`); env-var overrides; validators consume `loop/guardrails.BOUNDS`; `Config.create_test_config()` for tests
- **`camera_manager.py`**: dual-stream Picamera2 (high-res capture + low-res motion stream); `MockCameraManager` for tests
- **`motion_detector.py`**: MOG2 background subtraction, central-region weighting, consecutive-detection filter, optional color-variance filter
- **`species_identifier.py`**: SpeciesNet wrapper; assigns `DetectionStatus` (human gate, blank routing, etc.); `MockSpeciesIdentifier` for tests
- **`database_manager.py`**: SQLite (WAL) detection log, observability columns, `detection_feedback` labels
- **`notification_service.py`**: Telegram send + caption formatting
- **`feedback_protocol.py`**: shared inline-keyboard `callback_data` ↔ label mapping
- **`telegram_feedback.py`**: feedback sidecar (`wildlife-feedback.service`); sole owner of `getUpdates`; handles `/pause`, `/rollback`
- **`timelapse_writer.py`**: low-rate frame stream independent of motion, for false-negative audits
- **`resource_manager.py`**: memory monitoring, storage cleanup, human-photo purge
- **`data_models.py`**: `MotionResult`, `DetectionResult`, `IdentificationResult`, `DetectionRecord`, `DetectionStatus` (named `data_models` to avoid clashing with YOLO's `models`)
- **`exceptions.py`**: unified exception hierarchy
- **`utils.py`**: `PerformanceTimer`, `MotionVisualizer`, `SharpnessAnalyzer`, `SunChecker`, `extract_common_name`, taxonomy-label helpers (`is_blank_label`, `is_unnamed_animal_label`)
- **`src/loop/`**: deterministic, token-free loop tools — `nightgate` (pre-gate + heartbeat), `ingest`, `metrics` (FP/FN with Wilson CIs, owns the watermark), `report`, `deploy` (only writer of live config → `experiments/deployed_config.env`), `apply_pending_deploy` (pre-sunrise restart, `wildlife-deploy.timer`), `guardrails` (BOUNDS, FN-veto, freeze), `checkpoint`, `endtick`, `state`, `scene_watch` (camera re-aim watchdog), `replay` (stub)

### Data Flow

1. Low-res frames (640x480) captured continuously for motion analysis (5 FPS; daylight only; motion suppressed for a 300s MOG2 warm-up).
2. Motion: background subtraction → threshold → contours → central-region filter → (optional) color-variance filter → consecutive-detection filter.
3. On motion: a 5-frame high-res burst (2028x1520) is saved (`capture_TIMESTAMP_frame1..5.jpg`); the sharpest frame is selected.
4. SpeciesNet (MegaDetector + classifier + ensemble) assigns a `DetectionStatus`; review-class bursts get a burst human sweep.
5. Detection DB-logged with all observability fields — **always**, whatever the gates decide.
6. Notification gate chain decides: suppress, send to MAIN, or send with 🔍 REVIEW prefix (possibly deferred).
7. Cleanup: oldest bursts deleted as units; human-adjacent photos purged after 48h.

## Detection gates

Full reference — mechanisms, evidence, DB columns, log tags, rollback levers, origin experiments — is **`docs/detection-gates.md`**. Read it before changing or reasoning about any gate.

Statuses: `IDENTIFIED`, `ANIMAL_UNCERTAIN`, `NO_ANIMAL`, `UNCLASSIFIABLE`, `HUMAN`, `ERROR`. **Review-class** = `NO_ANIMAL`/`UNCLASSIFIABLE` (sent with 🔍 REVIEW prefix in the same channel). Routing in `species_identifier`: human gate first; SpeciesNet's `blank` verdict → `NO_ANIMAL`; `no cv result` → `UNCLASSIFIABLE`; the generic `;;;;;;animal` rollup is `IDENTIFIED`.

Notification precedence — implemented once, in `notification_gate.decide()` (first match wins, exactly one suppression log per burst; every gate fails open except HUMAN and FAIL-CLOSED):

1. `[HUMAN-GATE]` Human/Privacy — HUMAN status never notifies (upstream: `[HUMAN-SWEEP]` escalates review-class bursts whose sibling frames hold a person)
   - `[FAIL-CLOSED]` — processing error on a review-class / unnamed-animal burst, or on any burst (incl. a species-ID failure) inside a human window/density: muted instead of sent as an ERROR photo
2. `[HUMAN-PROXIMITY]` Human-Proximity — window OR density OR demoted-band; review-class, ERROR and ANIMAL_UNCERTAIN bursts, plus `;;;;;;animal` IDENTIFIED bursts
3. `[BLUR]` Blur — below sharpness floor AND luma ≥ 70, review-class only
4. `[BLANK-CONF]` Confident-Blank — raw top-1 `blank` ≥ 0.92, review-class only
5. `[REVIEW-SAMPLE]` Review Sampling — deterministic fraction sent
6. `[REVIEW-DEFER]` Deferred send — surviving REVIEW sends held 240s, cancelled if any HUMAN burst lands in that window

MAIN-channel (non-review, non-human) alerts are never delayed. Photos of HUMAN and human-adjacent/human-proximity-muted bursts are purged after 48h; DB rows are kept.

### Standing rulings the loop must obey

- **No human pre-approval step exists.** If a change passes the guardrails, ship it (PROTOCOL.md "Autonomy"). Daniel's levers are post-hoc: `/pause`, `/rollback`, `git revert`.
- **Scene gate, Unnamed-Animal Blank-Raw gate and Animal-Proximity exemption were retired by Daniel 2026-10-03** (docs/detection-gates.md §15). The earlier "scene gate is human-ruled ON" ruling is void; do not re-add them without new evidence. DB columns remain (NULL on new rows).
- **Review sampling is a volume lever, not an FP lever.** Never credit it with an fp_rate change; fewer labels is intended, not a freeze trigger (PROTOCOL.md "Review sampling").
- **Mute thresholds the loop may not cross**: `BLANK_CONFIDENCE_MUTE_THRESHOLD` loop range `[0.87, 1.0]`; `HUMAN_DETECTION_CONFIDENCE` `[0.3, 0.7]`. Setting the blank-confidence threshold to `0` (disable) is human-only. All ranges: `src/loop/guardrails.py::BOUNDS`.
- **Auto-labels are not truth.** Headline fp_rate uses human labels only; `cant_tell` and `person` are excluded from its denominator.
- **No second Telegram channel** — likely-FPs use the same-channel 🔍 REVIEW prefix.
- **Analyze existing data first** (DB rows + frames on disk + labels) before proposing new instrumentation (`experiments/loop.md`).
- Deploys go live only on a camera restart stamped via `pending_restart_at` (tz-aware, ≤ 03:30); verify via `systemctl`, not `state.json`.

## Configuration

All parameters have env-var overrides with prefixes `CAMERA_`, `MOTION_`, `PERFORMANCE_`, `STORAGE_`, `SPECIES_`, `LOCATION_` + field name (see `src/config.py`; gate settings tabulated in `docs/detection-gates.md`). `.env` holds secrets and hand overrides; **`experiments/deployed_config.env` is rendered by `loop.deploy` from `state.json` — never hand-edit it** — and overrides `.env` for `MOTION_*`, `PERFORMANCE_*`, `SPECIES_*`.

`.env` must define `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`. Values worth knowing (code default → live value where different):

- Motion: `MOTION_THRESHOLD` 2000 → **800** (.env); `MOTION_MIN_CONTOUR_AREA` 50; `MOTION_CONSECUTIVE_REQUIRED` 2; color filtering off (`MOTION_ENABLE_COLOR_FILTERING`, `MOTION_MIN_COLOR_VARIANCE` 200).
- Camera: auto-exposure (`CAMERA_EXPOSURE_TIME`/`CAMERA_ANALOGUE_GAIN` unset). `CAMERA_AE_EXPOSURE_MODE` code default `short` → **`normal`** (.env; exp #7 rolled back — AE bias didn't fix dusk sharpness). Degrades gracefully if libcamera's enum is unavailable.
- Timing/storage: `PERFORMANCE_COOLDOWN_PERIOD` 30s; `PERFORMANCE_MAX_IMAGES` 300 bursts; `STORAGE_LOG_DIR` `data/logs` (rotating `wildlife.log`, 5MB×5, INFO+; DEBUG stays console-only; never crashes if unwritable).
- Species: `SPECIES_COUNTRY_CODE` DEU, `SPECIES_ADMIN1_REGION` NW (Bonn geofence), `SPECIES_UNKNOWN_SPECIES_THRESHOLD` 0.5, `SPECIES_HUMAN_DETECTION_CONFIDENCE` 0.3 → **0.5** (deployed, exp #14).
- Debug: `PERFORMANCE_SEND_ANNOTATED_IMAGE` (motion overlay alongside the photo; default false).

## Species Identification

- Google SpeciesNet 5.0.2 package, model `kaggle:google/speciesnet/pyTorch/v4.0.1a/1` (auto-downloaded). Uses the `SpeciesNet` class (not `SpeciesNetEnsemble`), `predict(filepaths=..., country=..., admin1_region=...)`.
- Lazy-loaded on first detection (~6s); ~11s per image after that on the Pi 5 CPU; ~2-3GB RAM during inference.
- MegaDetector box floor `min_detection_confidence` 0.2; classifier `min_classification_confidence` 0.5; `return_top_k` 5.
- Never crashes: always returns a valid `IdentificationResult` (`ERROR` status on failure).
- Dependencies: `ml-dtypes>=0.5.0` (float4_e2m1fn) → `numpy>=2.1.0` → `opencv-python>=4.10.0`.

## Motion Detection Notes

- MOG2 with `detectShadows=True`; foreground threshold 200 drops shadow markers (127), suppressing moving tree shadows.
- The background model is **not reset** after detections — MOG2 adaptation (history 500) is relied on so the learned shadow distribution survives triggers.
- Color-variance filtering (off) captures RGB instead of YUV420 grayscale and drops low-variance motion (leaves, grass).

## Hardware & Environment

- Raspberry Pi 5, 8GB RAM (required for SpeciesNet); Pi Camera Module (tested IMX477); ~2GB storage for models + images.
- Python 3.13 venv at `.venv` **with system site packages** (`include-system-site-packages = true`) so the apt-installed `python3-libcamera` is importable. Recreate: `uv venv --python /usr/bin/python3 --system-site-packages && uv sync`.
- systemd units: `wildlife-camera`, `wildlife-feedback`, `wildlife-loop` (+ timer), `wildlife-deploy` (+ timer).

## Testing

pytest + asyncio; files `tests/test_*.py`. Mocks (`MockCameraManager`, `MockSpeciesIdentifier`) allow running without hardware or SpeciesNet. Gate behaviour is covered mainly in `test_wildlife_system.py` (incl. the precedence golden test), `test_notification_gate.py`, `test_species_identifier.py` and `test_loop_*.py`; run the full suite after any change.
