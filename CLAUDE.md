# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Raspberry Pi 5-based wildlife camera system that automatically detects motion, captures photos, identifies species using Google SpeciesNet AI, and sends notifications to a Telegram channel. The system uses OpenCV for motion detection, Picamera2 for camera control, and SpeciesNet for AI-powered species identification.

### Active initiative: autonomous detection-tuning loop

An in-progress effort to reduce false positives **and** false negatives via a self-improving daily loop (human feedback over Telegram + a git-backed lab notebook + on-Pi Claude Code `/loop`). The design is converged; **Phase 1 is the next thing to build**. See **`docs/ADR-004-autonomous-tuning-loop.md`** — start with its "Status & how to resume" section.

## Development Workflow (default — no need to restate per session)

- **Subagent-driven development is the default** for any multi-step implementation: the main session orchestrates (superpowers:subagent-driven-development); implementation, test runs, and code review are delegated to subagents. Don't do heavy implementation work in the main session.
- **Model selection**: Sonnet for coding/implementation subagents; Opus for planning/design/architecture agents. Escalate a coder to Opus only if the task is genuinely hard (cross-cutting design, subtle concurrency, repeated Sonnet failures).
- **Plans describe intent and structure, not literal code** — let the coding subagents write the code and run the deterministic test suite (`uv run pytest tests/ -v`).

## Key Commands

### Camera preview for focus adjustment
```bash
python3 scripts/camera_preview.py
```
Starts MJPEG web stream on port 8000 for adjusting camera focus and positioning. Access at `http://<pi-ip>:8000`.

### Running the system
```bash
uv run python src/wildlife_system.py
```

### Package Management

This project uses **UV** for fast, reliable Python package management with Python 3.13.

**Note**: UV binary is located at `~/.local/bin/uv` and should already be in your PATH from `.bashrc`. If running scripts in non-interactive shells (e.g., cron jobs, systemd services), you may need to explicitly set PATH:
```bash
export PATH="$HOME/.local/bin:/usr/bin:$PATH"
```

**Installing/syncing dependencies**:
```bash
uv sync
```

**Running Python scripts with UV**:
```bash
uv run python scripts/test_classification.py
uv run python src/wildlife_system.py
```

**Running tests**:
```bash
uv run pytest tests/ -v
```

**Testing species classification**:
```bash
uv run python scripts/test_classification.py
```
This captures a photo and runs the full SpeciesNet pipeline (MegaDetector + species classifier). First run downloads ~214MB model files from Kaggle.

### Running specific test files
```bash
uv run pytest tests/test_config.py -v
uv run pytest tests/test_camera_manager.py -v
uv run pytest tests/test_motion_detector.py -v
```

### Text editor preference
Use VIM instead of nano for console edits.

## Architecture

The system follows a modular architecture with these main components:

- **`wildlife_system.py`**: Main orchestrator that coordinates all components and manages the event loop
- **`config.py`**: Centralized configuration management using environment variables (.env file)
- **`camera_manager.py`**: Handles dual-stream camera operations (high-res capture + low-res motion detection)
- **`motion_detector.py`**: OpenCV-based motion detection with central region weighting and consecutive detection filtering
- **`species_identifier.py`**: SpeciesNet AI integration for wildlife species identification
- **`database_manager.py`**: SQLite database for detection logging
- **`notification_service.py`**: Telegram notification service with message formatting
- **`resource_manager.py`**: Memory management, storage cleanup, and system monitoring
- **`data_models.py`**: Consolidated data models (MotionResult, DetectionResult, IdentificationResult, DetectionRecord) - named `data_models` to avoid conflict with YOLO's internal `models` package
- **`exceptions.py`**: Unified exception hierarchy for all components
- **`utils.py`**: Utilities (PerformanceTimer, MotionVisualizer, SharpnessAnalyzer, SunChecker, `extract_common_name`)
- **`scripts/camera_preview.py`**: MJPEG streaming server for live camera preview (focus adjustment tool)

### Data Flow

1. **Motion Detection Loop**: Low-resolution frames (640x480) captured continuously for motion analysis
   - Uses RGB format when color filtering is enabled, YUV420 grayscale otherwise
2. **Motion Processing**: Background subtraction → thresholding → contour analysis → central region filtering → color variance filtering (if enabled)
3. **Photo Capture**: High-resolution frames (1920x1080) captured only when motion is detected
4. **Burst Frame Saving**: All captured frames saved for debugging (e.g., `capture_TIMESTAMP_frame1.jpg` through `_frame5.jpg`)
5. **Species Identification**: SpeciesNet AI analyzes best/sharpest frame (~5-10 seconds processing time)
6. **Database Logging**: Detection stored with species name, confidence score, and metadata
7. **Telegram Notification**: Async notification with species information
8. **Cleanup**: Automatic old image cleanup to manage storage (deletes entire bursts as units)

### Key Configuration Parameters

All configuration is centralized in `Config` class with nested dataclasses:

- **Motion Detection**:
  - `motion_threshold` (default: 2000, current: 800 px) - minimum motion area to trigger detection
  - `min_contour_area` (50) - minimum size of individual contours
  - `consecutive_detections_required` (2) - reduces momentary false positives
  - **Color Filtering** (disabled by default): Reduces false positives from uniform vegetation
    - `enable_color_filtering` (false) - when enabled, captures RGB instead of grayscale for color analysis
    - `min_color_variance` (200.0) - motion from low-variance objects (leaves/grass) is filtered out
- **Camera**: Dual resolution streams with frame rate limiting (`frame_duration`: 100000 microseconds)
  - **Exposure Control**: `exposure_time` (2000μs = 1/500s) and `analogue_gain` (2.5x) for motion freeze
  - Set either to `None` to enable auto-exposure mode (currently: auto-exposure enabled)
  - **Auto-Exposure Bias**: `ae_exposure_mode` (default `"short"`, one of `normal|short|long`) — only applies when auto-exposure is active (i.e. `exposure_time`/`analogue_gain` are `None`); mapped to libcamera's `AeExposureModeEnum` in `PiCameraManager._apply_ae_exposure_mode`. Biases AE toward shorter exposures at dusk/low light so bursts don't fall below `min_sharpness_threshold` (see dusk-exposure fix, `experiments/runs/0006-dusk-short-exposure.md`). Degrades gracefully (logs a warning, no `AeExposureMode` control set) if libcamera is unavailable or the enum lookup fails. Rollback lever: `CAMERA_AE_EXPOSURE_MODE=normal` + service restart.
- **Timing**: `cooldown_period` (30s), `frame_interval` (0.2s for 5 FPS)
- **Storage**: `max_images` (300 bursts, raised from 100 on 2026-07-27 — on a busy day the old default deleted burst frames within hours, so the nightly tuning loop could no longer visually adjudicate its own muted/sampled-out bursts; disk headroom is ample, ~282MB per 100 bursts vs 8.6GB free) with automatic cleanup of oldest bursts
  - **File Logging**: `log_dir` (default `data/logs`, env `STORAGE_LOG_DIR`) - `configure_logging(config)` (called from `wildlife_system.py`'s `__main__`) installs a `RotatingFileHandler` at `<log_dir>/wildlife.log` (5MB × 5 backups) alongside the existing console/journald stream handler, so INFO+ history survives a Pi reboot (journald's does not). DEBUG stays console-only to avoid the DIAG-MOTION firehose churning rotation. If the log directory can't be created/written, a warning is logged via the stream handler and setup continues without the file handler (never crashes). `StorageConfig.logs_dir` (used by `resource_manager.ensure_directories`) is now a property aliasing `log_dir`, so a `STORAGE_LOG_DIR` override moves both the file handler's target and the directory `ensure_directories` creates.
- **Debug**: `send_annotated_image` (false) - when enabled, sends motion detection overlay image alongside the original photo in Telegram
- **Species Identification**: `model_version` (v4.0.1a), `country_code` (DEU), `admin1_region` (NW), `unknown_species_threshold` (0.5)
- **Human/Privacy Gate**: `human_detection_confidence` (0.3) - MegaDetector person-category confidence that, alone or together with a `homo` taxonomy segment in the SpeciesNet ensemble prediction, classifies a burst as `DetectionStatus.HUMAN`; evaluated *before* the animal branch, so a frame with both a person and a confident animal still routes to HUMAN. `suppress_human_alerts` (true) - HUMAN-status detections are still species-ID'd and DB-logged, but no Telegram notification is sent (not REVIEW-tagged, suppressed entirely). `human_retention_hours` (48) - saved burst photos for HUMAN-status detections are purged this many hours after capture; the DB row is kept as a metadata-only record. `human_retention_proximity_seconds` (default 240.0, exp #11 extension 2026-08-01) extends that purge **symmetrically in time** to review-class (NO_ANIMAL/UNCLASSIFIABLE) bursts that sit within this many seconds of a HUMAN-status detection *in either direction*: a burst that really contains a person but was misclassified `no_animal` (the leak class the Human-Proximity Gate below exists for) would otherwise keep recognizable frames on disk for the full `max_images` rotation (days), defeating the 48h human-photo policy. The purge runs ≥48h after capture, so unlike the notification gates it may look forward as well as backward. Implemented as `DatabaseManager.get_human_adjacent_review_detections(cutoff, window_seconds)` (adjacency matched in Python over two indexed queries — a correlated SQL `EXISTS`/`strftime` subquery measured 3.9s per call on a 4k-row DB and `purge_human_bursts()` runs after every detection) plus a second sweep in `StorageManager.purge_human_bursts()`. DB rows are kept (metadata-only) exactly as for HUMAN rows; only image files are deleted. Rollback lever: `PERFORMANCE_HUMAN_RETENTION_PROXIMITY_SECONDS=0` + service restart.
- **Blur Gate**: `min_sharpness_threshold` (11.0) - a burst whose best frame scores below this is no longer silently discarded. It still gets a real image path + `sharpness_info`, flows through species ID, and is always DB-logged. The notification layer then decides: an animal found in a below-floor burst still alerts (with `below_sharpness_floor` noted); a below-floor burst with a review-class status (NO_ANIMAL/UNCLASSIFIABLE, no animal found) is DB-logged but *not* sent to Telegram, so REVIEW-channel volume doesn't rise. The human-privacy gate takes precedence over the blur mute (a blurry human burst is suppressed as HUMAN, not as blur).
- **Scene-Unchanged Gate**: `scene_gate_enabled` (code default in `config.py` is still **`False`**) - a second, independent mute path for review-class (NO_ANIMAL/UNCLASSIFIABLE) bursts: `scene_gate.py::compute_scene_similarity` compares the burst's best frame (grayscale, downsized, mean/std-normalized to resist exposure/AE shifts) against a rolling `SceneReferenceSet` of the last `scene_gate_ref_count` (3) review-class best frames within `scene_gate_ref_max_age_hours` (6h); a similarity >= `scene_gate_similarity_threshold` (0.97) sets `scene_gate_muted=True` - the burst is still species-ID'd and DB-logged (`scene_similarity` REAL, `scene_gate_muted` BOOLEAN, both NULL when the gate is disabled or the status isn't review-class; `scene_gate_muted` is False (not NULL) when evaluated against an empty/failed reference set) but no Telegram notification is sent, same pattern as the Blur Gate. Fails open (never mutes) if disabled, if the reference set is empty, or on any frame-read/comparison error - `compute_scene_similarity` never raises. **Precedence**: Human/Privacy Gate > Human-Proximity Gate > Blur Gate > Scene Gate (> Review Sampling Gate, see below) - a below-floor or HUMAN or human-proximity-muted burst gets exactly one suppression log (the earliest-precedence gate wins); `scene_similarity`/`scene_gate_muted` are still computed and DB-logged for every review-class burst regardless of blur/proximity status. The reference set is seeded from the DB at `WildlifeSystem` startup (`DatabaseManager.get_recent_review_detections`) and updated on every subsequent review-class detection; HUMAN-status frames are never added as references. **Enabled by human override since 2026-07-26**: the gate shipped disabled (Task 5, 2026-07-17) because `scripts/validate_scene_gate.py`'s offline replay found zero human `animal`/`animal_wrong_id`-labeled review-class rows with a frame still on disk - all 17 such rows corpus-wide predate the ~100-burst image retention window - so per the spec's FN-veto acceptance rule no threshold could be validated from evidence. On 2026-07-26 Daniel explicitly overrode that HOLD as an accepted-risk decision to cut REVIEW volume (paired with the Review Sampling Gate below), deploying `PERFORMANCE_SCENE_GATE_ENABLED=true` at the placeholder threshold 0.97 despite the still-empty animal-labeled bucket - see `experiments/PROTOCOL.md` → "Scene-gate ownership" (SUPERSEDED 2026-07-26 block) and `experiments/runs/0009-review-volume-reduction.md`. The loop must not re-derive the threshold or disable the gate on the grounds that the animal bucket is empty - that condition is known and already ruled on. Rollback lever: `PERFORMANCE_SCENE_GATE_ENABLED=false` + service restart.
- **Review Sampling Gate**: `review_sample_rate` (default 0.25) - a fifth, last-precedence mute path for review-class (NO_ANIMAL/UNCLASSIFIABLE) bursts that survive the Human/Proximity/Blur/Scene gates above: only this fraction of them are actually sent to Telegram, to cut REVIEW-channel volume (deployed alongside the Scene-Unchanged Gate override, 2026-07-26 - see `experiments/runs/0009-review-volume-reduction.md`). The decision is deterministic per burst, not random per run: `wildlife_system._review_sample_fraction(detection_id)` hashes `detection_id` with sha256 into a stable `[0, 1)` fraction, and `is_review_sampled_out(detection_id, review_sample_rate)` mutes when that fraction `>= review_sample_rate` (so a given detection_id always lands on the same side of the gate). Fails open: a `None` detection_id (e.g. the DB write itself failed) always sends; `review_sample_rate >= 1.0` never samples out (the rollback lever); `<= 0.0` always samples out. Computed in `WildlifeSystem.process_detection` for every review-class row immediately after the DB INSERT (via a follow-up `DatabaseManager.update_review_sampled_out` UPDATE, since `detection_id` doesn't exist until the INSERT returns) and persisted as the `review_sampled_out` BOOLEAN column - `True`/`False` for every review-class row, `NULL` for non-review-class rows and rows logged before this gate existed. The burst is still species-ID'd and DB-logged either way; only the Telegram send is skipped, with a `[REVIEW-SAMPLE]` log line. **Precedence**: Human/Privacy Gate > Human-Proximity Gate > Blur Gate > Scene Gate > Review Sampling Gate - it is evaluated and logged last, so a burst any earlier gate would already suppress produces exactly one suppression log, never a redundant second one. `loop/metrics.py`'s `n_sampled_out` count (surfaced in the nightly report as "• Not sent (review sampling): N", same treatment as `n_cant_tell`) excludes these rows from the "Not yet labelled" backlog line, since they were never sent for a human to label - it does not affect `fp_rate` or the per-tier FP buckets, which remain label-conditioned as before. Rollback lever: `PERFORMANCE_REVIEW_SAMPLE_RATE=1.0` + service restart.
- **Human-Proximity Mute Gate**: `human_proximity_window_seconds` (default 120.0) - a second, independent mute path layered on top of the Human/Privacy Gate for review-class (NO_ANIMAL/UNCLASSIFIABLE) bursts: MegaDetector scores extreme close-up / motion-blurred partial human bodies at only ~0.02-0.15 person confidence, too low to trip the Human/Privacy Gate itself, so such bursts slip through as `no_animal` and leak a recognizable person to REVIEW. Three such leaks occurred on 2026-07-27 (detection ids 3544, 3553, 3554), each 76-108s after a correctly-classified `human`-status burst. `WildlifeSystem` tracks `self._last_human_detection_at` in memory (seeded at startup from `DatabaseManager.get_last_human_detection_time()`, updated every time a detection is classified `DetectionStatus.HUMAN`); a review-class burst is muted when it lands within `human_proximity_window_seconds` after that timestamp (`0 <= (burst_time - last_human).total_seconds() <= window`). Validated against all 12 human-labelled `animal`/`animal_wrong_id` review-class rows since the human gate went live (2026-07-08): the closest is 329s from a preceding human-status burst, so the 120s default window costs zero known false negatives. The decision is persisted as the `human_proximity_muted` BOOLEAN column - `True`/`False` for every review-class row, `NULL` for non-review-class rows (same convention as `scene_gate_muted`) - written on the same DB write as the other observability fields (no follow-up UPDATE needed, unlike `review_sampled_out`, since it doesn't depend on this row's own id). The burst is still species-ID'd and DB-logged either way; only the Telegram send is skipped, with a `[HUMAN-PROXIMITY]` log line. Fails open: any error computing proximity, a `0.0` window, or no prior human detection all result in no mute, never a blocked notification. **Density condition (exp #11 mechanism extension, 2026-07-28)**: a second, OR-ed condition on the SAME gate — tonight's adjudication found recognizable-person review-class bursts that fell OUTSIDE the 120s window (gaps of 432s and 732s past the last human burst) during a long gardening session, so a single "time since last human" window can't catch a garden that stays occupied longer than the window. `human_density_window_seconds` (default 1800.0) and `human_density_count` (default 8) add a "garden is occupied" test: `WildlifeSystem` also tracks `self._recent_human_detection_times` (a list of HUMAN-status timestamps, seeded at startup from `DatabaseManager.get_recent_human_detection_times(since)`, appended to on every HUMAN classification, pruned to `human_density_window_seconds` before every use via `_count_recent_human_detections` so it can't grow unbounded); a review-class burst is also muted when at least `human_density_count` HUMAN-status detections occurred in `[burst_time - human_density_window_seconds, burst_time]`. Same `human_proximity_muted` DB column (no new column), same `[HUMAN-PROXIMITY]` log line (now noting `window` vs `density` in the message text), same precedence. `human_density_count=0` disables the density condition (rollback lever) while leaving the window condition untouched; fails open identically (any exception, or `window=0` AND `count=0`, or no prior human detections → no mute). **Precedence**: Human/Privacy Gate > Human-Proximity Gate (window OR density) > Blur Gate > Scene Gate > Review Sampling Gate - evaluated right after the Human/Privacy Gate so a HUMAN-status burst produces exactly one suppression log ([HUMAN-GATE], not [HUMAN-PROXIMITY]), and a burst this gate mutes never also produces a [BLUR]/[SCENE-GATE]/[REVIEW-SAMPLE] log. Rollback levers: `PERFORMANCE_HUMAN_PROXIMITY_WINDOW_SECONDS=0` (disables the window condition) and/or `PERFORMANCE_HUMAN_DENSITY_COUNT=0` (disables the density condition) + service restart.
- **Deferred REVIEW Send Gate (cancel-on-human)**: `review_defer_seconds` (default 240.0, exp #11 extension 2026-08-01) - the Human-Proximity Gate above is *backward-looking by construction*, so it can never mute the **leading edge** of a human visit: the first burst of a visit is often a close-up motion smear classified `no_animal`, with the visit's first HUMAN-status burst arriving seconds *later*. Three such bursts reached REVIEW (3829 on 07-29 at 75s, 3867 on 07-30 at 51s, 3909 on 07-31 at 81s before the first HUMAN burst); 3909's saved frames contain a clearly recognizable face. Fix: a review-class burst that survives every gate above does **not** send immediately — `WildlifeSystem._schedule_deferred_review_send` builds the annotated image synchronously (it depends on `self.last_motion_frame`, which mutates on the next loop iteration) and then hands a background asyncio task everything the send needs; `_deferred_review_send` sleeps `review_defer_seconds` and cancels the Telegram send if a HUMAN-status detection landed in the meantime (`burst_time < self._last_human_detection_at <= burst_time + review_defer_seconds`), logging `[REVIEW-DEFER]` and persisting the mute via `DatabaseManager.update_human_proximity_muted` (same column as a backward mute — a follow-up UPDATE, since the decision isn't known at INSERT time). The main detection loop never awaits the deferral; pending tasks are tracked in `self._pending_review_tasks` and cancelled in `run()`'s shutdown path. Only review-class sends are deferred — MAIN/animal notifications are never delayed. Fails open: any exception sends the notification (`asyncio.CancelledError` is a `BaseException` and deliberately propagates, so a shutdown cancel never triggers a spurious send). FN cost measured, not assumed: of the 12 human-labelled `animal`/`animal_wrong_id` review-class rows since the human gate went live (2026-07-08), the closest sits 1846s before the next HUMAN burst - 7.7x the 240s window - so the gate costs zero known false negatives; it cancels ~18% of otherwise-sent review-class bursts. **Precedence**: evaluated last, after the Review Sampling Gate, as the final step before an actual review-class send. Rollback lever: `PERFORMANCE_REVIEW_DEFER_SECONDS=0` + service restart (reviews then send immediately, as before).
- **Observability columns (ADR-004, nightly tuning loop)**: the `detections` table has five nullable columns, populated on every write from 2026-07-09 onward (NULL on rows logged before that date — there is no backfill). `sharpness_score` (REAL) and `below_sharpness_floor` (BOOLEAN) mirror the Blur Gate's `sharpness_info` dict. `person_confidence` (REAL) is the max MegaDetector person-category confidence from `species_identifier`'s parsed metadata — recorded on every *parsed* identification result (HUMAN, NO_ANIMAL, UNCLASSIFIABLE, ANIMAL_UNCERTAIN, IDENTIFIED, and parse-path ERROR; error responses for unreadable images carry no metadata, so those rows store NULL), not only ones that tripped the Human/Privacy Gate, so sub-threshold person scores are visible too (0.0 when no person box was present). `top_species_raw` (TEXT) and `top_species_score` (REAL) are the species classifier's raw top-1 prediction (label/score), captured independently of the (possibly rolled-up) ensemble `species_name` — see `utils.extract_common_name` and the "Best guess" caption line below. All five round-trip through `DatabaseManager.log_detection` and `WildlifeSystem.process_detection`, which reads `sharpness_info` (now an explicit `process_detection` parameter) and `IdentificationResult.metadata` to populate them.
- **"Best guess" caption line**: when the ensemble's final species label is a generic rollup (blank/empty genus or species taxonomy segments, e.g. `aves;;;;;bird` or `;;;;;;animal`) but the classifier's raw top-1 prediction (`metadata['top_classifier_prediction']`) names something more specific and non-generic, the Telegram caption appends a line `Best guess: <common name> (NN%)` (built by `WildlifeSystem._best_guess_line`, using `utils.extract_common_name` to pull the last non-empty semicolon-delimited segment). Shown even at low confidence — that's the point. Wrapped defensively so a formatting error never blocks the notification; a non-dict/malformed `top_classifier_prediction` (legacy classifier shape) degrades to no caption line and NULL DB columns rather than crashing.
- **Feedback labels (`src/feedback_protocol.py`, 2026-07-09 redesign)**: each notification carries a 5-button, 2-row inline keyboard — row 1 "✅ Animal" / "🐦 Animal, wrong ID" / "👤 Human", row 2 "❌ Nothing there" / "🤷 Can't tell" — mapped via `CODE_TO_LABEL` to `detection_feedback.label`: `a`→`animal`, `wid`→`animal_wrong_id`, `p`→`person`, `fp`→`false_positive`, `ct`→`cant_tell`. Button label is `person`, not `human`, to avoid colliding with `human` as a labeller tier (`detection_feedback.source='human'` vs tier1/tier2 auto-labels). `ws`→`wrong_species` is legacy: `parse_callback_data` still accepts it and remains valid in `VALID_FEEDBACK_LABELS` (all 6 labels), but is never shown on new keyboards — predates the `animal_wrong_id`/`person` split (heterogeneous history). `cant_tell` wins reconciliation in `loop/ingest.py`, blocking tier-1/tier-2 auto-label backfill, but `loop/metrics.py` excludes it from the fp_rate denominator and every per-tier bucket; the nightly `loop/report.py` summary lists it as "• Can't tell (unusable image): N", excluded from "Not yet labelled". False negatives have no dedicated button: the signal is a human `animal`/`animal_wrong_id` label on a `no_animal`/`unclassifiable`-status row, read at query time by joining `detections` against `detection_feedback` (no dedicated code path).

### Configuration Architecture

The system uses a sophisticated configuration system with:
- **Type-safe dataclasses**: `CameraConfig`, `MotionConfig`, `PerformanceConfig`, `StorageConfig`, `SpeciesConfig`
- **Environment variable overrides**: All parameters can be overridden via env vars
- **Validation**: Configuration validation with meaningful error messages
- **Test configuration factory**: `Config.create_test_config()` for unit tests

### Species Identification Architecture

The species identification system integrates Google SpeciesNet v5.0.2:

- **SpeciesIdentifier**: Main class that wraps SpeciesNet (uses `SpeciesNet` class, not `SpeciesNetEnsemble`)
- **API**: Uses `predict()` method with `filepaths`, `country`, and `admin1_region` parameters
- **Model**: Default is `kaggle:google/speciesnet/pyTorch/v4.0.1a/1` (auto-downloaded from Kaggle on first use)
- **Components**: Loads detector (MegaDetector), classifier, and ensemble combiner
- **Lazy Loading**: Model loads on first identification request (not at startup) - takes ~6 seconds
- **Geographic Filtering**: Configured for Bonn, Germany (DEU/NW region) via geofencing
- **Confidence Thresholds**: Two-stage filtering (detection @ 0.5, classification @ 0.5)
- **Blank-prediction routing** (exp #13, 2026-08-02, commit `55234f1`): SpeciesNet's explicit empty-frame verdict is a fully-generic taxonomy label ending in `blank` (e.g. `<uuid>;;;;;;blank`) emitted at ~0.99 confidence, so it used to clear `unknown_species_threshold` and return `DetectionStatus.IDENTIFIED` — a MAIN-channel species alert on a frame the model called empty, and one that bypassed every review-class mute path (human-proximity, deferral, blur, scene, sampling). `SpeciesIdentifier._is_blank_prediction` (last segment `blank` **and** all taxonomy segments empty) now routes it to `DetectionStatus.NO_ANIMAL` with `animals_detected=False`, mirroring the adjacent `no cv result` → UNCLASSIFIABLE branch; observability metadata is preserved so `top_species_raw` still records the blank label. A populated taxonomy ending in `blank` is unaffected. Rollback: `git revert 55234f1` + restart.
- **Error Resilience**: Always returns valid IdentificationResult, never crashes
- **MockSpeciesIdentifier**: Test implementation for development without SpeciesNet

**SpeciesNet Dependencies**:
- Requires `ml-dtypes>=0.5.0` for float4_e2m1fn support
- Requires `numpy>=2.1.0` (ml-dtypes 0.5+ dependency)
- Requires `opencv-python>=4.10.0` (for NumPy 2.x compatibility)
- Uses ONNX for model inference (PyTorch backend)

### Camera Manager Architecture

The camera system supports multiple implementations through the `CameraInterface`:
- **PiCameraManager**: Production implementation using Picamera2 with error handling and resource management
- **MockCameraManager**: Test implementation for development without hardware
- **Dual-stream capture**: Separate low-res motion detection and high-res photo capture
- **Resource management**: Automatic cleanup and memory management for Pi Zero compatibility
- **Camera Preview Tool**: `scripts/camera_preview.py` provides MJPEG web streaming for focus adjustment (based on official Picamera2 example)

### Motion Detection Strategy

- **Background subtraction**: Uses MOG2 algorithm that adapts to lighting changes
  - **Shadow detection** is enabled (`detectShadows=True`); the foreground threshold is set to 200 to drop MOG2's shadow markers (value 127) and keep only true foreground (value 255). This suppresses false positives from moving tree shadows.
  - The background model is **not reset** after detections — MOG2's natural adaptation (history=500) plus shadow detection are relied upon instead, so the learned shadow distribution survives across triggers.
- **Central region weighting**: Emphasizes motion in the center of the frame
- **Consecutive detection filtering**: Requires multiple consecutive detections to reduce momentary false positives
- **Contour analysis**: Validates motion based on size and position
- **Color variance filtering** (disabled by default): Analyzes color distribution in motion regions
  - Captures RGB frames instead of grayscale when enabled
  - Filters out motion from uniform-color objects (e.g., wind-blown leaves, grass)
  - Helps distinguish vegetation movement from actual animals with varied coloring

### Environment Setup

Requires `.env` file with:
- `TELEGRAM_BOT_TOKEN`: Bot token for Telegram integration
- `TELEGRAM_CHAT_ID`: Target chat/channel ID

Additional optional environment variables for fine-tuning:
- Motion: `MOTION_THRESHOLD`, `MOTION_CONSECUTIVE_REQUIRED`, `MOTION_FRAME_INTERVAL`, `MOTION_ENABLE_COLOR_FILTERING`, `MOTION_MIN_COLOR_VARIANCE`
- Camera: `CAMERA_MAIN_RESOLUTION`, `CAMERA_MOTION_RESOLUTION`, `CAMERA_EXPOSURE_TIME`, `CAMERA_ANALOGUE_GAIN`, `CAMERA_AE_EXPOSURE_MODE` (auto-exposure bias, one of `normal|short|long`, only applies when `CAMERA_EXPOSURE_TIME`/`CAMERA_ANALOGUE_GAIN` are unset; default `short` — dusk-exposure fix, rollback via `normal`)
- Performance: `PERFORMANCE_COOLDOWN`, `PERFORMANCE_MAX_IMAGES` (burst image retention; default 300, raised from 100 on 2026-07-27 — see the Storage bullet above), `PERFORMANCE_SEND_ANNOTATED_IMAGE` (debug: send motion overlay alongside original, default false)
  `PERFORMANCE_REVIEW_PREFIX_ENABLED` (prefix likely-false-positive notifications — NO_ANIMAL/UNCLASSIFIABLE — with a 🔍 REVIEW header in the same channel; default true)
  `PERFORMANCE_SUPPRESS_HUMAN_ALERTS` (skip the Telegram notification entirely for HUMAN-status detections; still species-ID'd and DB-logged; default true)
  `PERFORMANCE_HUMAN_RETENTION_HOURS` (purge saved burst photos of HUMAN-status detections after this many hours; DB row kept as metadata-only; default 48)
  `PERFORMANCE_SCENE_GATE_ENABLED` (Scene-Unchanged Gate on/off; code default **false**, but deployed as **true** since 2026-07-26 by explicit human override — see the Scene-Unchanged Gate bullet above), `PERFORMANCE_SCENE_GATE_SIMILARITY_THRESHOLD` (mute threshold, bounded `[0.80, 1.0]`; default 0.97), `PERFORMANCE_SCENE_GATE_REF_COUNT` (rolling reference-set size; default 3), `PERFORMANCE_SCENE_GATE_REF_MAX_AGE_HOURS` (reference max age; default 6.0)
  `PERFORMANCE_REVIEW_SAMPLE_RATE` (Review Sampling Gate: fraction of review-class bursts surviving Human/Proximity/Blur/Scene that are actually sent to Telegram, bounded `[0.0, 1.0]`; default 0.25; `1.0` = send all, the rollback lever)
  `PERFORMANCE_HUMAN_PROXIMITY_WINDOW_SECONDS` (Human-Proximity Mute Gate: look-back window after the most recent HUMAN-status detection during which review-class bursts are muted, bounded `[0.0, 600.0]`; default 120.0; `0` = disabled, the rollback lever)
  `PERFORMANCE_HUMAN_DENSITY_WINDOW_SECONDS` (Human-Proximity Mute Gate's OR-ed density condition: trailing window over which HUMAN-status detections are counted, bounded `[0.0, 7200.0]`; default 1800.0), `PERFORMANCE_HUMAN_DENSITY_COUNT` (minimum HUMAN-status detection count in that window that mutes a review-class burst, bounded `[0, 100]`; default 8; `0` = density condition disabled, the rollback lever)
  `PERFORMANCE_REVIEW_DEFER_SECONDS` (Deferred REVIEW Send Gate: how long a surviving review-class notification is held before sending, cancelled if a HUMAN-status detection lands within it, bounded `[0.0, 600.0]`; default 240.0; `0` = send immediately, the rollback lever), `PERFORMANCE_HUMAN_RETENTION_PROXIMITY_SECONDS` (symmetric ± window around a HUMAN-status detection within which review-class burst photos are also purged at `human_retention_hours`, bounded `[0.0, 3600.0]`; default 240.0; `0` = disabled, the rollback lever)
- Species: `SPECIES_COUNTRY_CODE`, `SPECIES_REGION`, `SPECIES_UNKNOWN_THRESHOLD`, `SPECIES_HUMAN_DETECTION_CONFIDENCE` (MegaDetector person-category confidence, combined with a `homo` taxonomy check, that fires `DetectionStatus.HUMAN`; default 0.3)
- Storage: `STORAGE_LOG_DIR` (rotating file log destination, `<log_dir>/wildlife.log`; default `data/logs`)

### Hardware Dependencies

- **Raspberry Pi 5 with 8GB RAM** (required for SpeciesNet)
- **Raspberry Pi Camera Module** (any compatible module, tested with IMX477)
- **Storage**: ~2GB for models and images (SpeciesNet models: ~214MB)
- **Python 3.13** with system site packages access (for libcamera)
- Uses Picamera2 for native Pi camera control (requires libcamera system package)
- OpenCV 4.11+ for computer vision processing
- SpeciesNet 5.0.2 for AI species identification (PyTorch-based, ONNX runtime)

### Error Handling

The system includes comprehensive error handling:
- **Camera recovery**: Automatic restart on repeated errors
- **Resource cleanup**: Proper memory management with SystemMonitor
- **Graceful degradation**: System continues on individual component failures
- **Species ID fallback**: Returns "Unknown species" on any identification error
- **Logging**: Comprehensive logging for debugging all components

### Testing Strategy

The codebase includes extensive unit tests:
- **Configuration validation tests**: Ensure proper parameter validation including SpeciesConfig
- **Camera manager tests**: Test both production and mock implementations
- **Motion detection tests**: Validate detection algorithms
- **Species identification tests**: Mock SpeciesNet calls for unit testing
- **Database tests**: Validate detection logging and queries
- **Integration tests**: End-to-end system testing

Test files follow the pattern `test_*.py` and use pytest with asyncio support.
MockSpeciesIdentifier is available for testing without SpeciesNet dependency.

## Development Notes

- The system uses async/await for Telegram operations while maintaining synchronous camera and AI operations
- Motion detection uses weighted masks to prioritize central regions
- Background subtraction model automatically adapts to lighting changes
- Camera manager provides both production (Picamera2) and mock implementations for development
- **Species identification uses lazy loading** - model loads on first detection, not at startup
- SpeciesNet ensemble combines MegaDetector (object detection) with species classifier
- Geographic filtering automatically restricts predictions to region-appropriate species
- Configuration system supports both defaults and environment-based overrides
- All major components are thoroughly unit tested
- **Performance**: SpeciesNet inference takes ~17 seconds on Pi 5 CPU (no GPU acceleration)
  - Model loading: ~6 seconds (first time only, cached afterward)
  - Detection + classification: ~11 seconds per image
- **Memory**: System uses ~2-3GB RAM during species identification

### UV and Virtual Environment Setup

The project uses UV with a Python 3.13 virtual environment that has system site packages enabled (required for libcamera access):

```bash
# Virtual environment is at .venv with system-site-packages = true
# This allows access to system-installed libcamera Python bindings
# To recreate if needed:
uv venv --python /usr/bin/python3 --system-site-packages
uv sync
```

The `.venv/pyvenv.cfg` file should have `include-system-site-packages = true`.

### Common Dependency Issues

**ml-dtypes and NumPy compatibility**:
- ml-dtypes 0.5.0+ requires NumPy 2.1.0+
- Older OpenCV versions (< 4.10) don't support NumPy 2.x
- Solution: Use opencv-python >= 4.10.0 with numpy >= 2.1.0

**PATH issues in non-interactive shells**:
- UV is at `~/.local/bin/uv` (already in PATH for interactive shells)
- For cron jobs or systemd services, explicitly set: `PATH=$HOME/.local/bin:/usr/bin:$PATH`
- Note: Interactive terminal sessions via `.bashrc` have this configured correctly

**libcamera access**:
- libcamera is a system package (python3-libcamera) from apt
- UV venv must have system-site-packages enabled
- Python 3.13 is required (matches system Python version)