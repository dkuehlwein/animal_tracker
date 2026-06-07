# Phase 1 Implementation Spec — Ground Truth (FP *and* FN)

Implements the "smallest correct slice" from ADR-004 §Phased Roadmap → Phase 1.
Goal: start collecting human-labeled ground truth (false positives *and* false
negatives) with near-zero operator effort, without changing what the system
*sends* today. Nothing here tunes anything — it only **measures and records**.

**Settled architecture decision (was open in ADR-004):** the Telegram receive
path runs as a **separate sidecar process** (`src/telegram_feedback.py`) sharing
the SQLite DB via WAL. The async camera loop in `wildlife_system.py` is left
untouched except for threading a `detection_id` through to the send path.

## Scope (in order)

### 1. `detection_feedback` table + richer detection logging  (`database_manager.py`)

- **Enable WAL** at init (`PRAGMA journal_mode=WAL`) so the sidecar can write
  feedback while the main process writes detections.
- **New table `detection_feedback`** (append-only — never UPDATE/DELETE rows;
  anti-self-poisoning rule):
  ```sql
  CREATE TABLE detection_feedback (
      id           INTEGER PRIMARY KEY AUTOINCREMENT,
      detection_id INTEGER NOT NULL,        -- event key = detections.id
      label        TEXT    NOT NULL,        -- 'animal' | 'false_positive' | 'wrong_species'
      source       TEXT    NOT NULL DEFAULT 'human',  -- labeling tier
      created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (detection_id) REFERENCES detections(id)
  );
  ```
  Multiple rows per detection are allowed (a re-tap appends; reconciliation takes
  the latest human row — that's a later phase's concern).
- **Richer columns on `detections`** (idempotent `ALTER TABLE ADD COLUMN`
  migration, guarded by `PRAGMA table_info`): `animals_detected`,
  `detection_count`, `max_detection_confidence`, `contour_count`,
  `largest_contour_area`, `foreground_pixel_count`, `hour_of_day`,
  `gate_would_suppress`, `frame_stability`. All nullable / default — old rows stay
  valid. These are values *already computed* and currently dropped; this is pure
  schema + plumbing.
- **API**:
  - `log_detection(...)` gains optional kwargs for the new fields (back-compatible
    defaults). `hour_of_day` is derived from the insert time.
  - `add_feedback(detection_id, label, source='human') -> int` — append a row.
  - `get_feedback(detection_id) -> list` — for tests / later reconciliation.

### 2. Telegram receive path — sidecar  (`telegram_feedback.py` + `notification_service.py`)

- **Outgoing**: every detection notification gains an inline keyboard:
  `[✅ Animal] [❌ False positive] [🐦 Wrong species]`. `callback_data` =
  `fb:<detection_id>:<code>` with `code ∈ {a, fp, ws}` (short, well under
  Telegram's 64-byte limit). Built by `build_feedback_keyboard(detection_id)`.
  - `NotificationService.send_photo_with_caption` / `send_text_message` gain an
    optional `reply_markup`. Media groups can't carry a keyboard (Telegram API
    limitation) so the debug annotated-image path sends the keyboard on a short
    follow-up message.
  - `wildlife_system.process_detection` returns the `detection_id`; `run()`
    threads it into `send_notification`, which attaches the keyboard.
- **Incoming (sidecar)**: `src/telegram_feedback.py` runs a long-lived
  `telegram.ext.Application` polling for `callback_query` updates. A
  `CallbackQueryHandler` parses `fb:<id>:<code>`, calls `database.add_feedback`,
  answers the callback (toast), and edits the message to confirm the recorded
  label. The pure parse+record step is factored into
  `record_feedback_callback(data, database)` so it is unit-testable without
  network. Only the sidecar polls `getUpdates`; the main process is send-only, so
  there is no getUpdates conflict.

### 3. Shadow-mode notification gate  (`wildlife_system.py`)

- Compute `gate_would_suppress = not species_result.animals_detected` and persist
  it (step 1). **Send behaviour is unchanged** — all detections still go to the
  main channel. The gate only records what it *would* have suppressed, so we can
  later measure its FN cost before flipping it live (Phase 2).

### 4. Lightweight timelapse FN-audit writer  (`timelapse_writer.py`)

- Independent low-rate capture so missed animals (never triggered motion) are
  auditable. Reuses the motion frame the loop already grabs every tick — no extra
  camera work: `TimelapseWriter.maybe_capture(frame, now)` saves a 640×480
  grayscale JPEG every `timelapse_interval` seconds into `data/timelapse/`, with
  count-bounded rotation (`timelapse_max_files`). Called once per loop tick.
- Config (in `PerformanceConfig`): `enable_timelapse` (default True),
  `timelapse_interval` (20.0 s), `timelapse_max_files` (10000 ≈ ~2 days @ 20 s).

## Out of scope (later phases)
Flipping the gate live / review channel (Phase 2), ROI masking, threshold
experiments, the Layer-A replay harness + RDE (Phase 3), the `/loop` runner and
git notebook (Phase 4). No `MotionConfig` bounds validators yet — that lands with
the autonomous loop that actually writes config (Phase 4); nothing in Phase 1
mutates config.

## Test plan
- `test_database_manager.py`: WAL enabled; migration adds columns idempotently on
  a pre-existing old-schema DB; `log_detection` persists new fields; `add_feedback`
  / `get_feedback` round-trip; feedback is append-only (two taps → two rows).
- `test_telegram_feedback.py`: `record_feedback_callback` parses each code → right
  label and writes a row; malformed/unknown data is rejected without writing;
  `build_feedback_keyboard` emits the 3 expected `callback_data` strings.
- `test_timelapse_writer.py`: respects interval (no double-write inside one
  window); rotates to honour `timelapse_max_files`; disabled → no writes.
- Existing suite must stay green (notably `test_wildlife_system.py`).
