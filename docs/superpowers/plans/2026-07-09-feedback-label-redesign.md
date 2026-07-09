# Telegram Feedback Label Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-07-09-feedback-label-redesign-design.md` — read it first; it defines the semantics.

**Goal:** Replace the 3-button feedback keyboard with a 5-button, 2-row keyboard (`animal`, `animal_wrong_id`, `person`, `false_positive`, `cant_tell`), keep legacy `ws` taps working, and exclude `cant_tell` from the tuning loop's fp_rate denominator and per-tier buckets.

**Architecture:** `src/feedback_protocol.py` is the single source of truth for wire codes and button layout (send path `notification_service` and receive sidecar `telegram_feedback` both import it — that stays true). Metrics semantics change only in `src/loop/metrics.py`; reconciliation in `src/loop/ingest.py` is intentionally unchanged (a truthy human `cant_tell` already wins and blocks auto-label backfill).

**Tech Stack:** Python 3.13, python-telegram-bot, pytest (`uv run pytest tests/ -v`).

**Plan style note (per user preference):** tasks specify intent, exact interfaces, and exact test expectations — not literal code. The implementing subagent writes the code, follows TDD (failing test first), and runs the deterministic suite.

## Global Constraints

- Stored label for person-in-frame is `person`, NEVER `human` (`human` already means the labeller tier: `source='human'`, per-tier `human` bucket).
- Legacy wire code `ws` → `wrong_species` must keep parsing (old messages in the channel have live buttons) but must NOT appear on new keyboards.
- All labels are non-empty strings (documented invariant in `metrics.py` `_per_tier_partition` relies on truthiness).
- `callback_data` format stays `fb:<detection_id>:<code>` and must stay under Telegram's 64-byte limit.
- Every task: run the touched test file, then the full suite `uv run pytest tests/ -v`; commit per task.

---

### Task 1: Protocol — new vocabulary + 2-row keyboard

**Files:**
- Modify: `src/feedback_protocol.py`
- Test: `tests/test_feedback_protocol.py`

**Interfaces:**
- Produces: `CODE_TO_LABEL` mapping exactly `{"a": "animal", "wid": "animal_wrong_id", "p": "person", "fp": "false_positive", "ct": "cant_tell", "ws": "wrong_species"}`.
- Produces: `DISPLAYED_CODES = ["a", "wid", "p", "fp", "ct"]` semantics — `ws` is parse-only. (Implementation detail free; the contract is: keyboard shows exactly the 5 new codes, parser accepts all 6.)
- Produces: `build_feedback_keyboard(detection_id)` returns an `InlineKeyboardMarkup` with **two rows**: row 1 = `✅ Animal`, `🐦 Animal, wrong ID`, `👤 Human`; row 2 = `❌ Nothing there`, `🤷 Can't tell` (button text exact, in this order).
- `parse_callback_data` contract unchanged: returns `(detection_id, label)`, raises `ValueError` on unknown codes/malformed data.

**Steps:**

- [ ] **Step 1: Write failing tests** in `tests/test_feedback_protocol.py` (update existing tests + add new ones):
  - `test_build_keyboard_callback_data`: update — keyboard has 2 rows (3 + 2 buttons); each button's `callback_data` is `fb:<id>:<code>` for codes `a, wid, p` (row 1) and `fp, ct` (row 2); no button carries `ws`.
  - New `test_keyboard_button_texts`: exact button texts and row grouping as in Interfaces above.
  - `test_parse_callback_data_each_code`: update — parametrize over all 6 codes including legacy `ws` → `wrong_species`.
  - New `test_legacy_ws_not_on_keyboard`: `ws` parses but appears on no button.
  - `test_parse_callback_data_rejects_malformed`: keep; add unknown-code case e.g. `fb:1:xx`.
  - `test_callback_data_within_telegram_limit`: keep, now covers longest code `wid` with a large detection id.
- [ ] **Step 2: Run** `uv run pytest tests/test_feedback_protocol.py -v` — new/updated tests FAIL (old 3-button keyboard).
- [ ] **Step 3: Implement** in `src/feedback_protocol.py`: extend `CODE_TO_LABEL`; restructure `_BUTTONS` as a list of rows; update `build_feedback_keyboard` to build 2 rows. Update the module docstring (still one source of truth; note `ws` parse-only legacy).
- [ ] **Step 4: Run** the file's tests, then full suite — all PASS.
- [ ] **Step 5: Commit** `feat(feedback): 5-button 2-row label keyboard, ws parse-only legacy`.

---

### Task 2: Sidecar — confirmation messages for new labels

**Files:**
- Modify: `src/telegram_feedback.py` (`_LABEL_CONFIRMATION` and its comment/docstring mention of the button set)
- Test: `tests/test_feedback_protocol.py` (the `record_feedback_callback` tests live here)

**Interfaces:**
- Consumes: Task 1's `CODE_TO_LABEL`.
- Produces: `_LABEL_CONFIRMATION` covering all 6 labels: `animal` → `"✅ Recorded: animal"`, `animal_wrong_id` → `"🐦 Recorded: animal, wrong ID"`, `person` → `"👤 Recorded: human in frame"`, `false_positive` → `"❌ Recorded: nothing there"`, `cant_tell` → `"🤷 Recorded: can't tell"`, `wrong_species` → keep existing `"🐦 Recorded: wrong species"` (legacy taps).

**Steps:**

- [ ] **Step 1: Write failing test** `test_record_feedback_callback_confirmations`: parametrize over all 6 codes; `record_feedback_callback(f"fb:<id>:<code>", db)` writes a `detection_feedback` row with the mapped label and `source="human"`, and returns the exact confirmation string above (no fallback `Recorded: <label>` path hit).
- [ ] **Step 2: Run** `uv run pytest tests/test_feedback_protocol.py -v` — FAILS for the 3 new labels (fallback string returned).
- [ ] **Step 3: Implement** the `_LABEL_CONFIRMATION` additions; update the module docstring's button list.
- [ ] **Step 4: Run** file tests + full suite — PASS.
- [ ] **Step 5: Commit** `feat(feedback): confirmation messages for new label vocabulary`.

---

### Task 3: Metrics — exclude `cant_tell` from denominator and tier buckets

**Files:**
- Modify: `src/loop/metrics.py` (`compute_metrics` labeled-set filter; `_per_tier_partition`; the invariant comment)
- Modify: `src/loop/ingest.py` (comment only: extend the vocabulary note — human `cant_tell` intentionally wins reconciliation so auto-labels can't backfill; no behavior change)
- Test: `tests/test_loop_metrics.py`

**Interfaces:**
- Consumes: reconciled row dicts from `ingest.reconcile` (keys `reconciled_label`, `human`, `tier2`, `tier1`, `detection_status`, ...).
- Produces: `compute_metrics` — `labeled_triggers` counts rows with `reconciled_label` not None **and** not `"cant_tell"`; fp numerator unchanged (`== "false_positive"`). `_per_tier_partition` — a row whose winning label (first non-None of human/tier2/tier1, same precedence as now) equals `"cant_tell"` is skipped entirely (no bucket). Check the winning label uniformly across tiers, not only for human rows. Invariant `n_human + n_claude + n_md == labeled_triggers` must still hold under the new definitions — update the comment to say so.

**Semantics reminders (from spec):** `animal`, `animal_wrong_id`, `person`, legacy `wrong_species` are all non-FP real triggers — no code needed, but tests must pin it. `error_count`/`error_rate` remain over ALL rows (unchanged).

**Steps:**

- [ ] **Step 1: Write failing tests** in `tests/test_loop_metrics.py`:
  - New `test_cant_tell_excluded_from_fp_denominator`: rows = 1 human `false_positive`, 1 human `cant_tell`, 1 human `animal` → `labeled_triggers == 2`, `fp_count == 1`, `fp_rate == 0.5`.
  - New `test_cant_tell_excluded_from_tier_buckets`: same rows → `n_human == 2`; and `n_human + n_claude + n_md == labeled_triggers`.
  - New `test_cant_tell_blocks_lower_tiers`: row with `human="cant_tell"`, `tier2="false_positive"`, `tier1="animal"` (reconciled_label `"cant_tell"`, as ingest produces) → counted nowhere: `labeled_triggers == 0`, all bucket `n` == 0.
  - New `test_new_labels_count_as_non_fp`: parametrize `animal_wrong_id`, `person`, `wrong_species` as human label → in denominator, `fp_count == 0`.
  - Existing partition/invariant tests (`test_per_tier_partition_no_overlap_sums_to_labeled` etc.) must keep passing unmodified.
- [ ] **Step 2: Run** `uv run pytest tests/test_loop_metrics.py -v` — new tests FAIL (cant_tell currently lands in denominator/buckets).
- [ ] **Step 3: Implement** in `metrics.py`; add the one-paragraph comment in `ingest.py`. Keep the change minimal — do not touch CSV schema, report, or state handling.
- [ ] **Step 4: Run** file tests + full suite — PASS.
- [ ] **Step 5: Commit** `feat(loop): exclude cant_tell labels from fp denominator and tier buckets`.

---

### Task 4: Documentation — CLAUDE.md label vocabulary

**Files:**
- Modify: `CLAUDE.md` (add a short "Feedback labels" bullet block under the Key Configuration Parameters / ADR-004 observability area, wherever it reads most naturally)

**Content to convey (prose, concise, match surrounding style):**
- The 5-button keyboard (texts + stored labels + wire codes), one tap per notification.
- `person` vs `human` naming rule (label vs labeller tier).
- Legacy `ws`/`wrong_species`: parse-only, heterogeneous history, never displayed.
- `cant_tell`: wins reconciliation (blocks tier-1/tier-2 backfill) but is excluded from the fp_rate denominator and per-tier buckets.
- FN signal = `animal`/`animal_wrong_id` label on a `no_animal`/`unclassifiable` status row (query-time join; no code).

**Steps:**

- [ ] **Step 1: Write the CLAUDE.md section** (no tests; docs task).
- [ ] **Step 2: Run** full suite once more `uv run pytest tests/ -v` — PASS (sanity, nothing should change).
- [ ] **Step 3: Commit** `docs: document feedback label vocabulary and cant_tell semantics`.

---

## Deployment note (manual, after merge)

The feedback sidecar (`src/telegram_feedback.py`) and main process must be restarted to pick up the new keyboard/labels. Old messages keep their old keyboards — their `a`/`fp`/`ws` taps all still parse. This is a Daniel-visible change to the Telegram channel; mention it in the completion summary.
