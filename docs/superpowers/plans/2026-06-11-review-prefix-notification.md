# REVIEW-prefix Notification Flagging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prefix likely-false-positive detection notifications (`NO_ANIMAL`, `UNCLASSIFIABLE`) with a `🔍 REVIEW` header line in the existing Telegram channel, behind a default-on env toggle.

**Architecture:** A pure caption render. A small predicate `is_review_detection(status)` in `data_models.py` decides whether a detection is review-class; `_build_caption` in `wildlife_system.py` prepends the header when the predicate is true and the toggle is on. No DB, schema, labelling, or send-path changes.

**Tech Stack:** Python 3.13, pydantic-settings `BaseSettings` (config), pytest. Run everything with `PYTHONPATH=src uv run ...` from repo root `/home/daniel/animal_tracker`.

---

## File Structure

- `src/data_models.py` — add module-level `is_review_detection(status)` next to `DetectionStatus`.
- `src/config.py` — add `review_prefix_enabled: bool = True` to `PerformanceConfig` (env var `PERFORMANCE_REVIEW_PREFIX_ENABLED`).
- `src/wildlife_system.py` — import the predicate; prepend the header in `_build_caption` before `return caption`.
- `tests/test_detection_status.py` — add predicate tests + caption prefix tests (this file already owns `_build_caption` caption tests and the `_make_system` / `_species_result_for_status` helpers).
- Docs: `CLAUDE.md`, `experiments/runs/0001-notification-gate-live.md`, `experiments/JOURNAL.md`.

**Note on env var name:** `PerformanceConfig` uses `env_prefix='PERFORMANCE_'`, so the real env var is `PERFORMANCE_REVIEW_PREFIX_ENABLED` (consistent with `PERFORMANCE_SEND_ANNOTATED_IMAGE`). The field is `review_prefix_enabled`.

---

### Task 1: `is_review_detection` predicate

**Files:**
- Modify: `src/data_models.py` (after the `DetectionStatus` class, which ends with `ERROR = "error"` around line 42)
- Test: `tests/test_detection_status.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_detection_status.py` (after the `DetectionStatus constants` section, before the `_build_caption` section):

```python
# ===========================================================================
# 1b. is_review_detection predicate
# ===========================================================================

def test_is_review_detection_true_for_no_animal_and_unclassifiable():
    from data_models import DetectionStatus, is_review_detection
    assert is_review_detection(DetectionStatus.NO_ANIMAL) is True
    assert is_review_detection(DetectionStatus.UNCLASSIFIABLE) is True


def test_is_review_detection_false_for_non_review_statuses():
    from data_models import DetectionStatus, is_review_detection
    assert is_review_detection(DetectionStatus.IDENTIFIED) is False
    assert is_review_detection(DetectionStatus.ANIMAL_UNCERTAIN) is False
    assert is_review_detection(DetectionStatus.ERROR) is False


def test_is_review_detection_false_for_unknown_value():
    from data_models import is_review_detection
    assert is_review_detection(None) is False
    assert is_review_detection("something_else") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_detection_status.py -k is_review_detection -v`
Expected: FAIL with `ImportError: cannot import name 'is_review_detection'`.

- [ ] **Step 3: Implement the predicate**

In `src/data_models.py`, immediately after the `DetectionStatus` class (after the `ERROR = "error"` line and its docstring comment), add:

```python
#: Detection statuses that are routed to human "review" (likely false
#: positives). Used by the notification layer to prepend a 🔍 REVIEW header.
#: Kept independent of the DB `gate_would_suppress` shadow column.
_REVIEW_STATUSES = frozenset({DetectionStatus.NO_ANIMAL, DetectionStatus.UNCLASSIFIABLE})


def is_review_detection(status) -> bool:
    """True if `status` is a review-class (likely false-positive) detection."""
    return status in _REVIEW_STATUSES
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_detection_status.py -k is_review_detection -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data_models.py tests/test_detection_status.py
git commit -m "feat(data_models): is_review_detection predicate (NO_ANIMAL, UNCLASSIFIABLE)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `review_prefix_enabled` config flag

**Files:**
- Modify: `src/config.py:154` (inside `PerformanceConfig`, right after `send_annotated_image`)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py` (follow the existing test style in that file — it builds configs via `Config` / `create_test_config`):

```python
def test_review_prefix_enabled_defaults_true():
    from config import Config
    cfg = Config.create_test_config()
    assert cfg.performance.review_prefix_enabled is True


def test_review_prefix_enabled_env_override(monkeypatch):
    monkeypatch.setenv("PERFORMANCE_REVIEW_PREFIX_ENABLED", "false")
    from config import PerformanceConfig
    assert PerformanceConfig().review_prefix_enabled is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_config.py -k review_prefix -v`
Expected: FAIL with `AttributeError: 'PerformanceConfig' object has no attribute 'review_prefix_enabled'`.

- [ ] **Step 3: Add the config field**

In `src/config.py`, inside `PerformanceConfig`, immediately after line 154 (`send_annotated_image: bool = False ...`), add:

```python
    review_prefix_enabled: bool = True  # Prefix likely-FP (NO_ANIMAL/UNCLASSIFIABLE) captions with 🔍 REVIEW header
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_config.py -k review_prefix -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat(config): REVIEW_PREFIX_ENABLED toggle (PerformanceConfig, default on)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Prepend the `🔍 REVIEW` header in `_build_caption`

**Files:**
- Modify: `src/wildlife_system.py:27` (import) and the end of `_build_caption` (the `return caption` around line 452)
- Test: `tests/test_detection_status.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_detection_status.py` (in the `_build_caption` section, after the existing caption tests; reuses the `_make_system` and `_species_result_for_status` helpers already defined in that file):

```python
def test_caption_review_prefix_on_no_animal(monkeypatch, tmp_path):
    from data_models import DetectionStatus
    sys_obj = _make_system(monkeypatch, tmp_path)
    sys_obj.config.performance.review_prefix_enabled = True
    sr = _species_result_for_status(DetectionStatus.NO_ANIMAL)
    caption = sys_obj._build_caption(sr, 5000, datetime(2026, 6, 9, 14, 30, 0))
    assert caption.startswith("🔍 REVIEW")
    # Original body still present after the header
    assert "false positive" in caption.lower()


def test_caption_review_prefix_on_unclassifiable(monkeypatch, tmp_path):
    from data_models import DetectionStatus
    sys_obj = _make_system(monkeypatch, tmp_path)
    sys_obj.config.performance.review_prefix_enabled = True
    sr = _species_result_for_status(DetectionStatus.UNCLASSIFIABLE)
    caption = sys_obj._build_caption(sr, 5000, datetime(2026, 6, 9, 14, 30, 0))
    assert caption.startswith("🔍 REVIEW")


def test_caption_no_review_prefix_when_disabled(monkeypatch, tmp_path):
    from data_models import DetectionStatus
    sys_obj = _make_system(monkeypatch, tmp_path)
    sys_obj.config.performance.review_prefix_enabled = False
    sr = _species_result_for_status(DetectionStatus.NO_ANIMAL)
    caption = sys_obj._build_caption(sr, 5000, datetime(2026, 6, 9, 14, 30, 0))
    assert "🔍 REVIEW" not in caption


def test_caption_no_review_prefix_for_identified(monkeypatch, tmp_path):
    from data_models import DetectionStatus
    sys_obj = _make_system(monkeypatch, tmp_path)
    sys_obj.config.performance.review_prefix_enabled = True
    sr = _species_result_for_status(DetectionStatus.IDENTIFIED, "Red Fox", 0.85)
    caption = sys_obj._build_caption(sr, 5000, datetime(2026, 6, 9, 14, 30, 0))
    assert "🔍 REVIEW" not in caption
    assert "Red Fox" in caption
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/test_detection_status.py -k review_prefix -v`
Expected: the three "on"/"identified" assertions FAIL (no `🔍 REVIEW` produced yet); `test_caption_no_review_prefix_when_disabled` and `..._for_identified` may already pass since no prefix exists. Confirm the prefix-present tests fail.

- [ ] **Step 3: Wire the prefix**

In `src/wildlife_system.py`, change the import on line 27 from:

```python
from data_models import DetectionStatus
```

to:

```python
from data_models import DetectionStatus, is_review_detection
```

Then, at the end of `_build_caption`, replace the final `return caption` with:

```python
        # REVIEW prefix (ADR-004 exp #1, labeling variant): flag likely-FP
        # detections with a scannable header in the same channel. FN-safe —
        # nothing is dropped, only labeled.
        if self.config.performance.review_prefix_enabled and is_review_detection(status):
            caption = f"🔍 REVIEW — likely false positive\n{caption}"

        return caption
```

(`status` is already bound at the top of `_build_caption`; `self.config.performance` is available.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_detection_status.py -k review_prefix -v`
Expected: 4 PASS.

- [ ] **Step 5: Run the full caption + status suite to confirm no regression**

Run: `PYTHONPATH=src uv run pytest tests/test_detection_status.py -v`
Expected: all PASS (existing caption tests assert on body lines, which are unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/wildlife_system.py tests/test_detection_status.py
git commit -m "feat(notify): prefix likely-FP captions with 🔍 REVIEW header (exp #1 labeling variant)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Full suite + docs/notebook

**Files:**
- Modify: `CLAUDE.md` (env var docs), `experiments/runs/0001-notification-gate-live.md` (record shipped variant), `experiments/JOURNAL.md` (one line)

- [ ] **Step 1: Run the full test suite (verification gate)**

Run: `cd /home/daniel/animal_tracker && PYTHONPATH=src uv run pytest tests/ -v 2>&1 | tail -20`
Expected: all tests pass (prior baseline was 242 passed before these new tests; expect 242 + the new tests). Record the final count.

- [ ] **Step 2: Document the env var in `CLAUDE.md`**

In the "Performance" env-vars list under **Environment Setup** (the line that lists `PERFORMANCE_COOLDOWN`, `PERFORMANCE_MAX_IMAGES`, `PERFORMANCE_SEND_ANNOTATED_IMAGE`), append:

```
`PERFORMANCE_REVIEW_PREFIX_ENABLED` (prefix likely-false-positive notifications — NO_ANIMAL/UNCLASSIFIABLE — with a 🔍 REVIEW header in the same channel; default true)
```

- [ ] **Step 3: Record the shipped variant in the experiment run file**

Append a dated section to `experiments/runs/0001-notification-gate-live.md` under "Daily observations" (do NOT rewrite prior observations — append only). Use this text, filling in the actual commit SHA from `git rev-parse --short HEAD` after Task 3:

```markdown
### 2026-06-11 — shipped as SAME-CHANNEL LABELING (not routing)

Deployed the gate as **labeling, not routing**: review-class detections
(`NO_ANIMAL`, `UNCLASSIFIABLE`) now get a `🔍 REVIEW — likely false positive`
header prepended to their Telegram caption in the existing channel. Nothing is
dropped or moved → **FN-safe** (a misclassified real animal is still fully
visible with its feedback buttons, merely labeled "review"). This sidesteps the
two blockers that held the routing variant: no second Telegram channel needed,
and no FN-veto (suppression was the vetoed part; labeling is not suppression).

- Predicate: `is_review_detection(status)` in `data_models.py`
  (`{NO_ANIMAL, UNCLASSIFIABLE}`). The DB `gate_would_suppress` shadow column is
  unchanged (still `not animals_detected`) — deliberately independent.
- Toggle: `PERFORMANCE_REVIEW_PREFIX_ENABLED` (default true); instant off-switch.
- Commit: `<SHA from Task 3>`. Reversal: `git revert` + env flag.
- Spec: `docs/superpowers/specs/2026-06-11-review-prefix-notification-design.md`.

Goes live on the next camera restart (code change). Experiment #1 is realized as
this labeling variant; a future real channel split remains optional follow-up.
```

- [ ] **Step 4: Add a JOURNAL line**

Append to `experiments/JOURNAL.md`:

```markdown
- 2026-06-11 (human-directed) — Shipped exp #1 (notification-gate-live) as
  SAME-CHANNEL LABELING: 🔍 REVIEW header on NO_ANIMAL/UNCLASSIFIABLE captions
  (`is_review_detection` in data_models; `PERFORMANCE_REVIEW_PREFIX_ENABLED`
  default on). FN-safe (labels, doesn't drop/route) → no 2nd channel, no FN-veto.
  gate_would_suppress column untouched. Live on next camera restart. See
  runs/0001-notification-gate-live.md.
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md experiments/runs/0001-notification-gate-live.md experiments/JOURNAL.md
git commit -m "docs(exp): record exp #1 shipped as same-channel REVIEW labeling

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Review predicate `{NO_ANIMAL, UNCLASSIFIABLE}` → Task 1. ✓
- Env toggle default on → Task 2. ✓
- Caption header prepend, body unchanged, flag-off path → Task 3. ✓
- Feedback buttons unchanged → no code touched in the send path (verified: `_build_caption` only builds text; `send_notification` / feedback keyboard untouched). ✓
- No DB/labelling/`gate_would_suppress` change → no task touches the DB or that column. ✓
- Tests for predicate + caption → Tasks 1, 3. ✓
- Docs (CLAUDE.md, run 0001, JOURNAL) → Task 4. ✓
- Out of scope (2nd channel, BOUNDS/deploy-gate) → not present in any task. ✓

**Placeholder scan:** none — all steps have concrete code/commands. The only fill-in is the commit SHA in Task 4 Step 3, with explicit instructions to obtain it.

**Type consistency:** `is_review_detection` signature and name identical across Tasks 1 and 3; `review_prefix_enabled` field name identical across Tasks 2 and 3; env var `PERFORMANCE_REVIEW_PREFIX_ENABLED` consistent in Tasks 2 and 4.
