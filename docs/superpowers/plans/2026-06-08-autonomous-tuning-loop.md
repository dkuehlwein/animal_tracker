# Autonomous Detection-Tuning Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-improving nightly detection-tuning loop: deterministic Python tools under `src/loop/` (ingest, metrics, replay-stub, report, deploy, guardrails) plus a committed `experiments/` git notebook and systemd plumbing, so an on-Pi Claude Code `/loop` session can reduce false positives and false negatives with bounds + FN-veto guardrails.

**Architecture:** Two layers. (1) **Deterministic local Python** in `src/loop/` — each module is single-purpose, unit-tested, and exposes a `python -m loop.<name>` CLI that prints a JSON result to stdout. (2) **Judgment layer** — the `experiments/loop.md` + `PROTOCOL.md` prompt that `/loop` runs, reconstructing state from the git notebook each tick. The loop reads the SQLite DB read-only (WAL), writes config only via `deploy.py` (renders `experiments/deployed_config.env`, an overlay env-file `config.py` layers over `.env`), and applies it by a pre-sunrise camera restart driven by a systemd timer.

**Tech Stack:** Python 3.13, pydantic-settings (`BaseSettings`), SQLite (WAL), `astral` (via `SunChecker`), `python-telegram-bot` 22.7, pytest + pytest-asyncio, UV, systemd. Tests reuse `tests/conftest.py` isolation; `src` is already on `sys.path` (conftest inserts it), so `src/loop/` is importable as `loop.<name>`.

**Conventions used throughout this plan:**
- New runtime code: `src/loop/<name>.py`. Tests: `tests/test_loop_<name>.py`.
- Each `src/loop/<name>.py` ends with a `main()` + `if __name__ == "__main__":` block that prints JSON to stdout and exits non-zero with `{"error": ...}` on failure, so `python -m loop.<name>` is a clean CLI.
- Run tests with: `uv run pytest tests/test_loop_<name>.py -v` (run from repo root `/home/daniel/animal_tracker`).
- Commit after every green step. Conventional-commit messages, each ending with:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  ```
- Never read/write the production `data/` DB in tests — `tests/conftest.py` already redirects `STORAGE_*` to a temp dir and disables `.env`. Loop tests build their own temp SQLite DBs and temp `experiments/` dirs via pytest `tmp_path`.

---

## File Structure

**Created (runtime):**
- `src/loop/__init__.py` — package marker (empty).
- `src/loop/guardrails.py` — single bounds dict + fast guards + FN-veto + freeze logic.
- `src/loop/ingest.py` — read detections+feedback past watermark; reconcile labels.
- `src/loop/metrics.py` — Wilson-CI paired FP/FN; append `daily.csv` idempotently.
- `src/loop/replay.py` — STUB Layer-A replay seam.
- `src/loop/report.py` — Telegram daily summary + heartbeat (send-only).
- `src/loop/deploy.py` — validate→write `state.json`→render env→stamp restart; rollback.
- `src/loop/state.py` — small helper: atomic JSON read/write of `experiments/state.json` (shared by deploy + ingest + report).

**Modified (runtime):**
- `src/config.py` — overlay env-file on tunable sub-configs + bounds field validators.
- `src/telegram_feedback.py` — add `/pause` and `/rollback` command handlers.

**Created (tests):**
- `tests/test_loop_guardrails.py`, `tests/test_loop_ingest.py`, `tests/test_loop_metrics.py`, `tests/test_loop_replay.py`, `tests/test_loop_report.py`, `tests/test_loop_deploy.py`, `tests/test_loop_state.py`, `tests/test_telegram_feedback_commands.py`.
- `tests/test_config.py` — add overlay-precedence + bounds tests.

**Created (notebook, committed to `main`):**
- `experiments/PROTOCOL.md`, `experiments/loop.md`, `experiments/JOURNAL.md`, `experiments/LEARNINGS.md`, `experiments/state.json`, `experiments/runs/0001-notification-gate-live.md`, `experiments/metrics/.gitkeep`, `experiments/gold/.gitkeep`.

**Created (ops):**
- `wildlife-loop.service`, `wildlife-deploy.service`, `wildlife-deploy.timer`.
- `.gitignore` line for `experiments/deployed_config.env`.
- README section documenting installation.

**Shared types (defined once, referenced everywhere):**
- `BOUNDS: dict[str, tuple[float, float]]` in `guardrails.py` — keys are env-var names (e.g. `"MOTION_THRESHOLD"`), values are `(low, high)` inclusive ranges.
- `ReplayResult` dataclass in `replay.py` — fields `status: str`, `reason: str`, `metrics: dict | None`.
- `WilsonCI` is a plain `tuple[float, float]` (low, high).
- `state.json` schema (see Task 6) — read/written only via `src/loop/state.py` helpers `load_state(path)` / `save_state(path, state)`.

---

## Task 1: `guardrails.py` — bounds dict + guards + FN-veto + freeze

**Files:**
- Create: `src/loop/__init__.py`
- Create: `src/loop/guardrails.py`
- Test: `tests/test_loop_guardrails.py`

- [ ] **Step 1: Create the package marker**

Create `src/loop/__init__.py` with a single line:

```python
"""Deterministic, token-free tools for the autonomous tuning loop (ADR-004 Phase 4)."""
```

- [ ] **Step 2: Write the failing test for the bounds dict + `validate_param`**

Create `tests/test_loop_guardrails.py`:

```python
"""Tests for src/loop/guardrails.py — bounds, fast guards, FN-veto, freeze."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from loop import guardrails


def test_bounds_dict_has_seeded_tunables():
    # Bounds are keyed by env-var name (same keys deploy.py validates against).
    assert "MOTION_THRESHOLD" in guardrails.BOUNDS
    low, high = guardrails.BOUNDS["MOTION_THRESHOLD"]
    assert low < high


def test_validate_param_accepts_in_range():
    guardrails.validate_param("MOTION_THRESHOLD", 2000)  # no raise


def test_validate_param_rejects_out_of_range():
    low, high = guardrails.BOUNDS["MOTION_THRESHOLD"]
    with pytest.raises(ValueError, match="out of bounds"):
        guardrails.validate_param("MOTION_THRESHOLD", high + 1)


def test_validate_param_unknown_key_rejected():
    with pytest.raises(ValueError, match="not a tunable"):
        guardrails.validate_param("NOT_A_REAL_KEY", 1)
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_loop_guardrails.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'loop.guardrails'` (or AttributeError).

- [ ] **Step 4: Implement bounds dict + `validate_param`**

Create `src/loop/guardrails.py`:

```python
"""Guardrails for the autonomous tuning loop (ADR-004 Phase 4).

Single source of truth for:
- BOUNDS: allowed ranges for every tunable param (consumed by config.py field
  validators at load-time AND by deploy.py before any write).
- Fast guards: capture-volume collapse / explosion vs a trailing baseline.
- FN-veto: reject an FP win that comes with an FN rise beyond CI noise; HOLD a
  change when FN is unmeasured and the change could plausibly raise FN.
- Feedback-starved freeze: stop tuning when no fresh human labels for N days.
"""

from __future__ import annotations

# Keys are env-var names so config.py validators and deploy.py share one map.
# (low, high) inclusive. Ranges are deliberately conservative — the loop tunes
# within these; out-of-range is rejected by the SYSTEM, not merely discouraged.
BOUNDS: dict[str, tuple[float, float]] = {
    "MOTION_THRESHOLD": (200, 8000),
    "MOTION_MIN_CONTOUR_AREA": (10, 2000),
    "MOTION_CONSECUTIVE_REQUIRED": (1, 6),
    "MOTION_MIN_COLOR_VARIANCE": (0.0, 2000.0),
    "SPECIES_UNKNOWN_THRESHOLD": (0.3, 0.95),
}

FEEDBACK_STARVED_DAYS = 3


def validate_param(key: str, value: float) -> None:
    """Raise ValueError if `key` is not tunable or `value` is out of bounds."""
    if key not in BOUNDS:
        raise ValueError(f"{key!r} is not a tunable parameter")
    low, high = BOUNDS[key]
    if not (low <= value <= high):
        raise ValueError(
            f"{key}={value} out of bounds [{low}, {high}]"
        )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_loop_guardrails.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add src/loop/__init__.py src/loop/guardrails.py tests/test_loop_guardrails.py
git commit -m "feat(loop): add guardrails bounds dict + validate_param

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 7: Write the failing test for fast volume guards**

Append to `tests/test_loop_guardrails.py`:

```python
def test_volume_collapse_flags_near_zero():
    # Baseline ~40/night, tonight ~0 → collapse → rollback recommended.
    verdict = guardrails.check_volume(tonight=0, baseline=40.0)
    assert verdict["rollback"] is True
    assert "collapse" in verdict["reason"]


def test_volume_explosion_flags_spike():
    verdict = guardrails.check_volume(tonight=400, baseline=40.0)
    assert verdict["rollback"] is True
    assert "explos" in verdict["reason"]


def test_volume_normal_ok():
    verdict = guardrails.check_volume(tonight=45, baseline=40.0)
    assert verdict["rollback"] is False
```

- [ ] **Step 8: Run the test to verify it fails**

Run: `uv run pytest tests/test_loop_guardrails.py -v`
Expected: FAIL with `AttributeError: module 'loop.guardrails' has no attribute 'check_volume'`.

- [ ] **Step 9: Implement `check_volume`**

Add to `src/loop/guardrails.py`:

```python
# Multipliers off the trailing baseline that signal something broke, not tuned.
VOLUME_COLLAPSE_FRACTION = 0.1   # < 10% of baseline ⇒ camera/loop likely dead
VOLUME_EXPLOSION_FACTOR = 5.0    # > 5x baseline ⇒ runaway false positives


def check_volume(tonight: int, baseline: float) -> dict:
    """Detect capture-volume collapse (~0) or explosion vs trailing baseline.

    Returns {"rollback": bool, "reason": str}. A baseline <= 0 is treated as
    "no baseline yet" → never recommends rollback (avoids div-by-zero / false
    alarm on a fresh install).
    """
    if baseline <= 0:
        return {"rollback": False, "reason": "no baseline yet"}
    if tonight <= baseline * VOLUME_COLLAPSE_FRACTION:
        return {
            "rollback": True,
            "reason": f"volume collapse: {tonight} vs baseline {baseline:.1f}",
        }
    if tonight >= baseline * VOLUME_EXPLOSION_FACTOR:
        return {
            "rollback": True,
            "reason": f"volume explosion: {tonight} vs baseline {baseline:.1f}",
        }
    return {"rollback": False, "reason": "volume within normal range"}
```

- [ ] **Step 10: Run the test to verify it passes**

Run: `uv run pytest tests/test_loop_guardrails.py -v`
Expected: PASS (7 passed).

- [ ] **Step 11: Commit**

```bash
git add src/loop/guardrails.py tests/test_loop_guardrails.py
git commit -m "feat(loop): add volume collapse/explosion fast guard

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 12: Write the failing test for the FN-veto**

Append to `tests/test_loop_guardrails.py`:

```python
def test_fn_veto_accepts_fp_win_with_stable_fn():
    # FP improved; FN measured and CIs overlap (no significant FN rise) → accept.
    verdict = guardrails.fn_veto(
        fp_before=0.40, fp_after=0.25,
        fn_before=0.10, fn_before_ci=(0.05, 0.18),
        fn_after=0.11, fn_after_ci=(0.06, 0.19),
    )
    assert verdict["decision"] == "accept"


def test_fn_veto_rejects_fp_win_with_fn_rise_beyond_ci():
    # FN jumps and the after-CI lower bound clears the before-CI upper bound.
    verdict = guardrails.fn_veto(
        fp_before=0.40, fp_after=0.25,
        fn_before=0.10, fn_before_ci=(0.05, 0.15),
        fn_after=0.30, fn_after_ci=(0.22, 0.40),
    )
    assert verdict["decision"] == "reject"
    assert "FN" in verdict["reason"]


def test_fn_veto_holds_when_fn_unmeasured_and_change_risky():
    # FN unmeasured (None) + change could raise FN → HOLD, never guess deploy.
    verdict = guardrails.fn_veto(
        fp_before=0.40, fp_after=0.25,
        fn_before=None, fn_before_ci=None,
        fn_after=None, fn_after_ci=None,
        fn_risk="high",
    )
    assert verdict["decision"] == "hold"
```

- [ ] **Step 13: Run the test to verify it fails**

Run: `uv run pytest tests/test_loop_guardrails.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'fn_veto'`.

- [ ] **Step 14: Implement `fn_veto`**

Add to `src/loop/guardrails.py`:

```python
from typing import Optional


def fn_veto(
    fp_before: float,
    fp_after: float,
    fn_before: Optional[float],
    fn_before_ci: Optional[tuple[float, float]],
    fn_after: Optional[float],
    fn_after_ci: Optional[tuple[float, float]],
    fn_risk: str = "low",
) -> dict:
    """Veto an FP win that worsens (or risks worsening) FN.

    Decision values:
      - "accept": FP improved and FN did not rise beyond CI noise.
      - "reject": FP improved but FN rose beyond CI noise (after-CI low > before-CI high).
      - "hold":   FN unmeasured and the change could plausibly raise FN (fn_risk != "low").

    A zero/None FN must NOT silently clear the veto — that is exactly the failure
    the spec warns about ("a zero would falsely clear the FN-veto").
    """
    fp_improved = fp_after < fp_before

    fn_unmeasured = fn_after is None or fn_after_ci is None or fn_before_ci is None
    if fn_unmeasured:
        if fn_risk != "low":
            return {
                "decision": "hold",
                "reason": "FN unmeasured and change could raise FN; holding",
            }
        return {
            "decision": "accept" if fp_improved else "reject",
            "reason": "FN unmeasured but change is low FN-risk",
        }

    # Both FN intervals known: a real rise = after's lower bound above before's upper.
    fn_rose_significantly = fn_after_ci[0] > fn_before_ci[1]
    if fp_improved and fn_rose_significantly:
        return {
            "decision": "reject",
            "reason": (
                f"FN rose beyond CI: after {fn_after_ci} > before {fn_before_ci}"
            ),
        }
    if fp_improved:
        return {"decision": "accept", "reason": "FP improved, FN stable within CI"}
    return {"decision": "reject", "reason": "no FP improvement"}
```

- [ ] **Step 15: Run the test to verify it passes**

Run: `uv run pytest tests/test_loop_guardrails.py -v`
Expected: PASS (10 passed).

- [ ] **Step 16: Commit**

```bash
git add src/loop/guardrails.py tests/test_loop_guardrails.py
git commit -m "feat(loop): add FN-veto (reject/hold on FN rise or unmeasured risk)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 17: Write the failing test for feedback-starved freeze**

Append to `tests/test_loop_guardrails.py`:

```python
def test_freeze_when_no_labels_for_three_days():
    assert guardrails.is_feedback_starved(days_since_last_label=3) is True
    assert guardrails.is_feedback_starved(days_since_last_label=4) is True


def test_not_frozen_with_recent_labels():
    assert guardrails.is_feedback_starved(days_since_last_label=2) is False
    assert guardrails.is_feedback_starved(days_since_last_label=0) is False
```

- [ ] **Step 18: Run the test to verify it fails**

Run: `uv run pytest tests/test_loop_guardrails.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'is_feedback_starved'`.

- [ ] **Step 19: Implement `is_feedback_starved`**

Add to `src/loop/guardrails.py`:

```python
def is_feedback_starved(days_since_last_label: int) -> bool:
    """Freeze tuning if no fresh human labels for >= FEEDBACK_STARVED_DAYS days."""
    return days_since_last_label >= FEEDBACK_STARVED_DAYS
```

- [ ] **Step 20: Run the test to verify it passes**

Run: `uv run pytest tests/test_loop_guardrails.py -v`
Expected: PASS (12 passed).

- [ ] **Step 21: Commit**

```bash
git add src/loop/guardrails.py tests/test_loop_guardrails.py
git commit -m "feat(loop): add feedback-starved freeze at N=3 days

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `config.py` — overlay env-file + bounds field validators

**Files:**
- Modify: `src/config.py` (sub-configs `MotionConfig` ~57, `PerformanceConfig` ~108, `SpeciesConfig` ~157; add validators on `MotionConfig`/`SpeciesConfig`)
- Test: `tests/test_config.py` (append a new `TestOverlayAndBounds` class)

- [ ] **Step 1: Write the failing test for overlay precedence**

Append to `tests/test_config.py`:

```python
class TestOverlayAndBounds:
    """ADR-004 Phase 4: deployed_config.env overlay + bounds validators."""

    def _write_overlay(self, tmp_path, body: str) -> str:
        overlay = tmp_path / "deployed_config.env"
        overlay.write_text(body)
        return str(overlay)

    def test_overlay_overrides_defaults(self, tmp_path, monkeypatch):
        # No real OS env for the key; overlay file sets it → overlay wins over default.
        monkeypatch.delenv("MOTION_THRESHOLD", raising=False)
        overlay = self._write_overlay(tmp_path, "MOTION_THRESHOLD=2500\n")
        from config import MotionConfig
        cfg = MotionConfig(_env_file=(None, overlay))
        assert cfg.threshold == 2500

    def test_os_env_overrides_overlay(self, tmp_path, monkeypatch):
        # Real OS env beats the overlay file (manual override preserved).
        monkeypatch.setenv("MOTION_THRESHOLD", "3000")
        overlay = self._write_overlay(tmp_path, "MOTION_THRESHOLD=2500\n")
        from config import MotionConfig
        cfg = MotionConfig(_env_file=(None, overlay))
        assert cfg.threshold == 3000

    def test_missing_overlay_is_safe(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MOTION_THRESHOLD", raising=False)
        missing = str(tmp_path / "does_not_exist.env")
        from config import MotionConfig
        cfg = MotionConfig(_env_file=(None, missing))
        assert cfg.threshold == 2000  # documented default
```

NOTE on test isolation: `tests/conftest.py` sets every config class's
`model_config["env_file"] = None` before each test, so these tests pass
`_env_file=(None, overlay)` explicitly to exercise the overlay precedence
without depending on the production `.env`. The `None` first element stands in
for `.env` (disabled in tests). In production the tuple is `('.env', 'experiments/deployed_config.env')`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_config.py::TestOverlayAndBounds -v`
Expected: FAIL — `test_overlay_overrides_defaults` fails because `MotionConfig` currently uses `env_file='.env'` (a string), so the tuple-form `_env_file` is the only thing making the overlay take effect; this passes only after the production default is the tuple AND pydantic layers it. The first failure will be an assertion mismatch (threshold still 2000) or that conftest set env_file to None — confirm at least one assertion fails.

- [ ] **Step 3: Add the overlay env-file to tunable sub-configs**

In `src/config.py`, change the three tunable sub-configs' `model_config` to use the overlay tuple. pydantic-settings applies env_files left→right with the LAST winning, and real OS env beats all env_files.

`MotionConfig` (line ~57):

```python
class MotionConfig(BaseSettings):
    """Motion detection configuration settings."""
    # Overlay precedence (pydantic-settings): real OS env > deployed_config.env
    # > .env > field defaults. The loop's deploy step renders deployed_config.env;
    # a human OS env var still wins. A missing overlay file is ignored.
    model_config = SettingsConfigDict(
        env_prefix='MOTION_',
        env_file=('.env', 'experiments/deployed_config.env'),
        extra='ignore',
    )
```

`PerformanceConfig` (line ~108):

```python
class PerformanceConfig(BaseSettings):
    """Performance and resource management configuration."""
    model_config = SettingsConfigDict(
        env_prefix='PERFORMANCE_',
        env_file=('.env', 'experiments/deployed_config.env'),
        extra='ignore',
    )
```

`SpeciesConfig` (line ~157):

```python
class SpeciesConfig(BaseSettings):
    """Species identification configuration."""
    model_config = SettingsConfigDict(
        env_prefix='SPECIES_',
        env_file=('.env', 'experiments/deployed_config.env'),
        extra='ignore',
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_config.py::TestOverlayAndBounds -v`
Expected: PASS (3 passed). If `test_overlay_overrides_defaults` still fails, confirm pydantic-settings layers the explicit `_env_file` tuple (it does — passing `_env_file` overrides `model_config['env_file']` entirely, so the `(None, overlay)` tuple is what's read).

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat(config): layer experiments/deployed_config.env overlay over .env

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 6: Write the failing test for bounds validators**

Append to `tests/test_config.py` inside `TestOverlayAndBounds`:

```python
    def test_out_of_range_motion_threshold_raises(self, monkeypatch):
        from pydantic import ValidationError
        from config import MotionConfig
        # 100000 is outside guardrails.BOUNDS["MOTION_THRESHOLD"] = (200, 8000).
        monkeypatch.setenv("MOTION_THRESHOLD", "100000")
        with pytest.raises(ValidationError):
            MotionConfig(_env_file=None)

    def test_in_range_motion_threshold_ok(self, monkeypatch):
        from config import MotionConfig
        monkeypatch.setenv("MOTION_THRESHOLD", "2500")
        cfg = MotionConfig(_env_file=None)
        assert cfg.threshold == 2500

    def test_out_of_range_unknown_threshold_raises(self, monkeypatch):
        from pydantic import ValidationError
        from config import SpeciesConfig
        # 1.5 is outside (0.3, 0.95).
        monkeypatch.setenv("SPECIES_UNKNOWN_THRESHOLD", "1.5")
        with pytest.raises(ValidationError):
            SpeciesConfig(_env_file=None)
```

- [ ] **Step 7: Run the test to verify it fails**

Run: `uv run pytest tests/test_config.py::TestOverlayAndBounds -v`
Expected: FAIL — `test_out_of_range_motion_threshold_raises` does NOT raise (no validator yet); 100000 is accepted.

- [ ] **Step 8: Add bounds field validators importing ranges from `guardrails`**

At the top of `src/config.py`, after the existing imports (after line ~10), add:

```python
import sys as _sys
from pathlib import Path as _Path
# guardrails lives in src/loop; ensure src is importable when config is imported
# standalone (production runs with src on the path; this is belt-and-braces).
_SRC = _Path(__file__).resolve().parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))
from loop.guardrails import BOUNDS as _BOUNDS
```

Add a validator to `MotionConfig` (inside the class, after the `motion_threshold` property block, ~line 81):

```python
    @field_validator('threshold')
    @classmethod
    def validate_threshold_bounds(cls, v):
        low, high = _BOUNDS["MOTION_THRESHOLD"]
        if not (low <= v <= high):
            raise ValueError(
                f"MOTION_THRESHOLD={v} out of allowed bounds [{low}, {high}]"
            )
        return v
```

Add a validator to `SpeciesConfig` (inside the class, after `validate_model`, ~line 174):

```python
    @field_validator('unknown_species_threshold')
    @classmethod
    def validate_unknown_threshold_bounds(cls, v):
        low, high = _BOUNDS["SPECIES_UNKNOWN_THRESHOLD"]
        if not (low <= v <= high):
            raise ValueError(
                f"SPECIES_UNKNOWN_THRESHOLD={v} out of allowed bounds [{low}, {high}]"
            )
        return v
```

- [ ] **Step 9: Run the test to verify it passes**

Run: `uv run pytest tests/test_config.py::TestOverlayAndBounds -v`
Expected: PASS (6 passed).

- [ ] **Step 10: Run the full config suite to confirm no regressions**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (all existing tests still green; the default `MOTION_THRESHOLD=2000` and `SPECIES_UNKNOWN_THRESHOLD=0.5` are inside bounds).

- [ ] **Step 11: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat(config): add bounds field validators sourced from guardrails.BOUNDS

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `state.py` — atomic JSON helper for `state.json`

**Files:**
- Create: `src/loop/state.py`
- Test: `tests/test_loop_state.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_loop_state.py`:

```python
"""Tests for src/loop/state.py — atomic state.json read/write."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loop import state as state_mod


def test_load_missing_returns_default(tmp_path):
    p = tmp_path / "state.json"
    loaded = state_mod.load_state(p)
    assert loaded == {}


def test_save_then_load_roundtrips(tmp_path):
    p = tmp_path / "state.json"
    data = {"watermark": 5, "deployed": {"MOTION_THRESHOLD": 2500}}
    state_mod.save_state(p, data)
    assert state_mod.load_state(p) == data


def test_save_is_atomic_no_temp_left_behind(tmp_path):
    p = tmp_path / "state.json"
    state_mod.save_state(p, {"a": 1})
    leftovers = [f.name for f in tmp_path.iterdir() if f.name != "state.json"]
    assert leftovers == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_loop_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'loop.state'`.

- [ ] **Step 3: Implement `state.py`**

Create `src/loop/state.py`:

```python
"""Atomic read/write of experiments/state.json.

state.json is the COMMITTED source of truth for the loop (deployed config,
active experiment id, seeded backlog, baselines, best_known_good + history,
ingest watermark, paused flag, pending-deploy stamp). Only deterministic tools
write it. Writes are atomic (temp file + os.replace) so a crashed write never
leaves a half-file that would crash the camera service.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def load_state(path: str | Path) -> dict[str, Any]:
    """Load state.json. A missing file returns {} (fresh checkout is safe)."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(path: str | Path, state: dict[str, Any]) -> None:
    """Atomically write state as pretty JSON (temp file + os.replace)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".state-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_loop_state.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/loop/state.py tests/test_loop_state.py
git commit -m "feat(loop): add atomic state.json read/write helper

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `ingest.py` — watermark + label reconciliation (read-only)

**Files:**
- Create: `src/loop/ingest.py`
- Test: `tests/test_loop_ingest.py`

Background — reconciliation precedence is **human > tier-2 > tier-1**. Tier-1 =
`detections.animals_detected` (MegaDetector: True→"animal", False→"false_positive").
Tier-2 = a `detection_feedback` row with `source='tier2'`. Human = a
`detection_feedback` row with `source='human'`; the LATEST human row wins.

- [ ] **Step 1: Write the failing test (uses DatabaseManager fixtures)**

Create `tests/test_loop_ingest.py`:

```python
"""Tests for src/loop/ingest.py — watermark + reconciliation, read-only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from config import Config
from database_manager import DatabaseManager
from loop import ingest


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A real DatabaseManager pointed at a temp SQLite file."""
    db_file = tmp_path / "detections.db"
    monkeypatch.setenv("STORAGE_DATABASE_PATH", str(db_file))
    monkeypatch.setenv("STORAGE_DATA_DIR", str(tmp_path))
    cfg = Config.create_test_config()
    return DatabaseManager(cfg)


def _add_detection(db, animals_detected):
    return db.log_detection(
        image_path="/x.jpg", motion_area=1000,
        animals_detected=animals_detected, detection_count=1 if animals_detected else 0,
        gate_would_suppress=not animals_detected,
    )


def test_tier1_label_from_animals_detected(db):
    did = _add_detection(db, animals_detected=False)
    rows = ingest.reconcile(db, since_id=0)
    row = next(r for r in rows if r["detection_id"] == did)
    assert row["tier1"] == "false_positive"
    assert row["reconciled_label"] == "false_positive"


def test_human_overrides_tier1(db):
    did = _add_detection(db, animals_detected=False)
    db.add_feedback(did, "animal", source="human")
    rows = ingest.reconcile(db, since_id=0)
    row = next(r for r in rows if r["detection_id"] == did)
    assert row["human"] == "animal"
    assert row["reconciled_label"] == "animal"  # human > tier1


def test_latest_human_row_wins(db):
    did = _add_detection(db, animals_detected=True)
    db.add_feedback(did, "animal", source="human")
    db.add_feedback(did, "false_positive", source="human")  # corrected
    rows = ingest.reconcile(db, since_id=0)
    row = next(r for r in rows if r["detection_id"] == did)
    assert row["reconciled_label"] == "false_positive"


def test_watermark_filters_old_rows(db):
    first = _add_detection(db, animals_detected=True)
    second = _add_detection(db, animals_detected=False)
    rows = ingest.reconcile(db, since_id=first)
    ids = {r["detection_id"] for r in rows}
    assert second in ids and first not in ids


def test_max_id_reports_new_watermark(db):
    _add_detection(db, animals_detected=True)
    second = _add_detection(db, animals_detected=False)
    result = ingest.ingest(db, since_id=0)
    assert result["new_watermark"] == second
    assert len(result["rows"]) == 2


def test_reconcile_does_not_write_feedback(db):
    did = _add_detection(db, animals_detected=False)
    before = db.get_feedback(did)
    ingest.reconcile(db, since_id=0)
    after = db.get_feedback(did)
    assert before == after  # pure read
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_loop_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'loop.ingest'`.

- [ ] **Step 3: Implement `ingest.py`**

Create `src/loop/ingest.py`:

```python
"""Ingest detections + feedback past a watermark and reconcile labels.

Pure READ over the SQLite DB (WAL). Never writes labels — ground truth is
append-only and owned by the feedback sidecar (anti-self-poisoning). Output is a
list of per-detection records the judgment layer consumes; the watermark is the
max detections.id seen so the next tick only processes new rows.

Reconciliation precedence: human > tier-2 > tier-1.
  tier-1 = detections.animals_detected (MegaDetector).
  tier-2 = detection_feedback row with source='tier2'.
  human  = detection_feedback row with source='human' (latest wins).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import Config
from database_manager import DatabaseManager


def _tier1_label(animals_detected) -> str | None:
    if animals_detected is None:
        return None
    return "animal" if animals_detected else "false_positive"


def _read_connection(db: DatabaseManager) -> sqlite3.Connection:
    """Open a read-only connection (URI mode) to the WAL DB."""
    uri = f"file:{db.db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def reconcile(db: DatabaseManager, since_id: int) -> list[dict]:
    """Return reconciled per-detection records for detections.id > since_id."""
    conn = _read_connection(db)
    try:
        det_rows = conn.execute(
            """
            SELECT id, animals_detected, motion_area, contour_count,
                   largest_contour_area, foreground_pixel_count, hour_of_day,
                   gate_would_suppress
            FROM detections
            WHERE id > ?
            ORDER BY id ASC
            """,
            (since_id,),
        ).fetchall()

        results: list[dict] = []
        for d in det_rows:
            det_id = d["id"]
            fb = conn.execute(
                """
                SELECT label, source, created_at, id
                FROM detection_feedback
                WHERE detection_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (det_id,),
            ).fetchall()

            tier1 = _tier1_label(d["animals_detected"])
            tier2 = None
            human = None
            for row in fb:  # ascending; last assignment of each source wins
                if row["source"] == "tier2":
                    tier2 = row["label"]
                elif row["source"] == "human":
                    human = row["label"]

            reconciled = human or tier2 or tier1
            results.append(
                {
                    "detection_id": det_id,
                    "reconciled_label": reconciled,
                    "tier1": tier1,
                    "tier2": tier2,
                    "human": human,
                    "motion_area": d["motion_area"],
                    "contour_count": d["contour_count"],
                    "largest_contour_area": d["largest_contour_area"],
                    "foreground_pixel_count": d["foreground_pixel_count"],
                    "hour_of_day": d["hour_of_day"],
                    "gate_would_suppress": bool(d["gate_would_suppress"])
                    if d["gate_would_suppress"] is not None
                    else None,
                }
            )
        return results
    finally:
        conn.close()


def ingest(db: DatabaseManager, since_id: int) -> dict:
    """Reconcile and report the advanced watermark (max detections.id seen)."""
    rows = reconcile(db, since_id)
    new_watermark = max((r["detection_id"] for r in rows), default=since_id)
    return {"rows": rows, "new_watermark": new_watermark, "count": len(rows)}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ingest detections past a watermark")
    parser.add_argument("--since-id", type=int, default=0)
    args = parser.parse_args()
    try:
        db = DatabaseManager(Config())
        result = ingest(db, args.since_id)
        print(json.dumps(result))
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": str(e)}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_loop_ingest.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/loop/ingest.py tests/test_loop_ingest.py
git commit -m "feat(loop): add read-only ingest with human>tier2>tier1 reconciliation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: `metrics.py` — Wilson CI, FN "unmeasured", idempotent `daily.csv`

**Files:**
- Create: `src/loop/metrics.py`
- Test: `tests/test_loop_metrics.py`

- [ ] **Step 1: Write the failing test for the Wilson CI math**

Create `tests/test_loop_metrics.py`:

```python
"""Tests for src/loop/metrics.py — Wilson CI, FN unmeasured, idempotent CSV."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loop import metrics


def test_wilson_ci_known_value():
    # 40 successes of 100 at 95%: classic Wilson interval ≈ (0.307, 0.500).
    low, high = metrics.wilson_ci(40, 100)
    assert abs(low - 0.3066) < 0.005
    assert abs(high - 0.5000) < 0.005
    assert low < 0.40 < high


def test_wilson_ci_zero_trials_is_full_interval():
    low, high = metrics.wilson_ci(0, 0)
    assert low == 0.0
    assert high == 1.0


def test_wilson_ci_all_successes():
    low, high = metrics.wilson_ci(10, 10)
    assert high == 1.0
    assert 0.0 < low < 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_loop_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'loop.metrics'`.

- [ ] **Step 3: Implement `wilson_ci`**

Create `src/loop/metrics.py`:

```python
"""Compute paired FP/FN with Wilson 95% CIs and append to daily.csv.

FP comes from the labeled/captured set (triggers marked false_positive ÷ labeled
triggers). FN comes from the timelapse audit channel; until a timelapse detector
pass exists, FN is reported as "unmeasured" (NOT 0 — a zero would falsely clear
the FN-veto). daily.csv is idempotent per date: a re-run for the same date
overwrites that date's row, never duplicates.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# z for 95% two-sided.
_Z = 1.959963984540054


def wilson_ci(successes: int, trials: int) -> tuple[float, float]:
    """Wilson score interval (95%). trials==0 → (0.0, 1.0) (no information)."""
    if trials <= 0:
        return (0.0, 1.0)
    p = successes / trials
    z = _Z
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    margin = (
        z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    ) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_loop_metrics.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/loop/metrics.py tests/test_loop_metrics.py
git commit -m "feat(loop): add Wilson 95% CI helper

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 6: Write the failing test for `compute_metrics` (FP + FN unmeasured)**

Append to `tests/test_loop_metrics.py`:

```python
def test_compute_metrics_fp_from_labeled_rows():
    # 3 of 5 labeled triggers are false_positive → fp_rate 0.6.
    rows = [
        {"reconciled_label": "false_positive"},
        {"reconciled_label": "false_positive"},
        {"reconciled_label": "false_positive"},
        {"reconciled_label": "animal"},
        {"reconciled_label": "animal"},
        {"reconciled_label": None},  # unlabeled → excluded from FP denominator
    ]
    m = metrics.compute_metrics(rows, fn_audit=None)
    assert m["labeled_triggers"] == 5
    assert abs(m["fp_rate"] - 0.6) < 1e-9
    assert m["fp_ci"][0] < 0.6 < m["fp_ci"][1]


def test_compute_metrics_fn_unmeasured_when_no_audit():
    rows = [{"reconciled_label": "animal"}]
    m = metrics.compute_metrics(rows, fn_audit=None)
    assert m["fn_rate"] == "unmeasured"
    assert m["fn_ci"] is None


def test_compute_metrics_fn_measured_when_audit_present():
    rows = [{"reconciled_label": "animal"}]
    # 2 missed animals of 8 animal-present frames → fn_rate 0.25.
    m = metrics.compute_metrics(rows, fn_audit={"missed": 2, "animal_frames": 8})
    assert abs(m["fn_rate"] - 0.25) < 1e-9
    assert m["fn_ci"] is not None
```

- [ ] **Step 7: Run the test to verify it fails**

Run: `uv run pytest tests/test_loop_metrics.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'compute_metrics'`.

- [ ] **Step 8: Implement `compute_metrics`**

Add to `src/loop/metrics.py`:

```python
def compute_metrics(rows: list[dict], fn_audit: Optional[dict]) -> dict:
    """Paired FP/FN with Wilson CIs over reconciled rows.

    FP denominator = rows with a non-None reconciled_label (labeled triggers).
    FN is "unmeasured" unless fn_audit={"missed": int, "animal_frames": int} is
    supplied by a timelapse detector pass (NOT implemented this build).
    """
    labeled = [r for r in rows if r.get("reconciled_label") is not None]
    fp_count = sum(1 for r in labeled if r["reconciled_label"] == "false_positive")
    fp_rate = (fp_count / len(labeled)) if labeled else 0.0
    fp_ci = wilson_ci(fp_count, len(labeled))

    if fn_audit and fn_audit.get("animal_frames", 0) > 0:
        missed = fn_audit["missed"]
        frames = fn_audit["animal_frames"]
        fn_rate: object = missed / frames
        fn_ci: object = wilson_ci(missed, frames)
    else:
        fn_rate = "unmeasured"
        fn_ci = None

    return {
        "labeled_triggers": len(labeled),
        "total_triggers": len(rows),
        "fp_count": fp_count,
        "fp_rate": fp_rate,
        "fp_ci": fp_ci,
        "fn_rate": fn_rate,
        "fn_ci": fn_ci,
    }
```

- [ ] **Step 9: Run the test to verify it passes**

Run: `uv run pytest tests/test_loop_metrics.py -v`
Expected: PASS (6 passed).

- [ ] **Step 10: Commit**

```bash
git add src/loop/metrics.py tests/test_loop_metrics.py
git commit -m "feat(loop): compute paired FP/FN metrics with FN 'unmeasured' default

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 11: Write the failing test for idempotent `daily.csv`**

Append to `tests/test_loop_metrics.py`:

```python
def test_daily_csv_appends_one_row_per_date(tmp_path):
    csv_path = tmp_path / "daily.csv"
    m = {"labeled_triggers": 5, "total_triggers": 6, "fp_count": 3,
         "fp_rate": 0.6, "fp_ci": (0.3, 0.8), "fn_rate": "unmeasured", "fn_ci": None}
    metrics.append_daily(csv_path, "2026-06-10", m)
    metrics.append_daily(csv_path, "2026-06-11", m)
    with open(csv_path) as f:
        data_rows = list(csv.DictReader(f))
    assert len(data_rows) == 2
    assert {r["date"] for r in data_rows} == {"2026-06-10", "2026-06-11"}


def test_daily_csv_rerun_same_date_overwrites(tmp_path):
    csv_path = tmp_path / "daily.csv"
    m1 = {"labeled_triggers": 5, "total_triggers": 6, "fp_count": 3,
          "fp_rate": 0.6, "fp_ci": (0.3, 0.8), "fn_rate": "unmeasured", "fn_ci": None}
    m2 = {"labeled_triggers": 10, "total_triggers": 11, "fp_count": 2,
          "fp_rate": 0.2, "fp_ci": (0.05, 0.5), "fn_rate": "unmeasured", "fn_ci": None}
    metrics.append_daily(csv_path, "2026-06-10", m1)
    metrics.append_daily(csv_path, "2026-06-10", m2)  # same date re-run
    with open(csv_path) as f:
        data_rows = list(csv.DictReader(f))
    assert len(data_rows) == 1
    assert data_rows[0]["fp_rate"] == "0.2"
```

- [ ] **Step 12: Run the test to verify it fails**

Run: `uv run pytest tests/test_loop_metrics.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'append_daily'`.

- [ ] **Step 13: Implement `append_daily`**

Add to `src/loop/metrics.py`:

```python
_CSV_FIELDS = [
    "date", "total_triggers", "labeled_triggers", "fp_count", "fp_rate",
    "fp_ci_low", "fp_ci_high", "fn_rate", "fn_ci_low", "fn_ci_high",
]


def _row_for_csv(date: str, m: dict) -> dict:
    fp_ci = m["fp_ci"]
    fn_ci = m["fn_ci"]
    return {
        "date": date,
        "total_triggers": m["total_triggers"],
        "labeled_triggers": m["labeled_triggers"],
        "fp_count": m["fp_count"],
        "fp_rate": m["fp_rate"],
        "fp_ci_low": fp_ci[0],
        "fp_ci_high": fp_ci[1],
        "fn_rate": m["fn_rate"],
        "fn_ci_low": fn_ci[0] if fn_ci else "",
        "fn_ci_high": fn_ci[1] if fn_ci else "",
    }


def append_daily(csv_path, date: str, m: dict) -> None:
    """Write one row per date; a re-run for the same date overwrites that row."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists():
        with open(path, newline="") as f:
            existing = [r for r in csv.DictReader(f) if r["date"] != date]
    existing.append(_row_for_csv(date, m))
    existing.sort(key=lambda r: r["date"])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(existing)
```

Also add the module CLI at the bottom of `src/loop/metrics.py`:

```python
def main() -> None:
    import argparse
    import json
    import sys as _sys
    from pathlib import Path as _Path
    from datetime import date as _date

    _src = _Path(__file__).resolve().parent.parent
    if str(_src) not in _sys.path:
        _sys.path.insert(0, str(_src))
    from config import Config
    from database_manager import DatabaseManager
    from loop import ingest
    from loop import state as state_mod

    parser = argparse.ArgumentParser(description="Compute daily FP/FN metrics")
    parser.add_argument("--state", default="experiments/state.json")
    parser.add_argument("--csv", default="experiments/metrics/daily.csv")
    parser.add_argument("--date", default=_date.today().isoformat())
    args = parser.parse_args()
    try:
        st = state_mod.load_state(args.state)
        watermark = int(st.get("watermark", 0))
        db = DatabaseManager(Config())
        ing = ingest.ingest(db, watermark)
        m = compute_metrics(ing["rows"], fn_audit=None)
        append_daily(args.csv, args.date, m)
        print(json.dumps({"date": args.date, "metrics": _jsonable(m)}))
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": str(e)}))
        raise SystemExit(1)


def _jsonable(m: dict) -> dict:
    out = dict(m)
    out["fp_ci"] = list(m["fp_ci"])
    out["fn_ci"] = list(m["fn_ci"]) if m["fn_ci"] else None
    return out


if __name__ == "__main__":
    main()
```

- [ ] **Step 14: Run the test to verify it passes**

Run: `uv run pytest tests/test_loop_metrics.py -v`
Expected: PASS (8 passed).

- [ ] **Step 15: Commit**

```bash
git add src/loop/metrics.py tests/test_loop_metrics.py
git commit -m "feat(loop): idempotent daily.csv writer + metrics CLI

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: `replay.py` — Layer-A seam (STUB)

**Files:**
- Create: `src/loop/replay.py`
- Test: `tests/test_loop_replay.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_loop_replay.py`:

```python
"""Tests for src/loop/replay.py — STUB Layer-A seam (interface shape stable)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loop import replay


def test_replay_returns_skipped():
    result = replay.replay(candidate_config={"MOTION_THRESHOLD": 2500}, labeled_set=[])
    assert result.status == "skipped"
    assert result.reason == "not implemented"
    assert result.metrics is None


def test_replay_result_is_dataclass_with_stable_fields():
    result = replay.replay(candidate_config={}, labeled_set=[])
    # Interface contract the rest of the system relies on.
    assert hasattr(result, "status")
    assert hasattr(result, "reason")
    assert hasattr(result, "metrics")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_loop_replay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'loop.replay'`.

- [ ] **Step 3: Implement the stub**

Create `src/loop/replay.py`:

```python
"""Layer-A offline replay seam (ADR-004 Phase 3) — STUB for this build.

When filled in, replay() will re-run MegaDetector + classifier over saved
high-res images with a candidate config and score against labels, giving offline
evidence before a live deploy. Until then it returns status="skipped", which the
rest of the system treats as "no offline evidence available — this experiment
cannot be validated offline yet" (replay-gated experiments stay parked).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class ReplayResult:
    status: str            # "ok" | "skipped" | "error"
    reason: str
    metrics: Optional[dict] = None


def replay(candidate_config: dict, labeled_set: list) -> ReplayResult:
    """STUB: returns skipped. Real implementation is a later task."""
    return ReplayResult(status="skipped", reason="not implemented", metrics=None)


def main() -> None:
    result = replay(candidate_config={}, labeled_set=[])
    print(json.dumps({"status": result.status, "reason": result.reason,
                      "metrics": result.metrics}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_loop_replay.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/loop/replay.py tests/test_loop_replay.py
git commit -m "feat(loop): add replay.py Layer-A seam as a skipped stub

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: `report.py` — Telegram daily summary + heartbeat (send-only)

**Files:**
- Create: `src/loop/report.py`
- Test: `tests/test_loop_report.py`

Note: `report.py` must NOT poll (no `getUpdates`) — that would conflict with the
feedback sidecar. It only *sends* via `NotificationService.send_text_message`.
`/pause` and `/rollback` are handled in the sidecar (Task 11), never here.

- [ ] **Step 1: Write the failing test for summary rendering**

Create `tests/test_loop_report.py`:

```python
"""Tests for src/loop/report.py — summary + heartbeat text, paused suppression."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loop import report


def _metrics(fp_rate=0.38, fn="unmeasured"):
    return {
        "date": "2026-06-10",
        "total_triggers": 42,
        "labeled_triggers": 30,
        "fp_count": 11,
        "fp_rate": fp_rate,
        "fp_ci": (0.30, 0.46),
        "fn_rate": fn,
        "fn_ci": None if fn == "unmeasured" else (0.05, 0.18),
    }


def test_summary_includes_fp_and_fn():
    text = report.render_summary(
        metrics=_metrics(), state={"active_experiment_id": 1, "paused": False},
        active_experiment={"slug": "notification-gate-live", "status": "running"},
    )
    assert "FP" in text
    assert "FN" in text
    assert "unmeasured" in text
    assert "38%" in text or "0.38" in text
    assert "notification-gate-live" in text


def test_summary_suppresses_tuning_lines_when_paused():
    text = report.render_summary(
        metrics=_metrics(), state={"active_experiment_id": 1, "paused": True},
        active_experiment={"slug": "notification-gate-live", "status": "running"},
    )
    assert "PAUSED" in text
    # When paused we do not advertise active tuning as if it's progressing.
    assert "tuning frozen" in text.lower() or "paused" in text.lower()


def test_heartbeat_text_mentions_alive_and_timestamp():
    text = report.render_heartbeat(last_tick_iso="2026-06-10T23:14:00")
    assert "alive" in text.lower()
    assert "2026-06-10T23:14:00" in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_loop_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'loop.report'`.

- [ ] **Step 3: Implement render functions**

Create `src/loop/report.py`:

```python
"""Build + send the Telegram daily summary (FP+FN) and heartbeat.

Send-only: reuses NotificationService (no getUpdates → no conflict with the
feedback sidecar). The daily summary IS the deadman ping; render_heartbeat is the
terse fallback for no-op resume ticks. /pause and /rollback are handled by the
feedback sidecar, not here.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _fmt_pct(x) -> str:
    if isinstance(x, (int, float)):
        return f"{x * 100:.0f}%"
    return str(x)


def render_summary(metrics: dict, state: dict, active_experiment: dict) -> str:
    """Daily summary: FP + FN, active experiment, paused banner."""
    paused = bool(state.get("paused", False))
    lines = [f"🦊 Wildlife loop — daily summary {metrics.get('date', '')}"]
    if paused:
        lines.append("⏸️ PAUSED — tuning frozen until resumed")

    fp_ci = metrics["fp_ci"]
    lines.append(
        f"FP rate: {_fmt_pct(metrics['fp_rate'])} "
        f"(95% CI {_fmt_pct(fp_ci[0])}–{_fmt_pct(fp_ci[1])}, "
        f"{metrics['fp_count']}/{metrics['labeled_triggers']} labeled)"
    )
    fn = metrics["fn_rate"]
    if fn == "unmeasured":
        lines.append("FN rate: unmeasured (timelapse audit not yet wired)")
    else:
        fn_ci = metrics["fn_ci"]
        lines.append(
            f"FN rate: {_fmt_pct(fn)} (95% CI {_fmt_pct(fn_ci[0])}–{_fmt_pct(fn_ci[1])})"
        )

    lines.append(f"Triggers tonight: {metrics['total_triggers']}")

    if active_experiment and not paused:
        lines.append(
            f"Active experiment: {active_experiment['slug']} "
            f"[{active_experiment['status']}]"
        )
    elif not paused:
        lines.append("Active experiment: none")
    return "\n".join(lines)


def render_heartbeat(last_tick_iso: str) -> str:
    """Terse 'alive' ping for a no-op resume tick (no full summary)."""
    return f"💓 Wildlife loop alive, last tick OK @ {last_tick_iso}"


async def send(text: str) -> bool:
    """Send `text` to the configured Telegram chat via NotificationService."""
    from config import Config
    from notification_service import NotificationService

    service = NotificationService(Config())
    return await service.send_text_message(text)


def main() -> None:
    import argparse
    from loop import state as state_mod

    parser = argparse.ArgumentParser(description="Send the loop's Telegram report")
    parser.add_argument("--mode", choices=["summary", "heartbeat"], default="summary")
    parser.add_argument("--state", default="experiments/state.json")
    parser.add_argument("--last-tick", default="")
    args = parser.parse_args()
    try:
        st = state_mod.load_state(args.state)
        if args.mode == "heartbeat":
            text = render_heartbeat(args.last_tick)
        else:
            metrics = st.get("last_metrics")
            if metrics is None:
                raise RuntimeError("state.json has no last_metrics to report")
            # Tuples were serialised to lists in state.json; restore CI shape.
            metrics = dict(metrics)
            metrics["fp_ci"] = tuple(metrics["fp_ci"])
            metrics["fn_ci"] = tuple(metrics["fn_ci"]) if metrics["fn_ci"] else None
            active_id = st.get("active_experiment_id")
            active = next(
                (e for e in st.get("backlog", []) if e.get("id") == active_id), {}
            )
            text = render_summary(metrics, st, active)
        ok = asyncio.run(send(text))
        print(json.dumps({"sent": ok, "mode": args.mode}))
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": str(e)}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_loop_report.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/loop/report.py tests/test_loop_report.py
git commit -m "feat(loop): add send-only daily summary + heartbeat report

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: `deploy.py` — validate, write state, render env, stamp restart, rollback

**Files:**
- Create: `src/loop/deploy.py`
- Test: `tests/test_loop_deploy.py`

`state.json` schema (written here, read by metrics/report; values are JSON, so
tuples become lists):

```json
{
  "watermark": 0,
  "paused": false,
  "deployed": { "MOTION_THRESHOLD": 2000 },
  "best_known_good": { "MOTION_THRESHOLD": 2000 },
  "history": [ { "config": { "MOTION_THRESHOLD": 2000 }, "at": "2026-06-10T22:00:00" } ],
  "pending_restart_at": "2026-06-11T04:30:00+02:00",
  "active_experiment_id": 1,
  "backlog": [],
  "baselines": { "volume_per_night": 40.0 },
  "last_metrics": null
}
```

- [ ] **Step 1: Write the failing test for bounds rejection (no write)**

Create `tests/test_loop_deploy.py`:

```python
"""Tests for src/loop/deploy.py — validate, write, render env, rollback."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from loop import deploy
from loop import state as state_mod


@pytest.fixture
def paths(tmp_path):
    return {
        "state": tmp_path / "state.json",
        "env": tmp_path / "deployed_config.env",
    }


def _seed_state(paths, deployed=None):
    deployed = deployed or {"MOTION_THRESHOLD": 2000}
    state_mod.save_state(paths["state"], {
        "watermark": 0, "paused": False,
        "deployed": deployed, "best_known_good": deployed,
        "history": [], "pending_restart_at": None,
        "active_experiment_id": 1, "backlog": [], "baselines": {},
        "last_metrics": None,
    })


def test_out_of_bounds_rejected_no_write(paths):
    _seed_state(paths)
    before = paths["state"].read_text()
    with pytest.raises(ValueError, match="out of bounds"):
        deploy.deploy(
            {"MOTION_THRESHOLD": 999999},
            state_path=paths["state"], env_path=paths["env"],
            restart_at="2026-06-11T04:30:00+02:00",
        )
    assert paths["state"].read_text() == before  # unchanged
    assert not paths["env"].exists()             # no env rendered
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_loop_deploy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'loop.deploy'`.

- [ ] **Step 3: Implement `deploy()` validation + state + env render + stamp**

Create `src/loop/deploy.py`:

```python
"""Deploy a candidate config delta (the only writer of live config).

Steps, in order:
  1. Validate every key/value against guardrails.BOUNDS (reject out-of-range,
     no write).
  2. Update state.json: new deployed = old deployed merged with delta; push the
     previous deployed onto history; keep best_known_good.
  3. Render experiments/deployed_config.env from state.json.deployed.
  4. Stamp pending_restart_at (computed by the caller, ~60 min pre-sunrise); the
     wildlife-deploy.timer applies it pre-sunrise.

rollback() restores best_known_good, re-renders the env, and stamps a restart.
Writes go through state.save_state (atomic temp+rename) and an atomic env render.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from loop import guardrails
from loop import state as state_mod


def _render_env(deployed: dict, env_path) -> None:
    """Atomically write KEY=value lines from the deployed config."""
    p = Path(env_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{k}={v}\n" for k, v in sorted(deployed.items()))
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".env-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("# Rendered from experiments/state.json — do not edit by hand.\n")
            f.write(body)
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def deploy(delta: dict, state_path, env_path, restart_at: str) -> dict:
    """Validate `delta`, merge into deployed, render env, stamp restart."""
    for key, value in delta.items():
        guardrails.validate_param(key, value)  # raises ValueError out-of-bounds

    st = state_mod.load_state(state_path)
    previous = dict(st.get("deployed", {}))
    new_deployed = {**previous, **delta}

    st["history"] = st.get("history", []) + [
        {"config": previous, "replaced_at": restart_at}
    ]
    st["deployed"] = new_deployed
    st["pending_restart_at"] = restart_at
    state_mod.save_state(state_path, st)

    _render_env(new_deployed, env_path)
    return {"deployed": new_deployed, "pending_restart_at": restart_at}


def rollback(state_path, env_path, restart_at: str) -> dict:
    """Restore best_known_good → re-render env → stamp restart."""
    st = state_mod.load_state(state_path)
    bkg = dict(st.get("best_known_good", {}))
    st["history"] = st.get("history", []) + [
        {"config": dict(st.get("deployed", {})), "rolled_back_at": restart_at}
    ]
    st["deployed"] = bkg
    st["pending_restart_at"] = restart_at
    state_mod.save_state(state_path, st)
    _render_env(bkg, env_path)
    return {"deployed": bkg, "rolled_back": True, "pending_restart_at": restart_at}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Deploy a config delta or rollback")
    parser.add_argument("--state", default="experiments/state.json")
    parser.add_argument("--env", default="experiments/deployed_config.env")
    parser.add_argument("--restart-at", required=True)
    parser.add_argument("--delta", help='JSON object, e.g. {"MOTION_THRESHOLD": 2500}')
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    try:
        if args.rollback:
            result = rollback(args.state, args.env, args.restart_at)
        else:
            delta = json.loads(args.delta) if args.delta else {}
            result = deploy(delta, args.state, args.env, args.restart_at)
        print(json.dumps(result))
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": str(e)}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_loop_deploy.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/loop/deploy.py tests/test_loop_deploy.py
git commit -m "feat(loop): add deploy with bounds validation, state update, env render

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 6: Write the failing tests for env render, history, best_known_good, rollback**

Append to `tests/test_loop_deploy.py`:

```python
def test_valid_deploy_renders_env_and_updates_state(paths):
    _seed_state(paths)
    result = deploy.deploy(
        {"MOTION_THRESHOLD": 2500},
        state_path=paths["state"], env_path=paths["env"],
        restart_at="2026-06-11T04:30:00+02:00",
    )
    assert result["deployed"]["MOTION_THRESHOLD"] == 2500
    env_text = paths["env"].read_text()
    assert "MOTION_THRESHOLD=2500" in env_text
    st = state_mod.load_state(paths["state"])
    assert st["deployed"]["MOTION_THRESHOLD"] == 2500
    assert st["pending_restart_at"] == "2026-06-11T04:30:00+02:00"


def test_deploy_pushes_previous_onto_history(paths):
    _seed_state(paths, deployed={"MOTION_THRESHOLD": 2000})
    deploy.deploy(
        {"MOTION_THRESHOLD": 2500},
        state_path=paths["state"], env_path=paths["env"],
        restart_at="2026-06-11T04:30:00+02:00",
    )
    st = state_mod.load_state(paths["state"])
    assert any(h["config"].get("MOTION_THRESHOLD") == 2000 for h in st["history"])


def test_best_known_good_preserved_across_deploy(paths):
    _seed_state(paths, deployed={"MOTION_THRESHOLD": 2000})
    deploy.deploy(
        {"MOTION_THRESHOLD": 2500},
        state_path=paths["state"], env_path=paths["env"],
        restart_at="2026-06-11T04:30:00+02:00",
    )
    st = state_mod.load_state(paths["state"])
    assert st["best_known_good"]["MOTION_THRESHOLD"] == 2000  # untouched by deploy


def test_rollback_restores_best_known_good(paths):
    _seed_state(paths, deployed={"MOTION_THRESHOLD": 2000})
    deploy.deploy(
        {"MOTION_THRESHOLD": 2500},
        state_path=paths["state"], env_path=paths["env"],
        restart_at="2026-06-11T04:30:00+02:00",
    )
    result = deploy.rollback(
        state_path=paths["state"], env_path=paths["env"],
        restart_at="2026-06-11T05:00:00+02:00",
    )
    assert result["deployed"]["MOTION_THRESHOLD"] == 2000
    assert "MOTION_THRESHOLD=2000" in paths["env"].read_text()
    st = state_mod.load_state(paths["state"])
    assert st["deployed"]["MOTION_THRESHOLD"] == 2000
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_loop_deploy.py -v`
Expected: PASS (5 passed). All behaviour is already implemented in Step 3; these tests lock the contract.

- [ ] **Step 8: Commit**

```bash
git add tests/test_loop_deploy.py
git commit -m "test(loop): lock deploy env-render, history, best_known_good, rollback

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: `experiments/` scaffold (committed to `main`)

**Files:**
- Create: `experiments/state.json`, `experiments/PROTOCOL.md`, `experiments/loop.md`, `experiments/JOURNAL.md`, `experiments/LEARNINGS.md`, `experiments/runs/0001-notification-gate-live.md`, `experiments/metrics/.gitkeep`, `experiments/gold/.gitkeep`
- Modify: `.gitignore` (add `experiments/deployed_config.env`)

- [ ] **Step 1: Add the gitignore line for the rendered overlay**

Edit `.gitignore`, append:

```
# Loop overlay is rendered from experiments/state.json (the committed source of truth)
experiments/deployed_config.env
```

- [ ] **Step 2: Create `experiments/state.json` with the seeded backlog**

Create `experiments/state.json`:

```json
{
  "watermark": 0,
  "paused": false,
  "deployed": {},
  "best_known_good": {},
  "history": [],
  "pending_restart_at": null,
  "active_experiment_id": null,
  "last_metrics": null,
  "baselines": {
    "volume_per_night": 0.0
  },
  "backlog": [
    {
      "id": 1,
      "slug": "notification-gate-live",
      "validation": "live",
      "status": "proposed",
      "hypothesis": "Flip the shadow gate to route no-animal triggers to a review channel; cuts FP without raising FN."
    },
    {
      "id": 2,
      "slug": "unknown-species-threshold",
      "validation": "parked",
      "status": "parked",
      "hypothesis": "Raise SPECIES_UNKNOWN_THRESHOLD 0.5 -> 0.75. Replay-gated; parked until replay.py is real."
    },
    {
      "id": 3,
      "slug": "roi-masking",
      "validation": "live",
      "status": "proposed",
      "hypothesis": "Restrict motion ROI to suppress edge vegetation; incremental FP reduction."
    }
  ]
}
```

- [ ] **Step 3: Create the example experiment record documenting the front-matter schema**

Create `experiments/runs/0001-notification-gate-live.md`:

```markdown
---
id: 1
slug: notification-gate-live
status: proposed          # proposed | running | concluded | rolled_back | parked
validation: live          # live | replay | parked
hypothesis: "Route no-animal triggers to a review channel; cuts FP w/o raising FN"
param_delta: { notification_gate: "shadow -> live" }
predicted_effect: { fp_rate: "-15pp", fn_risk: "low" }
created: 2026-06-08
started: null
concluded: null
decision: null            # keep | rollback | inconclusive
baseline: { fp_rate: null, fp_ci: null, fn: unmeasured }
result:   { fp_rate: null, fp_ci: null, fn: null }
confidence: null
---

## Hypothesis & method

The shadow notification gate already records, per detection, whether it *would*
suppress (no animal found by MegaDetector) via `detections.gate_would_suppress`.
This experiment flips the gate from shadow to live: no-animal triggers route to a
review channel instead of the main channel. We expect a large FP reduction in the
main channel with low FN risk (MegaDetector misses are rare and caught by the
review channel). Validation is live (no offline replay this build); FN-veto
applies — because FN is currently *unmeasured*, a change that could raise FN is
HELD rather than auto-deployed.

## Daily observations

(Append-only. The agent records each night's FP/FN deltas and CI overlap here.
Never rewrite prior observations — anti-self-poisoning.)

## Decision & rationale

(Filled in when the experiment concludes: keep / rollback / inconclusive, with
the CI-based reasoning and the FN-veto outcome.)
```

- [ ] **Step 4: Create `experiments/JOURNAL.md`**

Create `experiments/JOURNAL.md`:

```markdown
# Loop Journal

Thin, append-only chronological index. One line per event, linking run files.
Cross-experiment notes live here; per-experiment detail lives in `runs/NNNN-<slug>.md`.

- 2026-06-08 — Notebook scaffolded. Seeded backlog: #1 notification-gate-live (live),
  #2 unknown-species-threshold (parked/replay), #3 roi-masking (live).
```

- [ ] **Step 5: Create `experiments/LEARNINGS.md`**

Create `experiments/LEARNINGS.md`:

```markdown
# Learnings

Distilled, firm conclusions (semantic memory) — only things proven on a real
corpus with CI support. Keep terse; link the run file that established each.

(empty — no concluded experiments yet)
```

- [ ] **Step 6: Create `experiments/PROTOCOL.md` (the SOP)**

Create `experiments/PROTOCOL.md`:

```markdown
# PROTOCOL — Autonomous Tuning Loop SOP

Read this FIRST every `/loop` tick. You (the judgment layer) reconstruct all
state from this git notebook; there is no hidden stage machine.

## When to spend tokens vs invoke Python
- Deterministic work (SQL reads, metric math, config writes, env render, restarts,
  Telegram sends) is done by `src/loop/*.py`. INVOKE them, read their JSON.
- Spend tokens only on: tier-2 adjudication of ambiguous crops, experiment design,
  self-audit, journaling.

## Daily cycle (one nightly run, resumable)
1. **Gate** — `SunChecker` says night? `state.json` says tonight's run already done?
   If not night or already done → stop (cheap no-op; send heartbeat only if asked).
2. **Ingest** — `python -m loop.ingest --since-id <watermark>`; reconcile labels.
3. **Label** — adjudicate ambiguous crops (tier-2); append to `gold/`. Never
   UPDATE/DELETE existing labels.
4. **Measure** — `python -m loop.metrics`; paired FP/FN + CIs → `metrics/daily.csv`.
5. **Check** — does the active experiment's prediction still hold (CI-based)? done?
6. **Self-audit (cadence)** — auto-labels vs the day's human labels; re-check past
   wins on the larger corpus; note confidence in `runs/`.
7. **Decide** — keep / rollback; if concluded, pick next from backlog / OFAT within
   bounds. Respect freeze + one-experiment-at-a-time + `paused`.
8. **Validate** — Layer A = `python -m loop.replay` (STUB → "skipped"). Layer B =
   bounds + predicted live effect. FN-veto: reject FP wins that worsen (or risk, if
   FN unmeasured) FN.
9. **Deploy** — `python -m loop.deploy --delta '{...}' --restart-at <pre-sunrise>`;
   writes state.json + renders env + stamps the restart.
10. **Record** — update `runs/NNNN-<slug>.md` (front matter + observations), append
    a `JOURNAL.md` line, update `state.json` pointers.
11. **Report** — `python -m loop.report --mode summary`; commit + push.

Budget exhausted mid-run → the next 2h tick resumes from committed state.

## Guardrail contract (hard rules)
- BOUNDS in `src/loop/guardrails.py` are enforced by the system (config validators
  + deploy). Never propose out-of-range values.
- FN-veto: an FP win with an FN rise beyond CI is rejected; if FN is unmeasured and
  the change could raise FN, HOLD.
- Volume collapse/explosion vs baseline → rollback.
- Feedback-starved freeze: no human labels for 3 days → freeze, hold best_known_good.
- One active experiment at a time. Respect `state.json.paused`.

## Anti-self-poisoning & self-skepticism
- Ground truth is append-only; never rewrite `detection_feedback`, `gold/`, or prior
  `runs/` observations.
- Treat your own auto-labels with suspicion; reconcile against human labels in the
  self-audit step before trusting a "win."
```

- [ ] **Step 7: Create `experiments/loop.md` (the per-tick prompt)**

Create `experiments/loop.md`:

```markdown
# /loop — per-tick prompt

You are the judgment layer of the wildlife detection-tuning loop. Run on Opus,
every 2h, gated to the night window.

1. Read `experiments/PROTOCOL.md` fully (it is the SOP).
2. **Night gate:** decide if it is night and whether tonight's run is already
   complete (see `state.json` — a `last_metrics` row dated today means done).
   If gated out, optionally `python -m loop.report --mode heartbeat
   --last-tick <now>` and STOP.
3. Run the deterministic CLIs in order: `loop.ingest` → adjudicate → `loop.metrics`
   → `loop.replay` (skipped) → decide → `loop.deploy` (only if validated and not
   FN-vetoed/paused/frozen).
4. Write the notebook: update the active `runs/NNNN-<slug>.md`, append to
   `JOURNAL.md`, update `state.json` (watermark, active_experiment_id, last_metrics,
   pending_restart_at).
5. `python -m loop.report --mode summary`, then commit + push the notebook.

Never poll Telegram here — the feedback sidecar owns getUpdates. `/pause` and
`/rollback` arrive via the sidecar and land in `state.json` / a rollback; honor
`state.json.paused` on the next tick.
```

- [ ] **Step 8: Create `.gitkeep` files for empty dirs**

Create `experiments/metrics/.gitkeep` (empty file) and `experiments/gold/.gitkeep` (empty file) so the directories are committed.

- [ ] **Step 9: Verify state.json is valid JSON and load_state reads it**

Run:
```bash
uv run python -c "import sys; sys.path.insert(0,'src'); from loop.state import load_state; s=load_state('experiments/state.json'); print(len(s['backlog']), 'backlog entries')"
```
Expected: `3 backlog entries`.

- [ ] **Step 10: Commit**

```bash
git add experiments/ .gitignore
git commit -m "feat(loop): scaffold experiments/ notebook with seeded backlog

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 10: systemd units (committed, mirror existing service style)

**Files:**
- Create: `wildlife-loop.service`, `wildlife-deploy.service`, `wildlife-deploy.timer`

- [ ] **Step 1: Create `wildlife-loop.service`**

Mirrors `wildlife-feedback.service` (User=daniel, /home/daniel paths, journal
logging). Keeps the on-Pi `claude`/`/loop` session alive and re-arms the 7-day
`/loop` expiry across reboots.

Create `wildlife-loop.service`:

```ini
[Unit]
Description=Wildlife Autonomous Tuning Loop (on-Pi Claude /loop session)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=daniel
WorkingDirectory=/home/daniel/animal_tracker
Environment="PATH=/home/daniel/.local/bin:/usr/bin:/bin"
# Keep-alive wrapper that (re-)launches `claude` and arms `/loop` every 2h,
# night-gated. Re-arms the 7-day /loop expiry on restart/reboot.
ExecStart=/home/daniel/.local/bin/claude --continue --dangerously-skip-permissions "/loop"
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

# Resource limits
Nice=10

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 2: Create `wildlife-deploy.service` (oneshot)**

Pre-sunrise oneshot: checks `state.json` for a pending deploy and, if its
`pending_restart_at` is due, restarts the camera service. Keeps the restart off
the `/loop` session's critical path.

Create `wildlife-deploy.service`:

```ini
[Unit]
Description=Wildlife pre-sunrise deploy (apply pending config + restart camera)
After=network.target

[Service]
Type=oneshot
User=daniel
WorkingDirectory=/home/daniel/animal_tracker
Environment="PATH=/home/daniel/.local/bin:/usr/bin:/bin"
# Apply any pending deploy stamped in experiments/state.json, then restart the
# camera so it reloads experiments/deployed_config.env on startup. The wrapper
# exits 0 with no-op if no deploy is due.
ExecStart=/home/daniel/.local/bin/uv run python -m loop.apply_pending_deploy
StandardOutput=journal
StandardError=journal
```

NOTE: `loop.apply_pending_deploy` is a thin entrypoint added in Step 4 below;
the camera restart itself uses `systemctl restart wildlife-camera.service` issued
from that entrypoint (daniel must have a polkit rule or run the service as a user
unit; document in README — out of scope to configure here).

- [ ] **Step 3: Create `wildlife-deploy.timer`**

Fires daily; the oneshot itself checks whether a deploy is actually due (it is a
no-op otherwise). A fixed early-morning calendar time keeps it simple; the
`/loop` step stamps `pending_restart_at` ~60 min pre-sunrise and the entrypoint
honors it.

Create `wildlife-deploy.timer`:

```ini
[Unit]
Description=Run the pre-sunrise wildlife deploy check daily

[Timer]
# Fire well before the earliest plausible sunrise; the service is a no-op unless
# state.json has a pending deploy whose restart time has arrived.
OnCalendar=*-*-* 03:30:00
Persistent=true

[Install]
WantedBy=timers.target
```

- [ ] **Step 4: Create the `apply_pending_deploy` entrypoint + its test**

Create `tests/test_loop_apply_pending.py`:

```python
"""Tests for loop.apply_pending_deploy — only restarts when a deploy is due."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loop import apply_pending_deploy as app
from loop import state as state_mod


def test_no_restart_when_no_pending(tmp_path):
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {"pending_restart_at": None})
    calls = []
    result = app.apply(sp, now_iso="2026-06-11T03:30:00+02:00",
                       restart_fn=lambda: calls.append("restart"))
    assert result["restarted"] is False
    assert calls == []


def test_restarts_and_clears_stamp_when_due(tmp_path):
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {"pending_restart_at": "2026-06-11T03:00:00+02:00"})
    calls = []
    result = app.apply(sp, now_iso="2026-06-11T03:30:00+02:00",
                       restart_fn=lambda: calls.append("restart"))
    assert result["restarted"] is True
    assert calls == ["restart"]
    # Stamp cleared so we don't restart again on the next timer fire.
    assert state_mod.load_state(sp)["pending_restart_at"] is None


def test_no_restart_when_pending_in_future(tmp_path):
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {"pending_restart_at": "2026-06-11T05:00:00+02:00"})
    calls = []
    result = app.apply(sp, now_iso="2026-06-11T03:30:00+02:00",
                       restart_fn=lambda: calls.append("restart"))
    assert result["restarted"] is False
    assert calls == []
```

- [ ] **Step 5: Run the test to verify it fails**

Run: `uv run pytest tests/test_loop_apply_pending.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'loop.apply_pending_deploy'`.

- [ ] **Step 6: Implement `apply_pending_deploy.py`**

Create `src/loop/apply_pending_deploy.py`:

```python
"""Apply a pending pre-sunrise deploy: restart the camera if a deploy is due.

Run by wildlife-deploy.service (a daily oneshot). No-op unless state.json has a
pending_restart_at whose time has arrived. Restarting the camera makes it reload
experiments/deployed_config.env on startup (Config is built once at startup).
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from loop import state as state_mod


def _restart_camera() -> None:
    subprocess.run(
        ["systemctl", "restart", "wildlife-camera.service"], check=True
    )


def apply(state_path, now_iso: str, restart_fn=_restart_camera) -> dict:
    """Restart the camera iff pending_restart_at <= now; then clear the stamp."""
    st = state_mod.load_state(state_path)
    pending = st.get("pending_restart_at")
    if not pending:
        return {"restarted": False, "reason": "no pending deploy"}
    if datetime.fromisoformat(pending) > datetime.fromisoformat(now_iso):
        return {"restarted": False, "reason": "pending deploy not due yet"}

    restart_fn()
    st["pending_restart_at"] = None
    state_mod.save_state(state_path, st)
    return {"restarted": True, "reason": f"applied deploy stamped {pending}"}


def main() -> None:
    try:
        result = apply("experiments/state.json", now_iso=datetime.now().astimezone().isoformat())
        print(json.dumps(result))
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": str(e)}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `uv run pytest tests/test_loop_apply_pending.py -v`
Expected: PASS (3 passed).

- [ ] **Step 8: Commit**

```bash
git add wildlife-loop.service wildlife-deploy.service wildlife-deploy.timer src/loop/apply_pending_deploy.py tests/test_loop_apply_pending.py
git commit -m "feat(loop): add systemd loop service + pre-sunrise deploy timer/service

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 11: feedback sidecar `/pause` + `/rollback` handling

**Files:**
- Modify: `src/telegram_feedback.py` (add command handlers; wire into `build_application` ~line 73)
- Test: `tests/test_telegram_feedback_commands.py`

The sidecar already runs `telegram.ext.Application` and polls. We add two
`CommandHandler`s: `/pause` writes `paused: true` to `state.json`; `/rollback`
calls `deploy.rollback()`. Both are pure-logic functions (Telegram-free) so they
are unit-testable, mirroring `record_feedback_callback`.

- [ ] **Step 1: Write the failing test for the pure command logic**

Create `tests/test_telegram_feedback_commands.py`:

```python
"""Tests for /pause and /rollback pure-logic helpers in telegram_feedback."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loop import state as state_mod
import telegram_feedback as tf


def test_handle_pause_sets_paused_flag(tmp_path):
    sp = tmp_path / "state.json"
    state_mod.save_state(sp, {"paused": False})
    msg = tf.handle_pause(state_path=sp)
    assert state_mod.load_state(sp)["paused"] is True
    assert "paus" in msg.lower()


def test_handle_rollback_invokes_deploy_rollback(tmp_path):
    sp = tmp_path / "state.json"
    env = tmp_path / "deployed_config.env"
    state_mod.save_state(sp, {
        "deployed": {"MOTION_THRESHOLD": 2500},
        "best_known_good": {"MOTION_THRESHOLD": 2000},
        "history": [], "pending_restart_at": None,
    })
    msg = tf.handle_rollback(state_path=sp, env_path=env,
                             restart_at="2026-06-11T05:00:00+02:00")
    assert state_mod.load_state(sp)["deployed"]["MOTION_THRESHOLD"] == 2000
    assert "MOTION_THRESHOLD=2000" in env.read_text()
    assert "rollback" in msg.lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_telegram_feedback_commands.py -v`
Expected: FAIL with `AttributeError: module 'telegram_feedback' has no attribute 'handle_pause'`.

- [ ] **Step 3: Add the pure helpers + handlers to `telegram_feedback.py`**

In `src/telegram_feedback.py`, add to the imports block (after line ~23):

```python
import sys as _sys
from pathlib import Path as _Path

_SRC = _Path(__file__).resolve().parent
if str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))
from loop import state as _state_mod
from loop import deploy as _deploy
from telegram.ext import CommandHandler
```

Add the pure helpers (after `record_feedback_callback`, ~line 45):

```python
DEFAULT_STATE_PATH = "experiments/state.json"
DEFAULT_ENV_PATH = "experiments/deployed_config.env"


def handle_pause(state_path: str = DEFAULT_STATE_PATH) -> str:
    """Set paused=true in state.json. Pure of Telegram I/O for testability."""
    st = _state_mod.load_state(state_path)
    st["paused"] = True
    _state_mod.save_state(state_path, st)
    logger.info("Loop paused via /pause command")
    return "⏸️ Tuning paused. The loop will hold best_known_good until resumed."


def handle_rollback(
    state_path: str = DEFAULT_STATE_PATH,
    env_path: str = DEFAULT_ENV_PATH,
    restart_at: str = None,
) -> str:
    """Roll back to best_known_good via deploy.rollback()."""
    from datetime import datetime, timedelta

    if restart_at is None:
        # Default: ASAP (next minute) — the deploy timer applies it pre-sunrise,
        # but a manual rollback should restart at the next opportunity.
        restart_at = (datetime.now().astimezone() + timedelta(minutes=1)).isoformat()
    result = _deploy.rollback(state_path, env_path, restart_at)
    logger.info(f"Rollback requested via /rollback: {result}")
    return (
        "↩️ Rollback queued: restored best_known_good "
        f"({result['deployed']}); camera restart stamped."
    )
```

Add the async command handlers (after `_on_callback`, ~line 71):

```python
async def _on_pause(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = handle_pause()
    await update.message.reply_text(msg)


async def _on_rollback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = handle_rollback()
    await update.message.reply_text(msg)
```

Wire them into `build_application` (inside the function, after the existing
`add_handler` call, ~line 79):

```python
    application.add_handler(CommandHandler("pause", _on_pause))
    application.add_handler(CommandHandler("rollback", _on_rollback))
```

Finally, broaden the polling allowed-updates so commands are delivered. Change
the `run_polling` line in `main()` (~line 96):

```python
    application.run_polling(allowed_updates=["callback_query", "message"])
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_telegram_feedback_commands.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the existing feedback test to confirm no regression**

Run: `uv run pytest tests/ -k feedback -v`
Expected: PASS (existing feedback callback tests still green).

- [ ] **Step 6: Commit**

```bash
git add src/telegram_feedback.py tests/test_telegram_feedback_commands.py
git commit -m "feat(loop): add /pause and /rollback veto commands to feedback sidecar

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 12: README documentation for the loop + services

**Files:**
- Modify: `README.md` (add an "Autonomous tuning loop" section)

- [ ] **Step 1: Read the existing README service-install section to match style**

Run: `grep -n "systemctl\|wildlife-camera.service\|wildlife-feedback" README.md`
Expected: shows the existing install pattern (`sudo cp ... /etc/systemd/system/`, `systemctl enable --now`).

- [ ] **Step 2: Add the loop documentation section**

Append to `README.md` (match the existing service-install wording from Step 1):

```markdown
## Autonomous tuning loop (ADR-004 Phase 4)

A nightly self-improving loop reduces false positives and false negatives. It is
two layers: deterministic tools in `src/loop/` (invoked as `python -m loop.<name>`,
JSON to stdout) and a judgment layer (`experiments/loop.md` + `PROTOCOL.md`) run
by an on-Pi Claude Code `/loop` session.

### Components
- `loop.ingest` — read new detections + feedback past the watermark; reconcile
  labels (human > tier-2 > tier-1).
- `loop.metrics` — paired FP/FN with Wilson 95% CIs → `experiments/metrics/daily.csv`.
- `loop.replay` — Layer-A offline replay seam (STUB: returns "skipped" this build).
- `loop.report` — Telegram daily summary (FP+FN) + heartbeat (send-only).
- `loop.deploy` — validate against bounds, update `experiments/state.json`, render
  `experiments/deployed_config.env`, stamp a pre-sunrise restart; `rollback()`.
- `loop.guardrails` — bounds dict (shared with config validators), volume guards,
  FN-veto, feedback-starved freeze (N=3 days).

### Config overlay
Tunable sub-configs load `('.env', 'experiments/deployed_config.env')`. Precedence:
real OS env > `deployed_config.env` > `.env` > defaults. `deployed_config.env` is
gitignored and regenerated from `experiments/state.json` (the committed source of
truth). Out-of-bounds deployed values raise at `Config()` construction, so the
camera refuses to start on a bad config.

### Deploy mechanism
The loop never hot-reloads. `loop.deploy` stamps `pending_restart_at` ~60 min
before sunrise; `wildlife-deploy.timer` fires `wildlife-deploy.service`
(`loop.apply_pending_deploy`), which restarts `wildlife-camera.service` when the
stamp is due so MOG2 warmup finishes while still dark.

### Veto commands
`/pause` (freeze tuning, hold best_known_good) and `/rollback` (restore
best_known_good) are sent to the Telegram bot and handled by the feedback sidecar
(`wildlife-feedback.service`).

### Installing the services
```bash
sudo cp wildlife-loop.service /etc/systemd/system/
sudo cp wildlife-deploy.service /etc/systemd/system/
sudo cp wildlife-deploy.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now wildlife-loop.service
sudo systemctl enable --now wildlife-deploy.timer
```
`wildlife-deploy.service` runs `systemctl restart wildlife-camera.service`; grant
user `daniel` permission via a polkit rule or run the camera as a user unit.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(loop): document autonomous tuning loop, overlay, services

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 13: Full-suite green + end-to-end dry run on the Pi

**Files:** none (verification only)

- [ ] **Step 1: Run the entire test suite**

Run: `uv run pytest tests/ -v`
Expected: PASS — all new `test_loop_*` tests plus the existing suite green. If any
existing test fails, the most likely cause is the config overlay tuple
interacting with conftest; confirm `tests/conftest.py` still sets
`model_config["env_file"] = None` (it does — its loop only checks that
`"env_file" in obj.model_config`, which remains true for the tuple form).

- [ ] **Step 2: Smoke-test each CLI prints JSON**

Run (from repo root, requires a valid `.env` for `ingest`/`metrics`/`report`;
`replay` and `deploy --rollback` work without one):
```bash
PYTHONPATH=src uv run python -m loop.replay
PYTHONPATH=src uv run python -m loop.ingest --since-id 0
PYTHONPATH=src uv run python -m loop.metrics --date 2026-06-08
```
Expected: each prints a single JSON object to stdout (or `{"error": ...}` + exit 1
if the DB/.env is absent — that is the documented failure contract).

- [ ] **Step 3: Dry-run a bounded no-op deploy and rollback**

Run:
```bash
PYTHONPATH=src uv run python -m loop.deploy --delta '{"MOTION_THRESHOLD": 2000}' --restart-at 2026-06-09T04:30:00+02:00
cat experiments/deployed_config.env
PYTHONPATH=src uv run python -m loop.deploy --rollback --restart-at 2026-06-09T05:00:00+02:00
git checkout experiments/state.json   # discard the dry-run state mutation
rm -f experiments/deployed_config.env # gitignored artifact
```
Expected: first command prints `{"deployed": {"MOTION_THRESHOLD": 2000}, ...}` and
writes `experiments/deployed_config.env` containing `MOTION_THRESHOLD=2000`;
rollback prints `{"rolled_back": true, ...}`.

- [ ] **Step 4: Verify the deployed env actually flows into Config on the Pi**

Run:
```bash
PYTHONPATH=src uv run python -m loop.deploy --delta '{"MOTION_THRESHOLD": 2345}' --restart-at 2026-06-09T04:30:00+02:00
uv run python -c "import sys; sys.path.insert(0,'src'); from config import MotionConfig; print(MotionConfig().threshold)"
git checkout experiments/state.json
rm -f experiments/deployed_config.env
```
Expected: prints `2345` (overlay flowed into `MotionConfig`). This confirms the
end-to-end deploy→overlay→Config path before arming nightly.

- [ ] **Step 5: Confirm the loop service / timer files are valid unit syntax**

Run:
```bash
systemd-analyze verify wildlife-loop.service wildlife-deploy.service wildlife-deploy.timer 2>&1 || true
```
Expected: no fatal syntax errors (warnings about missing binaries on a dev box are
acceptable; on the Pi the paths resolve).

- [ ] **Step 6: Final commit (only if dry-run produced doc-worthy notes)**

If the dry run revealed nothing to change, no commit is needed. Otherwise capture
fixes:

```bash
git add -A
git commit -m "chore(loop): dry-run fixes from end-to-end verification

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review (run by the plan author; findings already folded in above)

**1. Spec coverage** — every spec section maps to a task:
- guardrails bounds + config overlay + bounds validators → Tasks 1, 2.
- ingest + metrics (seeded DB fixtures) → Tasks 4, 5.
- replay stub → Task 6.
- report + heartbeat → Task 7.
- deploy + rollback + state.json schema → Tasks 3, 8.
- experiments/ scaffold (seeded backlog, PROTOCOL, loop.md, JOURNAL, LEARNINGS,
  gold/, metrics/, runs/ example, gitignore) → Task 9.
- systemd units + pre-sunrise apply + README → Tasks 10, 12.
- feedback sidecar /pause + /rollback → Task 11.
- end-to-end dry run → Task 13.
- Build-order (spec §"Build order") followed: guardrails/config → ingest/metrics →
  replay → report → deploy → scaffold → systemd+sidecar → dry run.

**2. Placeholder scan** — no "TBD"/"add appropriate X"; every code step has full
code; the only intentional stub is `replay.py` (required by the spec) and it has a
complete body + test.

**3. Type consistency** — names are stable across tasks: `BOUNDS` (keyed by env-var
name), `validate_param`, `check_volume`, `fn_veto`, `is_feedback_starved`,
`load_state`/`save_state`, `reconcile`/`ingest`, `wilson_ci`/`compute_metrics`/
`append_daily`, `ReplayResult(status, reason, metrics)`/`replay`,
`render_summary`/`render_heartbeat`/`send`, `deploy`/`rollback`/`_render_env`,
`apply` (apply_pending_deploy), `handle_pause`/`handle_rollback`. The `fn_veto`
signature in Task 1 matches its test; `compute_metrics(rows, fn_audit)` matches
its test and the report's `last_metrics` shape; deploy's `state.json` schema
(Task 8) is the same one report (Task 7) and apply_pending (Task 10) read.

**Known sharp edge flagged for the implementer:** `tests/conftest.py` neutralises
`.env` by setting `model_config["env_file"] = None` on each config class. For the
tuple-form overlay this means production loads `('.env', 'experiments/deployed_config.env')`
but tests load `None` unless they pass `_env_file=` explicitly — which the Task 2
tests do. If a future test constructs a tunable sub-config without `_env_file` and
expects the overlay, it will not see it under conftest; pass `_env_file=(None, overlay)`
as shown.
