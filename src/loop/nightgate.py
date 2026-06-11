"""Deterministic pre-gate for the autonomous tuning loop.

Decides — without invoking any LLM — whether the loop should run this tick.

Gate semantics (designed for shell && / || chaining):
  exit 0  → the loop SHOULD run  (it is night AND tonight's run is not yet done)
  exit 1  → skip this tick       (it is daytime OR last_tick_completed_day == loop_day())

On a SKIP, a heartbeat is sent once per loop-day (Fix #3):
  - If state.json["last_heartbeat_loopday"] != loop_day(), send a heartbeat and
    set last_heartbeat_loopday = loop_day() in state.json.
  - The heartbeat send is BEST-EFFORT: a failure is logged but never changes the
    gate's exit code.

A one-line reason is printed to stdout in both cases, e.g.:
  proceed: night, run not done
  skip: daytime
  skip: tonight's run already done (2026-06-08)

Usage (from repo root with PYTHONPATH=src):
  uv run python -m loop.nightgate [--state experiments/state.json]

Imports are intentionally light: Config, utils.SunChecker, loop.state only.
Heavy modules (species_identifier, camera_manager, etc.) are never imported.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is importable when run as __main__ or from tests.
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import Config  # noqa: E402
from utils import SunChecker  # noqa: E402
from loop import state as state_mod  # noqa: E402

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seams for testing (monkeypatch these in unit tests)
# ---------------------------------------------------------------------------

def _get_is_daytime() -> bool:
    """Return True if it is currently daytime (delegates to SunChecker)."""
    config = Config()
    return SunChecker(config).is_daytime()


def _get_loop_day() -> str:
    """Return the current loop-day as a YYYY-MM-DD string."""
    return state_mod.loop_day()


def _send_heartbeat(state_path: str, loop_day_str: str) -> None:
    """Send a heartbeat via loop.report and stamp last_heartbeat_loopday.

    Monkeypatch this in tests to avoid real Telegram calls.
    """
    import asyncio
    from loop import report as report_mod

    text = report_mod.render_heartbeat(loop_day_str)
    # Fire-and-forget: best-effort send (may fail on network error).
    asyncio.run(report_mod.send(text))

    # Stamp heartbeat sent into state.json so we don't double-send.
    st = state_mod.load_state(state_path)
    st["last_heartbeat_loopday"] = loop_day_str
    state_mod.save_state(state_path, st)


# ---------------------------------------------------------------------------
# Pure decision function — fully testable without I/O
# ---------------------------------------------------------------------------

def should_run(
    is_daytime: bool,
    last_tick_completed_day: str | None,
    current_loop_day: str,
) -> tuple[bool, str]:
    """Return (should_run, reason) based on current conditions.

    Args:
        is_daytime:             True if it is currently between sunrise and sunset.
        last_tick_completed_day: state.json["last_tick_completed_day"], or None if
                                 absent (first-ever run is safe to proceed).
        current_loop_day:       The current loop-day from loop_day() — stable across
                                the whole 18:00→06:00 overnight window.

    Returns:
        (True,  "proceed: night, run not done")                       — loop should fire
        (False, "skip: daytime")                                      — skip, not night yet
        (False, "skip: tonight's run already done (YYYY-MM-DD)")      — skip
    """
    if is_daytime:
        return False, "skip: daytime"
    if last_tick_completed_day == current_loop_day:
        return False, f"skip: tonight's run already done ({current_loop_day})"
    return True, "proceed: night, run not done"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Parse args, evaluate gate, send heartbeat on skip (once per loop-day),
    print reason, exit 0 (run) or 1 (skip)."""
    parser = argparse.ArgumentParser(
        description="Deterministic pre-gate: exit 0 = loop should run, exit 1 = skip"
    )
    parser.add_argument(
        "--state",
        default="experiments/state.json",
        help="Path to state.json (default: experiments/state.json)",
    )
    args = parser.parse_args(argv)

    is_daytime = _get_is_daytime()
    current_loop_day = _get_loop_day()

    # Load state and extract last_tick_completed_day (None is safe — fresh checkout).
    st = state_mod.load_state(args.state)
    last_tick_completed_day: str | None = st.get("last_tick_completed_day")

    run, reason = should_run(
        is_daytime=is_daytime,
        last_tick_completed_day=last_tick_completed_day,
        current_loop_day=current_loop_day,
    )

    print(reason)

    if not run:
        # Send a heartbeat once per loop-day (deadman / silence-is-the-alarm).
        last_heartbeat = st.get("last_heartbeat_loopday")
        if last_heartbeat != current_loop_day:
            try:
                _send_heartbeat(args.state, current_loop_day)
            except Exception:  # noqa: BLE001
                log.warning(
                    "nightgate: heartbeat send failed (best-effort — skipping cleanly)",
                    exc_info=True,
                )

    raise SystemExit(0 if run else 1)


if __name__ == "__main__":
    main()
