"""Deterministic pre-gate for the autonomous tuning loop.

Decides — without invoking any LLM — whether the loop should run this tick.

Gate semantics (designed for shell && / || chaining):
  exit 0  → the loop SHOULD run  (it is night AND tonight's run is not yet done)
  exit 1  → skip this tick       (it is daytime OR last_metrics.date == today)

A one-line reason is printed to stdout in both cases, e.g.:
  proceed: night, run not done
  skip: daytime
  skip: tonight's run already done (2026-06-09)

Usage (from repo root with PYTHONPATH=src):
  uv run python -m loop.nightgate [--state experiments/state.json]

Imports are intentionally light: Config, utils.SunChecker, loop.state only.
Heavy modules (species_identifier, camera_manager, etc.) are never imported.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

# Ensure src/ is importable when run as __main__ or from tests.
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import Config
from utils import SunChecker
from loop import state as state_mod


# ---------------------------------------------------------------------------
# Seams for testing (monkeypatch these in unit tests)
# ---------------------------------------------------------------------------

def _get_is_daytime() -> bool:
    """Return True if it is currently daytime (delegates to SunChecker)."""
    config = Config()
    return SunChecker(config).is_daytime()


def _get_today() -> str:
    """Return today's local date as a YYYY-MM-DD string."""
    return date.today().isoformat()


# ---------------------------------------------------------------------------
# Pure decision function — fully testable without I/O
# ---------------------------------------------------------------------------

def should_run(
    is_daytime: bool,
    last_metrics_date: str | None,
    today: str,
) -> tuple[bool, str]:
    """Return (should_run, reason) based on current conditions.

    Args:
        is_daytime:        True if it is currently between sunrise and sunset.
        last_metrics_date: The 'date' field from state['last_metrics'], or None
                           if state.json is absent / has no last_metrics yet.
        today:             Today's local date as a YYYY-MM-DD string.

    Returns:
        (True,  "proceed: night, run not done")     — loop should fire
        (False, "skip: daytime")                    — skip, not night yet
        (False, "skip: tonight's run already done (YYYY-MM-DD)")  — skip
    """
    if is_daytime:
        return False, "skip: daytime"
    if last_metrics_date == today:
        return False, f"skip: tonight's run already done ({today})"
    return True, "proceed: night, run not done"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Parse args, evaluate gate, print reason, exit 0 (run) or 1 (skip)."""
    parser = argparse.ArgumentParser(
        description="Deterministic pre-gate: exit 0 = loop should run, exit 1 = skip"
    )
    parser.add_argument(
        "--state",
        default="experiments/state.json",
        help="Path to state.json (default: experiments/state.json)",
    )
    args = parser.parse_args(argv)

    # Load state and extract last_metrics date (None is safe — fresh checkout).
    st = state_mod.load_state(args.state)
    last_metrics = st.get("last_metrics") or {}
    last_metrics_date: str | None = last_metrics.get("date")

    run, reason = should_run(
        is_daytime=_get_is_daytime(),
        last_metrics_date=last_metrics_date,
        today=_get_today(),
    )

    print(reason)
    raise SystemExit(0 if run else 1)


if __name__ == "__main__":
    main()
