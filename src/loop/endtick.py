"""Stamp loop completion into state.json after a successful claude tick.

Sets state.json["last_tick_completed_day"] and state.json["last_heartbeat_loopday"]
to the current loop_day() so that:
- nightgate skips re-running the same loop-day (Fix #1).
- nightgate does not double-send a heartbeat after the substantive summary (Fix #3).

This module runs AFTER a successful `claude -p` tick (see wildlife-loop.service).
It is idempotent — calling it twice for the same loop-day is a harmless no-op.

Usage (from repo root with PYTHONPATH=src):
    uv run python -m loop.endtick [--state experiments/state.json]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from loop import state as state_mod  # noqa: E402


def stamp(state_path: str | Path, now: datetime | None = None) -> None:
    """Atomically stamp last_tick_completed_day and last_heartbeat_loopday in state.json.

    Args:
        state_path: Path to state.json.
        now:        Optional datetime for testing (uses loop_day(now)).
    """
    day = state_mod.loop_day(now)
    st = state_mod.load_state(state_path)
    st["last_tick_completed_day"] = day
    st["last_heartbeat_loopday"] = day
    state_mod.save_state(state_path, st)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: stamp completion, exit 0."""
    parser = argparse.ArgumentParser(
        description="Stamp loop completion into state.json (runs after a successful claude tick)"
    )
    parser.add_argument(
        "--state",
        default="experiments/state.json",
        help="Path to state.json (default: experiments/state.json)",
    )
    args = parser.parse_args(argv)
    stamp(args.state)
    print(f"endtick: stamped loop-day {state_mod.loop_day()} in {args.state}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
