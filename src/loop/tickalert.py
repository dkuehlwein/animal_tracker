"""Telegram alert when the nightly `claude -p` run exits non-zero.

Invoked by wildlife-loop.service only on a claude failure, with the exit code
and a path to the captured run output. Classifies auth failures (expired OAuth
login — the cause of two multi-night silent outages) vs other failures, and
sends one Telegram alert per kind per loop-day (deduped via state.json).

BEST-EFFORT: never raises, always exits 0 — the service script preserves
claude's own exit status itself.

Usage: python -m loop.tickalert --exit-code N --log PATH [--state PATH]
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from loop import nightgate  # noqa: E402
from loop import state as state_mod  # noqa: E402

log = logging.getLogger(__name__)

AUTH_MARKERS = ("failed to authenticate", "oauth", "not logged in", "/login")
STAMP_KEYS = {
    "auth": "last_auth_alert_loopday",
    "other": "last_tick_failure_alert_loopday",
}
TAIL_LINES = 8
MAX_LINE = 200
_SECRET_RE = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+\S+|[A-Za-z0-9_\-]{32,})")


def _get_loop_day() -> str:
    return state_mod.loop_day()


def _send_alert(state_path: str, loop_day_str: str, text: str, stamp_key: str) -> None:
    """Seam over nightgate's shared send+stamp helper (monkeypatch in tests)."""
    nightgate._send_alert(state_path, loop_day_str, text, stamp_key)


def classify_failure(output: str) -> str:
    low = output.lower()
    return "auth" if any(m in low for m in AUTH_MARKERS) else "other"


def render_message(kind: str, exit_code: int, output: str) -> str:
    if kind == "auth":
        return (
            "🔑 Wildlife nightly loop FAILED: the Claude login on the Pi has "
            "expired, so tonight's tuning run did not happen.\n"
            "Fix: SSH to the Pi, run `claude`, then `/login` "
            "(or `claude auth login`)."
        )
    lines = [ln.strip()[:MAX_LINE] for ln in output.splitlines() if ln.strip()]
    tail = _SECRET_RE.sub("[redacted]", "\n".join(lines[-TAIL_LINES:]))
    return (
        f"⚠️ Wildlife nightly loop: claude exited with code {exit_code}.\n"
        f"Last output:\n{tail or '(no output captured)'}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Alert on nightly claude failure")
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--log", required=True, help="captured claude output file")
    parser.add_argument("--state", default="experiments/state.json")
    args = parser.parse_args(argv)

    try:
        try:
            output = Path(args.log).read_text(encoding="utf-8", errors="replace")
        except OSError:
            output = ""
        kind = classify_failure(output)
        key = STAMP_KEYS[kind]
        day = _get_loop_day()
        if state_mod.load_state(args.state).get(key) == day:
            return
        _send_alert(args.state, day, render_message(kind, args.exit_code, output), key)
    except Exception:  # noqa: BLE001
        log.warning("tickalert: alert failed (best-effort)", exc_info=True)


if __name__ == "__main__":
    main()
