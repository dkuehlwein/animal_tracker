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

_JOURNAL_TELEGRAM_LIMIT = 4000  # Telegram max is 4096; leave a small margin


def latest_journal_entry(journal_path: str | Path) -> str | None:
    """Return the last top-level bullet entry from JOURNAL.md, or None.

    A top-level entry line starts with ``- `` at column 0.  Continuation
    lines (indented or blank within a block) are collected until the next
    top-level bullet or EOF.  Returns the raw block text, trimmed of leading/
    trailing whitespace, or ``None`` if the file is missing or has no bullets.
    """
    path = Path(journal_path)
    if not path.exists():
        return None

    lines = path.read_text(encoding="utf-8").splitlines()

    last_entry_lines: list[str] = []
    current: list[str] = []

    for line in lines:
        if line.startswith("- "):
            # Start of a new top-level entry — save previous block, open new one.
            last_entry_lines = current
            current = [line]
        elif current:
            # Continuation of the current top-level entry.
            current.append(line)
        # Lines before the first bullet (header prose) are ignored.

    # After the loop `current` holds the last started entry (or [] if none).
    if current:
        last_entry_lines = current

    if not last_entry_lines:
        return None

    text = "\n".join(last_entry_lines).strip()
    return text if text else None


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

    # Fix #2: alert when FP rate is untrustworthy due to high error rate.
    # fp_trustworthy may be absent in old state.json shapes — default to True
    # (backward-compatible: no alert for legacy metrics that lack the field).
    fp_trustworthy = metrics.get("fp_trustworthy", True)
    if not fp_trustworthy:
        error_rate = metrics.get("error_rate", 0.0)
        error_count = metrics.get("error_count", "?")
        total = metrics.get("total_triggers", "?")
        lines.append(
            f"⚠️ FP rate UNTRUSTWORTHY this period — "
            f"error_rate {error_rate:.0%} ({error_count}/{total})"
        )

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
    parser.add_argument(
        "--no-send", "--dry-run",
        dest="no_send",
        action="store_true",
        default=False,
        help="Render the report text and print it to stdout; do NOT call Telegram.",
    )
    parser.add_argument(
        "--journal",
        default="experiments/JOURNAL.md",
        help="Path to JOURNAL.md for the second summary message (default: experiments/JOURNAL.md).",
    )
    args = parser.parse_args()
    try:
        st = state_mod.load_state(args.state)
        if args.mode == "heartbeat":
            text = render_heartbeat(args.last_tick)
            if args.no_send:
                print(json.dumps({"sent": False, "mode": args.mode, "rendered": text}))
            else:
                ok = asyncio.run(send(text))
                print(json.dumps({"sent": ok, "mode": args.mode}))
            return

        # --- summary mode ---
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

        # Fetch the latest journal entry (may be None if file missing / no bullets).
        entry = latest_journal_entry(args.journal)
        if entry and len(entry) > _JOURNAL_TELEGRAM_LIMIT:
            entry = entry[:_JOURNAL_TELEGRAM_LIMIT] + "… (truncated)"

        if args.no_send:
            print(json.dumps({
                "sent": False,
                "mode": args.mode,
                "rendered": text,
                "journal_entry": entry,
            }))
        else:
            ok = asyncio.run(send(text))
            journal_sent: bool | None = None
            if entry is not None:
                try:
                    journal_sent = asyncio.run(send(entry))
                except Exception:  # noqa: BLE001
                    journal_sent = False
            print(json.dumps({"sent": ok, "mode": args.mode, "journal_sent": journal_sent}))
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": str(e)}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
