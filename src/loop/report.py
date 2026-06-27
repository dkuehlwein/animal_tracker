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
    """Daily summary: per-tier labelling breakdown. Plain English, no CI/jargon."""
    paused = bool(state.get("paused", False))
    total = metrics.get("total_triggers", 0)

    lines = [f"🦊 Last night: {total} images captured."]

    if paused:
        lines.append("⏸️ PAUSED — tuning frozen until resumed")

    # Alert when FP rate is untrustworthy due to high error rate.
    # fp_trustworthy may be absent in old state.json shapes — default to True.
    fp_trustworthy = metrics.get("fp_trustworthy", True)
    if not fp_trustworthy:
        error_rate = metrics.get("error_rate", 0.0)
        error_count = metrics.get("error_count", "?")
        lines.append(
            f"⚠️ FP rate UNTRUSTWORTHY this period — "
            f"error_rate {error_rate:.0%} ({error_count}/{total})"
        )

    # Per-tier lines — backward-compat: .get defaults to 0 for old metrics shapes.
    n_human = metrics.get("n_human", 0)
    fp_human = metrics.get("fp_human_count", 0)
    n_claude = metrics.get("n_claude", 0)
    fp_claude = metrics.get("fp_claude_count", 0)
    n_md = metrics.get("n_md", 0)
    fp_md = metrics.get("fp_md_count", 0)

    def _tier_line(label: str, n: int, fp: int) -> str:
        if n == 0:
            return f"• {label} 0"
        alarm_word = "false alarm" if fp == 1 else "false alarms"
        return f"• {label} {n} — {fp} {alarm_word}"

    lines.append(_tier_line("You labelled", n_human, fp_human))
    lines.append(_tier_line("Claude labelled", n_claude, fp_claude))

    # MegaDetector always tagged (auto, unverified).
    if n_md == 0:
        lines.append("• MegaDetector (auto, unverified): 0")
    else:
        alarm_word = "false alarm" if fp_md == 1 else "false alarms"
        lines.append(f"• MegaDetector (auto, unverified): {n_md} — {fp_md} {alarm_word}")

    # Remainder: images not yet labelled by any tier.
    remainder = total - (n_human + n_claude + n_md)
    if remainder > 0:
        lines.append(f"• Not yet labelled: {remainder}")

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
        # Tuples were serialised to lists in state.json; restore CI shape if present
        # (old state shapes have fp_ci/fn_ci; new per-tier shapes may not).
        metrics = dict(metrics)
        if "fp_ci" in metrics:
            metrics["fp_ci"] = tuple(metrics["fp_ci"])
        if metrics.get("fn_ci") is not None:
            metrics["fn_ci"] = tuple(metrics["fn_ci"])
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
