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
