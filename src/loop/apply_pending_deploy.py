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

from loop import state as state_mod  # noqa: E402


def _as_aware(ts: str) -> datetime:
    """Parse an ISO stamp, treating an offset-naive one as local time.

    pending_restart_at is stamped by loop.deploy for env deltas but written by
    hand for code-only experiments, and a hand-written stamp easily omits the
    UTC offset. Comparing naive against aware raises TypeError, which crashed
    wildlife-deploy.service on 2026-09-21 and silently skipped the restart
    (the stamp is only cleared on the success path, so it would have failed
    every night thereafter). Local time is the right reading: every stamp this
    loop writes is a local pre-sunrise wall-clock time.
    """
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt


def _restart_camera() -> None:
    subprocess.run(
        ["sudo", "-n", "systemctl", "restart", "wildlife-camera.service"], check=True
    )


def apply(state_path, now_iso: str, restart_fn=_restart_camera) -> dict:
    """Restart the camera iff pending_restart_at <= now; then clear the stamp."""
    st = state_mod.load_state(state_path)
    pending = st.get("pending_restart_at")
    if not pending:
        return {"restarted": False, "reason": "no pending deploy"}
    if _as_aware(pending) > _as_aware(now_iso):
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
