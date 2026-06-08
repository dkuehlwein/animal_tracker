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
