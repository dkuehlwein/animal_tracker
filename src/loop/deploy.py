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

from loop import guardrails  # noqa: E402
from loop import state as state_mod  # noqa: E402


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
