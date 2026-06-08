"""Atomic read/write of experiments/state.json.

state.json is the COMMITTED source of truth for the loop (deployed config,
active experiment id, seeded backlog, baselines, best_known_good + history,
ingest watermark, paused flag, pending-deploy stamp). Only deterministic tools
write it. Writes are atomic (temp file + os.replace) so a crashed write never
leaves a half-file that would crash the camera service.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def load_state(path: str | Path) -> dict[str, Any]:
    """Load state.json. A missing file returns {} (fresh checkout is safe)."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(path: str | Path, state: dict[str, Any]) -> None:
    """Atomically write state as pretty JSON (temp file + os.replace)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".state-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
