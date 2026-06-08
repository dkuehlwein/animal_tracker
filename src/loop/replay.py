"""Layer-A offline replay seam (ADR-004 Phase 3) — STUB for this build.

When filled in, replay() will re-run MegaDetector + classifier over saved
high-res images with a candidate config and score against labels, giving offline
evidence before a live deploy. Until then it returns status="skipped", which the
rest of the system treats as "no offline evidence available — this experiment
cannot be validated offline yet" (replay-gated experiments stay parked).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class ReplayResult:
    status: str            # "ok" | "skipped" | "error"
    reason: str
    metrics: Optional[dict] = None


def replay(candidate_config: dict, labeled_set: list) -> ReplayResult:
    """STUB: returns skipped. Real implementation is a later task."""
    return ReplayResult(status="skipped", reason="not implemented", metrics=None)


def main() -> None:
    result = replay(candidate_config={}, labeled_set=[])
    print(json.dumps({"status": result.status, "reason": result.reason,
                      "metrics": result.metrics}))


if __name__ == "__main__":
    main()
