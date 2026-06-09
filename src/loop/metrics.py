"""Compute paired FP/FN with Wilson 95% CIs and append to daily.csv.

FP comes from the labeled/captured set (triggers marked false_positive ÷ labeled
triggers). FN comes from the timelapse audit channel; until a timelapse detector
pass exists, FN is reported as "unmeasured" (NOT 0 — a zero would falsely clear
the FN-veto). daily.csv is idempotent per date: a re-run for the same date
overwrites that date's row, never duplicates.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# z for 95% two-sided.
_Z = 1.959963984540054


def wilson_ci(successes: int, trials: int) -> tuple[float, float]:
    """Wilson score interval (95%). trials==0 → (0.0, 1.0) (no information)."""
    if trials <= 0:
        return (0.0, 1.0)
    p = successes / trials
    z = _Z
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    margin = (
        z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    ) / denom
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    # Clamp tiny floating-point overshoots that keep high just under 1.0
    # when successes == trials (e.g., 0.9999999999999999 → 1.0).
    if round(high, 10) == 1.0:
        high = 1.0
    if round(low, 10) == 0.0:
        low = 0.0
    return (low, high)


def compute_metrics(rows: list[dict], fn_audit: Optional[dict]) -> dict:
    """Paired FP/FN with Wilson CIs over reconciled rows.

    FP denominator = rows with a non-None reconciled_label (labeled triggers).
    FN is "unmeasured" unless fn_audit={"missed": int, "animal_frames": int} is
    supplied by a timelapse detector pass (NOT implemented this build).
    """
    labeled = [r for r in rows if r.get("reconciled_label") is not None]
    fp_count = sum(1 for r in labeled if r["reconciled_label"] == "false_positive")
    fp_rate = (fp_count / len(labeled)) if labeled else 0.0
    fp_ci = wilson_ci(fp_count, len(labeled))

    # Count ERROR-status rows.  They have reconciled_label=None (excluded from
    # the FP denominator above) but we surface them so errors are visible.
    error_count = sum(
        1 for r in rows
        if r.get("detection_status") == "error"
    )

    if fn_audit and fn_audit.get("animal_frames", 0) > 0:
        missed = fn_audit["missed"]
        frames = fn_audit["animal_frames"]
        fn_rate: object = missed / frames
        fn_ci: object = wilson_ci(missed, frames)
    else:
        fn_rate = "unmeasured"
        fn_ci = None

    return {
        "labeled_triggers": len(labeled),
        "total_triggers": len(rows),
        "fp_count": fp_count,
        "fp_rate": fp_rate,
        "fp_ci": fp_ci,
        "fn_rate": fn_rate,
        "fn_ci": fn_ci,
        "error_count": error_count,
    }


_CSV_FIELDS = [
    "date", "total_triggers", "labeled_triggers", "fp_count", "fp_rate",
    "fp_ci_low", "fp_ci_high", "fn_rate", "fn_ci_low", "fn_ci_high",
]


def _row_for_csv(date: str, m: dict) -> dict:
    fp_ci = m["fp_ci"]
    fn_ci = m["fn_ci"]
    return {
        "date": date,
        "total_triggers": m["total_triggers"],
        "labeled_triggers": m["labeled_triggers"],
        "fp_count": m["fp_count"],
        "fp_rate": m["fp_rate"],
        "fp_ci_low": fp_ci[0],
        "fp_ci_high": fp_ci[1],
        "fn_rate": m["fn_rate"],
        "fn_ci_low": fn_ci[0] if fn_ci else "",
        "fn_ci_high": fn_ci[1] if fn_ci else "",
    }


def append_daily(csv_path, date: str, m: dict) -> None:
    """Write one row per date; a re-run for the same date overwrites that row."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists():
        with open(path, newline="") as f:
            existing = [r for r in csv.DictReader(f) if r["date"] != date]
    existing.append(_row_for_csv(date, m))
    existing.sort(key=lambda r: r["date"])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(existing)


def _jsonable(m: dict) -> dict:
    out = dict(m)
    out["fp_ci"] = list(m["fp_ci"])
    out["fn_ci"] = list(m["fn_ci"]) if m["fn_ci"] else None
    return out


def main() -> None:
    import argparse
    import json
    import sys as _sys
    from pathlib import Path as _Path
    from datetime import date as _date

    _src = _Path(__file__).resolve().parent.parent
    if str(_src) not in _sys.path:
        _sys.path.insert(0, str(_src))
    from config import Config
    from database_manager import DatabaseManager
    from loop import ingest
    from loop import state as state_mod

    parser = argparse.ArgumentParser(description="Compute daily FP/FN metrics")
    parser.add_argument("--state", default="experiments/state.json")
    parser.add_argument("--csv", default="experiments/metrics/daily.csv")
    parser.add_argument("--date", default=_date.today().isoformat())
    args = parser.parse_args()
    try:
        st = state_mod.load_state(args.state)
        watermark = int(st.get("watermark", 0))
        db = DatabaseManager(Config())
        ing = ingest.ingest(db, watermark)

        # No-data tick: watermark has already caught up to the latest detection;
        # no new rows have arrived.  Do NOT overwrite last_metrics or append to
        # daily.csv — either action would clobber the standing baseline with a
        # degenerate 0-trigger row (the camera is daytime-only; every overnight
        # tick is a no-data tick).
        if ing["count"] == 0:
            print(json.dumps({
                "status": "no_data",
                "measured": 0,
                "baseline_preserved": True,
            }))
            return

        m = compute_metrics(ing["rows"], fn_audit=None)
        append_daily(args.csv, args.date, m)
        # Build the flat last_metrics dict that report.render_summary consumes.
        # Shape: date + all metric fields, fp_ci/fn_ci as JSON lists.
        flat = _jsonable(m)
        flat["date"] = args.date
        # Persist into state.json so report.main() can read it without manual glue.
        st["last_metrics"] = flat
        state_mod.save_state(args.state, st)
        print(json.dumps({"date": args.date, "metrics": flat}))
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": str(e)}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
