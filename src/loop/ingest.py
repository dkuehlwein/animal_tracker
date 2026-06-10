"""Ingest detections + feedback past a watermark and reconcile labels.

Pure READ over the SQLite DB (WAL). Never writes labels — ground truth is
append-only and owned by the feedback sidecar (anti-self-poisoning). Output is a
list of per-detection records the judgment layer consumes; the watermark is the
max detections.id seen so the next tick only processes new rows.

Reconciliation precedence: human > tier-2 > tier-1.
  tier-1 = detections.animals_detected (MegaDetector).
  tier-2 = detection_feedback row with source='tier2'.
  human  = detection_feedback row with source='human' (latest wins).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import Config
from database_manager import DatabaseManager


def _coerce_int(value):
    """Return value as a Python int, decoding little-endian BLOB if needed.

    SQLite stores numpy.int64 scalars as raw 8-byte little-endian BLOBs when
    no sqlite3 adapter is registered for the NumPy type.  This function undoes
    that serialisation so already-corrupted rows remain readable.
    """
    if isinstance(value, (bytes, bytearray)):
        return int.from_bytes(value, "little")
    if value is None:
        return None
    return int(value)


def _tier1_label(animals_detected) -> str | None:
    if animals_detected is None:
        return None
    return "animal" if animals_detected else "false_positive"


# Status → tier-1 label mapping.  ERROR rows get None so they are excluded
# from the FP/FN denominator (a pipeline crash is not a labelled event).
#
# `unclassifiable` → false_positive (exp #4, 2026-06-10): MegaDetector boxes a
# region but the classifier cannot ID a species.  Empirically this is the camera
# boxing wind-blown vegetation / the swinging bird-feeder, NOT a real animal:
# across all human-labelled history, unclassifiable rows are 27/27 false_positive
# (0 animal, 0 wrong_species).  Mapping these to "animal" was the dominant source
# of the tier-1 label-trust gap (auto-labels only ~29-36% concordant with humans,
# biased toward calling FPs "animal"), which made the reconciled FP rate a
# systematic UNDER-estimate on unlabelled rows.  This is a metrics-reconciliation
# change only — it does not touch the live detection/notification pipeline, so it
# carries zero FN risk to wildlife capture.  Revert if a real animal ever lands an
# `unclassifiable` human label.
_STATUS_TO_TIER1: dict[str, str | None] = {
    "identified": "animal",
    "animal_uncertain": "animal",
    "unclassifiable": "false_positive",
    "no_animal": "false_positive",
    "error": None,
}


def _tier1_label_from_status(detection_status, animals_detected) -> str | None:
    """Return tier-1 label, preferring detection_status over legacy animals_detected.

    - If detection_status is set (new rows): use _STATUS_TO_TIER1 mapping.
    - If detection_status is NULL/None (legacy rows): fall back to animals_detected.
    """
    if detection_status is not None:
        return _STATUS_TO_TIER1.get(str(detection_status).lower())
    return _tier1_label(animals_detected)


def _read_connection(db: DatabaseManager) -> sqlite3.Connection:
    """Open a read-only connection (URI mode) to the WAL DB."""
    uri = f"file:{db.db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def reconcile(db: DatabaseManager, since_id: int) -> list[dict]:
    """Return reconciled per-detection records for detections.id > since_id."""
    conn = _read_connection(db)
    try:
        det_rows = conn.execute(
            """
            SELECT id, animals_detected, detection_status, motion_area, contour_count,
                   largest_contour_area, foreground_pixel_count, hour_of_day,
                   gate_would_suppress
            FROM detections
            WHERE id > ?
            ORDER BY id ASC
            """,
            (since_id,),
        ).fetchall()

        results: list[dict] = []
        for d in det_rows:
            det_id = d["id"]
            fb = conn.execute(
                """
                SELECT label, source, created_at, id
                FROM detection_feedback
                WHERE detection_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (det_id,),
            ).fetchall()

            tier1 = _tier1_label_from_status(d["detection_status"], d["animals_detected"])
            tier2 = None
            human = None
            for row in fb:  # ascending; last assignment of each source wins
                if row["source"] == "tier2":
                    tier2 = row["label"]
                elif row["source"] == "human":
                    human = row["label"]

            reconciled = human or tier2 or tier1
            results.append(
                {
                    "detection_id": det_id,
                    "reconciled_label": reconciled,
                    "tier1": tier1,
                    "tier2": tier2,
                    "human": human,
                    "detection_status": d["detection_status"],
                    "motion_area": _coerce_int(d["motion_area"]),
                    "contour_count": _coerce_int(d["contour_count"]),
                    "largest_contour_area": _coerce_int(d["largest_contour_area"]),
                    "foreground_pixel_count": _coerce_int(d["foreground_pixel_count"]),
                    "hour_of_day": _coerce_int(d["hour_of_day"]),
                    "gate_would_suppress": bool(d["gate_would_suppress"])
                    if d["gate_would_suppress"] is not None
                    else None,
                }
            )
        return results
    finally:
        conn.close()


def ingest(db: DatabaseManager, since_id: int) -> dict:
    """Reconcile and report the advanced watermark (max detections.id seen)."""
    rows = reconcile(db, since_id)
    new_watermark = max((r["detection_id"] for r in rows), default=since_id)
    return {"rows": rows, "new_watermark": new_watermark, "count": len(rows)}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ingest detections past a watermark")
    parser.add_argument("--since-id", type=int, default=0)
    args = parser.parse_args()
    try:
        db = DatabaseManager(Config())
        result = ingest(db, args.since_id)
        print(json.dumps(result))
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": str(e)}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
