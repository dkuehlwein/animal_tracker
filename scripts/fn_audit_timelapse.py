"""Timelapse false-negative audit: did an animal appear that never triggered?

The tuning loop's hardest blind spot is FN measurement: with no animal-labelled
rows, every trigger-side lever is FN-vetoed for lack of counter-evidence, and
"zero animals today" cannot be distinguished from "the camera stopped seeing
animals". The 20-second timelapse stream (`data/timelapse/`, daylight only,
same framing as the detector) answers that independently of the trigger stream.

Method: a raw frame-diff is useless here — the fountain and bamboo move in every
frame. Instead each pixel's deviation from its own local temporal median is
normalized by its own local temporal std, so chronically-moving pixels (water,
wind-blown leaves) are suppressed while a one-off intruder survives. Frames are
ranked by the largest surviving connected component and each candidate is
reported with the nearest real trigger, so "big transient object, no trigger
within minutes" rises to the top for human/tier-2 inspection.

Calibration note (2026-09-07, first run over 10k frames / 09-03..09-07): the
top-25 candidates were all either coincident with a real trigger, inside the
fountain/bamboo quadrant, or whole-scene sun/shade illumination transitions
(inspect them — the two largest non-trigger hits were dappled-light shifts).
Illumination transitions are this detector's dominant false alarm; it does not
try to suppress them, because doing so would risk suppressing a real intruder.

Usage (repo root):  PYTHONPATH=src uv run python scripts/fn_audit_timelapse.py
Runtime ~3 min for 10k frames on the Pi 5. Read the output, then LOOK at the
top candidates — the ranking is a triage aid, not a verdict.
"""
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

TIMELAPSE_DIR = Path("data/timelapse")
DB_PATH = Path("data/detections.db")
W, H = 160, 120
WINDOW = 31          # +-15 frames ~ +-5 min of local background
STD_MULT = 2.5       # deviation must exceed this many local stds ...
SLACK = 8.0          # ... plus this many grey levels, to beat sensor noise
DEDUPE_SECONDS = 180  # one candidate per 3-minute cluster
TOP_N = 25


def load_frames():
    frames = []
    for p in TIMELAPSE_DIR.iterdir():
        m = re.match(r"timelapse_(\d+)\.jpg$", p.name)
        if m:
            frames.append((int(m.group(1)) / 1000.0, p))
    frames.sort()
    return frames


def main() -> int:
    frames = load_frames()
    if not frames:
        print(f"no timelapse frames under {TIMELAPSE_DIR}")
        return 1
    print(f"{len(frames)} frames, {datetime.fromtimestamp(frames[0][0])} "
          f"-> {datetime.fromtimestamp(frames[-1][0])}")

    kept = []
    for ts, p in frames:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        small = cv2.GaussianBlur(cv2.resize(img, (W, H)), (5, 5), 0)
        kept.append((ts, p, small.astype(np.float32)))
    print(f"loaded {len(kept)}")

    arr = np.stack([k[2] for k in kept])
    half = WINDOW // 2
    results = []
    for i in range(len(kept)):
        win = arr[max(0, i - half):i + half + 1]
        z = np.abs(arr[i] - np.median(win, axis=0)) - STD_MULT * np.std(win, axis=0) - SLACK
        n, _lab, stats, cent = cv2.connectedComponentsWithStats((z > 0).astype(np.uint8), 8)
        best, centroid = 0, (0.0, 0.0)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] > best:
                best, centroid = stats[k, cv2.CC_STAT_AREA], cent[k]
        results.append((best, kept[i][0], kept[i][1], centroid))

    results.sort(reverse=True)
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    triggers = [(datetime.fromisoformat(t).timestamp(), i, s) for i, t, s in
                conn.execute("SELECT id, timestamp, detection_status FROM detections")]

    print("\n blob_px  time                 cx,cy      nearest trigger")
    seen: list[float] = []
    for blob, ts, path, c in results:
        if any(abs(ts - s) < DEDUPE_SECONDS for s in seen):
            continue
        seen.append(ts)
        near = min(triggers, key=lambda d: abs(d[0] - ts)) if triggers else None
        gap = f"id={near[1]} {near[2]} {ts - near[0]:+.0f}s" if near else "no triggers in DB"
        print(f"  {blob:6d}  {datetime.fromtimestamp(ts):%Y-%m-%d %H:%M:%S}  "
              f"{c[0] / W:.2f},{c[1] / H:.2f}   {gap}  {path.name}")
        if len(seen) >= TOP_N:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
