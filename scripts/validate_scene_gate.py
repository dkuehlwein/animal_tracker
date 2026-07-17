#!/usr/bin/env python3
"""Offline validation for the scene-unchanged gate (Task 5, feat/scene-gate).

Replays the review-class detection history chronologically, reproducing the
live gate's reference-set semantics (`SceneReferenceSet`, K=`--ref-count`
refs within `--ref-max-age-hours`), and buckets the resulting similarity
scores by ground truth:

  - human labels are truth: 'animal'/'animal_wrong_id' -> animal-present,
    'false_positive' -> FP. 'person' is reported separately (a person is not
    an "empty scene" either, but the brief's acceptance rule is scoped to
    animal-labeled rows only). 'cant_tell' is excluded entirely (unusable
    image). Legacy 'wrong_species' is excluded too and reported separately:
    per project history (see MEMORY: wrong_species is heterogeneous) it can
    mean either "animal, wrong species" or "actually a human" and can't be
    resolved without eyeballing each frame, so it is never folded into the
    animal-present bucket.
  - tier-2 (Claude) auto-labels are reported separately, never folded into
    the truth number (project rule, see MEMORY: auto-labels are not truth).
  - everything else is 'unlabeled'.

Read-only: never writes to the DB. Retention keeps only ~100 most recent
bursts of images on disk, so most DB rows have no frame; those are skipped
and counted.

Usage (from repo root):
    PYTHONPATH=src uv run python scripts/validate_scene_gate.py
"""

import argparse
import math
import sqlite3
import sys
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path so this also runs without PYTHONPATH=src set explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from config import Config  # noqa: E402
import scene_gate as scene_gate_module  # noqa: E402
from scene_gate import SceneReferenceSet  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

REVIEW_STATUSES = ("no_animal", "unclassifiable")

HUMAN_ANIMAL_LABELS = {"animal", "animal_wrong_id"}
HUMAN_FP_LABEL = "false_positive"
HUMAN_PERSON_LABEL = "person"
HUMAN_CANT_TELL_LABEL = "cant_tell"
HUMAN_WRONG_SPECIES_LABEL = "wrong_species"  # legacy, heterogeneous — excluded

# Low-texture diagnostic threshold (task 1 review carry-over).
LOW_TEXTURE_STD = 5.0


def parse_args() -> argparse.Namespace:
    cfg = Config()
    parser = argparse.ArgumentParser(
        description="Offline validation + threshold selection for the scene-unchanged gate"
    )
    parser.add_argument("--db", default=cfg.storage.database_path,
                         help="Path to the detections SQLite DB (read-only).")
    parser.add_argument("--images-root", default=str(cfg.storage.image_dir),
                         help="Directory to fall back to (by basename) when a stored "
                              "image_path no longer resolves as-is.")
    parser.add_argument("--ref-count", type=int, default=3,
                         help="K: max reference frames kept (mirrors scene_gate_ref_count).")
    parser.add_argument("--ref-max-age-hours", type=float, default=6.0,
                         help="Max reference age in hours (mirrors scene_gate_ref_max_age_hours).")
    return parser.parse_args()


def open_readonly(db_path: str) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def resolve_image_path(image_path: str, images_root: Path) -> Optional[Path]:
    """Return an existing Path for image_path, or None if it can't be found.

    Tries the stored path as-is first (relative to cwd, which is how it was
    written by the live system), then falls back to images_root/basename in
    case the DB was copied without its matching image directory alongside.
    """
    p = Path(image_path)
    if p.exists():
        return p
    alt = images_root / p.name
    if alt.exists():
        return alt
    return None


def fetch_review_detections(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    placeholders = ", ".join("?" for _ in REVIEW_STATUSES)
    cursor = conn.execute(
        f"""
        SELECT id, timestamp, image_path, detection_status
        FROM detections
        WHERE detection_status IN ({placeholders})
        ORDER BY timestamp ASC, id ASC
        """,
        REVIEW_STATUSES,
    )
    return cursor.fetchall()


def fetch_feedback(conn: sqlite3.Connection, detection_id: int) -> List[sqlite3.Row]:
    cursor = conn.execute(
        """
        SELECT label, source, created_at, id
        FROM detection_feedback
        WHERE detection_id = ?
        ORDER BY created_at ASC, id ASC
        """,
        (detection_id,),
    )
    return cursor.fetchall()


def classify_ground_truth(feedback_rows: List[sqlite3.Row]) -> Tuple[str, Optional[str]]:
    """Return (bucket, raw_label) for one detection's feedback history.

    Precedence: human label (latest wins) > tier2 label (latest wins) >
    unlabeled. Mirrors src/loop/ingest.py's reconciliation precedence.
    """
    human_label = None
    tier2_label = None
    for row in feedback_rows:
        if row["source"] == "human":
            human_label = row["label"]
        elif row["source"] == "tier2":
            tier2_label = row["label"]

    if human_label is not None:
        if human_label in HUMAN_ANIMAL_LABELS:
            return "human_animal", human_label
        if human_label == HUMAN_FP_LABEL:
            return "human_fp", human_label
        if human_label == HUMAN_PERSON_LABEL:
            return "human_person", human_label
        if human_label == HUMAN_CANT_TELL_LABEL:
            return "excluded_cant_tell", human_label
        if human_label == HUMAN_WRONG_SPECIES_LABEL:
            return "excluded_wrong_species", human_label
        return "human_other", human_label  # shouldn't happen given known vocabulary

    if tier2_label is not None:
        if tier2_label in HUMAN_ANIMAL_LABELS:
            return "auto_tier2_animal", tier2_label
        if tier2_label == HUMAN_FP_LABEL:
            return "auto_tier2_fp", tier2_label
        return "auto_tier2_other", tier2_label

    return "unlabeled", None


_std_cache: Dict[str, Optional[float]] = {}


def frame_std(path: Path) -> Optional[float]:
    """Std-dev of the resized grayscale frame, matching scene_gate's own
    downsizing so the low-texture diagnostic reflects what the comparator
    actually sees. Cached per path (images are re-used as references)."""
    key = str(path)
    if key in _std_cache:
        return _std_cache[key]
    img = cv2.imread(key, cv2.IMREAD_GRAYSCALE)
    if img is None:
        _std_cache[key] = None
        return None
    small = cv2.resize(img, scene_gate_module._COMPARE_SIZE).astype(np.float64)
    std = float(small.std())
    _std_cache[key] = std
    return std


def prune_mirror(refs: List[Tuple[Path, datetime]], now: datetime,
                  max_age_hours: float) -> List[Tuple[Path, datetime]]:
    max_age = timedelta(hours=max_age_hours)
    return [(p, t) for p, t in refs if now - t <= max_age]


def fmt_stats(values: List[float]) -> str:
    if not values:
        return "n=0"
    return (f"n={len(values)}  min={min(values):.4f}  mean={statistics.mean(values):.4f}  "
            f"median={statistics.median(values):.4f}  max={max(values):.4f}")


def main() -> None:
    args = parse_args()
    images_root = Path(args.images_root)

    conn = open_readonly(args.db)
    try:
        review_rows = fetch_review_detections(conn)
        total_review = len(review_rows)

        # Full-corpus label tally (all review-class rows, regardless of
        # whether the frame still exists) — gives context for *why* a bucket
        # may end up empty once we restrict to on-disk frames.
        full_corpus_buckets: Dict[str, int] = {}
        for row in review_rows:
            fb = fetch_feedback(conn, row["id"])
            bucket, _ = classify_ground_truth(fb)
            full_corpus_buckets[bucket] = full_corpus_buckets.get(bucket, 0) + 1

        ref_set = SceneReferenceSet(max_refs=args.ref_count, max_age_hours=args.ref_max_age_hours)
        mirror_refs: List[Tuple[Path, datetime]] = []  # for the low-texture diagnostic only

        skipped_missing = 0
        skipped_no_reference = 0  # image exists but no ref within window (no score)

        # bucket -> list of similarity scores
        bucket_scores: Dict[str, List[float]] = {}
        # bucket -> list of (detection_id, timestamp, score) for reporting
        bucket_rows: Dict[str, List[Tuple[int, str, float]]] = {}

        compared_frame_stds: List[float] = []  # every distinct frame that took part in >=1 comparison
        compared_frame_paths_seen: set = set()
        low_texture_pairs: List[Tuple[int, str, float, float, float]] = []  # (id, ref_path, q_std, r_std, pair_sim)

        for row in review_rows:
            det_id = row["id"]
            timestamp = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
            resolved = resolve_image_path(row["image_path"], images_root)

            if resolved is None:
                skipped_missing += 1
                continue

            fb = fetch_feedback(conn, det_id)
            bucket, raw_label = classify_ground_truth(fb)

            active_refs = prune_mirror(mirror_refs, timestamp, args.ref_max_age_hours)

            score = ref_set.best_similarity(resolved, timestamp)

            if score is None:
                skipped_no_reference += 1
            else:
                bucket_scores.setdefault(bucket, []).append(score)
                bucket_rows.setdefault(bucket, []).append((det_id, row["timestamp"], score))

                # Low-texture diagnostic: std of the query frame + every
                # active reference frame it was compared against.
                q_std = frame_std(resolved)
                if q_std is not None:
                    key = str(resolved)
                    if key not in compared_frame_paths_seen:
                        compared_frame_paths_seen.add(key)
                        compared_frame_stds.append(q_std)
                for ref_path, _ref_ts in active_refs:
                    r_std = frame_std(ref_path)
                    if r_std is not None:
                        rkey = str(ref_path)
                        if rkey not in compared_frame_paths_seen:
                            compared_frame_paths_seen.add(rkey)
                            compared_frame_stds.append(r_std)
                    if q_std is not None and r_std is not None and q_std < LOW_TEXTURE_STD and r_std < LOW_TEXTURE_STD:
                        pair_sim = scene_gate_module.compute_scene_similarity(resolved, ref_path)
                        low_texture_pairs.append((det_id, str(ref_path), q_std, r_std, pair_sim))

            # Every existing review-class frame becomes a future reference,
            # regardless of its (unknown-at-capture-time) label — mirrors
            # wildlife_system.py's live behaviour exactly.
            ref_set.add(resolved, timestamp)
            mirror_refs.append((resolved, timestamp))
            mirror_refs.sort(key=lambda e: e[1])
            mirror_refs = prune_mirror(mirror_refs, timestamp, args.ref_max_age_hours)
            if len(mirror_refs) > args.ref_count:
                mirror_refs = mirror_refs[-args.ref_count:]

        scored_total = sum(len(v) for v in bucket_scores.values())

        # ---------------------------------------------------------------
        # Report
        # ---------------------------------------------------------------
        print("=" * 78)
        print("SCENE-GATE OFFLINE VALIDATION")
        print("=" * 78)
        print(f"DB: {args.db}")
        print(f"Images root: {images_root}")
        print(f"Replay params: ref_count(K)={args.ref_count}  ref_max_age_hours={args.ref_max_age_hours}")
        print()
        print(f"Total review-class detections (no_animal/unclassifiable): {total_review}")
        print(f"  Skipped (image_path missing on disk): {skipped_missing}")
        print(f"  Have frame on disk: {total_review - skipped_missing}")
        print(f"    Skipped (no reference frame within window, e.g. first-of-run/gap): {skipped_no_reference}")
        print(f"    Scored (have a best-similarity score): {scored_total}")
        print()
        print("-" * 78)
        print("Full-corpus ground-truth tally (ALL review-class rows, incl. rows whose")
        print("frame is gone — shows what the label mix looks like before the on-disk")
        print("filter shrinks it):")
        for k in sorted(full_corpus_buckets):
            print(f"  {k:28s} {full_corpus_buckets[k]}")
        print()

        print("-" * 78)
        print("SCORED bucket distributions (similarity, higher = more similar to a")
        print("recent empty-scene reference):")
        bucket_order = [
            "human_animal", "human_fp", "human_person",
            "excluded_cant_tell", "excluded_wrong_species", "human_other",
            "auto_tier2_animal", "auto_tier2_fp", "auto_tier2_other",
            "unlabeled",
        ]
        for b in bucket_order:
            scores = bucket_scores.get(b, [])
            print(f"  {b:28s} {fmt_stats(scores)}")
        # Any bucket key that showed up but isn't in bucket_order (defensive)
        for b in bucket_scores:
            if b not in bucket_order:
                print(f"  {b:28s} {fmt_stats(bucket_scores[b])}")
        print()

        print("-" * 78)
        print("Low-texture diagnostic (Task 1 review carry-over):")
        print(f"  {fmt_stats(compared_frame_stds)} (frame std, over distinct frames used in >=1 comparison)")
        print(f"  Flagged pairs where BOTH frames had std < {LOW_TEXTURE_STD}: {len(low_texture_pairs)}")
        for det_id, ref_path, q_std, r_std, pair_sim in low_texture_pairs[:50]:
            sim_str = f"{pair_sim:.4f}" if pair_sim is not None else "None"
            print(f"    detection_id={det_id}  ref={ref_path}  q_std={q_std:.2f}  ref_std={r_std:.2f}  similarity={sim_str}")
        if len(low_texture_pairs) > 50:
            print(f"    ... and {len(low_texture_pairs) - 50} more")
        print()

        # -----------------------------------------------------------
        # Threshold selection
        # -----------------------------------------------------------
        print("-" * 78)
        print("THRESHOLD SELECTION")
        animal_scores = bucket_scores.get("human_animal", [])
        fp_scores = bucket_scores.get("human_fp", [])
        person_scores = bucket_scores.get("human_person", [])
        all_scored = [s for v in bucket_scores.values() for s in v]

        def yield_at(scores: List[float], t: float) -> Tuple[int, int, float]:
            if not scores:
                return 0, 0, 0.0
            hits = sum(1 for s in scores if s >= t)
            return hits, len(scores), hits / len(scores)

        if not animal_scores:
            print("Animal-labeled (human 'animal'/'animal_wrong_id') bucket is EMPTY among")
            print("scored (on-disk) rows -- no evidence available either way about whether")
            print("any threshold T would ever mute a real animal.")
            print()
            print("EXCEPTION per brief: cannot pick a threshold from thin air.")
            print("Recommendation: scene_gate_enabled=False (keep threshold placeholder at 0.97).")
        else:
            max_animal = max(animal_scores)
            # Highest-yield threshold that still keeps zero animal-labeled
            # violations: the tightest T strictly greater than max_animal
            # (mute condition is score >= T, so T must exceed max_animal to
            # guarantee zero animal-labeled mutes). Rounded up to 2dp to
            # match the config's precision and keep a small buffer above the
            # exact boundary.
            t = round(math.ceil(max_animal * 100) / 100, 2)
            if t <= max_animal:
                t = round(t + 0.01, 2)
            t_margin = round(t - 0.02, 2)

            fp_hits_t, fp_n_t, fp_yield_t = yield_at(fp_scores, t)
            fp_hits_m, fp_n_m, fp_yield_m = yield_at(fp_scores, t_margin)
            all_hits_t, all_n_t, all_yield_t = yield_at(all_scored, t)
            all_hits_m, all_n_m, all_yield_m = yield_at(all_scored, t_margin)
            animal_violations_at_margin = sum(1 for s in animal_scores if s >= t_margin)

            print(f"Animal-labeled bucket: n={len(animal_scores)}  max={max_animal:.4f}")
            print(f"Chosen T (tightest safe threshold, zero animal-labeled rows score >= T): {t}")
            print(f"  FP-mute yield at T:      {fp_hits_t}/{fp_n_t} = {fp_yield_t:.1%} (human false_positive bucket)")
            print(f"  Overall mute rate at T:  {all_hits_t}/{all_n_t} = {all_yield_t:.1%} (all scored review-class rows)")
            print(f"Safety-margin T-0.02 = {t_margin}:")
            print(f"  FP-mute yield at T-0.02:     {fp_hits_m}/{fp_n_m} = {fp_yield_m:.1%}")
            print(f"  Overall mute rate at T-0.02: {all_hits_m}/{all_n_m} = {all_yield_m:.1%}")
            print(f"  Animal-labeled violations at T-0.02: {animal_violations_at_margin}")

        if person_scores:
            print()
            print(f"NOTE: human 'person' bucket is non-empty (n={len(person_scores)}, "
                  f"max={max(person_scores):.4f}). Not part of the brief's formal acceptance "
                  f"rule (scoped to animal-labeled rows), but a muted person-in-frame row is "
                  f"also an undesirable concealment -- reported for awareness.")

        excluded_wrong_species = full_corpus_buckets.get("excluded_wrong_species", 0)
        if excluded_wrong_species:
            print()
            print(f"NOTE: {excluded_wrong_species} legacy 'wrong_species' human labels exist on "
                  f"review-class rows corpus-wide (heterogeneous: animal-wrong-ID OR human, per "
                  f"project history) -- excluded from the animal/FP buckets rather than guessed.")

        print()
        print("=" * 78)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
