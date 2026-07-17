"""Tests for the scene-unchanged gate comparator + rolling reference set.

Synthetic images only (no camera hardware fixtures): frames are generated
with numpy/cv2 and written to tmp_path.
"""
from datetime import datetime, timedelta

import cv2
import numpy as np


def _make_scene(path, w=200, h=200, seed=0):
    """Write a deterministic textured (non-flat) grayscale scene to path."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 30, size=(h, w), dtype=np.uint8)
    gradient = np.tile(np.linspace(80, 180, w, dtype=np.uint8), (h, 1))
    img = cv2.add(gradient, noise)
    cv2.imwrite(str(path), img)
    return img


def _make_brightened(base_img, path, delta=40):
    """Write a copy of base_img with a uniform brightness shift applied."""
    shift = np.full(base_img.shape, delta, dtype=np.uint8)
    brightened = cv2.add(base_img, shift)
    cv2.imwrite(str(path), brightened)
    return brightened


def _make_blob(base_img, path, size=20):
    """Write a copy of base_img with a dark size x size blob pasted on it."""
    blobbed = base_img.copy()
    h, w = blobbed.shape
    y0, x0 = h // 2 - size // 2, w // 2 - size // 2
    blobbed[y0:y0 + size, x0:x0 + size] = 0
    cv2.imwrite(str(path), blobbed)
    return blobbed


# --- compute_scene_similarity ------------------------------------------------

def test_identical_frames_are_highly_similar(tmp_path):
    from scene_gate import compute_scene_similarity

    base = _make_scene(tmp_path / "a.jpg")
    cv2.imwrite(str(tmp_path / "b.jpg"), base)

    sim = compute_scene_similarity(tmp_path / "a.jpg", tmp_path / "b.jpg")
    assert sim is not None
    assert sim > 0.98


def test_uniform_brightness_shift_still_similar(tmp_path):
    """Global exposure/AE shifts should not dominate thanks to normalization."""
    from scene_gate import compute_scene_similarity

    base = _make_scene(tmp_path / "a.jpg")
    _make_brightened(base, tmp_path / "bright.jpg", delta=40)

    sim = compute_scene_similarity(tmp_path / "a.jpg", tmp_path / "bright.jpg")
    assert sim is not None
    assert sim > 0.95


def test_blob_lowers_similarity_measurably(tmp_path):
    """A pasted-on dark blob ('bird') should score measurably below an
    identical-frame comparison."""
    from scene_gate import compute_scene_similarity

    base = _make_scene(tmp_path / "a.jpg")
    cv2.imwrite(str(tmp_path / "b_identical.jpg"), base)
    _make_blob(base, tmp_path / "b_blob.jpg", size=20)

    sim_identical = compute_scene_similarity(tmp_path / "a.jpg", tmp_path / "b_identical.jpg")
    sim_blob = compute_scene_similarity(tmp_path / "a.jpg", tmp_path / "b_blob.jpg")

    assert sim_identical is not None
    assert sim_blob is not None
    assert sim_blob < sim_identical - 0.01


def test_similarity_is_bounded_zero_to_one(tmp_path):
    from scene_gate import compute_scene_similarity

    base = _make_scene(tmp_path / "a.jpg", seed=0)
    _make_scene(tmp_path / "c.jpg", seed=1)  # unrelated noise pattern

    sim = compute_scene_similarity(tmp_path / "a.jpg", tmp_path / "c.jpg")
    assert sim is not None
    assert 0.0 <= sim <= 1.0


def test_missing_frame_path_returns_none(tmp_path):
    from scene_gate import compute_scene_similarity

    _make_scene(tmp_path / "a.jpg")
    sim = compute_scene_similarity(tmp_path / "does_not_exist.jpg", tmp_path / "a.jpg")
    assert sim is None


def test_missing_reference_path_returns_none(tmp_path):
    from scene_gate import compute_scene_similarity

    _make_scene(tmp_path / "a.jpg")
    sim = compute_scene_similarity(tmp_path / "a.jpg", tmp_path / "does_not_exist.jpg")
    assert sim is None


def test_unreadable_file_returns_none(tmp_path, caplog):
    from scene_gate import compute_scene_similarity

    _make_scene(tmp_path / "a.jpg")
    garbage = tmp_path / "garbage.jpg"
    garbage.write_bytes(b"not an image")

    with caplog.at_level("WARNING"):
        sim = compute_scene_similarity(tmp_path / "a.jpg", garbage)
    assert sim is None
    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_compute_scene_similarity_never_raises(tmp_path):
    """Even a directory path (not a file) must degrade to None, not raise."""
    from scene_gate import compute_scene_similarity

    a_dir = tmp_path / "a_dir"
    a_dir.mkdir()
    _make_scene(tmp_path / "a.jpg")

    sim = compute_scene_similarity(a_dir, tmp_path / "a.jpg")
    assert sim is None


# --- SceneReferenceSet --------------------------------------------------------

def test_best_similarity_with_empty_set_returns_none(tmp_path):
    from scene_gate import SceneReferenceSet

    _make_scene(tmp_path / "frame.jpg")
    ref_set = SceneReferenceSet(max_refs=5, max_age_hours=24)

    result = ref_set.best_similarity(tmp_path / "frame.jpg", now=datetime(2026, 1, 1))
    assert result is None


def test_best_similarity_returns_max_over_refs_skipping_none(tmp_path):
    from scene_gate import SceneReferenceSet

    frame = _make_scene(tmp_path / "frame.jpg", seed=0)
    # ref1: identical to frame -> high similarity
    cv2.imwrite(str(tmp_path / "ref1.jpg"), frame)
    # ref2: unrelated scene -> lower similarity
    _make_scene(tmp_path / "ref2.jpg", seed=1)

    ref_set = SceneReferenceSet(max_refs=5, max_age_hours=24)
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    ref_set.add(tmp_path / "ref2.jpg", t0)
    ref_set.add(tmp_path / "ref1.jpg", t0 + timedelta(minutes=1))
    # ref3: path that will fail to read -> compute_scene_similarity returns None,
    # must be skipped, not blow up the max().
    ref_set.add(tmp_path / "does_not_exist.jpg", t0 + timedelta(minutes=2))

    best = ref_set.best_similarity(tmp_path / "frame.jpg", now=t0 + timedelta(minutes=3))
    assert best is not None
    assert best > 0.98


def test_add_evicts_oldest_beyond_max_refs(tmp_path):
    from scene_gate import SceneReferenceSet

    ref_set = SceneReferenceSet(max_refs=3, max_age_hours=1000)
    t0 = datetime(2026, 1, 1, 0, 0, 0)
    paths = []
    for i in range(5):
        p = tmp_path / f"ref{i}.jpg"
        _make_scene(p, seed=i)
        paths.append(p)
        ref_set.add(p, t0 + timedelta(minutes=i))

    assert len(ref_set._refs) == 3
    kept_paths = [entry[0] for entry in ref_set._refs]
    # Only the 3 most-recently-added refs (ref2, ref3, ref4) should survive.
    assert str(paths[2]) in kept_paths
    assert str(paths[3]) in kept_paths
    assert str(paths[4]) in kept_paths
    assert str(paths[0]) not in kept_paths
    assert str(paths[1]) not in kept_paths


def test_add_evicts_refs_older_than_max_age_relative_to_newest_add(tmp_path):
    from scene_gate import SceneReferenceSet

    ref_set = SceneReferenceSet(max_refs=10, max_age_hours=2)
    t0 = datetime(2026, 1, 1, 0, 0, 0)

    old_ref = tmp_path / "old.jpg"
    new_ref = tmp_path / "new.jpg"
    _make_scene(old_ref, seed=0)
    _make_scene(new_ref, seed=1)

    ref_set.add(old_ref, t0)
    # This add is > max_age_hours after old_ref -> old_ref should be evicted.
    ref_set.add(new_ref, t0 + timedelta(hours=3))

    assert len(ref_set._refs) == 1
    assert ref_set._refs[0][0] == str(new_ref)


def test_best_similarity_prunes_stale_refs_against_now(tmp_path):
    from scene_gate import SceneReferenceSet

    ref_set = SceneReferenceSet(max_refs=10, max_age_hours=2)
    t0 = datetime(2026, 1, 1, 0, 0, 0)

    ref = tmp_path / "ref.jpg"
    _make_scene(ref, seed=0)
    ref_set.add(ref, t0)

    frame = tmp_path / "frame.jpg"
    _make_scene(frame, seed=0)

    # `now` is far enough past t0 that the only ref should be pruned as stale.
    result = ref_set.best_similarity(frame, now=t0 + timedelta(hours=5))
    assert result is None
    assert len(ref_set._refs) == 0


def test_seed_bulk_loads_oldest_to_newest_with_eviction(tmp_path):
    from scene_gate import SceneReferenceSet

    ref_set = SceneReferenceSet(max_refs=2, max_age_hours=1000)
    t0 = datetime(2026, 1, 1, 0, 0, 0)

    paths = []
    for i in range(3):
        p = tmp_path / f"seed{i}.jpg"
        _make_scene(p, seed=i)
        paths.append(p)

    rows = [(str(paths[i]), t0 + timedelta(minutes=i)) for i in range(3)]
    ref_set.seed(rows)

    assert len(ref_set._refs) == 2
    kept_paths = [entry[0] for entry in ref_set._refs]
    assert str(paths[1]) in kept_paths
    assert str(paths[2]) in kept_paths
    assert str(paths[0]) not in kept_paths


def test_seed_skips_missing_files(tmp_path):
    from scene_gate import SceneReferenceSet

    ref_set = SceneReferenceSet(max_refs=5, max_age_hours=1000)
    t0 = datetime(2026, 1, 1, 0, 0, 0)

    existing = tmp_path / "exists.jpg"
    _make_scene(existing, seed=0)
    missing = tmp_path / "missing.jpg"

    rows = [
        (str(missing), t0),
        (str(existing), t0 + timedelta(minutes=1)),
    ]
    ref_set.seed(rows)

    assert len(ref_set._refs) == 1
    assert ref_set._refs[0][0] == str(existing)
