"""
Unit tests for the timelapse FN-audit writer (ADR-004 Phase 1).
"""

import sys
from types import SimpleNamespace

import numpy as np

sys.path.append('src')

from timelapse_writer import TimelapseWriter


def _make_writer(tmp_path, enabled=True, interval=20.0, max_files=10000):
    config = SimpleNamespace(
        performance=SimpleNamespace(
            enable_timelapse=enabled,
            timelapse_interval=interval,
            timelapse_max_files=max_files,
        ),
        storage=SimpleNamespace(data_dir=tmp_path),
    )
    return TimelapseWriter(config), tmp_path / "timelapse"


def _frame():
    return np.zeros((480, 640), dtype=np.uint8)


def test_respects_interval(tmp_path):
    writer, out = _make_writer(tmp_path, interval=20.0)
    assert writer.maybe_capture(_frame(), now=100.0) is True
    # Same window → no second write
    assert writer.maybe_capture(_frame(), now=110.0) is False
    assert len(list(out.glob("*.jpg"))) == 1


def test_writes_after_interval(tmp_path):
    writer, out = _make_writer(tmp_path, interval=20.0)
    assert writer.maybe_capture(_frame(), now=100.0) is True
    assert writer.maybe_capture(_frame(), now=121.0) is True
    assert len(list(out.glob("*.jpg"))) == 2


def test_rotation_bounds_file_count(tmp_path):
    writer, out = _make_writer(tmp_path, interval=0.0, max_files=3)
    for i in range(6):
        # interval=0 → every call writes; distinct timestamps keep names unique
        assert writer.maybe_capture(_frame(), now=100.0 + i) is True
    assert len(list(out.glob("*.jpg"))) == 3


def test_disabled_writes_nothing(tmp_path):
    writer, out = _make_writer(tmp_path, enabled=False)
    assert writer.maybe_capture(_frame(), now=100.0) is False
    assert not out.exists() or len(list(out.glob("*.jpg"))) == 0


def test_none_frame_is_noop(tmp_path):
    writer, out = _make_writer(tmp_path)
    assert writer.maybe_capture(None, now=100.0) is False
    assert len(list(out.glob("*.jpg"))) == 0


def test_rgb_frame_is_converted(tmp_path):
    writer, out = _make_writer(tmp_path)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    assert writer.maybe_capture(rgb, now=100.0) is True
    assert len(list(out.glob("*.jpg"))) == 1
