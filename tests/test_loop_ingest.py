"""Tests for src/loop/ingest.py — watermark + reconciliation, read-only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from config import Config
from database_manager import DatabaseManager
from loop import ingest


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A real DatabaseManager pointed at a temp SQLite file."""
    db_file = tmp_path / "detections.db"
    monkeypatch.setenv("STORAGE_DATABASE_PATH", str(db_file))
    monkeypatch.setenv("STORAGE_DATA_DIR", str(tmp_path))
    cfg = Config.create_test_config()
    return DatabaseManager(cfg)


def _add_detection(db, animals_detected):
    return db.log_detection(
        image_path="/x.jpg", motion_area=1000,
        animals_detected=animals_detected, detection_count=1 if animals_detected else 0,
        gate_would_suppress=not animals_detected,
    )


def test_tier1_label_from_animals_detected(db):
    did = _add_detection(db, animals_detected=False)
    rows = ingest.reconcile(db, since_id=0)
    row = next(r for r in rows if r["detection_id"] == did)
    assert row["tier1"] == "false_positive"
    assert row["reconciled_label"] == "false_positive"


def test_human_overrides_tier1(db):
    did = _add_detection(db, animals_detected=False)
    db.add_feedback(did, "animal", source="human")
    rows = ingest.reconcile(db, since_id=0)
    row = next(r for r in rows if r["detection_id"] == did)
    assert row["human"] == "animal"
    assert row["reconciled_label"] == "animal"  # human > tier1


def test_latest_human_row_wins(db):
    did = _add_detection(db, animals_detected=True)
    db.add_feedback(did, "animal", source="human")
    db.add_feedback(did, "false_positive", source="human")  # corrected
    rows = ingest.reconcile(db, since_id=0)
    row = next(r for r in rows if r["detection_id"] == did)
    assert row["reconciled_label"] == "false_positive"


def test_watermark_filters_old_rows(db):
    first = _add_detection(db, animals_detected=True)
    second = _add_detection(db, animals_detected=False)
    rows = ingest.reconcile(db, since_id=first)
    ids = {r["detection_id"] for r in rows}
    assert second in ids and first not in ids


def test_max_id_reports_new_watermark(db):
    _add_detection(db, animals_detected=True)
    second = _add_detection(db, animals_detected=False)
    result = ingest.ingest(db, since_id=0)
    assert result["new_watermark"] == second
    assert len(result["rows"]) == 2


def test_reconcile_does_not_write_feedback(db):
    did = _add_detection(db, animals_detected=False)
    before = db.get_feedback(did)
    ingest.reconcile(db, since_id=0)
    after = db.get_feedback(did)
    assert before == after  # pure read
