"""TDD tests for the DetectionStatus taxonomy.

Tests are written before implementation so they fail first, then pass once
the feature is wired in. Covers:
  - status constants defined on DetectionStatus
  - species_identifier branches set correct status
  - _build_caption renders correct lines per status
  - database round-trip for detection_status column
  - ingest tier-1 label mapping
  - metrics error_count + ERROR excluded from FP denominator
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ===========================================================================
# 1. DetectionStatus constants
# ===========================================================================

def test_detection_status_constants_exist():
    from data_models import DetectionStatus
    assert DetectionStatus.IDENTIFIED == "identified"
    assert DetectionStatus.ANIMAL_UNCERTAIN == "animal_uncertain"
    assert DetectionStatus.NO_ANIMAL == "no_animal"
    assert DetectionStatus.UNCLASSIFIABLE == "unclassifiable"
    assert DetectionStatus.ERROR == "error"


def test_identification_result_has_status_field():
    from data_models import IdentificationResult, DetectionStatus
    result = IdentificationResult(
        species_name="Fox",
        confidence=0.9,
        api_success=True,
        processing_time=0.5,
        status=DetectionStatus.IDENTIFIED,
    )
    assert result.status == DetectionStatus.IDENTIFIED


def test_identification_result_status_defaults_to_identified():
    """Default status should be IDENTIFIED so existing callers don't break."""
    from data_models import IdentificationResult
    result = IdentificationResult(
        species_name="Fox",
        confidence=0.9,
        api_success=True,
        processing_time=0.5,
    )
    from data_models import DetectionStatus
    assert result.status == DetectionStatus.IDENTIFIED


# ===========================================================================
# 2. species_identifier branch → status mapping
# ===========================================================================

def _make_config():
    """Minimal config stub for SpeciesIdentifier tests."""
    from config import Config
    return Config.create_test_config()


def _make_identifier():
    from config import Config
    from species_identifier import SpeciesIdentifier
    cfg = Config.create_test_config()
    identifier = SpeciesIdentifier(cfg)
    return identifier, cfg


def test_no_predictions_returned_sets_error_status(tmp_path):
    """When SpeciesNet returns empty predictions, status must be ERROR."""
    from data_models import DetectionStatus
    from species_identifier import SpeciesIdentifier
    from config import Config

    identifier, cfg = _make_identifier()
    identifier._model_loaded = True
    identifier._model = MagicMock()
    identifier._model.predict.return_value = {}  # empty — no 'predictions' key

    img = tmp_path / "test.jpg"
    img.write_bytes(b"fake")
    result = identifier.identify_species(img)
    assert result.status == DetectionStatus.ERROR


def test_no_animal_detected_sets_no_animal_status(tmp_path):
    """When MegaDetector finds no animals, status must be NO_ANIMAL."""
    from data_models import DetectionStatus
    from species_identifier import SpeciesIdentifier

    identifier, cfg = _make_identifier()
    identifier._model_loaded = True
    identifier._model = MagicMock()
    identifier._model.predict.return_value = {
        "predictions": [
            {
                "detections": [],  # no detections
                "prediction": "blank",
                "prediction_score": 0.0,
                "prediction_source": "classifier",
                "classifications": {},
            }
        ]
    }

    img = tmp_path / "test.jpg"
    img.write_bytes(b"fake")
    result = identifier.identify_species(img)
    assert result.status == DetectionStatus.NO_ANIMAL


def test_low_confidence_sets_animal_uncertain_status(tmp_path):
    """Animal detected but confidence < threshold → ANIMAL_UNCERTAIN."""
    from data_models import DetectionStatus
    from species_identifier import SpeciesIdentifier

    identifier, cfg = _make_identifier()
    # unknown_species_threshold default is 0.5
    identifier._model_loaded = True
    identifier._model = MagicMock()
    identifier._model.predict.return_value = {
        "predictions": [
            {
                "detections": [{"category": "animal", "conf": 0.9, "bbox": [0, 0, 1, 1]}],
                "prediction": "some;species;path;here;genus;specificname;Common Name",
                "prediction_score": 0.2,  # below threshold 0.5
                "prediction_source": "classifier",
                "classifications": {
                    "classes": ["some;species;path;here;genus;specificname;Common Name"],
                    "scores": [0.2],
                },
            }
        ]
    }

    img = tmp_path / "test.jpg"
    img.write_bytes(b"fake")
    result = identifier.identify_species(img)
    assert result.status == DetectionStatus.ANIMAL_UNCERTAIN


def test_high_confidence_sets_identified_status(tmp_path):
    """Animal detected and confidence ≥ threshold → IDENTIFIED."""
    from data_models import DetectionStatus
    from species_identifier import SpeciesIdentifier

    identifier, cfg = _make_identifier()
    identifier._model_loaded = True
    identifier._model = MagicMock()
    identifier._model.predict.return_value = {
        "predictions": [
            {
                "detections": [{"category": "animal", "conf": 0.9, "bbox": [0, 0, 1, 1]}],
                "prediction": "abc;mammalia;carnivora;canidae;vulpes;vulpes;Red Fox",
                "prediction_score": 0.85,  # above threshold 0.5
                "prediction_source": "classifier",
                "classifications": {
                    "classes": ["abc;mammalia;carnivora;canidae;vulpes;vulpes;Red Fox"],
                    "scores": [0.85],
                },
            }
        ]
    }

    img = tmp_path / "test.jpg"
    img.write_bytes(b"fake")
    result = identifier.identify_species(img)
    assert result.status == DetectionStatus.IDENTIFIED


def test_no_cv_result_sentinel_sets_unclassifiable_status(tmp_path):
    """When SpeciesNet returns 'no cv result' as species name, status = UNCLASSIFIABLE."""
    from data_models import DetectionStatus
    from species_identifier import SpeciesIdentifier

    identifier, cfg = _make_identifier()
    identifier._model_loaded = True
    identifier._model = MagicMock()
    identifier._model.predict.return_value = {
        "predictions": [
            {
                "detections": [{"category": "animal", "conf": 0.9, "bbox": [0, 0, 1, 1]}],
                "prediction": "no cv result",
                "prediction_score": 0.0,
                "prediction_source": "no_cv",
                "classifications": {},
            }
        ]
    }

    img = tmp_path / "test.jpg"
    img.write_bytes(b"fake")
    result = identifier.identify_species(img)
    assert result.status == DetectionStatus.UNCLASSIFIABLE


def test_exception_in_identifier_sets_error_status(tmp_path):
    """A pipeline exception in identify_species must yield status=ERROR."""
    from data_models import DetectionStatus
    from species_identifier import SpeciesIdentifier

    identifier, cfg = _make_identifier()
    identifier._model_loaded = True
    identifier._model = MagicMock()
    identifier._model.predict.side_effect = RuntimeError("unexpected crash")

    img = tmp_path / "test.jpg"
    img.write_bytes(b"fake")
    result = identifier.identify_species(img)
    assert result.status == DetectionStatus.ERROR


def test_image_not_found_sets_error_status(tmp_path):
    """Missing image file → status=ERROR."""
    from data_models import DetectionStatus
    from species_identifier import SpeciesIdentifier

    identifier, cfg = _make_identifier()
    result = identifier.identify_species(tmp_path / "nonexistent.jpg")
    assert result.status == DetectionStatus.ERROR


# ===========================================================================
# 3. _build_caption renders correct lines per status
# ===========================================================================

def _make_system(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    for mod in ("wildlife_system", "config"):
        sys.modules.pop(mod, None)
    from wildlife_system import WildlifeSystem
    from database_manager import DatabaseManager
    sys_obj = WildlifeSystem.__new__(WildlifeSystem)
    sys_obj.config = __import__("config").Config.create_test_config()
    return sys_obj


def _species_result_for_status(status, species_name="Red Fox", confidence=0.85):
    """Build a minimal species_result dict for caption tests."""
    from data_models import DetectionResult, IdentificationResult, DetectionStatus
    boxes = [{"confidence": 0.9, "bbox": [0.1, 0.1, 0.5, 0.5]}] if status != DetectionStatus.NO_ANIMAL else []
    animals_detected = status not in (DetectionStatus.NO_ANIMAL, DetectionStatus.ERROR)
    det = DetectionResult(
        animals_detected=animals_detected,
        detection_count=len(boxes),
        bounding_boxes=boxes,
        detections=[],
        processing_time=0.1,
    )
    result = IdentificationResult(
        species_name=species_name,
        confidence=confidence,
        api_success=status not in (DetectionStatus.ERROR,),
        processing_time=0.5,
        fallback_reason="processing error" if status == DetectionStatus.ERROR else None,
        detection_result=det,
        animals_detected=animals_detected,
        status=status,
    )
    # Build dict as process_detection does
    return {
        "species_name": result.species_name,
        "confidence": result.confidence,
        "api_success": result.api_success,
        "processing_time": result.processing_time,
        "fallback_reason": result.fallback_reason,
        "animals_detected": result.animals_detected,
        "detection_count": result.detection_count if hasattr(result, "detection_count") else 1,
        "detection_result": result.detection_result,
        "metadata": result.metadata,
        "detection_id": 42,
        "detection_status": result.status,
    }


def test_caption_identified_shows_species_and_conf(monkeypatch, tmp_path):
    from data_models import DetectionStatus
    sys_obj = _make_system(monkeypatch, tmp_path)
    sr = _species_result_for_status(DetectionStatus.IDENTIFIED, "Red Fox", 0.85)
    caption = sys_obj._build_caption(sr, 5000, datetime(2026, 6, 9, 14, 30, 0))
    assert "Red Fox" in caption
    assert "85%" in caption
    # Should NOT contain "uncertain" or "failed"
    assert "uncertain" not in caption.lower()
    assert "failed" not in caption.lower()


def test_caption_animal_uncertain_shows_animal_line(monkeypatch, tmp_path):
    from data_models import DetectionStatus
    sys_obj = _make_system(monkeypatch, tmp_path)
    sr = _species_result_for_status(DetectionStatus.ANIMAL_UNCERTAIN, "Unknown species", 0.2)
    caption = sys_obj._build_caption(sr, 5000, datetime(2026, 6, 9, 14, 30, 0))
    assert "uncertain" in caption.lower()
    assert "🐾" in caption


def test_caption_no_animal_shows_false_positive_line(monkeypatch, tmp_path):
    from data_models import DetectionStatus
    sys_obj = _make_system(monkeypatch, tmp_path)
    sr = _species_result_for_status(DetectionStatus.NO_ANIMAL)
    caption = sys_obj._build_caption(sr, 5000, datetime(2026, 6, 9, 14, 30, 0))
    assert "No animal" in caption or "no animal" in caption.lower()
    assert "false positive" in caption.lower()
    assert "👁️" in caption


def test_caption_unclassifiable_shows_quality_line(monkeypatch, tmp_path):
    from data_models import DetectionStatus
    sys_obj = _make_system(monkeypatch, tmp_path)
    sr = _species_result_for_status(DetectionStatus.UNCLASSIFIABLE, "no cv result", 0.0)
    caption = sys_obj._build_caption(sr, 5000, datetime(2026, 6, 9, 14, 30, 0))
    assert "🐾" in caption
    assert "classif" in caption.lower()  # "classify" or "classifiable"


def test_caption_error_shows_warning_line(monkeypatch, tmp_path):
    from data_models import DetectionStatus
    sys_obj = _make_system(monkeypatch, tmp_path)
    sr = _species_result_for_status(DetectionStatus.ERROR)
    sr["fallback_reason"] = "processing error"
    caption = sys_obj._build_caption(sr, 5000, datetime(2026, 6, 9, 14, 30, 0))
    assert "⚠️" in caption
    assert "failed" in caption.lower() or "error" in caption.lower()


# ===========================================================================
# 4. DB round-trip: detection_status column
# ===========================================================================

def _make_db(tmp_path):
    from database_manager import DatabaseManager
    db_path = tmp_path / "detections.db"
    config = SimpleNamespace(storage=SimpleNamespace(database_path=str(db_path)))
    return DatabaseManager(config), str(db_path)


def test_detection_status_column_in_extra_columns():
    from database_manager import DatabaseManager
    assert "detection_status" in DatabaseManager._DETECTION_EXTRA_COLUMNS


def test_log_detection_persists_detection_status(tmp_path):
    db, db_path = _make_db(tmp_path)
    det_id = db.log_detection(
        image_path="test.jpg",
        motion_area=1000,
        detection_status="no_animal",
    )
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT detection_status FROM detections WHERE id=?",
                          (det_id,)).fetchone()
    assert row["detection_status"] == "no_animal"


def test_log_detection_status_defaults_to_none(tmp_path):
    """Old callers that don't pass detection_status get NULL."""
    db, db_path = _make_db(tmp_path)
    det_id = db.log_detection(image_path="test.jpg", motion_area=1000)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT detection_status FROM detections WHERE id=?",
                          (det_id,)).fetchone()
    assert row["detection_status"] is None


def test_migration_adds_detection_status_to_old_db(tmp_path):
    """Migration must add detection_status to a pre-existing DB."""
    from database_manager import DatabaseManager

    db_path = tmp_path / "old.db"
    # Create minimal old schema without detection_status
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("""
            CREATE TABLE detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT NOT NULL,
                motion_area INTEGER,
                species_name TEXT DEFAULT 'Unknown species',
                confidence_score REAL DEFAULT 0.0,
                processing_time REAL,
                api_success BOOLEAN DEFAULT FALSE
            )
        """)
        conn.execute("INSERT INTO detections (image_path, motion_area) VALUES (?, ?)",
                    ("old.jpg", 500))
        conn.commit()

    config = SimpleNamespace(storage=SimpleNamespace(database_path=str(db_path)))
    DatabaseManager(config)

    with sqlite3.connect(str(db_path)) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(detections)")}
    assert "detection_status" in cols


# ===========================================================================
# 5. ingest.py tier-1 label mapping
# ===========================================================================

@pytest.fixture
def ingest_db(tmp_path, monkeypatch):
    db_file = tmp_path / "detections.db"
    monkeypatch.setenv("STORAGE_DATABASE_PATH", str(db_file))
    monkeypatch.setenv("STORAGE_DATA_DIR", str(tmp_path))
    from config import Config
    from database_manager import DatabaseManager
    cfg = Config.create_test_config()
    return DatabaseManager(cfg)


def _add_detection_with_status(db, animals_detected, detection_status=None):
    return db.log_detection(
        image_path="/x.jpg",
        motion_area=1000,
        animals_detected=animals_detected,
        detection_count=1 if animals_detected else 0,
        gate_would_suppress=not animals_detected,
        detection_status=detection_status,
    )


def test_ingest_error_status_yields_none_tier1(ingest_db):
    """ERROR status rows must have tier1=None so they're excluded from labeling."""
    from loop import ingest
    did = _add_detection_with_status(ingest_db, False, detection_status="error")
    rows = ingest.reconcile(ingest_db, since_id=0)
    row = next(r for r in rows if r["detection_id"] == did)
    assert row["tier1"] is None
    assert row["reconciled_label"] is None


def test_ingest_no_animal_status_yields_false_positive(ingest_db):
    """NO_ANIMAL status → tier1=false_positive."""
    from loop import ingest
    did = _add_detection_with_status(ingest_db, False, detection_status="no_animal")
    rows = ingest.reconcile(ingest_db, since_id=0)
    row = next(r for r in rows if r["detection_id"] == did)
    assert row["tier1"] == "false_positive"
    assert row["reconciled_label"] == "false_positive"


def test_ingest_identified_status_yields_animal(ingest_db):
    """IDENTIFIED status → tier1=animal."""
    from loop import ingest
    did = _add_detection_with_status(ingest_db, True, detection_status="identified")
    rows = ingest.reconcile(ingest_db, since_id=0)
    row = next(r for r in rows if r["detection_id"] == did)
    assert row["tier1"] == "animal"
    assert row["reconciled_label"] == "animal"


def test_ingest_animal_uncertain_status_yields_animal(ingest_db):
    """ANIMAL_UNCERTAIN → tier1=animal."""
    from loop import ingest
    did = _add_detection_with_status(ingest_db, True, detection_status="animal_uncertain")
    rows = ingest.reconcile(ingest_db, since_id=0)
    row = next(r for r in rows if r["detection_id"] == did)
    assert row["tier1"] == "animal"


def test_ingest_unclassifiable_status_yields_false_positive(ingest_db):
    """UNCLASSIFIABLE → tier1=false_positive (exp #4, 2026-06-10).

    MegaDetector boxes a region the classifier cannot ID; empirically these are
    wind-blown vegetation / the swinging bird-feeder (27/27 human labels = FP, 0
    animal), not real wildlife.  Mapping them to "animal" was the dominant
    label-trust bias that under-counted FP on unlabelled rows.
    """
    from loop import ingest
    did = _add_detection_with_status(ingest_db, True, detection_status="unclassifiable")
    rows = ingest.reconcile(ingest_db, since_id=0)
    row = next(r for r in rows if r["detection_id"] == did)
    assert row["tier1"] == "false_positive"


def test_ingest_legacy_null_status_uses_animals_detected(ingest_db):
    """NULL detection_status (legacy rows) → fallback to animals_detected behavior."""
    from loop import ingest
    # Legacy row: no detection_status, animals_detected=False → false_positive
    did = _add_detection_with_status(ingest_db, False, detection_status=None)
    rows = ingest.reconcile(ingest_db, since_id=0)
    row = next(r for r in rows if r["detection_id"] == did)
    assert row["tier1"] == "false_positive"


def test_ingest_error_status_human_override_still_works(ingest_db):
    """Even an ERROR row can be overridden by a human label."""
    from loop import ingest
    did = _add_detection_with_status(ingest_db, False, detection_status="error")
    ingest_db.add_feedback(did, "false_positive", source="human")
    rows = ingest.reconcile(ingest_db, since_id=0)
    row = next(r for r in rows if r["detection_id"] == did)
    assert row["tier1"] is None           # still None
    assert row["human"] == "false_positive"
    assert row["reconciled_label"] == "false_positive"  # human wins


def test_ingest_detection_status_present_in_row_dict(ingest_db):
    """reconcile() must expose detection_status on each row dict."""
    from loop import ingest
    did = _add_detection_with_status(ingest_db, False, detection_status="no_animal")
    rows = ingest.reconcile(ingest_db, since_id=0)
    row = next(r for r in rows if r["detection_id"] == did)
    assert "detection_status" in row
    assert row["detection_status"] == "no_animal"


# ===========================================================================
# 6. metrics: ERROR rows excluded from FP denominator, error_count reported
# ===========================================================================

def test_compute_metrics_error_rows_excluded_from_fp_denominator():
    """ERROR rows have reconciled_label=None, so they must not count toward FP denominator."""
    from loop import metrics
    rows = [
        {"reconciled_label": "false_positive"},
        {"reconciled_label": "false_positive"},
        {"reconciled_label": "animal"},
        {"reconciled_label": None, "detection_status": "error"},  # must be excluded
        {"reconciled_label": None, "detection_status": "error"},
    ]
    m = metrics.compute_metrics(rows, fn_audit=None)
    # labeled_triggers = 3 (not 5); fp_count = 2; fp_rate = 2/3
    assert m["labeled_triggers"] == 3
    assert m["fp_count"] == 2
    assert abs(m["fp_rate"] - 2 / 3) < 1e-9


def test_compute_metrics_reports_error_count():
    """compute_metrics must include error_count for ERROR-status rows."""
    from loop import metrics
    rows = [
        {"reconciled_label": "animal", "detection_status": "identified"},
        {"reconciled_label": None, "detection_status": "error"},
        {"reconciled_label": None, "detection_status": "error"},
        {"reconciled_label": "false_positive", "detection_status": "no_animal"},
    ]
    m = metrics.compute_metrics(rows, fn_audit=None)
    assert m["error_count"] == 2


def test_compute_metrics_error_count_zero_when_no_errors():
    """error_count must be 0 (not absent) when no errors exist."""
    from loop import metrics
    rows = [
        {"reconciled_label": "animal"},
        {"reconciled_label": "false_positive"},
    ]
    m = metrics.compute_metrics(rows, fn_audit=None)
    assert m["error_count"] == 0


def test_compute_metrics_error_count_in_last_metrics(tmp_path, monkeypatch):
    """metrics main() must include error_count in last_metrics written to state.json."""
    import json
    import sys

    state_path = tmp_path / "state.json"
    db_file = tmp_path / "detections.db"
    csv_path = tmp_path / "daily.csv"

    monkeypatch.setenv("STORAGE_DATABASE_PATH", str(db_file))
    monkeypatch.setenv("STORAGE_DATA_DIR", str(tmp_path))
    from config import Config
    from database_manager import DatabaseManager
    from loop import state as state_mod

    cfg = Config.create_test_config()
    db = DatabaseManager(cfg)
    # Insert one error and one animal detection
    db.log_detection(
        image_path="/a.jpg", motion_area=500,
        animals_detected=False, detection_status="error",
    )
    db.log_detection(
        image_path="/b.jpg", motion_area=800,
        animals_detected=True, detection_status="identified",
    )

    state_mod.save_state(state_path, {"watermark": 0, "paused": False})
    monkeypatch.setattr(
        sys, "argv",
        ["loop.metrics", "--state", str(state_path),
         "--csv", str(csv_path), "--date", "2026-06-09"],
    )
    from loop import metrics as metrics_mod
    metrics_mod.main()

    st = state_mod.load_state(state_path)
    lm = st.get("last_metrics")
    assert lm is not None
    assert "error_count" in lm
    assert lm["error_count"] == 1


# ===========================================================================
# 7. process_detection exception path sets ERROR status
# ===========================================================================

def test_process_detection_exception_returns_error_status(monkeypatch, tmp_path):
    """When process_detection catches an exception, result must carry ERROR status."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    for mod in ("wildlife_system", "config"):
        sys.modules.pop(mod, None)

    from wildlife_system import WildlifeSystem
    from database_manager import DatabaseManager
    from data_models import DetectionStatus

    sys_obj = WildlifeSystem.__new__(WildlifeSystem)
    sys_obj.config = __import__("config").Config.create_test_config()
    sys_obj.database = DatabaseManager(sys_obj.config)
    sys_obj.reference_drift = None
    sys_obj.reference_frame = None

    # Make species_identifier crash
    sys_obj.species_identifier = MagicMock()
    sys_obj.species_identifier.identify_species.side_effect = RuntimeError("boom")

    result, _ = sys_obj.process_detection(tmp_path / "test.jpg", 1000)
    assert result.get("detection_status") == DetectionStatus.ERROR


def test_caption_blank_shows_no_animal(monkeypatch, tmp_path):
    from data_models import DetectionStatus
    sys_obj = _make_system(monkeypatch, tmp_path)
    sr = _species_result_for_status(DetectionStatus.IDENTIFIED, "blank", 1.0)
    caption = sys_obj._build_caption(sr, 5000, datetime(2026, 6, 9, 14, 30, 0))
    assert "🚫 No animal" in caption
    # Must NOT fall back to a species emoji like the default 🦌
    assert "🦌" not in caption
    # Detector box confidence still surfaced
    assert "Box:" in caption
