"""
Integration test for wildlife_system main-loop behavior.

Verifies that the post-detection cooldown does NOT starve MOG2 — i.e.
motion_detector.detect() must keep being called every tick during the
cooldown window so the background model can track scene drift.
"""

import asyncio
import sqlite3
import sys
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock

import numpy as np
import pytest

sys.path.append('src')


@pytest.fixture
def system(monkeypatch, tmp_path):
    """A WildlifeSystem with an isolated temp DB and no real telegram/camera."""
    monkeypatch.setenv('TELEGRAM_BOT_TOKEN', 'test_token')
    monkeypatch.setenv('TELEGRAM_CHAT_ID', 'test_chat')
    monkeypatch.setenv('MOTION_WARMUP_SECONDS', '0')
    monkeypatch.setenv('PERFORMANCE_ENABLE_TIMELAPSE', 'false')
    for mod in ('wildlife_system', 'config'):
        sys.modules.pop(mod, None)

    from wildlife_system import WildlifeSystem
    from database_manager import DatabaseManager

    sys_obj = WildlifeSystem()
    # Isolate the database to a temp file.
    cfg = SimpleNamespace(storage=SimpleNamespace(database_path=str(tmp_path / "d.db")))
    sys_obj.database = DatabaseManager(cfg)
    sys_obj.reference_frame = None  # skip frame-stability imread
    return sys_obj


def _identification(animals_detected, boxes=None):
    from data_models import IdentificationResult, DetectionResult
    boxes = boxes or []
    det = DetectionResult(
        animals_detected=animals_detected,
        detection_count=len(boxes),
        bounding_boxes=boxes,
        detections=[],
        processing_time=0.1,
    )
    return IdentificationResult(
        species_name="Fox" if animals_detected else "Unknown species",
        confidence=0.9 if animals_detected else 0.0,
        api_success=True,
        processing_time=0.5,
        detection_result=det,
        animals_detected=animals_detected,
    )


def _identification_no_animal():
    """A review-class (NO_ANIMAL) result — status is what is_review_detection
    actually keys off, distinct from the plain `animals_detected=False` used
    by `_identification(False)` (which leaves status at its IDENTIFIED
    default and is not review-class).
    """
    from data_models import IdentificationResult, DetectionResult, DetectionStatus
    det = DetectionResult(
        animals_detected=False,
        detection_count=0,
        bounding_boxes=[],
        detections=[],
        processing_time=0.1,
    )
    return IdentificationResult(
        species_name="Unknown species",
        confidence=0.0,
        api_success=True,
        processing_time=0.5,
        detection_result=det,
        animals_detected=False,
        status=DetectionStatus.NO_ANIMAL,
    )


def _identification_human(confidence=0.9):
    from data_models import IdentificationResult, DetectionResult, DetectionStatus
    det = DetectionResult(
        animals_detected=True,
        detection_count=1,
        bounding_boxes=[{'confidence': confidence, 'category': '1'}],
        detections=[],
        processing_time=0.1,
    )
    return IdentificationResult(
        species_name="Homo sapiens",
        confidence=confidence,
        api_success=True,
        processing_time=0.5,
        detection_result=det,
        animals_detected=True,
        status=DetectionStatus.HUMAN,
    )


def test_process_detection_persists_richer_fields_and_id(system):
    from data_models import MotionResult
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification(True, boxes=[{'confidence': 0.7}, {'confidence': 0.85}])
    )
    motion = MotionResult(
        motion_detected=True, motion_area=5000, contour_count=3,
        largest_contour_area=2200, foreground_pixel_count=3300,
    )

    result, ts = system.process_detection("capture.jpg", 5000, motion)

    assert result['detection_id'] is not None
    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?",
                           (result['detection_id'],)).fetchone()
    assert row['animals_detected'] == 1
    assert row['detection_count'] == 2
    assert row['max_detection_confidence'] == pytest.approx(0.85)
    assert row['contour_count'] == 3
    assert row['largest_contour_area'] == 2200
    assert row['foreground_pixel_count'] == 3300
    assert row['gate_would_suppress'] == 0  # animal present → would NOT suppress


def test_process_detection_shadow_gate_records_suppression(system):
    """No animal → gate would suppress, but the row is still written (shadow mode)."""
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification(False)
    )
    result, _ = system.process_detection("capture.jpg", 800, None)

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?",
                           (result['detection_id'],)).fetchone()
    assert row['gate_would_suppress'] == 1
    assert row['animals_detected'] == 0


def test_process_detection_persists_sharpness_and_person_confidence(system):
    """Task 1 (ADR-004 observability): process_detection reads sharpness
    values from its sharpness_info parameter and person_confidence from the
    identification result's metadata, and logs all three to the DB."""
    from data_models import IdentificationResult, DetectionResult

    det = DetectionResult(
        animals_detected=True, detection_count=1,
        bounding_boxes=[{'confidence': 0.7}], detections=[],
        processing_time=0.1,
    )
    identification = IdentificationResult(
        species_name="Fox", confidence=0.9, api_success=True, processing_time=0.5,
        detection_result=det, animals_detected=True,
        metadata={'person_confidence': 0.22},
    )
    system.species_identifier.identify_species = MagicMock(return_value=identification)

    sharpness_info = {
        'sharpness_score': 18.4,
        'below_sharpness_floor': False,
        'selected_frame_index': 0,
        'frame_count': 5,
        'all_scores': [18.4] * 5,
        'meets_threshold': True,
        'all_frame_paths': [],
    }

    result, ts = system.process_detection("capture.jpg", 5000, None, sharpness_info)

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?",
                           (result['detection_id'],)).fetchone()
    assert row['sharpness_score'] == pytest.approx(18.4)
    assert row['below_sharpness_floor'] == 0
    assert row['person_confidence'] == pytest.approx(0.22)


def test_process_detection_observability_fields_null_without_sharpness_info(system):
    """No sharpness_info passed (single-frame capture path) → sharpness
    columns are NULL, not an error; person_confidence still flows from
    metadata when present."""
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification(True, boxes=[{'confidence': 0.7}])
    )

    result, ts = system.process_detection("capture.jpg", 5000, None)

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?",
                           (result['detection_id'],)).fetchone()
    assert row['sharpness_score'] is None
    assert row['below_sharpness_floor'] is None
    assert row['person_confidence'] is None


def test_process_detection_persists_top_species_guess(system):
    """Task 3 (ADR-004 observability): process_detection reads
    metadata['top_classifier_prediction'] and persists label/score as
    top_species_raw / top_species_score."""
    from data_models import IdentificationResult, DetectionResult

    det = DetectionResult(
        animals_detected=True, detection_count=1,
        bounding_boxes=[{'confidence': 0.7}], detections=[],
        processing_time=0.1,
    )
    identification = IdentificationResult(
        species_name="aves;;;;;bird", confidence=0.8, api_success=True, processing_time=0.5,
        detection_result=det, animals_detected=True,
        metadata={
            'top_classifier_prediction': {
                'label': 'def;aves;passeriformes;turdidae;turdus;merula;eurasian blackbird',
                'score': 0.34,
            },
        },
    )
    system.species_identifier.identify_species = MagicMock(return_value=identification)

    result, ts = system.process_detection("capture.jpg", 5000, None)

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?",
                           (result['detection_id'],)).fetchone()
    assert row['top_species_raw'] == 'def;aves;passeriformes;turdidae;turdus;merula;eurasian blackbird'
    assert row['top_species_score'] == pytest.approx(0.34)


def test_process_detection_top_species_guess_null_without_metadata(system):
    """No top_classifier_prediction in metadata (or no metadata at all) →
    columns stay NULL, not an error."""
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification(True, boxes=[{'confidence': 0.7}])
    )

    result, ts = system.process_detection("capture.jpg", 5000, None)

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?",
                           (result['detection_id'],)).fetchone()
    assert row['top_species_raw'] is None
    assert row['top_species_score'] is None


def test_process_detection_top_species_guess_null_when_metadata_value_not_dict(system):
    """Never-crash constraint: a malformed (non-dict) top_classifier_prediction
    in metadata must not raise — the detection is still logged, with the
    top-species columns NULL."""
    from data_models import IdentificationResult, DetectionResult

    det = DetectionResult(
        animals_detected=True, detection_count=1,
        bounding_boxes=[{'confidence': 0.7}], detections=[],
        processing_time=0.1,
    )
    identification = IdentificationResult(
        species_name="aves;;;;;bird", confidence=0.8, api_success=True, processing_time=0.5,
        detection_result=det, animals_detected=True,
        metadata={'top_classifier_prediction': "not-a-dict"},
    )
    system.species_identifier.identify_species = MagicMock(return_value=identification)

    result, ts = system.process_detection("capture.jpg", 5000, None)

    assert result['detection_id'] is not None  # row written, no crash
    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?",
                           (result['detection_id'],)).fetchone()
    assert row['top_species_raw'] is None
    assert row['top_species_score'] is None


def test_process_detection_fails_closed_to_human_on_db_error(system):
    """If species ID found a HUMAN but the DB write then blows up (e.g. disk
    full / WAL contention), the fallback must still report HUMAN so the
    Telegram suppression gate fires — NOT ERROR, which would leak the photo.
    """
    from data_models import DetectionStatus

    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_human()
    )
    system.database.log_detection = MagicMock(side_effect=Exception("disk full"))

    result, _ = system.process_detection("capture.jpg", 5000, None)

    assert result['detection_status'] == DetectionStatus.HUMAN
    assert result['detection_id'] is None


def test_process_detection_stays_error_when_identify_species_throws(system):
    """If identify_species itself raises, we never got a species_result, so
    we genuinely don't know if a human was present — fallback must stay ERROR.
    """
    from data_models import DetectionStatus

    system.species_identifier.identify_species = MagicMock(
        side_effect=Exception("model crashed")
    )

    result, _ = system.process_detection("capture.jpg", 5000, None)

    assert result['detection_status'] == DetectionStatus.ERROR
    assert result['detection_id'] is None


# ===========================================================================
# Task 3 (ADR-004 observability): "Best guess" caption line
#
# SpeciesNet's ensemble often rolls a low-confidence species-level guess up
# to a generic label (e.g. "aves;;;;;bird"). The classifier's raw top-1
# prediction (metadata['top_classifier_prediction']) is more specific and
# should be surfaced — but only when it actually adds information.
# ===========================================================================

def _identified_species_result(species_name_raw, confidence=0.8, metadata=None):
    from data_models import DetectionResult, DetectionStatus
    det = DetectionResult(
        animals_detected=True, detection_count=1,
        bounding_boxes=[{'confidence': 0.8}], detections=[],
        processing_time=0.1,
    )
    return {
        'species_name': species_name_raw,
        'confidence': confidence,
        'detection_status': DetectionStatus.IDENTIFIED,
        'detection_result': det,
        'metadata': metadata if metadata is not None else {},
        'fallback_reason': None,
    }


def test_build_caption_shows_best_guess_for_generic_rollup(system):
    """Ensemble rolled up to 'aves;;;;;bird' (genus/species empty) and the
    classifier's raw top-1 is species-level and non-generic → caption shows
    'Best guess: eurasian blackbird (34%)', even though the score is low."""
    species_result = _identified_species_result(
        "abc;aves;;;;;bird",
        confidence=0.8,
        metadata={
            'top_classifier_prediction': {
                'label': 'def;aves;passeriformes;turdidae;turdus;merula;eurasian blackbird',
                'score': 0.34,
            },
        },
    )
    caption = system._build_caption(species_result, 1000, datetime.now())
    assert "Best guess: eurasian blackbird (34%)" in caption


def test_build_caption_no_best_guess_when_ensemble_already_species_level(system):
    """Ensemble already resolved to a full species (genus+species present)
    — the best-guess line would add nothing, so it must not appear."""
    species_result = _identified_species_result(
        "abc;mammalia;carnivora;canidae;vulpes;vulpes;red fox",
        confidence=0.9,
        metadata={
            'top_classifier_prediction': {
                'label': 'abc;mammalia;carnivora;canidae;vulpes;vulpes;red fox',
                'score': 0.9,
            },
        },
    )
    caption = system._build_caption(species_result, 1000, datetime.now())
    assert "Best guess" not in caption


def test_build_caption_no_best_guess_when_top_classifier_prediction_missing(system):
    """Generic ensemble rollup but no classifier top-1 available at all."""
    species_result = _identified_species_result("abc;aves;;;;;bird", metadata={})
    caption = system._build_caption(species_result, 1000, datetime.now())
    assert "Best guess" not in caption


def test_build_caption_no_best_guess_when_top_prediction_itself_generic(system):
    """The classifier's own top-1 is itself a generic sentinel ('blank') —
    showing it would add no information, so it must be suppressed."""
    species_result = _identified_species_result(
        "abc;;;;;;animal",
        metadata={
            'top_classifier_prediction': {'label': 'def;;;;;;blank', 'score': 0.5},
        },
    )
    caption = system._build_caption(species_result, 1000, datetime.now())
    assert "Best guess" not in caption


def test_build_caption_never_crashes_on_malformed_top_classifier_prediction(system):
    """Never-crash constraint: a malformed metadata value must not block
    caption generation (or, by extension, the notification)."""
    species_result = _identified_species_result(
        "abc;aves;;;;;bird",
        metadata={'top_classifier_prediction': "not-a-dict"},
    )
    caption = system._build_caption(species_result, 1000, datetime.now())
    assert "📅" in caption  # caption still built despite the malformed value


def test_extract_common_name_helper():
    """Task 3: small utils helper — last non-empty semicolon segment."""
    from utils import extract_common_name

    assert extract_common_name(
        "def;aves;passeriformes;turdidae;turdus;merula;eurasian blackbird"
    ) == "eurasian blackbird"
    assert extract_common_name("aves;;;;;bird") == "bird"
    # Trailing empty segment (missing common name) — falls back further left.
    assert extract_common_name("abc;mammalia;carnivora;canidae;vulpes;vulpes;") == "vulpes"
    assert extract_common_name("") == ""
    assert extract_common_name(None) == ""


@pytest.mark.asyncio
async def test_send_notification_attaches_feedback_keyboard(system, tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.telegram_service = MagicMock()
    system.telegram_service.send_photo_with_caption = AsyncMock()
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0

    species_result = {'species_name': 'Fox', 'confidence': 0.9, 'animals_detected': True,
                      'detection_id': 123, 'detection_result': None}
    await system.send_notification(species_result, 5000, datetime.now(), image_path=img)

    _, kwargs = system.telegram_service.send_photo_with_caption.call_args
    keyboard = kwargs['reply_markup']
    data = [b.callback_data for b in keyboard.inline_keyboard[0]]
    assert data == ["fb:123:a", "fb:123:fp", "fb:123:ws"]


def _mock_telegram(system):
    system.telegram_service = MagicMock()
    system.telegram_service.send_photo_with_caption = AsyncMock()
    system.telegram_service.send_media_group = AsyncMock()
    system.telegram_service.send_text_message = AsyncMock()
    system.telegram_service.send_detection_notification = AsyncMock()
    system.telegram_service.send_document = AsyncMock()
    return system.telegram_service


@pytest.mark.asyncio
async def test_human_detection_suppresses_notification_but_still_cleans_up(system, tmp_path):
    """HUMAN detection + suppress_human_alerts=True (default): no Telegram
    call at all, but the DB row is still written (by process_detection) and
    cleanup_old_images still runs.
    """
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(return_value=_identification_human())
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    await system._process_and_notify_detection(img, 5000)

    telegram.send_photo_with_caption.assert_not_called()
    telegram.send_media_group.assert_not_called()
    telegram.send_detection_notification.assert_not_called()
    telegram.send_document.assert_not_called()
    system.cleanup_old_images.assert_called_once()

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    assert row['detection_status'] == 'human'


@pytest.mark.asyncio
async def test_non_human_detection_still_notifies(system, tmp_path):
    """Baseline: a normal animal detection is unaffected by the gate."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification(True, boxes=[{'confidence': 0.7}])
    )
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    await system._process_and_notify_detection(img, 5000)

    assert telegram.send_photo_with_caption.called or telegram.send_media_group.called
    system.cleanup_old_images.assert_called_once()


@pytest.mark.asyncio
async def test_human_detection_notifies_when_flag_disabled(system, tmp_path):
    """Escape hatch: PERFORMANCE_SUPPRESS_HUMAN_ALERTS=false still notifies on HUMAN."""
    system.config.performance.suppress_human_alerts = False
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(return_value=_identification_human())
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    await system._process_and_notify_detection(img, 5000)

    assert telegram.send_photo_with_caption.called or telegram.send_media_group.called
    system.cleanup_old_images.assert_called_once()


def _below_floor_sharpness_info(score=8.6):
    return {
        'sharpness_score': score,
        'selected_frame_index': 0,
        'frame_count': 5,
        'all_scores': [score] * 5,
        'meets_threshold': False,
        'below_sharpness_floor': True,
        'all_frame_paths': [],
    }


def _above_floor_sharpness_info(score=25.0):
    return {
        'sharpness_score': score,
        'selected_frame_index': 0,
        'frame_count': 5,
        'all_scores': [score] * 5,
        'meets_threshold': True,
        'below_sharpness_floor': False,
        'all_frame_paths': [],
    }


@pytest.mark.asyncio
async def test_blurry_animal_still_notifies(system, tmp_path):
    """Task 4: a below-floor burst that DID find an animal still alerts —
    a blurry bird beats no bird.
    """
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification(True, boxes=[{'confidence': 0.7}])
    )
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    await system._process_and_notify_detection(
        img, 5000, sharpness_info=_below_floor_sharpness_info()
    )

    assert telegram.send_photo_with_caption.called or telegram.send_media_group.called
    system.cleanup_old_images.assert_called_once()


@pytest.mark.asyncio
async def test_blurry_no_animal_suppresses_notification_but_logs_and_cleans_up(system, tmp_path, caplog):
    """Task 4: a below-floor burst with no animal found gets a DB row
    (process_detection logs unconditionally) but no Telegram send, and
    cleanup still runs — the blur gate no longer creates untracked drops
    and doesn't add REVIEW-channel volume either.
    """
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal()
    )
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    with caplog.at_level("INFO"):
        await system._process_and_notify_detection(
            img, 5000, sharpness_info=_below_floor_sharpness_info()
        )

    telegram.send_photo_with_caption.assert_not_called()
    telegram.send_media_group.assert_not_called()
    telegram.send_detection_notification.assert_not_called()
    telegram.send_document.assert_not_called()
    system.cleanup_old_images.assert_called_once()
    assert any("[BLUR]" in r.message for r in caplog.records)

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    assert row is not None
    assert row['detection_status'] == 'no_animal'


@pytest.mark.asyncio
async def test_sharp_no_animal_still_notifies_unchanged(system, tmp_path):
    """Baseline (must stay unchanged by Task 4): an above-floor burst with
    no animal found still gets a Telegram notification today (routed
    through the REVIEW-prefix path), it's just not the blur gate's job to
    suppress it.
    """
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal()
    )
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    await system._process_and_notify_detection(
        img, 5000, sharpness_info=_above_floor_sharpness_info()
    )

    assert telegram.send_photo_with_caption.called or telegram.send_media_group.called
    system.cleanup_old_images.assert_called_once()


@pytest.mark.asyncio
async def test_blurry_human_suppressed_via_human_gate_single_log(system, tmp_path, caplog):
    """Task 4: a below-floor HUMAN burst must be suppressed by the human
    gate (Task 2), not double-handled by the blur gate — exactly one
    suppression log line, and it's the [HUMAN-GATE] one, not [BLUR].
    """
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(return_value=_identification_human())
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    with caplog.at_level("INFO"):
        await system._process_and_notify_detection(
            img, 5000, sharpness_info=_below_floor_sharpness_info()
        )

    telegram.send_photo_with_caption.assert_not_called()
    telegram.send_media_group.assert_not_called()
    system.cleanup_old_images.assert_called_once()

    gate_logs = [r.message for r in caplog.records if "HUMAN-GATE" in r.message or "[BLUR]" in r.message]
    assert len(gate_logs) == 1
    assert "HUMAN-GATE" in gate_logs[0]


def test_capture_and_select_best_frame_below_floor_returns_path_not_none(system, tmp_path, monkeypatch):
    """Task 4: a below-floor best frame must NOT be silently discarded —
    the burst still yields a usable path + sharpness_info tagged
    below_sharpness_floor=True (previously this returned (None, None) and
    the whole burst vanished with no DB row).
    """
    import wildlife_system

    fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    system.camera.capture_burst_frames = MagicMock(return_value=[fake_frame] * 3)
    image_dir = system.config.storage.image_dir
    image_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = [image_dir / f"capture_x_frame{i}.jpg" for i in range(1, 4)]
    for p in saved_paths:
        p.write_bytes(b"fake")
    system.camera.save_burst_frames = MagicMock(return_value=saved_paths)

    below_floor_score = system.config.performance.min_sharpness_threshold - 2.0
    monkeypatch.setattr(
        wildlife_system.SharpnessAnalyzer,
        "select_sharpest_frame",
        staticmethod(lambda *a, **k: (fake_frame, 1, below_floor_score, [8.0, below_floor_score, 8.2])),
    )

    path, info = system._capture_and_select_best_frame()

    assert path == saved_paths[1]
    assert info is not None
    assert info['below_sharpness_floor'] is True
    assert info['meets_threshold'] is False
    assert info['sharpness_score'] == below_floor_score


def test_capture_and_select_best_frame_above_floor_flag_false(system, tmp_path, monkeypatch):
    """Baseline: an above-floor best frame is tagged below_sharpness_floor=False."""
    import wildlife_system

    fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    system.camera.capture_burst_frames = MagicMock(return_value=[fake_frame] * 3)
    image_dir = system.config.storage.image_dir
    image_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = [image_dir / f"capture_y_frame{i}.jpg" for i in range(1, 4)]
    for p in saved_paths:
        p.write_bytes(b"fake")
    system.camera.save_burst_frames = MagicMock(return_value=saved_paths)

    above_floor_score = system.config.performance.min_sharpness_threshold + 10.0
    monkeypatch.setattr(
        wildlife_system.SharpnessAnalyzer,
        "select_sharpest_frame",
        staticmethod(lambda *a, **k: (fake_frame, 0, above_floor_score, [above_floor_score] * 3)),
    )

    path, info = system._capture_and_select_best_frame()

    assert path == saved_paths[0]
    assert info['below_sharpness_floor'] is False
    assert info['meets_threshold'] is True


@pytest.mark.asyncio
async def test_cooldown_keeps_feeding_motion_detector(monkeypatch):
    """During post-detection cooldown the loop must keep calling
    motion_detector.detect() so MOG2 stays calibrated to the live scene.

    With the original (broken) cooldown gate, detect() is called exactly
    once: the first tick triggers a detection, sets last_detection_time,
    and every subsequent tick hits the early-`continue` and skips
    detect() entirely. After the fix detect() is called every tick,
    even while cooldown suppresses the heavy capture+ID+notify path.
    """
    monkeypatch.setenv('TELEGRAM_BOT_TOKEN', 'test_token')
    monkeypatch.setenv('TELEGRAM_CHAT_ID', 'test_chat')
    monkeypatch.setenv('MOTION_WARMUP_SECONDS', '0')
    monkeypatch.setenv('MOTION_FRAME_INTERVAL', '0.001')
    monkeypatch.setenv('PERFORMANCE_IDLE_SLEEP', '0')
    monkeypatch.setenv('PERFORMANCE_COOLDOWN_SLEEP', '0')
    monkeypatch.setenv('PERFORMANCE_DAYLIGHT_ONLY', 'false')
    monkeypatch.setenv('PERFORMANCE_COOLDOWN_PERIOD', '30')

    # Force config + wildlife_system reload so env vars are picked up fresh
    for mod in ('wildlife_system', 'config'):
        sys.modules.pop(mod, None)

    from wildlife_system import WildlifeSystem
    from data_models import MotionResult

    system = WildlifeSystem()

    detect_calls = []

    def detect_side_effect(frame):
        detect_calls.append(frame)
        # Tick 1 returns motion (arms cooldown). Subsequent ticks: no motion.
        first = len(detect_calls) == 1
        return MotionResult(
            motion_detected=first,
            motion_area=5000 if first else 0,
        )

    # Replace motion_detector with a mock we can observe
    system.motion_detector = MagicMock()
    system.motion_detector.detect.side_effect = detect_side_effect
    system.motion_detector.is_warming_up = False
    system._was_warming_up = False

    # Replace camera with a mock that yields a fake frame and a no-op session
    fake_frame = np.zeros((480, 640), dtype=np.uint8)
    system.camera = MagicMock()
    system.camera.capture_motion_frame.return_value = fake_frame
    system.camera.consume_restart_flag.return_value = False
    system.camera.capture_high_res_frame.return_value = None
    system.camera.capture_burst_frames.return_value = []  # forces image_path=None
    # MagicMock's default __enter__/__exit__ make camera_session() usable as a CM

    # Replace remaining components so the loop has no real side effects
    system.telegram_service = MagicMock()
    system.telegram_service.send_text_message = AsyncMock()
    system.telegram_service.send_photo_with_caption = AsyncMock()
    system.telegram_service.send_media_group = AsyncMock()
    system.telegram_service.send_detection_notification = AsyncMock()

    system.system_monitor = MagicMock()
    system.system_monitor.should_skip_processing.return_value = False
    system.system_monitor.memory_manager = MagicMock()

    system.sun_checker = MagicMock()
    system.sun_checker.is_daytime.return_value = True

    # Run the loop briefly, then cancel via timeout
    try:
        await asyncio.wait_for(system.run(), timeout=0.3)
    except asyncio.TimeoutError:
        pass

    # Cooldown is 30s, the loop ran for 0.3s — every tick after the first
    # is inside cooldown. detect() must still be called on every tick.
    assert len(detect_calls) >= 5, (
        f"Cooldown is starving MOG2: detect() called only {len(detect_calls)} "
        f"times in 0.3s. Expected >=5."
    )
