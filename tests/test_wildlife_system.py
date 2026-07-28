"""
Integration test for wildlife_system main-loop behavior.

Verifies that the post-detection cooldown does NOT starve MOG2 — i.e.
motion_detector.detect() must keep being called every tick during the
cooldown window so the background model can track scene drift.
"""

import asyncio
import sqlite3
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, ANY

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
    # Task 4 scene-gate tests below assume system.scene_reference_set is a
    # real (non-None) SceneReferenceSet. Force this independently of
    # PerformanceConfig's production default (Task 5, 2026-07-17, flipped
    # scene_gate_enabled's default to False) so this fixture's behavior
    # doesn't drift if that default changes again.
    monkeypatch.setenv('PERFORMANCE_SCENE_GATE_ENABLED', 'true')
    # REVIEW-sampling gate: default rate (0.25) would nondeterministically
    # (from this fixture's perspective) sample out review-class bursts,
    # since each test gets a fresh DB whose detection_id sequence restarts
    # at 1 — is_review_sampled_out(1, 0.25) is a fixed coin flip, not a
    # per-test-run random one, so it would silently flip pre-existing
    # blur/scene-gate tests that never asked to exercise sampling. Force
    # rate=1.0 (never sample out) as this fixture's default, same pattern as
    # PERFORMANCE_SCENE_GATE_ENABLED above; sampling-gate tests override
    # system.config.performance.review_sample_rate directly per-test.
    monkeypatch.setenv('PERFORMANCE_REVIEW_SAMPLE_RATE', '1.0')
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
    data = [b.callback_data for row in keyboard.inline_keyboard for b in row]
    assert data == ["fb:123:a", "fb:123:wid", "fb:123:p", "fb:123:fp", "fb:123:ct"]


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


_OMIT = object()


def _below_floor_sharpness_info(score=8.6, luma=80.0):
    info = {
        'sharpness_score': score,
        'selected_frame_index': 0,
        'frame_count': 5,
        'all_scores': [score] * 5,
        'meets_threshold': False,
        'below_sharpness_floor': True,
        'all_frame_paths': [],
    }
    if luma is not _OMIT:
        info['luma'] = luma
    return info


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
async def test_dark_blurry_no_animal_notifies_not_muted(system, tmp_path):
    """exp #8 (sharpness-floor-is-a-brightness-gate): a below-floor,
    no-animal burst captured at dusk (luma below blur_mute_min_luma) must
    NOT be muted — darkness, not blur, explains the low sharpness score,
    so it flows through as a normal REVIEW notification instead of being
    silently dropped (this is the FN the fix targets: a real dusk animal
    the classifier missed would otherwise vanish with no trace).
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
        img, 5000, sharpness_info=_below_floor_sharpness_info(luma=50.0)
    )

    assert telegram.send_photo_with_caption.called or telegram.send_media_group.called
    system.cleanup_old_images.assert_called_once()


@pytest.mark.asyncio
async def test_blurry_no_animal_missing_luma_notifies(system, tmp_path):
    """FN-safe fallback: if luma couldn't be computed (missing/None), the
    blur-mute must NOT fire — unknown light level defaults to notifying,
    never to silent suppression.
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
        img, 5000, sharpness_info=_below_floor_sharpness_info(luma=_OMIT)
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


@pytest.mark.asyncio
async def test_scene_unchanged_review_suppresses_notification(system, tmp_path, caplog):
    """Task 4 (a): a review-class burst whose similarity to a recent
    reference is >= threshold is muted — no Telegram send, a [SCENE-GATE]
    log line, and the DB row records scene_gate_muted + the similarity.
    """
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal()
    )
    system.scene_reference_set.best_similarity = MagicMock(return_value=0.99)
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    with caplog.at_level("INFO"):
        await system._process_and_notify_detection(img, 5000)

    telegram.send_photo_with_caption.assert_not_called()
    telegram.send_media_group.assert_not_called()
    telegram.send_detection_notification.assert_not_called()
    telegram.send_document.assert_not_called()
    system.cleanup_old_images.assert_called_once()
    assert any("[SCENE-GATE]" in r.message for r in caplog.records)

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    assert row is not None
    assert row['scene_gate_muted'] == 1
    assert row['scene_similarity'] == pytest.approx(0.99)


@pytest.mark.asyncio
async def test_scene_below_threshold_still_notifies(system, tmp_path):
    """Task 4 (b): a review-class burst whose similarity is below threshold
    is not muted by the scene gate — it still notifies (REVIEW-prefixed, as
    today).
    """
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal()
    )
    system.scene_reference_set.best_similarity = MagicMock(return_value=0.5)
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    await system._process_and_notify_detection(img, 5000)

    assert telegram.send_photo_with_caption.called or telegram.send_media_group.called
    system.cleanup_old_images.assert_called_once()


@pytest.mark.asyncio
async def test_scene_gate_never_touches_identified_animal(system, tmp_path):
    """Task 4 (c): an IDENTIFIED animal frame near-identical to a reference
    still notifies — the scene gate only ever evaluates review-class
    statuses, so best_similarity must not even be consulted.
    """
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification(True, boxes=[{'confidence': 0.7}])
    )
    system.scene_reference_set.best_similarity = MagicMock(return_value=0.99)
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    await system._process_and_notify_detection(img, 5000)

    assert telegram.send_photo_with_caption.called or telegram.send_media_group.called
    system.scene_reference_set.best_similarity.assert_not_called()


@pytest.mark.asyncio
async def test_scene_gate_would_match_human_suppressed_via_human_gate_single_log(system, tmp_path, caplog):
    """Task 4 (d): a HUMAN burst that would match the scene is suppressed by
    the human gate, not the scene gate — precedence, single log line, and
    the scene gate never evaluates a HUMAN status.
    """
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(return_value=_identification_human())
    system.scene_reference_set.best_similarity = MagicMock(return_value=0.99)
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    with caplog.at_level("INFO"):
        await system._process_and_notify_detection(img, 5000)

    telegram.send_photo_with_caption.assert_not_called()
    telegram.send_media_group.assert_not_called()
    system.cleanup_old_images.assert_called_once()
    system.scene_reference_set.best_similarity.assert_not_called()

    gate_logs = [
        r.message for r in caplog.records
        if "HUMAN-GATE" in r.message or "[BLUR]" in r.message or "[SCENE-GATE]" in r.message
    ]
    assert len(gate_logs) == 1
    assert "HUMAN-GATE" in gate_logs[0]


@pytest.mark.asyncio
async def test_scene_gate_would_match_blurry_review_blur_wins_single_log(system, tmp_path, caplog):
    """Task 4 (e): a blurry review-class burst that would also match the
    scene is suppressed via the blur gate — precedence, single log line.
    """
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal()
    )
    system.scene_reference_set.best_similarity = MagicMock(return_value=0.99)
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

    gate_logs = [
        r.message for r in caplog.records
        if "HUMAN-GATE" in r.message or "[BLUR]" in r.message or "[SCENE-GATE]" in r.message
    ]
    assert len(gate_logs) == 1
    assert "[BLUR]" in gate_logs[0]


@pytest.mark.asyncio
async def test_scene_gate_no_references_fails_open(system, tmp_path):
    """Task 4 (f): no references seeded (fresh reference set, nothing added
    yet) — best_similarity naturally returns None, so the gate never mutes
    and the review-class burst notifies as it does today.
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

    await system._process_and_notify_detection(img, 5000)

    assert telegram.send_photo_with_caption.called or telegram.send_media_group.called
    system.cleanup_old_images.assert_called_once()


@pytest.mark.asyncio
async def test_scene_gate_disabled_via_config_fails_open(system, tmp_path):
    """Task 4 (g): scene_gate_enabled=False — behavior identical to today
    even if the (unused) comparator would have matched.
    """
    system.config.performance.scene_gate_enabled = False
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal()
    )
    system.scene_reference_set.best_similarity = MagicMock(return_value=0.99)
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    await system._process_and_notify_detection(img, 5000)

    assert telegram.send_photo_with_caption.called or telegram.send_media_group.called
    system.scene_reference_set.best_similarity.assert_not_called()


@pytest.mark.asyncio
async def test_scene_gate_reference_set_update_review_yes_human_no(system, tmp_path):
    """Task 4 (h): a review-class detection joins the reference set for the
    next call; a HUMAN detection never does.
    """
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.scene_reference_set.add = MagicMock()
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal()
    )
    await system._process_and_notify_detection(img, 5000)
    system.scene_reference_set.add.assert_called_once_with(img, ANY)

    system.scene_reference_set.add.reset_mock()
    system.species_identifier.identify_species = MagicMock(return_value=_identification_human())
    await system._process_and_notify_detection(img, 5000)
    system.scene_reference_set.add.assert_not_called()


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


def test_capture_and_select_best_frame_populates_luma(system, tmp_path, monkeypatch):
    """exp #8: sharpness_info must carry a numeric 'luma' (mean best-frame
    brightness) so the blur-mute gate can tell darkness apart from real
    blur.
    """
    import wildlife_system

    fake_frame = np.full((10, 10, 3), 100, dtype=np.uint8)
    system.camera.capture_burst_frames = MagicMock(return_value=[fake_frame] * 3)
    image_dir = system.config.storage.image_dir
    image_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = [image_dir / f"capture_z_frame{i}.jpg" for i in range(1, 4)]
    for p in saved_paths:
        p.write_bytes(b"fake")
    system.camera.save_burst_frames = MagicMock(return_value=saved_paths)

    score = system.config.performance.min_sharpness_threshold + 10.0
    monkeypatch.setattr(
        wildlife_system.SharpnessAnalyzer,
        "select_sharpest_frame",
        staticmethod(lambda *a, **k: (fake_frame, 0, score, [score] * 3)),
    )

    path, info = system._capture_and_select_best_frame()

    assert path == saved_paths[0]
    assert 'luma' in info
    assert isinstance(info['luma'], float)
    assert info['luma'] > 0


# ---------------------------------------------------------------------------
# REVIEW-sampling gate (wildlife_system._review_sample_fraction /
# is_review_sampled_out / notification wiring). Precedence: Human > Blur >
# Scene > Sampling — a notification-volume lever only, the burst is still
# species-ID'd and DB-logged regardless of whether it's sent.
# ---------------------------------------------------------------------------

def test_review_sample_fraction_deterministic():
    from wildlife_system import _review_sample_fraction
    a = _review_sample_fraction(123)
    b = _review_sample_fraction(123)
    assert a == b


def test_review_sample_fraction_in_unit_interval():
    from wildlife_system import _review_sample_fraction
    for det_id in range(500):
        frac = _review_sample_fraction(det_id)
        assert 0.0 <= frac < 1.0


def test_review_sample_fraction_roughly_uniform_at_quarter_rate():
    """Not a proof of uniformity, just a sanity bound: ~1000 ids at rate
    0.25 should send roughly a quarter of them (0.20-0.30 tolerance)."""
    from wildlife_system import is_review_sampled_out
    n = 1000
    sent = sum(1 for i in range(n) if not is_review_sampled_out(i, 0.25))
    sent_fraction = sent / n
    assert 0.20 <= sent_fraction <= 0.30


def test_is_review_sampled_out_rate_one_never_samples_out():
    from wildlife_system import is_review_sampled_out
    for det_id in range(200):
        assert is_review_sampled_out(det_id, 1.0) is False


def test_is_review_sampled_out_rate_zero_always_samples_out():
    from wildlife_system import is_review_sampled_out
    for det_id in range(200):
        assert is_review_sampled_out(det_id, 0.0) is True


def test_is_review_sampled_out_none_id_fails_open():
    """Fail-open: a missing detection_id (e.g. the DB write itself failed)
    always sends, regardless of rate."""
    from wildlife_system import is_review_sampled_out
    assert is_review_sampled_out(None, 0.25) is False
    assert is_review_sampled_out(None, 0.0) is False
    assert is_review_sampled_out(None, 1.0) is False


@pytest.mark.asyncio
async def test_review_sampled_out_suppresses_notification(system, tmp_path, caplog):
    """rate=0.0 forces every detection_id to be sampled out: a review-class
    burst gets a DB row (species-ID'd and logged as always) but no Telegram
    send, and a [REVIEW-SAMPLE] log line."""
    system.config.performance.review_sample_rate = 0.0
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
        await system._process_and_notify_detection(img, 5000)

    telegram.send_photo_with_caption.assert_not_called()
    telegram.send_media_group.assert_not_called()
    telegram.send_detection_notification.assert_not_called()
    telegram.send_document.assert_not_called()
    system.cleanup_old_images.assert_called_once()
    assert any("[REVIEW-SAMPLE]" in r.message for r in caplog.records)

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    assert row is not None
    assert row['review_sampled_out'] == 1


@pytest.mark.asyncio
async def test_review_not_sampled_out_still_notifies(system, tmp_path):
    """rate=1.0 forces every detection_id to send — unchanged baseline
    behavior (the rollback lever)."""
    system.config.performance.review_sample_rate = 1.0
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal()
    )
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    await system._process_and_notify_detection(img, 5000)

    assert telegram.send_photo_with_caption.called or telegram.send_media_group.called
    system.cleanup_old_images.assert_called_once()

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    assert row['review_sampled_out'] == 0


@pytest.mark.asyncio
async def test_human_wins_over_sampling_single_log(system, tmp_path, caplog):
    """A HUMAN burst is suppressed by the human gate, not sampling — single
    suppression log, even at rate=0.0."""
    system.config.performance.review_sample_rate = 0.0
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(return_value=_identification_human())
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    with caplog.at_level("INFO"):
        await system._process_and_notify_detection(img, 5000)

    telegram.send_photo_with_caption.assert_not_called()
    telegram.send_media_group.assert_not_called()
    system.cleanup_old_images.assert_called_once()

    gate_logs = [
        r.message for r in caplog.records
        if "HUMAN-GATE" in r.message or "[BLUR]" in r.message
        or "[SCENE-GATE]" in r.message or "[REVIEW-SAMPLE]" in r.message
    ]
    assert len(gate_logs) == 1
    assert "HUMAN-GATE" in gate_logs[0]


@pytest.mark.asyncio
async def test_blur_wins_over_sampling_single_log(system, tmp_path, caplog):
    """A blurry review-class burst is suppressed via the blur gate, not
    double-suppressed or mis-attributed to sampling, even at rate=0.0."""
    system.config.performance.review_sample_rate = 0.0
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
    system.cleanup_old_images.assert_called_once()

    gate_logs = [
        r.message for r in caplog.records
        if "HUMAN-GATE" in r.message or "[BLUR]" in r.message
        or "[SCENE-GATE]" in r.message or "[REVIEW-SAMPLE]" in r.message
    ]
    assert len(gate_logs) == 1
    assert "[BLUR]" in gate_logs[0]


@pytest.mark.asyncio
async def test_scene_gate_wins_over_sampling_single_log(system, tmp_path, caplog):
    """A scene-gate-muted review-class burst is suppressed via the scene
    gate, not double-suppressed or mis-attributed to sampling, even at
    rate=0.0."""
    system.config.performance.review_sample_rate = 0.0
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal()
    )
    system.scene_reference_set.best_similarity = MagicMock(return_value=0.99)
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    with caplog.at_level("INFO"):
        await system._process_and_notify_detection(img, 5000)

    telegram.send_photo_with_caption.assert_not_called()
    telegram.send_media_group.assert_not_called()
    system.cleanup_old_images.assert_called_once()

    gate_logs = [
        r.message for r in caplog.records
        if "HUMAN-GATE" in r.message or "[BLUR]" in r.message
        or "[SCENE-GATE]" in r.message or "[REVIEW-SAMPLE]" in r.message
    ]
    assert len(gate_logs) == 1
    assert "[SCENE-GATE]" in gate_logs[0]


@pytest.mark.asyncio
async def test_sampled_out_flag_ignored_for_non_review_status(system, tmp_path):
    """Defence-in-depth: is_review_detection() gates the sampling branch
    just like the blur/scene gates above — even if review_sampled_out were
    somehow True on a non-review-class (e.g. identified) result, it must
    not suppress the notification.
    """
    from data_models import DetectionStatus

    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    fake_result = {
        'species_name': 'Fox',
        'confidence': 0.9,
        'api_success': True,
        'processing_time': 0.5,
        'fallback_reason': None,
        'animals_detected': True,
        'detection_count': 1,
        'detection_result': None,
        'metadata': {},
        'detection_id': 999,
        'detection_status': DetectionStatus.IDENTIFIED,
        'scene_similarity': None,
        'scene_gate_muted': False,
        'review_sampled_out': True,  # wrongly set — must be ignored here
    }
    system.process_detection = MagicMock(return_value=(fake_result, datetime.now()))
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    await system._process_and_notify_detection(img, 5000)

    assert telegram.send_photo_with_caption.called or telegram.send_media_group.called


# ---------------------------------------------------------------------------
# Human-Proximity Mute Gate (2026-07-27): MegaDetector scores extreme
# close-up / motion-blurred partial human bodies too low to trip the
# Human/Privacy Gate itself, so such bursts leak a recognizable person to
# REVIEW as no_animal. Mute review-class bursts that land within
# human_proximity_window_seconds of the most recent HUMAN-status detection.
# Precedence: Human > Human-Proximity > Blur > Scene > Sampling.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_human_proximity_mute_within_window(system, tmp_path, caplog):
    """A review-class burst landing shortly after a HUMAN-status detection is
    muted — no Telegram send, a [HUMAN-PROXIMITY] log line, and the DB row
    records human_proximity_muted."""
    system._last_human_detection_at = datetime.now() - timedelta(seconds=60)
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
        await system._process_and_notify_detection(img, 5000)

    telegram.send_photo_with_caption.assert_not_called()
    telegram.send_media_group.assert_not_called()
    telegram.send_detection_notification.assert_not_called()
    telegram.send_document.assert_not_called()
    system.cleanup_old_images.assert_called_once()
    assert any("[HUMAN-PROXIMITY]" in r.message for r in caplog.records)

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    assert row is not None
    assert row['human_proximity_muted'] == 1


@pytest.mark.asyncio
async def test_human_proximity_no_mute_outside_window(system, tmp_path):
    """A review-class burst well outside the look-back window is not muted —
    it still notifies (REVIEW-prefixed, as today)."""
    system._last_human_detection_at = datetime.now() - timedelta(seconds=200)
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal()
    )
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    await system._process_and_notify_detection(img, 5000)

    assert telegram.send_photo_with_caption.called or telegram.send_media_group.called
    system.cleanup_old_images.assert_called_once()

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    assert row['human_proximity_muted'] == 0


@pytest.mark.asyncio
async def test_human_proximity_no_mute_when_window_zero(system, tmp_path):
    """PERFORMANCE_HUMAN_PROXIMITY_WINDOW_SECONDS=0 disables the gate (the
    rollback lever) even with a very recent human detection."""
    system.config.performance.human_proximity_window_seconds = 0.0
    system._last_human_detection_at = datetime.now() - timedelta(seconds=1)
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal()
    )
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    await system._process_and_notify_detection(img, 5000)

    assert telegram.send_photo_with_caption.called or telegram.send_media_group.called


@pytest.mark.asyncio
async def test_human_proximity_no_mute_without_prior_human(system, tmp_path):
    """No prior HUMAN-status detection recorded (fresh system) — the gate
    never mutes."""
    assert system._last_human_detection_at is None
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal()
    )
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    await system._process_and_notify_detection(img, 5000)

    assert telegram.send_photo_with_caption.called or telegram.send_media_group.called


def test_human_proximity_muted_none_for_non_review_status(system):
    """process_detection only ever sets human_proximity_muted for review-class
    statuses — an IDENTIFIED animal always persists NULL, even with a very
    recent prior human detection."""
    system._last_human_detection_at = datetime.now() - timedelta(seconds=1)
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification(True, boxes=[{'confidence': 0.7}])
    )

    result, _ = system.process_detection("capture.jpg", 5000, None)

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?",
                           (result['detection_id'],)).fetchone()
    assert row['human_proximity_muted'] is None


def test_human_status_updates_last_human_detection_at(system):
    """Processing a HUMAN-status burst updates the in-memory tracker so the
    NEXT review-class burst (moments later) is measured against it."""
    assert system._last_human_detection_at is None
    system.species_identifier.identify_species = MagicMock(return_value=_identification_human())
    _, human_ts = system.process_detection("capture.jpg", 5000, None)
    assert system._last_human_detection_at == human_ts

    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal()
    )
    result, _ = system.process_detection("capture.jpg", 5000, None)
    assert result['human_proximity_muted'] is True


@pytest.mark.asyncio
async def test_human_wins_over_human_proximity_single_log(system, tmp_path, caplog):
    """A HUMAN-status burst is suppressed by the human gate itself, not the
    proximity gate — single suppression log (the proximity gate never
    evaluates a non-review-class status)."""
    system._last_human_detection_at = datetime.now() - timedelta(seconds=10)
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(return_value=_identification_human())
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    with caplog.at_level("INFO"):
        await system._process_and_notify_detection(img, 5000)

    telegram.send_photo_with_caption.assert_not_called()
    telegram.send_media_group.assert_not_called()
    system.cleanup_old_images.assert_called_once()

    gate_logs = [
        r.message for r in caplog.records
        if "HUMAN-GATE" in r.message or "[HUMAN-PROXIMITY]" in r.message
        or "[BLUR]" in r.message or "[SCENE-GATE]" in r.message
        or "[REVIEW-SAMPLE]" in r.message
    ]
    assert len(gate_logs) == 1
    assert "HUMAN-GATE" in gate_logs[0]


@pytest.mark.asyncio
async def test_human_proximity_wins_over_blur_single_log(system, tmp_path, caplog):
    """A below-floor review-class burst that also falls inside the
    human-proximity window is suppressed via the proximity gate, not
    double-handled by the blur gate — single suppression log."""
    system._last_human_detection_at = datetime.now() - timedelta(seconds=60)
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
    system.cleanup_old_images.assert_called_once()

    gate_logs = [
        r.message for r in caplog.records
        if "HUMAN-GATE" in r.message or "[HUMAN-PROXIMITY]" in r.message
        or "[BLUR]" in r.message or "[SCENE-GATE]" in r.message
        or "[REVIEW-SAMPLE]" in r.message
    ]
    assert len(gate_logs) == 1
    assert "[HUMAN-PROXIMITY]" in gate_logs[0]


@pytest.mark.asyncio
async def test_human_proximity_wins_over_sampling_single_log(system, tmp_path, caplog):
    """A review-class burst inside the proximity window is suppressed via
    the proximity gate, not double-attributed to sampling, even at
    review_sample_rate=0.0."""
    system.config.performance.review_sample_rate = 0.0
    system._last_human_detection_at = datetime.now() - timedelta(seconds=60)
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
        await system._process_and_notify_detection(img, 5000)

    telegram.send_photo_with_caption.assert_not_called()
    telegram.send_media_group.assert_not_called()
    system.cleanup_old_images.assert_called_once()

    gate_logs = [
        r.message for r in caplog.records
        if "HUMAN-GATE" in r.message or "[HUMAN-PROXIMITY]" in r.message
        or "[BLUR]" in r.message or "[SCENE-GATE]" in r.message
        or "[REVIEW-SAMPLE]" in r.message
    ]
    assert len(gate_logs) == 1
    assert "[HUMAN-PROXIMITY]" in gate_logs[0]


# ---------------------------------------------------------------------------
# Human-density condition (exp #11 mechanism extension, 2026-07-28): OR-ed
# onto the human-proximity gate above. Tonight's adjudication found
# recognizable-person review-class bursts OUTSIDE the window condition
# (gaps of 432s/732s past the last human burst) during a long gardening
# session — this condition mutes instead when the garden has been
# "occupied" (>= human_density_count HUMAN-status detections in the
# trailing human_density_window_seconds), regardless of how long ago the
# MOST RECENT one was.
# ---------------------------------------------------------------------------

def _seed_recent_humans(system, count, spacing_seconds=60, end_offset_seconds=500):
    """Populate system._recent_human_detection_times with `count` timestamps,
    the most recent `end_offset_seconds` in the past (outside the default
    120s window condition), spaced `spacing_seconds` apart before that."""
    now = datetime.now()
    latest = now - timedelta(seconds=end_offset_seconds)
    times = [latest - timedelta(seconds=spacing_seconds * i) for i in range(count)]
    times.reverse()
    system._recent_human_detection_times = times
    system._last_human_detection_at = times[-1] if times else None
    return times


@pytest.mark.asyncio
async def test_human_density_mute_at_threshold(system, tmp_path, caplog):
    """Exactly human_density_count HUMAN detections in the trailing window
    mutes via the density condition, even though the most recent one is well
    outside the (default 120s) window condition."""
    system.config.performance.human_density_count = 8
    system.config.performance.human_density_window_seconds = 1800.0
    _seed_recent_humans(system, count=8, end_offset_seconds=500)
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
        await system._process_and_notify_detection(img, 5000)

    telegram.send_photo_with_caption.assert_not_called()
    telegram.send_media_group.assert_not_called()
    system.cleanup_old_images.assert_called_once()
    gate_logs = [r.message for r in caplog.records if "[HUMAN-PROXIMITY]" in r.message]
    assert len(gate_logs) == 1
    assert "density" in gate_logs[0]

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    assert row['human_proximity_muted'] == 1


@pytest.mark.asyncio
async def test_human_density_no_mute_below_threshold(system, tmp_path):
    """One fewer than human_density_count, and the last human is outside the
    window condition — no mute from either condition."""
    system.config.performance.human_density_count = 8
    system.config.performance.human_density_window_seconds = 1800.0
    _seed_recent_humans(system, count=7, end_offset_seconds=500)
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal()
    )
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    await system._process_and_notify_detection(img, 5000)

    assert telegram.send_photo_with_caption.called or telegram.send_media_group.called

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    assert row['human_proximity_muted'] == 0


@pytest.mark.asyncio
async def test_window_condition_alone_still_mutes_regression(system, tmp_path, caplog):
    """Regression: the plain window condition (no density streak at all)
    still mutes on its own, reported as 'window' in the log line."""
    system._last_human_detection_at = datetime.now() - timedelta(seconds=60)
    system._recent_human_detection_times = [system._last_human_detection_at]
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
        await system._process_and_notify_detection(img, 5000)

    telegram.send_photo_with_caption.assert_not_called()
    telegram.send_media_group.assert_not_called()
    gate_logs = [r.message for r in caplog.records if "[HUMAN-PROXIMITY]" in r.message]
    assert len(gate_logs) == 1
    assert "window" in gate_logs[0]
    assert "density" not in gate_logs[0]


@pytest.mark.asyncio
async def test_human_density_mutes_when_last_human_gap_exceeds_window(system, tmp_path):
    """The measured failure mode: last HUMAN-status detection is 432s ago
    (well outside the 120s window condition), but the garden has been
    occupied (density condition) — still muted."""
    system.config.performance.human_density_count = 8
    system.config.performance.human_density_window_seconds = 1800.0
    _seed_recent_humans(system, count=8, end_offset_seconds=432)
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal()
    )
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    await system._process_and_notify_detection(img, 5000)

    telegram.send_photo_with_caption.assert_not_called()
    telegram.send_media_group.assert_not_called()


@pytest.mark.asyncio
async def test_human_density_count_zero_disables_density_condition(system, tmp_path):
    """PERFORMANCE_HUMAN_DENSITY_COUNT=0 disables the density condition (the
    rollback lever) even with a long occupied-garden streak — window
    condition (out of range here) doesn't mute either, so it notifies."""
    system.config.performance.human_density_count = 0
    system.config.performance.human_density_window_seconds = 1800.0
    _seed_recent_humans(system, count=20, end_offset_seconds=500)
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal()
    )
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    await system._process_and_notify_detection(img, 5000)

    assert telegram.send_photo_with_caption.called or telegram.send_media_group.called


def test_human_density_pruning_drops_out_of_window_timestamps(system):
    """_count_recent_human_detections prunes entries older than
    human_density_window_seconds relative to the reference time."""
    system.config.performance.human_density_window_seconds = 1800.0
    now = datetime.now()
    system._recent_human_detection_times = [
        now - timedelta(seconds=100),   # inside window
        now - timedelta(seconds=1000),  # inside window
        now - timedelta(seconds=2000),  # outside window -> pruned
    ]

    count = system._count_recent_human_detections(now)

    assert count == 2
    assert len(system._recent_human_detection_times) == 2
    assert all(
        (now - t).total_seconds() <= 1800.0
        for t in system._recent_human_detection_times
    )


def test_human_status_updates_recent_human_detection_times(system):
    """Processing a HUMAN-status burst appends to the density-condition
    list, not just the single last-human timestamp."""
    assert system._recent_human_detection_times == []
    system.species_identifier.identify_species = MagicMock(return_value=_identification_human())
    _, human_ts = system.process_detection("capture.jpg", 5000, None)
    assert system._recent_human_detection_times == [human_ts]


def test_recent_human_detection_times_seeded_at_startup(monkeypatch, tmp_path):
    """WildlifeSystem seeds self._recent_human_detection_times from the DB at
    startup, same pattern as _last_human_detection_at."""
    monkeypatch.setenv('TELEGRAM_BOT_TOKEN', 'test_token')
    monkeypatch.setenv('TELEGRAM_CHAT_ID', 'test_chat')
    monkeypatch.setenv('MOTION_WARMUP_SECONDS', '0')
    monkeypatch.setenv('PERFORMANCE_ENABLE_TIMELAPSE', 'false')
    monkeypatch.setenv('PERFORMANCE_SCENE_GATE_ENABLED', 'true')
    monkeypatch.setenv('PERFORMANCE_REVIEW_SAMPLE_RATE', '1.0')
    for mod in ('wildlife_system', 'config'):
        sys.modules.pop(mod, None)

    from wildlife_system import WildlifeSystem
    from database_manager import DatabaseManager

    seeded_times = [datetime.now() - timedelta(seconds=30)]

    class _FakeDB:
        def get_last_human_detection_time(self):
            return seeded_times[0]

        def get_recent_human_detection_times(self, since):
            return list(seeded_times)

    # Patch DatabaseManager construction so __init__'s seeding calls hit our
    # fake instead of a real (fresh, empty) DB.
    monkeypatch.setattr(
        'wildlife_system.DatabaseManager', lambda config: _FakeDB()
    )

    sys_obj = WildlifeSystem()

    assert sys_obj._recent_human_detection_times == seeded_times


def test_recent_human_detection_times_seeding_db_error_fails_open(monkeypatch):
    """A DB error while seeding the density-condition list must not crash
    startup — it just leaves the list empty (same fail-open pattern as the
    single-timestamp seeding above)."""
    monkeypatch.setenv('TELEGRAM_BOT_TOKEN', 'test_token')
    monkeypatch.setenv('TELEGRAM_CHAT_ID', 'test_chat')
    monkeypatch.setenv('MOTION_WARMUP_SECONDS', '0')
    monkeypatch.setenv('PERFORMANCE_ENABLE_TIMELAPSE', 'false')
    monkeypatch.setenv('PERFORMANCE_SCENE_GATE_ENABLED', 'true')
    monkeypatch.setenv('PERFORMANCE_REVIEW_SAMPLE_RATE', '1.0')
    for mod in ('wildlife_system', 'config'):
        sys.modules.pop(mod, None)

    from wildlife_system import WildlifeSystem

    class _FakeDB:
        def get_last_human_detection_time(self):
            return None

        def get_recent_human_detection_times(self, since):
            raise RuntimeError("db is on fire")

    monkeypatch.setattr(
        'wildlife_system.DatabaseManager', lambda config: _FakeDB()
    )

    sys_obj = WildlifeSystem()

    assert sys_obj._recent_human_detection_times == []


@pytest.mark.asyncio
async def test_human_density_non_review_status_unaffected(system, tmp_path):
    """Precedence/scope check: an IDENTIFIED animal result is never touched
    by the density condition, even with a long occupied-garden streak — it
    always notifies, and human_proximity_muted stays NULL (defense-in-depth,
    same as the window-condition test above)."""
    _seed_recent_humans(system, count=20, end_offset_seconds=10)
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification(True, boxes=[{'confidence': 0.7}])
    )

    result, _ = system.process_detection("capture.jpg", 5000, None)

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections WHERE id = ?",
                           (result['detection_id'],)).fetchone()
    assert row['human_proximity_muted'] is None


@pytest.mark.asyncio
async def test_human_density_precedence_unchanged_vs_human_gate(system, tmp_path, caplog):
    """A HUMAN-status burst is still suppressed by the human gate itself,
    not double-attributed to the density condition, even with a long
    occupied-garden streak already recorded — single suppression log."""
    _seed_recent_humans(system, count=20, end_offset_seconds=10)
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(return_value=_identification_human())
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    with caplog.at_level("INFO"):
        await system._process_and_notify_detection(img, 5000)

    telegram.send_photo_with_caption.assert_not_called()
    telegram.send_media_group.assert_not_called()

    gate_logs = [
        r.message for r in caplog.records
        if "HUMAN-GATE" in r.message or "[HUMAN-PROXIMITY]" in r.message
        or "[BLUR]" in r.message or "[SCENE-GATE]" in r.message
        or "[REVIEW-SAMPLE]" in r.message
    ]
    assert len(gate_logs) == 1
    assert "HUMAN-GATE" in gate_logs[0]


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
