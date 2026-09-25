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

import cv2
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
    # Deferred REVIEW send / cancel-on-human gate (2026-07-31 leading-edge
    # fix): default review_defer_seconds=240 would turn every pre-existing
    # "review-class burst notifies immediately" assertion below into a
    # scheduled-but-not-yet-sent background task instead. Force disabled
    # (0 = send immediately, as before this fix) as this fixture's default;
    # the deferral tests below override
    # system.config.performance.review_defer_seconds directly per-test.
    monkeypatch.setenv('PERFORMANCE_REVIEW_DEFER_SECONDS', '0')
    # Pin the human-proximity gate to its CODE defaults. Without this the
    # tests inherit whatever the loop last deployed via
    # experiments/deployed_config.env (a documented config-module-reload gap,
    # see tests/conftest.py) — the 2026-07-29 deploy of window=240 silently
    # broke test_human_proximity_no_mute_outside_window, which places a burst
    # 200s after a human detection and expects no mute under the 120s default.
    monkeypatch.setenv('PERFORMANCE_HUMAN_PROXIMITY_WINDOW_SECONDS', '120')
    monkeypatch.setenv('PERFORMANCE_HUMAN_DENSITY_COUNT', '8')
    monkeypatch.setenv('PERFORMANCE_HUMAN_DENSITY_WINDOW_SECONDS', '1800')
    # Burst human sweep (exp #21): enabled by default in production, but it
    # re-runs identify_species on sibling frames, which would turn the
    # single-call `identify_species` mocks below into multi-call ones. Force
    # it off as this fixture's default, same pattern as the gates above; the
    # sweep tests override system.config.performance.* directly per-test.
    monkeypatch.setenv('PERFORMANCE_HUMAN_SWEEP_DIVERGENCE_THRESHOLD', '0')
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


def _identification_no_animal_with_person(person_confidence):
    """Same as `_identification_no_animal()` but carrying a
    `person_confidence` in `metadata` — the value `process_detection` reads
    to drive the demoted-band window widening (exp #27) and also what it
    persists to the `person_confidence` DB column."""
    result = _identification_no_animal()
    result.metadata = {'person_confidence': person_confidence}
    return result


def _identification_no_animal_with_top1(label, score):
    """Same as `_identification_no_animal()` but carrying a
    `top_classifier_prediction` in `metadata` — the raw, pre-rollup
    classifier top-1 that `process_detection` reads to drive the
    Confident-Blank Mute Gate (exp #29) and persists to the
    `top_species_raw`/`top_species_score` DB columns."""
    result = _identification_no_animal()
    result.metadata = {'top_classifier_prediction': {'label': label, 'score': score}}
    return result


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


def _identification_unnamed_animal(confidence=0.72):
    """An IDENTIFIED result carrying SpeciesNet's fully-generic "an animal
    is there, but I cannot name it" rollup label (exp #26,
    unnamed-animal-main-leak) — status stays at its IDENTIFIED default, so
    is_review_detection() is False for this result, distinct from
    _identification_no_animal() above.
    """
    from data_models import IdentificationResult, DetectionResult
    det = DetectionResult(
        animals_detected=True,
        detection_count=1,
        bounding_boxes=[{'confidence': 0.4, 'category': '1'}],
        detections=[],
        processing_time=0.1,
    )
    return IdentificationResult(
        species_name="1f689929-d0e3-4ac6-8016-16aacd8d0dbe;;;;;;animal",
        confidence=confidence,
        api_success=True,
        processing_time=0.5,
        detection_result=det,
        animals_detected=True,
    )


def _identification_named_species(confidence=0.85):
    """An IDENTIFIED result naming a specific, non-generic species — must
    never be affected by the unnamed-animal widening of the human-proximity
    gate."""
    from data_models import IdentificationResult, DetectionResult
    det = DetectionResult(
        animals_detected=True,
        detection_count=1,
        bounding_boxes=[{'confidence': 0.8, 'category': '1'}],
        detections=[],
        processing_time=0.1,
    )
    return IdentificationResult(
        species_name="uuid;mammalia;carnivora;felidae;felis;catus;domestic cat",
        confidence=confidence,
        api_success=True,
        processing_time=0.5,
        detection_result=det,
        animals_detected=True,
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


def test_build_caption_prefers_geofenced_species_over_raw_top1(system):
    """exp #28: the raw top-1 is an out-of-region species (blue whistling-thrush,
    Himalayas) while the geofence-filtered candidate is the common blackbird that
    actually lives here — the caption must show the in-region species."""
    species_result = _identified_species_result(
        "abc;aves;;;;;bird",
        confidence=0.87,
        metadata={
            'top_classifier_prediction': {
                'label': 'x;aves;passeriformes;muscicapidae;myophonus;caeruleus;blue whistling-thrush',
                'score': 0.42,
            },
            'best_geofenced_species': {
                'label': 'y;aves;passeriformes;turdidae;turdus;merula;common blackbird',
                'score': 0.062,
            },
        },
    )
    caption = system._build_caption(species_result, 1000, datetime.now())
    assert "Best guess: common blackbird (6%)" in caption
    assert "whistling" not in caption


def test_build_caption_falls_back_to_raw_top1_without_geofenced_candidate(system):
    """No in-region candidate in the top-k (best_geofenced_species is None) —
    behaviour is unchanged from before exp #28: show the raw top-1."""
    species_result = _identified_species_result(
        "abc;aves;;;;;bird",
        metadata={
            'top_classifier_prediction': {
                'label': 'x;aves;passeriformes;turdidae;turdus;migratorius;american robin',
                'score': 0.33,
            },
            'best_geofenced_species': None,
        },
    )
    caption = system._build_caption(species_result, 1000, datetime.now())
    assert "Best guess: american robin (33%)" in caption


def test_build_caption_no_best_guess_when_guess_repeats_ensemble_name(system):
    """exp #28: the classifier's top-1 IS the same generic rollup the ensemble
    already reported ('bird') — 'Best guess: bird' under a bird verdict adds
    nothing and must be suppressed."""
    species_result = _identified_species_result(
        "abc;aves;;;;;bird",
        metadata={
            'top_classifier_prediction': {'label': 'abc;aves;;;;;bird', 'score': 0.55},
        },
    )
    caption = system._build_caption(species_result, 1000, datetime.now())
    assert "Best guess" not in caption


def test_build_caption_never_crashes_on_malformed_geofenced_species(system):
    """Never-crash constraint extends to the new candidate source: a malformed
    best_geofenced_species must fall through to the raw top-1, not raise."""
    species_result = _identified_species_result(
        "abc;aves;;;;;bird",
        metadata={
            'best_geofenced_species': "not-a-dict",
            'top_classifier_prediction': {
                'label': 'y;aves;passeriformes;turdidae;turdus;merula;common blackbird',
                'score': 0.2,
            },
        },
    )
    caption = system._build_caption(species_result, 1000, datetime.now())
    assert "Best guess: common blackbird (20%)" in caption


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
    still notifies — the scene gate only ever MUTES review-class statuses.
    Since 2026-09-04 the similarity is still measured (observability, see
    backlog #17), but it can never turn into a mute here.
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
    # Measured for observability, but never a mute on a non-review status.
    system.scene_reference_set.best_similarity.assert_called_once()
    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    assert row['scene_similarity'] == pytest.approx(0.99)
    assert row['scene_gate_muted'] is None


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
    # Measured (observability) but never muting: the HUMAN gate owns this row.
    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    assert row['scene_similarity'] == pytest.approx(0.99)
    assert row['scene_gate_muted'] is None

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

    # ... and neither does an IDENTIFIED animal, even though its similarity is
    # now measured for observability (2026-09-04, backlog #17).
    system.scene_reference_set.add.reset_mock()
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification(True, boxes=[{'confidence': 0.7}])
    )
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
    # exp #39 (leading-edge-animal-proximity): disable the forward
    # animal-proximity deferral so a sampled-out burst is still dropped
    # immediately here — this test is about the sampling gate in isolation,
    # not the deferral tested separately below.
    system.config.performance.animal_proximity_window_seconds = 0.0
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


# ---------------------------------------------------------------------------
# Demoted-band window (exp #27, 2026-09-15): the window condition above is
# widened to max(human_proximity_window_seconds, human_demoted_window_seconds)
# when the burst's OWN person_confidence clears human_demoted_person_floor
# (0.3) — MegaDetector saw something person-shaped, just not confidently
# enough to trip the Human/Privacy Gate. Burst 5305 (person_confidence
# 0.436, 480s since the last HUMAN detection) is the motivating case: both
# the flat 240s window and the density condition (5 < 8) missed it.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_human_demoted_band_mutes_at_480s(system, tmp_path, caplog):
    """The 5305 case: person_confidence (0.436) clears the demoted floor
    (0.3), so the look-back widens to human_demoted_window_seconds (1800s)
    and a burst 480s after the last HUMAN detection is muted, logged as
    'demoted-band window'."""
    system._last_human_detection_at = datetime.now() - timedelta(seconds=480)
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal_with_person(0.436)
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
    assert "demoted-band window" in gate_logs[0]

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    assert row['human_proximity_muted'] == 1


@pytest.mark.asyncio
async def test_human_demoted_band_no_mute_beyond_demoted_window(system, tmp_path):
    """Same elevated person_confidence, but 2000s since the last HUMAN
    detection — beyond even the widened 1800s demoted window — is NOT
    muted."""
    system._last_human_detection_at = datetime.now() - timedelta(seconds=2000)
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal_with_person(0.436)
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
async def test_human_demoted_band_no_mute_below_floor(system, tmp_path):
    """person_confidence below human_demoted_person_floor (0.3) does not
    widen the window — 480s since the last HUMAN detection is unchanged
    (not muted), same as today."""
    system._last_human_detection_at = datetime.now() - timedelta(seconds=480)
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal_with_person(0.1)
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
async def test_human_demoted_window_zero_restores_flat_window(system, tmp_path):
    """PERFORMANCE_HUMAN_DEMOTED_WINDOW_SECONDS=0 (the rollback lever)
    disables the widening even with an elevated person_confidence — flat
    240s window behaviour is restored, so 480s is NOT muted."""
    system.config.performance.human_demoted_window_seconds = 0.0
    system._last_human_detection_at = datetime.now() - timedelta(seconds=480)
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal_with_person(0.436)
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
async def test_human_demoted_band_person_confidence_none_fails_open(system, tmp_path):
    """A missing person_confidence (metadata absent, e.g. an error-path
    result) must not raise and must not mute at 480s — same as before this
    change existed."""
    system._last_human_detection_at = datetime.now() - timedelta(seconds=480)
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    result = _identification_no_animal()
    assert result.metadata is None
    system.species_identifier.identify_species = MagicMock(return_value=result)
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


# ---------------------------------------------------------------------------
# Deferred REVIEW send / cancel-on-human gate (leading-edge fix, 2026-07-31):
# the human-proximity gate above is backward-looking only — it mutes a
# review-class burst that lands AFTER a HUMAN-status detection, but can
# never catch the LEADING EDGE of a visit. Burst 3909 (2026-07-31, 18:22:42)
# was sent to REVIEW as no_animal 81s BEFORE the visit's first HUMAN burst
# (18:24:03), with a clearly recognisable face in its saved frames (two
# prior instances: 75s, 51s gaps). review_defer_seconds > 0 delays a
# review-class send; if a HUMAN-status detection lands within that window,
# the send is cancelled instead (see wildlife_system._deferred_review_send).
# ---------------------------------------------------------------------------

def _spy_process_detection(system):
    """Wrap system.process_detection so the real pipeline still runs (DB
    write, status, detection_id) but the exact full-precision timestamp it
    returns is captured. The deferred-send tests below need to place
    _last_human_detection_at a few milliseconds relative to the burst's own
    timestamp — the DB's stored timestamp only has whole-second resolution
    (see database_manager.log_detection), which isn't precise enough to do
    that race-free, so we capture the in-memory datetime object directly
    instead of round-tripping through the DB.
    """
    original = system.process_detection
    captured = {}

    def _spy(*args, **kwargs):
        result, ts = original(*args, **kwargs)
        captured['timestamp'] = ts
        captured['detection_id'] = result.get('detection_id')
        return result, ts

    system.process_detection = MagicMock(side_effect=_spy)
    return captured


@pytest.mark.asyncio
async def test_review_defer_schedules_task_not_immediate_send(system, tmp_path):
    """A review-class burst with review_defer_seconds > 0 is scheduled as a
    background task instead of sending synchronously — the main detection
    loop (_process_and_notify_detection) must not block on the defer
    window."""
    system.config.performance.review_defer_seconds = 5.0
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
    assert len(system._pending_review_tasks) == 1
    system.cleanup_old_images.assert_called_once()

    # Clean up the still-sleeping background task rather than leaking it
    # past this test.
    task = next(iter(system._pending_review_tasks))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_review_defer_sends_when_no_human_lands_in_window(system, tmp_path):
    """No HUMAN-status detection lands during the defer window — the review
    send goes out once the deferred task runs its course."""
    system.config.performance.review_defer_seconds = 0.01
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
    assert len(system._pending_review_tasks) == 1
    task = next(iter(system._pending_review_tasks))
    await task

    assert telegram.send_photo_with_caption.called or telegram.send_media_group.called


@pytest.mark.asyncio
async def test_review_defer_cancels_when_human_lands_in_window(system, tmp_path, caplog):
    """A HUMAN-status detection landing just after the burst, inside the
    defer window, cancels the send, logs [REVIEW-DEFER], and persists
    human_proximity_muted=1 — the leading-edge counterpart to the
    backward-looking proximity gate tested above."""
    system.config.performance.review_defer_seconds = 0.01
    captured = _spy_process_detection(system)
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
    assert len(system._pending_review_tasks) == 1

    # A HUMAN-status detection lands 1ms after this burst — comfortably
    # inside the 10ms defer window configured above, regardless of how long
    # the real asyncio.sleep(0.01) below actually takes wall-clock-wise.
    system._last_human_detection_at = captured['timestamp'] + timedelta(milliseconds=1)

    task = next(iter(system._pending_review_tasks))
    with caplog.at_level("INFO"):
        await task

    telegram.send_photo_with_caption.assert_not_called()
    telegram.send_media_group.assert_not_called()
    assert any("[REVIEW-DEFER]" in r.message for r in caplog.records)

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM detections WHERE id = ?", (captured['detection_id'],)
        ).fetchone()
    assert row['human_proximity_muted'] == 1


@pytest.mark.asyncio
async def test_review_defer_disabled_sends_immediately(system, tmp_path):
    """review_defer_seconds == 0 (the rollback lever) preserves pre-fix
    behaviour exactly: the review send happens immediately in-line, no
    background task is scheduled at all."""
    system.config.performance.review_defer_seconds = 0.0
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
    assert len(system._pending_review_tasks) == 0


@pytest.mark.asyncio
async def test_review_defer_never_applies_to_animal_detection(system, tmp_path):
    """A non-review (IDENTIFIED animal) detection is never deferred, even
    with a large review_defer_seconds configured — it always sends
    immediately, matching the "only review-class" scope of this gate."""
    system.config.performance.review_defer_seconds = 240.0
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
    assert len(system._pending_review_tasks) == 0


@pytest.mark.asyncio
async def test_review_defer_fail_open_on_internal_error(system, tmp_path, caplog):
    """Any exception inside the deferral wrapper (here: the DB persistence
    step itself raising, with a human otherwise landing inside the window)
    must still result in the notification being sent — fail-open, never a
    silently dropped detection."""
    system.config.performance.review_defer_seconds = 0.01
    captured = _spy_process_detection(system)
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
    assert len(system._pending_review_tasks) == 1

    # Human lands inside the window (would normally cancel the send)...
    system._last_human_detection_at = captured['timestamp'] + timedelta(milliseconds=1)
    # ...but persisting that decision is broken.
    system.database.update_human_proximity_muted = MagicMock(
        side_effect=RuntimeError("db is on fire")
    )

    task = next(iter(system._pending_review_tasks))
    with caplog.at_level("ERROR"):
        await task

    assert telegram.send_photo_with_caption.called or telegram.send_media_group.called
    assert any("Error in deferred REVIEW send" in r.message for r in caplog.records)


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


# ---------------------------------------------------------------------------
# Burst human sweep (exp #21, 2026-09-09)
#
# The human/privacy gate reads exactly one frame per burst — the sharpest —
# and sharpness is uncorrelated with whether a person is visible. Burst 5119
# leaked a recognisable face because the only frame of five that did NOT
# classify as human won selection by 13.57 vs 13.41 Laplacian variance.
# ---------------------------------------------------------------------------

def _texture(seed, mean=110.0, std=40.0, size=(135, 240)):
    """A deterministic texture: different seed -> different picture.

    `mean`/`std` set the exposure the picture is rendered at, independently of
    its content, so a test can render the same scene at noon and at dusk.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, size)
    base = cv2.GaussianBlur(base, (0, 0), 2.0)
    base = (base - base.mean()) / max(base.std(), 1e-6)
    return np.clip(base * std + mean, 0, 255).astype(np.uint8)


def _write_frames(tmp_path, specs, mean=110.0, std=40.0):
    """Write burst frames as textured JPEGs, one texture per level.

    `specs` maps filename -> level. Divergence is measured on contrast-
    normalised frames (exp #24), so frames that differ only in overall
    brightness are deliberately NOT divergent; the level therefore seeds the
    *content*, and two frames sharing a level are the same picture.
    """
    paths = []
    for name, level in specs.items():
        p = tmp_path / name
        img = _texture(level, mean=mean, std=std)
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        paths.append(str(p))
    return paths


def test_frame_divergence_identical_and_different(system, tmp_path):
    same_a, same_b, other = _write_frames(
        tmp_path, {'a.jpg': 100, 'b.jpg': 100, 'c.jpg': 200}
    )
    assert system._frame_divergence(same_a, same_b) == pytest.approx(0.0)
    assert system._frame_divergence(same_a, other) > 0.3


def test_frame_divergence_ignores_pure_exposure_shift(system, tmp_path):
    """A brightness/contrast change is not content, so it must not sweep.

    Auto-exposure moves between frames of a burst; before exp #24 a big enough
    shift alone could clear the threshold and pay for two SpeciesNet passes on
    a picture nothing had happened in.
    """
    (bright,) = _write_frames(tmp_path, {'bright.jpg': 7}, mean=150.0, std=45.0)
    (dim,) = _write_frames(tmp_path, {'dim.jpg': 7}, mean=40.0, std=15.0)
    assert system._frame_divergence(bright, dim) < 0.01


def test_frame_divergence_is_brightness_invariant(system, tmp_path):
    """exp #24: the same content change must score the same at dusk as at noon.

    Burst 5169 (2026-09-12 19:17) is the live counter-example the raw-level
    measure missed: five visually unrelated frames of a person walking through
    the garden, two of which the model reads as `human` at >=0.93, scored
    0.0005 because at a mean of 11/255 no pixel pair differs by 40 raw levels.
    The sweep never ran, and only the human-proximity window kept the burst out
    of REVIEW.
    """
    noon_a, noon_b = _write_frames(
        tmp_path, {'noon_a.jpg': 11, 'noon_b.jpg': 12}, mean=110.0, std=40.0
    )
    dusk_a, dusk_b = _write_frames(
        tmp_path, {'dusk_a.jpg': 11, 'dusk_b.jpg': 12}, mean=11.0, std=5.0
    )
    noon = system._frame_divergence(noon_a, noon_b)
    dusk = system._frame_divergence(dusk_a, dusk_b)

    threshold = 0.03
    assert noon > threshold
    assert dusk > threshold, (
        f"dusk divergence {dusk:.4f} under threshold — the sweep is blind in "
        f"low light again (noon scored {noon:.4f} on the same content change)"
    )
    assert dusk > noon / 2


def test_frame_divergence_unreadable_frame_returns_none(system, tmp_path):
    (a,) = _write_frames(tmp_path, {'a.jpg': 100})
    assert system._frame_divergence(a, tmp_path / 'missing.jpg') is None


def test_sweep_escalates_review_burst_to_human(system, tmp_path, caplog):
    """A person in a divergent sibling frame escalates the whole burst."""
    selected, sibling = _write_frames(tmp_path, {'f5.jpg': 100, 'f1.jpg': 200})
    system.config.performance.human_sweep_divergence_threshold = 0.03
    system.config.performance.human_sweep_max_frames = 2
    system.species_identifier = MagicMock()
    system.species_identifier.identify_species.return_value = _identification_human()

    with caplog.at_level('INFO'):
        result = system._burst_human_sweep(
            selected, {'all_frame_paths': [sibling, selected]}
        )

    from data_models import DetectionStatus
    assert result is not None and result.status == DetectionStatus.HUMAN
    system.species_identifier.identify_species.assert_called_once_with(sibling)
    assert '[HUMAN-SWEEP]' in caplog.text


def test_sweep_skips_near_identical_burst(system, tmp_path):
    """The common case — five near-identical frames — costs zero extra passes."""
    selected, sibling = _write_frames(tmp_path, {'f5.jpg': 100, 'f1.jpg': 100})
    system.config.performance.human_sweep_divergence_threshold = 0.03
    system.config.performance.human_sweep_max_frames = 2
    system.species_identifier = MagicMock()

    assert system._burst_human_sweep(
        selected, {'all_frame_paths': [sibling, selected]}
    ) is None
    system.species_identifier.identify_species.assert_not_called()


def test_sweep_returns_none_when_no_person_found(system, tmp_path):
    selected, sibling = _write_frames(tmp_path, {'f5.jpg': 100, 'f1.jpg': 200})
    system.config.performance.human_sweep_divergence_threshold = 0.03
    system.config.performance.human_sweep_max_frames = 2
    system.species_identifier = MagicMock()
    system.species_identifier.identify_species.return_value = _identification_no_animal()

    assert system._burst_human_sweep(
        selected, {'all_frame_paths': [sibling, selected]}
    ) is None


def test_sweep_respects_max_frames_cap(system, tmp_path):
    """Blind time is an FN source, so the sweep never exceeds the cap."""
    paths = _write_frames(
        tmp_path,
        {'f5.jpg': 100, 'f1.jpg': 200, 'f2.jpg': 210, 'f3.jpg': 220, 'f4.jpg': 230},
    )
    selected, siblings = paths[0], paths[1:]
    system.config.performance.human_sweep_divergence_threshold = 0.03
    system.config.performance.human_sweep_max_frames = 2
    system.species_identifier = MagicMock()
    system.species_identifier.identify_species.return_value = _identification_no_animal()

    system._burst_human_sweep(selected, {'all_frame_paths': siblings + [selected]})
    assert system.species_identifier.identify_species.call_count == 2


@pytest.mark.parametrize('threshold,max_frames', [(0.0, 2), (0.03, 0)])
def test_sweep_disabled_by_either_rollback_lever(system, tmp_path, threshold, max_frames):
    selected, sibling = _write_frames(tmp_path, {'f5.jpg': 100, 'f1.jpg': 200})
    system.config.performance.human_sweep_divergence_threshold = threshold
    system.config.performance.human_sweep_max_frames = max_frames
    system.species_identifier = MagicMock()

    assert system._burst_human_sweep(
        selected, {'all_frame_paths': [sibling, selected]}
    ) is None
    system.species_identifier.identify_species.assert_not_called()


def test_sweep_handles_missing_sharpness_info(system, tmp_path):
    (selected,) = _write_frames(tmp_path, {'f5.jpg': 100})
    system.config.performance.human_sweep_divergence_threshold = 0.03
    system.config.performance.human_sweep_max_frames = 2
    system.species_identifier = MagicMock()

    assert system._burst_human_sweep(selected, None) is None
    assert system._burst_human_sweep(selected, {}) is None
    system.species_identifier.identify_species.assert_not_called()


@pytest.mark.asyncio
async def test_process_detection_suppresses_swept_human_burst(system, tmp_path):
    """End to end: a review-class burst hiding a person is logged HUMAN and
    never notified — the leak burst 5119 exhibited."""
    selected, sibling = _write_frames(tmp_path, {'f5.jpg': 100, 'f1.jpg': 200})
    system.config.performance.human_sweep_divergence_threshold = 0.03
    system.config.performance.human_sweep_max_frames = 2
    system.config.performance.suppress_human_alerts = True
    system.species_identifier = MagicMock()
    system.species_identifier.identify_species.side_effect = [
        _identification_no_animal(),   # the selected frame: no person visible
        _identification_human(),       # the divergent sibling: a person
    ]
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    await system._process_and_notify_detection(
        selected, 5280,
        sharpness_info={'all_frame_paths': [sibling, selected],
                        'sharpness_score': 13.6,
                        'below_sharpness_floor': False},
    )

    telegram.send_photo_with_caption.assert_not_called()
    telegram.send_media_group.assert_not_called()
    telegram.send_detection_notification.assert_not_called()

    with sqlite3.connect(system.database.db_path) as conn:
        row = conn.execute(
            "SELECT detection_status FROM detections ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row[0] == 'human'


# ---------------------------------------------------------------------------
# Human-Proximity Gate widened to unnamed-animal IDENTIFIED bursts (exp #26,
# unnamed-animal-main-leak, 2026-09-14): SpeciesNet's ensemble sometimes
# returns a fully-generic "<uuid>;;;;;;animal" rollup label ("something is
# there, I cannot name it"), which routes to DetectionStatus.IDENTIFIED and
# so bypasses every review-class mute path (human-proximity, blur, scene,
# sampling, deferral) — a MAIN-channel "animal detected" alert. Measured
# over the whole corpus, gating these on the existing Human-Proximity Gate
# (window OR density) costs zero known human-labelled animal/animal_wrong_id
# false negatives. Named-species identifications must be completely
# unaffected.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unnamed_animal_muted_within_human_proximity_window(system, tmp_path, caplog):
    """An IDENTIFIED burst carrying the generic '<uuid>;;;;;;animal' label,
    landing shortly after a HUMAN-status detection, is muted via the
    human-proximity gate: no Telegram send, a [HUMAN-PROXIMITY] log line,
    and the DB row records human_proximity_muted=1 even though its status
    is 'identified', not review-class."""
    system._last_human_detection_at = datetime.now() - timedelta(seconds=60)
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_unnamed_animal()
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

    gate_logs = [
        r.message for r in caplog.records
        if "HUMAN-GATE" in r.message or "[HUMAN-PROXIMITY]" in r.message
        or "[BLUR]" in r.message or "[SCENE-GATE]" in r.message
        or "[REVIEW-SAMPLE]" in r.message
    ]
    assert len(gate_logs) == 1
    assert "[HUMAN-PROXIMITY]" in gate_logs[0]

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    assert row is not None
    assert row['detection_status'] == 'identified'
    assert row['human_proximity_muted'] == 1


@pytest.mark.asyncio
async def test_unnamed_animal_not_muted_without_recent_human(system, tmp_path):
    """The same generic '<uuid>;;;;;;animal' burst, with no recent
    HUMAN-status detection, still notifies as before and DB-records
    human_proximity_muted=0 (evaluated-but-not-muted), not NULL."""
    assert system._last_human_detection_at is None
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_unnamed_animal()
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
    assert row['detection_status'] == 'identified'
    assert row['human_proximity_muted'] == 0


@pytest.mark.asyncio
async def test_named_species_unaffected_by_unnamed_animal_widening(system, tmp_path):
    """A named-species IDENTIFIED burst (e.g. domestic cat) inside the
    human-proximity window is NOT muted — the widening only applies to the
    generic unnamed-animal rollup label, never to a real identification.
    human_proximity_muted stays NULL (not evaluated at all), same as any
    other non-review-class, non-unnamed-animal status."""
    system._last_human_detection_at = datetime.now() - timedelta(seconds=60)
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_named_species()
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
    assert row['detection_status'] == 'identified'
    assert row['human_proximity_muted'] is None


# ---------------------------------------------------------------------------
# Confident-Blank Mute Gate (exp #29, 2026-09-17): mute a review-class burst
# when the classifier's RAW top-1 prediction is SpeciesNet's fully-generic
# "blank" label at or above blank_confidence_mute_threshold (default 0.92).
# Precedence: Human/Privacy > Human-Proximity > Blur > Confident-Blank >
# Scene > Review Sampling > Deferred Send.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_blank_confidence_at_threshold_suppresses_notification(system, tmp_path, caplog):
    """A review-class burst whose raw top-1 is blank at exactly the
    threshold is muted: no Telegram send, a [BLANK-CONF] log line, and the
    DB row records blank_confidence_muted + top_species_raw/score."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal_with_top1(
            "f1856211-d0e3-4ac6-8016-16aacd8d0dbe;;;;;;blank", 0.92
        )
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
    assert any("[BLANK-CONF]" in r.message for r in caplog.records)

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    assert row is not None
    assert row['blank_confidence_muted'] == 1
    assert row['top_species_raw'] == "f1856211-d0e3-4ac6-8016-16aacd8d0dbe;;;;;;blank"
    assert row['top_species_score'] == pytest.approx(0.92)


@pytest.mark.asyncio
async def test_blank_confidence_above_threshold_suppresses_notification(system, tmp_path):
    """Comfortably above threshold also mutes."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal_with_top1("uuid;;;;;;blank", 0.98)
    )
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    await system._process_and_notify_detection(img, 5000)

    telegram.send_photo_with_caption.assert_not_called()
    telegram.send_media_group.assert_not_called()

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    assert row['blank_confidence_muted'] == 1


@pytest.mark.asyncio
async def test_blank_confidence_below_threshold_still_notifies(system, tmp_path):
    """Just below threshold does not mute — the burst still sends as a
    normal (REVIEW-prefixed) notification, and blank_confidence_muted is
    False (evaluated-but-not-muted), not NULL."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal_with_top1("uuid;;;;;;blank", 0.91)
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
    assert row['blank_confidence_muted'] == 0


@pytest.mark.asyncio
async def test_blank_confidence_not_muted_for_non_blank_label(system, tmp_path):
    """A high-confidence but NON-blank raw top-1 (e.g. a real species guess)
    must never trip this gate."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal_with_top1(
            "uuid;mammalia;carnivora;felidae;felis;catus;domestic cat", 0.99
        )
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
    assert row['blank_confidence_muted'] == 0


@pytest.mark.asyncio
async def test_blank_confidence_missing_score_does_not_mute(system, tmp_path):
    """No top_classifier_prediction in metadata (score is None) must not
    mute — fails open."""
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
    assert row['blank_confidence_muted'] == 0


@pytest.mark.asyncio
async def test_blank_confidence_threshold_zero_disables_gate(system, tmp_path):
    """threshold=0.0 disables the gate entirely (the rollback lever):
    blank_confidence_muted stays NULL ("gate didn't apply") even for a
    would-have-muted blank/high-score burst, and the burst still notifies."""
    system.config.performance.blank_confidence_mute_threshold = 0.0
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal_with_top1("uuid;;;;;;blank", 0.99)
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
    assert row['blank_confidence_muted'] is None


@pytest.mark.asyncio
async def test_blank_confidence_muted_none_for_non_review_status(system, tmp_path):
    """A non-review-class (IDENTIFIED animal) status leaves
    blank_confidence_muted NULL regardless of what the raw top-1 says —
    the gate only ever evaluates review-class bursts."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    identified = _identification(True, boxes=[{'confidence': 0.7}])
    identified.metadata = {
        'top_classifier_prediction': {'label': 'uuid;;;;;;blank', 'score': 0.99}
    }
    system.species_identifier.identify_species = MagicMock(return_value=identified)
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    await system._process_and_notify_detection(img, 5000)

    assert telegram.send_photo_with_caption.called or telegram.send_media_group.called

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    assert row['blank_confidence_muted'] is None


@pytest.mark.asyncio
async def test_blur_wins_over_blank_confidence_single_log(system, tmp_path, caplog):
    """A below-floor blurry review-class burst that is ALSO blank-confident
    is suppressed via the blur gate only — exactly one suppression log,
    [BLUR], not [BLANK-CONF]."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal_with_top1("uuid;;;;;;blank", 0.99)
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

    gate_logs = [
        r.message for r in caplog.records
        if "HUMAN-GATE" in r.message or "[HUMAN-PROXIMITY]" in r.message
        or "[BLUR]" in r.message or "[BLANK-CONF]" in r.message
        or "[SCENE-GATE]" in r.message or "[REVIEW-SAMPLE]" in r.message
    ]
    assert len(gate_logs) == 1
    assert "[BLUR]" in gate_logs[0]


@pytest.mark.asyncio
async def test_blank_confidence_wins_over_scene_gate_single_log(system, tmp_path, caplog):
    """A blank-confident review-class burst that would ALSO match the scene
    reference is suppressed via the blank-confidence gate only — exactly one
    suppression log, [BLANK-CONF], not [SCENE-GATE]."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal_with_top1("uuid;;;;;;blank", 0.99)
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

    gate_logs = [
        r.message for r in caplog.records
        if "HUMAN-GATE" in r.message or "[HUMAN-PROXIMITY]" in r.message
        or "[BLUR]" in r.message or "[BLANK-CONF]" in r.message
        or "[SCENE-GATE]" in r.message or "[REVIEW-SAMPLE]" in r.message
    ]
    assert len(gate_logs) == 1
    assert "[BLANK-CONF]" in gate_logs[0]

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    # process_detection computes scene_gate_muted independently of the
    # blank-confidence gate (same as it does for the blur gate) — it may
    # legitimately also be True here. What matters is precedence in the
    # notification layer, asserted above via the single [BLANK-CONF] log.
    assert row['blank_confidence_muted'] == 1


@pytest.mark.asyncio
async def test_blank_confidence_wins_over_sampling_single_log(system, tmp_path, caplog):
    """A blank-confident review-class burst is suppressed via the
    blank-confidence gate, not double-suppressed or mis-attributed to
    sampling, even at rate=0.0."""
    system.config.performance.review_sample_rate = 0.0
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal_with_top1("uuid;;;;;;blank", 0.99)
    )
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
        or "[BLUR]" in r.message or "[BLANK-CONF]" in r.message
        or "[SCENE-GATE]" in r.message or "[REVIEW-SAMPLE]" in r.message
    ]
    assert len(gate_logs) == 1
    assert "[BLANK-CONF]" in gate_logs[0]


@pytest.mark.asyncio
async def test_human_wins_over_blank_confidence_single_log(system, tmp_path, caplog):
    """A HUMAN burst is suppressed by the human gate, not blank-confidence —
    single suppression log, even with a blank/high-score top-1 in
    metadata."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    human = _identification_human()
    human.metadata = {
        'top_classifier_prediction': {'label': 'uuid;;;;;;blank', 'score': 0.99}
    }
    system.species_identifier.identify_species = MagicMock(return_value=human)
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
        or "[BLUR]" in r.message or "[BLANK-CONF]" in r.message
        or "[SCENE-GATE]" in r.message or "[REVIEW-SAMPLE]" in r.message
    ]
    assert len(gate_logs) == 1
    assert "HUMAN-GATE" in gate_logs[0]


# ---------------------------------------------------------------------------
# Unnamed-Animal Blank-Raw Mute Gate (exp #32, 2026-09-19): an IDENTIFIED
# burst carrying SpeciesNet's fully-generic "<uuid>;;;;;;animal" rollup fires
# a MAIN-channel species alert that bypasses every review-class mute path.
# When the classifier's RAW top-1 over the crop is "blank" BELOW
# unnamed_animal_blank_mute_threshold (default 0.90) the two models disagree
# and the classifier isn't even confident the crop is empty — mute it.
# Precedence: Human/Privacy > Human-Proximity > Unnamed-Animal-Blank > Blur >
# Confident-Blank > Scene > Review Sampling > Deferred Send.
# ---------------------------------------------------------------------------

def _unnamed_animal_with_top1(label, score, confidence=0.58):
    """The generic ';;;;;;animal' rollup plus a raw classifier top-1."""
    result = _identification_unnamed_animal(confidence=confidence)
    result.metadata = {'top_classifier_prediction': {'label': label, 'score': score}}
    return result


@pytest.mark.asyncio
async def test_unnamed_animal_blank_below_threshold_suppresses_notification(
    system, tmp_path, caplog
):
    """Burst 5374's shape: generic ';;;;;;animal' rollup with a low-confidence
    blank raw top-1. No Telegram send, one [UNNAMED-BLANK] log, DB records
    unnamed_animal_blank_muted."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_unnamed_animal_with_top1("f1856211;;;;;;blank", 0.0561)
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
    assert sum("[UNNAMED-BLANK]" in r.message for r in caplog.records) == 1

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    assert row['unnamed_animal_blank_muted'] == 1
    assert row['top_species_raw'] == "f1856211;;;;;;blank"


@pytest.mark.asyncio
async def test_unnamed_animal_blank_at_threshold_still_notifies(system, tmp_path):
    """The gate mutes strictly BELOW the threshold — a score exactly at it
    must still alert (the known animal counter-example sits above)."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_unnamed_animal_with_top1("uuid;;;;;;blank", 0.90)
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
    assert row['unnamed_animal_blank_muted'] == 0


@pytest.mark.asyncio
async def test_unnamed_animal_high_confidence_blank_still_notifies(system, tmp_path):
    """The n=1 animal counter-example (ids 2212/2213, blank @ 0.9722): a
    high-confidence blank raw top-1 on this shape is NOT muted."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_unnamed_animal_with_top1("uuid;;;;;;blank", 0.9722)
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
    assert row['unnamed_animal_blank_muted'] == 0


@pytest.mark.asyncio
async def test_unnamed_animal_named_raw_top1_never_muted(system, tmp_path):
    """34/34 labelled rows whose raw top-1 NAMES an animal are real animals —
    a named raw top-1 must never be muted, however low its score."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_unnamed_animal_with_top1(
            "87fdd451;aves;passeriformes;corvidae;corvus;brachyrhynchos;american crow",
            0.2306,
        )
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
    assert row['unnamed_animal_blank_muted'] == 0


@pytest.mark.asyncio
async def test_unnamed_animal_blank_threshold_zero_disables_gate(system, tmp_path):
    """0.0 DISABLES the gate (rollback lever) — it does not mean 'mute
    nothing by comparison': the column stays NULL ('gate didn't apply')."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.config.performance.unnamed_animal_blank_mute_threshold = 0.0
    system.species_identifier.identify_species = MagicMock(
        return_value=_unnamed_animal_with_top1("uuid;;;;;;blank", 0.05)
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
    assert row['unnamed_animal_blank_muted'] is None


@pytest.mark.asyncio
async def test_unnamed_animal_blank_muted_none_for_review_class(system, tmp_path):
    """A review-class burst is not this shape — the gate leaves the column
    NULL and the Confident-Blank gate owns that population instead."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal_with_top1("uuid;;;;;;blank", 0.05)
    )
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    await system._process_and_notify_detection(img, 5000)

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    assert row['unnamed_animal_blank_muted'] is None


@pytest.mark.asyncio
async def test_human_proximity_wins_over_unnamed_animal_blank_single_log(
    system, tmp_path, caplog
):
    """A burst this gate WOULD mute that the human-proximity gate already
    mutes produces exactly one suppression log ([HUMAN-PROXIMITY])."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system._last_human_detection_at = datetime.now()
    system.species_identifier.identify_species = MagicMock(
        return_value=_unnamed_animal_with_top1("uuid;;;;;;blank", 0.05)
    )
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()

    with caplog.at_level("INFO"):
        await system._process_and_notify_detection(img, 5000)

    telegram.send_photo_with_caption.assert_not_called()
    telegram.send_media_group.assert_not_called()
    assert any("[HUMAN-PROXIMITY]" in r.message for r in caplog.records)
    assert not any("[UNNAMED-BLANK]" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Animal-Proximity Review Exemption (exp #33, animal-proximity-review-
# exemption, 2026-09-20): a review-class (NO_ANIMAL/UNCLASSIFIABLE) burst
# landing within animal_proximity_window_seconds after the most recent
# named-animal IDENTIFIED detection is exempted from the Review Sampling
# Gate ONLY — it always sends as a REVIEW message instead of being sampled
# out. Every earlier-precedence mute gate (Human/Privacy, Human-Proximity,
# Blur, Confident-Blank, Scene) is unaffected.
# ---------------------------------------------------------------------------

def _identification_blank_species():
    """An IDENTIFIED result whose species_name is a populated blank verdict
    (uuid;;;;;;blank) — status stays IDENTIFIED (distinct from
    _identification_no_animal's NO_ANIMAL status), used to verify a blank
    label never anchors the Animal-Proximity Review Exemption even if it
    somehow reaches IDENTIFIED status."""
    from data_models import IdentificationResult, DetectionResult
    det = DetectionResult(
        animals_detected=True,
        detection_count=1,
        bounding_boxes=[{'confidence': 0.4, 'category': '1'}],
        detections=[],
        processing_time=0.1,
    )
    return IdentificationResult(
        species_name="uuid;;;;;;blank",
        confidence=0.4,
        api_success=True,
        processing_time=0.5,
        detection_result=det,
        animals_detected=True,
    )


@pytest.mark.asyncio
async def test_animal_proximity_exempt_within_window_sends_review(system, tmp_path, caplog):
    """A review-class burst landing shortly after a named-animal IDENTIFIED
    detection is exempted from the Review Sampling Gate — it sends as a
    REVIEW message even at review_sample_rate=0.0 (which would otherwise
    sample out every review-class burst), and review_sampled_out is
    persisted as False."""
    system.config.performance.review_sample_rate = 0.0
    system._last_animal_detection_at = datetime.now() - timedelta(seconds=25)
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

    assert telegram.send_photo_with_caption.called or telegram.send_media_group.called
    assert any("[ANIMAL-PROXIMITY]" in r.message for r in caplog.records)
    assert not any("[REVIEW-SAMPLE]" in r.message for r in caplog.records)

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    assert row['review_sampled_out'] == 0


@pytest.mark.asyncio
async def test_animal_proximity_no_exempt_outside_window(system, tmp_path, caplog):
    """A review-class burst well outside the animal-proximity window is not
    exempted by the backward exemption in process_detection — it is sampled
    out at review_sample_rate=0.0. Since exp #39 (leading-edge-animal-
    proximity), a sampled-out burst is no longer dropped immediately but
    handed to the forward deferral instead (animal_proximity_window_seconds
    shrunk to 0.01 here purely so the test doesn't block on a real 180s
    sleep); with no animal landing AFTER it either, the deferred task ends up
    logging the same [REVIEW-SAMPLE] suppression the immediate path used to,
    and review_sampled_out stays True — the end state this test asserts is
    unchanged, only the path to it is now asynchronous."""
    system.config.performance.review_sample_rate = 0.0
    system.config.performance.animal_proximity_window_seconds = 0.01
    system._last_animal_detection_at = datetime.now() - timedelta(seconds=200)
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
        assert not any("[REVIEW-SAMPLE]" in r.message for r in caplog.records)
        assert len(system._pending_review_tasks) == 1
        task = next(iter(system._pending_review_tasks))
        await task

    telegram.send_photo_with_caption.assert_not_called()
    telegram.send_media_group.assert_not_called()
    assert any("[REVIEW-SAMPLE]" in r.message for r in caplog.records)
    assert not any("[ANIMAL-PROXIMITY]" in r.message for r in caplog.records)
    assert not any("[ANIMAL-DEFER]" in r.message for r in caplog.records)

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    assert row['review_sampled_out'] == 1


@pytest.mark.asyncio
async def test_animal_proximity_zero_window_disables_exemption(system, tmp_path, caplog):
    """PERFORMANCE_ANIMAL_PROXIMITY_WINDOW_SECONDS=0 disables the exemption
    (the rollback lever) even with a very recent named-animal anchor."""
    system.config.performance.review_sample_rate = 0.0
    system.config.performance.animal_proximity_window_seconds = 0.0
    system._last_animal_detection_at = datetime.now() - timedelta(seconds=1)
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
    assert not any("[ANIMAL-PROXIMITY]" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_animal_proximity_no_exempt_without_prior_animal(system, tmp_path, caplog):
    """No prior named-animal IDENTIFIED detection recorded (fresh system) —
    the exemption never fires. Shrinks animal_proximity_window_seconds so the
    exp #39 forward deferral this now falls into (review_sample_rate=0.0 with
    no exemption) doesn't leave a ~180s background task sleeping past the end
    of this test."""
    assert system._last_animal_detection_at is None
    system.config.performance.review_sample_rate = 0.0
    system.config.performance.animal_proximity_window_seconds = 0.01
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
        assert len(system._pending_review_tasks) == 1
        task = next(iter(system._pending_review_tasks))
        await task

    telegram.send_photo_with_caption.assert_not_called()
    telegram.send_media_group.assert_not_called()
    assert not any("[ANIMAL-PROXIMITY]" in r.message for r in caplog.records)


def test_animal_proximity_exemption_does_not_apply_to_non_review_status(system, caplog):
    """The exemption only ever touches review_sampled_out for review-class
    statuses — an IDENTIFIED unnamed-animal burst is untouched (no
    [ANIMAL-PROXIMITY] log, review_sampled_out stays NULL) even with a very
    recent prior named-animal detection."""
    system._last_animal_detection_at = datetime.now() - timedelta(seconds=5)
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_unnamed_animal()
    )

    with caplog.at_level("INFO"):
        result, _ = system.process_detection("capture.jpg", 5000, None)

    assert result['detection_status'] == 'identified'
    assert result['review_sampled_out'] is None
    assert not any("[ANIMAL-PROXIMITY]" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_animal_proximity_exemption_does_not_override_blur_gate(system, tmp_path, caplog):
    """A below-floor review-class burst is suppressed by the blur gate even
    when it also qualifies for the animal-proximity exemption (recent
    named-animal anchor) — the exemption can only flip review_sampled_out,
    never bypass an earlier-precedence gate. Single suppression log."""
    system.config.performance.review_sample_rate = 0.0
    system._last_animal_detection_at = datetime.now() - timedelta(seconds=30)
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

    gate_logs = [
        r.message for r in caplog.records
        if "HUMAN-GATE" in r.message or "[HUMAN-PROXIMITY]" in r.message
        or "[BLUR]" in r.message or "[BLANK-CONF]" in r.message
        or "[SCENE-GATE]" in r.message or "[REVIEW-SAMPLE]" in r.message
    ]
    assert len(gate_logs) == 1
    assert "[BLUR]" in gate_logs[0]

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    # The exemption still flips review_sampled_out (it only ever touches
    # that one flag) but the blur gate's own independent flag is what
    # actually suppressed the send — confirming precedence held.
    assert row['review_sampled_out'] == 0
    assert row['below_sharpness_floor'] == 1


@pytest.mark.asyncio
async def test_animal_proximity_exemption_does_not_override_human_proximity_gate(
    system, tmp_path, caplog
):
    """Privacy-critical: a review-class burst inside BOTH the human-proximity
    window and the animal-proximity exemption window is still suppressed via
    the human-proximity gate — the exemption never overrides a privacy
    mute."""
    system.config.performance.review_sample_rate = 0.0
    system._last_human_detection_at = datetime.now() - timedelta(seconds=60)
    system._last_animal_detection_at = datetime.now() - timedelta(seconds=30)
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

    gate_logs = [
        r.message for r in caplog.records
        if "HUMAN-GATE" in r.message or "[HUMAN-PROXIMITY]" in r.message
        or "[BLUR]" in r.message or "[BLANK-CONF]" in r.message
        or "[SCENE-GATE]" in r.message or "[REVIEW-SAMPLE]" in r.message
    ]
    assert len(gate_logs) == 1
    assert "[HUMAN-PROXIMITY]" in gate_logs[0]


def test_named_species_updates_last_animal_detection_at(system):
    """Processing a named-species IDENTIFIED burst updates the in-memory
    tracker so the NEXT review-class burst (moments later) can be measured
    against it."""
    assert system._last_animal_detection_at is None
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_named_species()
    )
    _, ts = system.process_detection("capture.jpg", 5000, None)
    assert system._last_animal_detection_at == ts


def test_unnamed_animal_does_not_update_last_animal_detection_at(system):
    """The generic '<uuid>;;;;;;animal' rollup is IDENTIFIED-status but not a
    NAMED animal — must not anchor the exemption window."""
    assert system._last_animal_detection_at is None
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_unnamed_animal()
    )
    system.process_detection("capture.jpg", 5000, None)
    assert system._last_animal_detection_at is None


def test_blank_species_does_not_update_last_animal_detection_at(system):
    """A populated blank verdict ('uuid;;;;;;blank') must not anchor the
    exemption window even if it somehow reaches IDENTIFIED status."""
    assert system._last_animal_detection_at is None
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_blank_species()
    )
    system.process_detection("capture.jpg", 5000, None)
    assert system._last_animal_detection_at is None


def test_human_status_does_not_update_last_animal_detection_at(system):
    """A HUMAN-status detection must never anchor the exemption window,
    however confidently 'Homo sapiens' is named — status isn't IDENTIFIED at
    all, so the check short-circuits before the label is even inspected."""
    assert system._last_animal_detection_at is None
    system.species_identifier.identify_species = MagicMock(return_value=_identification_human())
    system.process_detection("capture.jpg", 5000, None)
    assert system._last_animal_detection_at is None


# ---------------------------------------------------------------------------
# Leading-edge Animal-Proximity Deferral (exp #39, leading-edge-animal-
# proximity, 2026-09-25): the backward Animal-Proximity Review Exemption
# above (exp #33) can only exempt a review-class burst landing AFTER a
# named-animal IDENTIFIED detection. Burst 5448 (07:36:43, a real calico cat
# at extreme close range, status=unclassifiable) was dropped by the Review
# Sampling Gate; burst 5449, the same cat, was correctly IDENTIFIED 37s
# later — the naming happened AFTER the sampled-out burst, exactly the case
# the backward exemption structurally cannot catch. A sampled-out
# review-class burst is now handed to the SAME deferred-send machinery the
# human leading-edge fix (exp #11) uses (`_deferred_review_send`), with
# `require_animal_proximity=True`: it sleeps animal_proximity_window_seconds,
# then only sends if a named-animal IDENTIFIED detection landed within that
# window afterward. Reuses the existing knob — no new config field, no new
# DB column; `0` disables both halves (this forward deferral and the exp #33
# backward exemption) at once, the single rollback lever.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_animal_proximity_deferral_recovers_sampled_out_burst(system, tmp_path, caplog):
    """A sampled-out review-class burst is recovered when a named-animal
    IDENTIFIED detection lands within animal_proximity_window_seconds
    afterward — the leading-edge mirror of burst 5448/5449 (tonight's cat).
    The notification is sent, [ANIMAL-DEFER] is logged, and
    update_review_sampled_out(id, False) is called exactly once so the DB
    row matches what actually happened (same convention exp #33 uses for its
    own, backward-looking exemption)."""
    system.config.performance.review_sample_rate = 0.0
    system.config.performance.animal_proximity_window_seconds = 0.01
    captured = _spy_process_detection(system)
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal()
    )
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()
    update_spy = MagicMock(side_effect=system.database.update_review_sampled_out)
    system.database.update_review_sampled_out = update_spy

    with caplog.at_level("INFO"):
        await system._process_and_notify_detection(img, 5000)
        assert len(system._pending_review_tasks) == 1

        # process_detection's own follow-up UPDATE already recorded the
        # initial sampled-out=True write above (see the "Observability
        # columns" bullet's review_sampled_out description) — reset the spy
        # so the assertion below isolates the call the DEFERRED recovery
        # itself makes, not that unrelated earlier one.
        update_spy.reset_mock()

        # A named-animal IDENTIFIED detection lands 1ms after this burst —
        # comfortably inside the 10ms animal-proximity window configured
        # above, regardless of how long the real asyncio.sleep(0.01) below
        # actually takes wall-clock-wise (same pattern as the human deferral
        # tests earlier in this file).
        system._last_animal_detection_at = captured['timestamp'] + timedelta(milliseconds=1)

        task = next(iter(system._pending_review_tasks))
        await task

    assert telegram.send_photo_with_caption.called or telegram.send_media_group.called
    assert any("[ANIMAL-DEFER]" in r.message for r in caplog.records)
    assert not any("[REVIEW-SAMPLE]" in r.message for r in caplog.records)
    update_spy.assert_called_once_with(captured['detection_id'], False)

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM detections WHERE id = ?", (captured['detection_id'],)
        ).fetchone()
    assert row['review_sampled_out'] == 0


@pytest.mark.asyncio
async def test_animal_proximity_deferral_no_recovery_stays_suppressed(system, tmp_path, caplog):
    """No named-animal IDENTIFIED detection lands within the window — the
    deferred task ends up suppressing the notification exactly like the
    immediate REVIEW-SAMPLE drop used to, and never calls
    update_review_sampled_out (the DB row stays sampled_out=True, matching
    what process_detection already wrote — this is the common case, and it
    must stay silent on Telegram)."""
    system.config.performance.review_sample_rate = 0.0
    system.config.performance.animal_proximity_window_seconds = 0.01
    captured = _spy_process_detection(system)
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal()
    )
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()
    update_spy = MagicMock(side_effect=system.database.update_review_sampled_out)
    system.database.update_review_sampled_out = update_spy

    with caplog.at_level("INFO"):
        await system._process_and_notify_detection(img, 5000)
        assert len(system._pending_review_tasks) == 1
        # Isolate the deferred task's own behaviour from process_detection's
        # unrelated initial sampled-out=True write, same as the recovery
        # test above.
        update_spy.reset_mock()
        task = next(iter(system._pending_review_tasks))
        await task

    telegram.send_photo_with_caption.assert_not_called()
    telegram.send_media_group.assert_not_called()
    assert any("[REVIEW-SAMPLE]" in r.message for r in caplog.records)
    assert not any("[ANIMAL-DEFER]" in r.message for r in caplog.records)
    update_spy.assert_not_called()

    with sqlite3.connect(system.database.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM detections WHERE id = ?", (captured['detection_id'],)
        ).fetchone()
    assert row['review_sampled_out'] == 1


@pytest.mark.asyncio
async def test_animal_proximity_deferral_disabled_immediate_drop(system, tmp_path, caplog):
    """animal_proximity_window_seconds == 0 (the rollback lever, shared with
    the exp #33 backward exemption) preserves pre-exp-#39 behaviour exactly:
    a sampled-out burst is dropped immediately with a [REVIEW-SAMPLE] log, no
    background task is ever scheduled."""
    system.config.performance.review_sample_rate = 0.0
    system.config.performance.animal_proximity_window_seconds = 0.0
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
    assert any("[REVIEW-SAMPLE]" in r.message for r in caplog.records)
    assert len(system._pending_review_tasks) == 0


@pytest.mark.asyncio
async def test_animal_proximity_deferral_still_cancelled_by_human(system, tmp_path, caplog):
    """Privacy precedence is preserved even for a recovered burst: a named
    animal lands within the animal-proximity window, recovering the burst
    (Phase 1), but a HUMAN-status detection then lands within the FULL
    review_defer_seconds window measured from the burst's own timestamp
    (Phase 2, unchanged) — the send is still cancelled, human_proximity_muted
    is persisted True, and nothing reaches Telegram. A person arriving after
    the burst wins even over a recovered animal."""
    system.config.performance.review_sample_rate = 0.0
    system.config.performance.animal_proximity_window_seconds = 0.01
    system.config.performance.review_defer_seconds = 0.03
    captured = _spy_process_detection(system)
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake")
    system.species_identifier.identify_species = MagicMock(
        return_value=_identification_no_animal()
    )
    telegram = _mock_telegram(system)
    system.system_monitor = MagicMock()
    system.system_monitor.get_cpu_temperature.return_value = 20.0
    system.cleanup_old_images = MagicMock()
    human_spy = MagicMock(side_effect=system.database.update_human_proximity_muted)
    system.database.update_human_proximity_muted = human_spy

    with caplog.at_level("INFO"):
        await system._process_and_notify_detection(img, 5000)
        assert len(system._pending_review_tasks) == 1

        # Both land immediately after scheduling, before the task's own
        # sleeps run their course — same pattern as the deferral tests
        # above. The animal lands well inside the 10ms animal-proximity
        # window (Phase 1); the human lands inside the FULL 30ms
        # review_defer_seconds window measured from the burst's own
        # timestamp (Phase 2's unchanged check), i.e. still inside the
        # ~20ms remaining after Phase 1 recovers and consumes its 10ms.
        system._last_animal_detection_at = captured['timestamp'] + timedelta(milliseconds=1)
        system._last_human_detection_at = captured['timestamp'] + timedelta(milliseconds=20)

        task = next(iter(system._pending_review_tasks))
        await task

    telegram.send_photo_with_caption.assert_not_called()
    telegram.send_media_group.assert_not_called()
    assert any("[ANIMAL-DEFER]" in r.message for r in caplog.records)
    assert any("[REVIEW-DEFER]" in r.message for r in caplog.records)
    human_spy.assert_called_once_with(captured['detection_id'], True)


@pytest.mark.asyncio
async def test_ordinary_review_burst_unaffected_by_animal_proximity_deferral(system, tmp_path):
    """A review-class burst that was NOT sampled out (review_sample_rate=1.0,
    the fixture default) takes the ordinary review_defer_seconds path
    unchanged by exp #39: require_animal_proximity is False for it, so
    Phase 1 never runs and the send proceeds exactly as it did before this
    change."""
    system.config.performance.review_sample_rate = 1.0
    system.config.performance.review_defer_seconds = 0.01
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
    assert len(system._pending_review_tasks) == 1
    task = next(iter(system._pending_review_tasks))
    await task

    assert telegram.send_photo_with_caption.called or telegram.send_media_group.called
