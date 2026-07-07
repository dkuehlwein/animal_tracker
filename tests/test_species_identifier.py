"""TDD tests for the human/privacy gate in SpeciesIdentifier._parse_predictions.

Covers task 1 of the human-gate initiative: MegaDetector 'person' detections
and/or an ensemble prediction resolving to genus "homo" must short-circuit to
DetectionStatus.HUMAN, ahead of (and instead of) the animal branch.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _make_identifier():
    from config import Config
    from species_identifier import SpeciesIdentifier
    cfg = Config.create_test_config()
    identifier = SpeciesIdentifier(cfg)
    return identifier, cfg


def _predict(identifier, tmp_path, detections, prediction="blank",
             prediction_score=0.0, classifications=None):
    identifier._model_loaded = True
    identifier._model = MagicMock()
    identifier._model.predict.return_value = {
        "predictions": [
            {
                "detections": detections,
                "prediction": prediction,
                "prediction_score": prediction_score,
                "prediction_source": "classifier",
                "classifications": classifications or {},
            }
        ]
    }
    img = tmp_path / "test.jpg"
    img.write_bytes(b"fake")
    return identifier.identify_species(img)


# ===========================================================================
# Person-box gate
# ===========================================================================

def test_person_high_confidence_no_animal_sets_human_status(tmp_path):
    """person conf 0.8, no animal box → HUMAN."""
    from data_models import DetectionStatus

    identifier, cfg = _make_identifier()
    result = _predict(
        identifier, tmp_path,
        detections=[{"category": "person", "conf": 0.8, "bbox": [0, 0, 1, 1]}],
    )
    assert result.status == DetectionStatus.HUMAN
    assert result.species_name == "human"
    assert result.confidence == 0.8
    assert result.detection_result is not None
    assert result.detection_result.detections == [
        {"category": "person", "conf": 0.8, "bbox": [0, 0, 1, 1]}
    ]


def test_person_below_threshold_leaves_no_animal_status_unchanged(tmp_path):
    """person conf 0.25 < default 0.3 threshold, no animal box → NO_ANIMAL (unchanged)."""
    from data_models import DetectionStatus

    identifier, cfg = _make_identifier()
    result = _predict(
        identifier, tmp_path,
        detections=[{"category": "person", "conf": 0.25, "bbox": [0, 0, 1, 1]}],
    )
    assert result.status == DetectionStatus.NO_ANIMAL


def test_person_category_variants_all_trigger_human(tmp_path):
    """category values 2, '2', 'person', 'human' must all be recognised."""
    from data_models import DetectionStatus

    for category in (2, '2', 'person', 'human'):
        identifier, cfg = _make_identifier()
        result = _predict(
            identifier, tmp_path,
            detections=[{"category": category, "conf": 0.9, "bbox": [0, 0, 1, 1]}],
        )
        assert result.status == DetectionStatus.HUMAN, f"category={category!r}"


# ===========================================================================
# Homo-taxon ensemble gate
# ===========================================================================

def test_homo_taxon_ensemble_with_zero_person_conf_sets_human_status(tmp_path):
    """Ensemble prediction resolves to genus 'homo', no person box at all → HUMAN."""
    from data_models import DetectionStatus

    identifier, cfg = _make_identifier()
    result = _predict(
        identifier, tmp_path,
        detections=[{"category": "animal", "conf": 0.9, "bbox": [0, 0, 1, 1]}],
        prediction="e3954aac;mammalia;primates;hominidae;homo;;homo species",
        prediction_score=0.994,
        classifications={
            "classes": ["e3954aac;mammalia;primates;hominidae;homo;;homo species"],
            "scores": [0.994],
        },
    )
    assert result.status == DetectionStatus.HUMAN
    assert result.species_name == "human"
    # No person-box confidence available -> falls back to ensemble confidence.
    assert result.confidence == 0.994


def test_non_homo_taxon_prediction_does_not_trigger_human_gate(tmp_path):
    """A normal species taxonomy string must not be mistaken for 'homo'."""
    from data_models import DetectionStatus

    identifier, cfg = _make_identifier()
    result = _predict(
        identifier, tmp_path,
        detections=[{"category": "animal", "conf": 0.9, "bbox": [0, 0, 1, 1]}],
        prediction="abc;mammalia;carnivora;canidae;vulpes;vulpes;Red Fox",
        prediction_score=0.85,
        classifications={
            "classes": ["abc;mammalia;carnivora;canidae;vulpes;vulpes;Red Fox"],
            "scores": [0.85],
        },
    )
    assert result.status == DetectionStatus.IDENTIFIED


def test_taxonomy_segment_must_equal_homo_not_substring(tmp_path):
    """A taxon segment merely containing 'homo' as a substring (e.g. 'homoptera')
    must NOT trigger the human gate — only an exact segment match does."""
    from data_models import DetectionStatus

    identifier, cfg = _make_identifier()
    result = _predict(
        identifier, tmp_path,
        detections=[{"category": "animal", "conf": 0.9, "bbox": [0, 0, 1, 1]}],
        prediction="abc;insecta;homoptera;aphididae;aphis;aphis;Aphid",
        prediction_score=0.85,
        classifications={
            "classes": ["abc;insecta;homoptera;aphididae;aphis;aphis;Aphid"],
            "scores": [0.85],
        },
    )
    assert result.status != DetectionStatus.HUMAN


# ===========================================================================
# Precedence: human gate wins over a confident animal detection
# ===========================================================================

def test_person_and_confident_animal_returns_human_precedence(tmp_path):
    """person conf 0.4 + animal conf 0.9 → HUMAN (privacy first)."""
    from data_models import DetectionStatus

    identifier, cfg = _make_identifier()
    result = _predict(
        identifier, tmp_path,
        detections=[
            {"category": "person", "conf": 0.4, "bbox": [0, 0, 0.3, 0.3]},
            {"category": "animal", "conf": 0.9, "bbox": [0.5, 0.5, 0.9, 0.9]},
        ],
        prediction="abc;mammalia;carnivora;canidae;vulpes;vulpes;Red Fox",
        prediction_score=0.9,
        classifications={
            "classes": ["abc;mammalia;carnivora;canidae;vulpes;vulpes;Red Fox"],
            "scores": [0.9],
        },
    )
    assert result.status == DetectionStatus.HUMAN
    assert result.confidence == 0.4


def test_max_person_confidence_used_when_multiple_person_boxes(tmp_path):
    from data_models import DetectionStatus

    identifier, cfg = _make_identifier()
    result = _predict(
        identifier, tmp_path,
        detections=[
            {"category": "person", "conf": 0.35, "bbox": [0, 0, 0.2, 0.2]},
            {"category": "person", "conf": 0.7, "bbox": [0.3, 0.3, 0.5, 0.5]},
        ],
    )
    assert result.status == DetectionStatus.HUMAN
    assert result.confidence == 0.7


# ===========================================================================
# Regression: existing no-detection and identified paths unchanged
# ===========================================================================

def test_no_detections_at_all_still_no_animal_status(tmp_path):
    from data_models import DetectionStatus

    identifier, cfg = _make_identifier()
    result = _predict(identifier, tmp_path, detections=[])
    assert result.status == DetectionStatus.NO_ANIMAL


def test_animal_only_high_confidence_still_identified(tmp_path):
    from data_models import DetectionStatus

    identifier, cfg = _make_identifier()
    result = _predict(
        identifier, tmp_path,
        detections=[{"category": "animal", "conf": 0.9, "bbox": [0, 0, 1, 1]}],
        prediction="abc;mammalia;carnivora;canidae;vulpes;vulpes;Red Fox",
        prediction_score=0.85,
        classifications={
            "classes": ["abc;mammalia;carnivora;canidae;vulpes;vulpes;Red Fox"],
            "scores": [0.85],
        },
    )
    assert result.status == DetectionStatus.IDENTIFIED
    assert result.species_name == "abc;mammalia;carnivora;canidae;vulpes;vulpes;Red Fox"
