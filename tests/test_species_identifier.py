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


def test_person_below_threshold_but_homo_taxon_fires_uses_ensemble_confidence(tmp_path):
    """Person box at conf 0.1 < threshold 0.3, homo-taxon ensemble at 0.85 → HUMAN with confidence 0.85.

    Regression: when the gate fires due to homo-taxon (not person-box), the reported
    confidence must come from the signal that fired it, not from the weak person-box conf.
    """
    from data_models import DetectionStatus

    identifier, cfg = _make_identifier()
    result = _predict(
        identifier, tmp_path,
        detections=[{"category": "person", "conf": 0.1, "bbox": [0, 0, 1, 1]}],
        prediction="e3954aac;mammalia;primates;hominidae;homo;;homo species",
        prediction_score=0.85,
        classifications={
            "classes": ["e3954aac;mammalia;primates;hominidae;homo;;homo species"],
            "scores": [0.85],
        },
    )
    assert result.status == DetectionStatus.HUMAN
    assert result.species_name == "human"
    # Gate fired due to homo-taxon (person conf 0.1 < 0.3 threshold), so confidence
    # must come from the ensemble prediction_score, not the weak person-box.
    assert result.confidence == 0.85


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
# Observability (Task 1, ADR-004): max person-category confidence is
# recorded in metadata on every parsed result, not only HUMAN ones, so the
# nightly tuning loop can attribute metric shifts even to sub-threshold
# person detections that never fired the privacy gate.
# ===========================================================================

def test_metadata_person_confidence_recorded_below_gate_threshold(tmp_path):
    """person conf 0.25 < default 0.3 threshold, no animal → status stays
    NO_ANIMAL, but metadata still carries the raw person confidence."""
    from data_models import DetectionStatus

    identifier, cfg = _make_identifier()
    result = _predict(
        identifier, tmp_path,
        detections=[{"category": "person", "conf": 0.25, "bbox": [0, 0, 1, 1]}],
    )
    assert result.status == DetectionStatus.NO_ANIMAL
    assert result.metadata is not None
    assert result.metadata['person_confidence'] == 0.25


def test_metadata_person_confidence_recorded_on_human_status(tmp_path):
    from data_models import DetectionStatus

    identifier, cfg = _make_identifier()
    result = _predict(
        identifier, tmp_path,
        detections=[{"category": "person", "conf": 0.8, "bbox": [0, 0, 1, 1]}],
    )
    assert result.status == DetectionStatus.HUMAN
    assert result.metadata is not None
    assert result.metadata['person_confidence'] == 0.8


def test_metadata_person_confidence_defaults_zero_when_no_person(tmp_path):
    """No person detection at all → metadata['person_confidence'] == 0.0
    (not missing, not None) on the IDENTIFIED path."""
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
    assert result.metadata is not None
    assert result.metadata['person_confidence'] == 0.0


def test_metadata_person_confidence_recorded_when_no_predictions_returned(tmp_path):
    """Even the early 'no predictions returned' ERROR path carries the key
    (as 0.0), so downstream code can always do metadata['person_confidence']
    without a None-check special case."""
    from data_models import DetectionStatus

    identifier, cfg = _make_identifier()
    identifier._model_loaded = True
    identifier._model = MagicMock()
    identifier._model.predict.return_value = {"predictions": []}
    img = tmp_path / "test.jpg"
    img.write_bytes(b"fake")
    result = identifier.identify_species(img)

    assert result.status == DetectionStatus.ERROR
    assert result.metadata is not None
    assert result.metadata['person_confidence'] == 0.0


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


# ===========================================================================
# Exp #13: SpeciesNet's explicit "blank" (empty frame) ensemble verdict scores
# ~0.99, so it used to clear the confidence threshold and be reported as an
# IDENTIFIED species — a MAIN-channel alert on a frame the model called empty,
# bypassing the whole review-class mute stack. It must route to NO_ANIMAL.
# ===========================================================================

def test_blank_ensemble_prediction_routes_to_no_animal(tmp_path):
    from data_models import DetectionStatus

    identifier, cfg = _make_identifier()
    result = _predict(
        identifier, tmp_path,
        detections=[{"category": "animal", "conf": 0.25, "bbox": [0, 0, 1, 1]}],
        prediction="f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank",
        prediction_score=0.9985,
        classifications={
            "classes": ["f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank"],
            "scores": [0.9985],
        },
    )
    assert result.status == DetectionStatus.NO_ANIMAL
    assert result.animals_detected is False
    assert result.species_name == "Unknown species"
    assert "blank" in (result.fallback_reason or "").lower()
    # observability metadata is still carried, so top_species_raw keeps the
    # blank label in the DB for the tuning loop.
    assert result.metadata is not None
    assert result.metadata["top_classifier_prediction"] == {
        "label": "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank",
        "score": 0.9985,
    }


def test_specific_animal_named_blank_suffix_not_treated_as_blank(tmp_path):
    """Only a fully-generic label ending in 'blank' is the empty-frame verdict;
    a populated taxonomy is still an identification."""
    from data_models import DetectionStatus

    identifier, cfg = _make_identifier()
    result = _predict(
        identifier, tmp_path,
        detections=[{"category": "animal", "conf": 0.9, "bbox": [0, 0, 1, 1]}],
        prediction="abc;mammalia;carnivora;canidae;vulpes;vulpes;blank",
        prediction_score=0.85,
        classifications={
            "classes": ["abc;mammalia;carnivora;canidae;vulpes;vulpes;blank"],
            "scores": [0.85],
        },
    )
    assert result.status == DetectionStatus.IDENTIFIED


def test_is_blank_prediction_edge_cases():
    from species_identifier import SpeciesIdentifier

    f = SpeciesIdentifier._is_blank_prediction
    assert f("f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank") is True
    assert f("uuid;;;;;;BLANK") is True
    assert f("") is False
    assert f(None) is False
    assert f("blank") is False  # single segment, no UUID prefix shape
    assert f("aves;;;;;bird") is False
    assert f("f2efdae9;no cv result;no cv result;no cv result;"
             "no cv result;no cv result;no cv result") is False


# ===========================================================================
# Task 3 (ADR-004 observability): the classifier's raw top-1 prediction
# (before geofence/rollup) must be carried in metadata even when the
# ensemble rolls the final label up to a generic class-level guess
# ("aves;;;;;bird"), so callers (caption, DB) can surface the more specific
# guess the classifier actually made.
# ===========================================================================

def test_top_classifier_prediction_in_metadata_for_generic_rollup(tmp_path):
    """Ensemble rolls up to 'aves;;;;;bird' but the classifier's raw top-1
    (species-level, low confidence) is still carried in metadata."""
    from data_models import DetectionStatus

    identifier, cfg = _make_identifier()
    result = _predict(
        identifier, tmp_path,
        detections=[{"category": "animal", "conf": 0.9, "bbox": [0, 0, 1, 1]}],
        prediction="aves;;;;;bird",
        prediction_score=0.8,
        classifications={
            "classes": [
                "def;aves;passeriformes;turdidae;turdus;merula;eurasian blackbird"
            ],
            "scores": [0.34],
        },
    )
    assert result.status == DetectionStatus.IDENTIFIED
    assert result.species_name == "aves;;;;;bird"
    assert result.metadata is not None
    assert result.metadata["top_classifier_prediction"] == {
        "label": "def;aves;passeriformes;turdidae;turdus;merula;eurasian blackbird",
        "score": 0.34,
    }


def test_top_classifier_prediction_list_branch_normalized_to_contract(tmp_path):
    """Legacy list-shaped classifications: the first entry is normalized to
    the {'label': str, 'score': float} contract when it is a conforming dict."""
    identifier, cfg = _make_identifier()
    result = _predict(
        identifier, tmp_path,
        detections=[{"category": "animal", "conf": 0.9, "bbox": [0, 0, 1, 1]}],
        prediction="aves;;;;;bird",
        prediction_score=0.8,
        classifications=[
            {"label": "def;aves;passeriformes;turdidae;turdus;merula;eurasian blackbird",
             "score": 0.34, "extra": "ignored"},
        ],
    )
    assert result.metadata is not None
    assert result.metadata["top_classifier_prediction"] == {
        "label": "def;aves;passeriformes;turdidae;turdus;merula;eurasian blackbird",
        "score": 0.34,
    }


def test_top_classifier_prediction_list_branch_non_dict_entry_yields_none(tmp_path):
    """Legacy list-shaped classifications whose first entry is not a dict
    (e.g. a bare label string) must NOT leak a non-dict into metadata —
    downstream .get() calls would crash the pipeline (never-crash constraint)."""
    identifier, cfg = _make_identifier()
    result = _predict(
        identifier, tmp_path,
        detections=[{"category": "animal", "conf": 0.9, "bbox": [0, 0, 1, 1]}],
        prediction="aves;;;;;bird",
        prediction_score=0.8,
        classifications=["def;aves;passeriformes;turdidae;turdus;merula;eurasian blackbird"],
    )
    assert result.metadata is not None
    assert result.metadata["top_classifier_prediction"] is None


def test_top_classifier_prediction_none_when_no_classifications(tmp_path):
    """No classifications payload at all → metadata key is present but None,
    so callers never need a KeyError special-case."""
    identifier, cfg = _make_identifier()
    result = _predict(
        identifier, tmp_path,
        detections=[{"category": "animal", "conf": 0.9, "bbox": [0, 0, 1, 1]}],
        prediction="abc;mammalia;carnivora;canidae;vulpes;vulpes;Red Fox",
        prediction_score=0.85,
        classifications={},
    )
    assert result.metadata is not None


# ===========================================================================
# Exp #9: raw-classifier homo leak. The ensemble sometimes rolls a
# homo-sapiens RAW classifier top-1 up into a generic/blank/unclassifiable
# label that itself contains no 'homo' segment, while the MegaDetector
# person box is sub-threshold. Both existing gate paths (person-box,
# ensemble-homo-taxon) miss it. A third trigger fires HUMAN when the raw
# top-1 is homo AND the ensemble did not confidently name a specific animal
# -- it must never override a confident, specific animal ID.
# ===========================================================================

def test_raw_classifier_homo_leak_generic_animal_rollup_sets_human(tmp_path):
    """Raw top-1 homo + generic ';;;;;;animal' ensemble + sub-threshold
    person box (0.1 < 0.3 default) -> HUMAN (DB id 1988 pattern)."""
    from data_models import DetectionStatus

    identifier, cfg = _make_identifier()
    result = _predict(
        identifier, tmp_path,
        detections=[{"category": "person", "conf": 0.1, "bbox": [0, 0, 1, 1]}],
        prediction=";;;;;;animal",
        prediction_score=0.55,
        classifications={
            "classes": ["e3954aac;mammalia;primates;hominidae;homo;sapiens;human"],
            "scores": [0.573],
        },
    )
    assert result.status == DetectionStatus.HUMAN
    assert result.species_name == "human"
    # Confidence should reflect the signal that actually fired the gate (the
    # raw classifier's homo score), not the weak person-box or ensemble score.
    assert result.confidence == 0.573
    assert result.metadata is not None
    assert result.metadata["person_confidence"] == 0.1
    assert "privacy gate" in (result.fallback_reason or "").lower()


def test_raw_classifier_homo_leak_unclassifiable_ensemble_sets_human(tmp_path):
    """Raw top-1 homo + unclassifiable/blank ensemble, no person box at all
    -> HUMAN (DB id 2548 pattern)."""
    from data_models import DetectionStatus

    identifier, cfg = _make_identifier()
    result = _predict(
        identifier, tmp_path,
        detections=[{"category": "animal", "conf": 0.9, "bbox": [0, 0, 1, 1]}],
        prediction="unclassifiable",
        prediction_score=0.0,
        classifications={
            "classes": ["e3954aac;mammalia;primates;hominidae;homo;sapiens;human"],
            "scores": [0.573],
        },
    )
    assert result.status == DetectionStatus.HUMAN
    assert result.species_name == "human"
    assert result.confidence == 0.573


def test_raw_classifier_homo_leak_does_not_override_confident_specific_animal(tmp_path):
    """Raw top-1 homo BUT the ensemble confidently names a specific animal
    (genus+species both non-empty) -> must remain the animal ID, never
    overridden to HUMAN. This is the critical never-false-suppress guard."""
    from data_models import DetectionStatus

    identifier, cfg = _make_identifier()
    result = _predict(
        identifier, tmp_path,
        detections=[{"category": "animal", "conf": 0.9, "bbox": [0, 0, 1, 1]}],
        prediction="abc;mammalia;carnivora;canidae;vulpes;vulpes;Red Fox",
        prediction_score=0.9,
        classifications={
            "classes": ["e3954aac;mammalia;primates;hominidae;homo;sapiens;human"],
            "scores": [0.573],
        },
    )
    assert result.status == DetectionStatus.IDENTIFIED
    assert result.species_name == "abc;mammalia;carnivora;canidae;vulpes;vulpes;Red Fox"


def test_non_homo_raw_classifier_top1_with_generic_ensemble_not_human(tmp_path):
    """Raw top-1 is a non-homo animal + generic ensemble rollup -> unchanged
    behavior, NOT human (guards against the new trigger being over-broad)."""
    from data_models import DetectionStatus

    identifier, cfg = _make_identifier()
    result = _predict(
        identifier, tmp_path,
        detections=[{"category": "animal", "conf": 0.9, "bbox": [0, 0, 1, 1]}],
        prediction="aves;;;;;bird",
        prediction_score=0.8,
        classifications={
            "classes": ["def;aves;passeriformes;turdidae;turdus;merula;eurasian blackbird"],
            "scores": [0.34],
        },
    )
    assert result.status != DetectionStatus.HUMAN


def test_raw_classifier_homo_leak_malformed_legacy_list_classifications_no_crash(tmp_path):
    """Legacy list-shaped classifications with a non-dict first entry must
    degrade to 'no raw-homo trigger' rather than raising, and must not
    spuriously fire HUMAN."""
    from data_models import DetectionStatus

    identifier, cfg = _make_identifier()
    result = _predict(
        identifier, tmp_path,
        detections=[{"category": "animal", "conf": 0.9, "bbox": [0, 0, 1, 1]}],
        prediction=";;;;;;animal",
        prediction_score=0.5,
        classifications=["e3954aac;mammalia;primates;hominidae;homo;sapiens;human"],
    )
    assert result.status != DetectionStatus.HUMAN
    assert result.metadata is not None
    assert result.metadata["top_classifier_prediction"] is None


def test_raw_classifier_homo_leak_none_classifications_no_crash(tmp_path):
    """classifications missing entirely (defaults to {}) -> no crash, no
    spurious HUMAN, top_classifier_prediction stays None."""
    from data_models import DetectionStatus

    identifier, cfg = _make_identifier()
    result = _predict(
        identifier, tmp_path,
        detections=[{"category": "animal", "conf": 0.9, "bbox": [0, 0, 1, 1]}],
        prediction=";;;;;;animal",
        prediction_score=0.5,
        classifications=None,
    )
    assert result.status != DetectionStatus.HUMAN
    assert result.metadata["top_classifier_prediction"] is None
