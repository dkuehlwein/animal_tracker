"""
Unit tests for standalone helpers in utils.py.
"""

import sys

sys.path.append('src')


# ---------------------------------------------------------------------------
# is_unnamed_animal_label (exp #26 unnamed-animal-main-leak, 2026-09-14):
# SpeciesNet's ensemble sometimes returns a fully-generic "something is
# there, I cannot name it" rollup label (last segment "animal", every
# taxonomy segment empty). That label currently routes to
# DetectionStatus.IDENTIFIED, producing a MAIN-channel alert that bypasses
# every review-class mute path. This helper is deliberately narrow, modeled
# on SpeciesIdentifier._is_blank_prediction: it must not fire for a real,
# named species (however generic-sounding, e.g. "bird"), and it must treat
# SpeciesNet's sentinel segments ("no cv result", "blank") as empty too.
# ---------------------------------------------------------------------------

def test_is_unnamed_animal_label_bare_uuid_rollup():
    from utils import is_unnamed_animal_label
    assert is_unnamed_animal_label(
        "1f689929-d0e3-4ac6-8016-16aacd8d0dbe;;;;;;animal"
    ) is True


def test_is_unnamed_animal_label_all_sentinel_segments():
    from utils import is_unnamed_animal_label
    assert is_unnamed_animal_label(
        "uuid;no cv result;no cv result;no cv result;"
        "no cv result;no cv result;animal"
    ) is True


def test_is_unnamed_animal_label_case_insensitive():
    from utils import is_unnamed_animal_label
    assert is_unnamed_animal_label("uuid;;;;;;ANIMAL") is True


def test_is_unnamed_animal_label_generic_class_rollup_not_unnamed_animal():
    """'aves;;;;;bird' names a class-level rollup (a bird), not the bare
    generic-animal label — must not match."""
    from utils import is_unnamed_animal_label
    assert is_unnamed_animal_label("uuid;aves;;;;;bird") is False


def test_is_unnamed_animal_label_specific_species_not_unnamed_animal():
    from utils import is_unnamed_animal_label
    assert is_unnamed_animal_label(
        "uuid;mammalia;carnivora;felidae;felis;catus;domestic cat"
    ) is False


def test_is_unnamed_animal_label_blank_prediction_not_unnamed_animal():
    from utils import is_unnamed_animal_label
    assert is_unnamed_animal_label("uuid;;;;;;blank") is False


def test_is_unnamed_animal_label_empty_string():
    from utils import is_unnamed_animal_label
    assert is_unnamed_animal_label("") is False


def test_is_unnamed_animal_label_none():
    from utils import is_unnamed_animal_label
    assert is_unnamed_animal_label(None) is False


def test_is_unnamed_animal_label_single_segment_no_uuid_shape():
    from utils import is_unnamed_animal_label
    assert is_unnamed_animal_label("animal") is False


def test_is_unnamed_animal_label_malformed_non_string_input_no_crash():
    from utils import is_unnamed_animal_label
    assert is_unnamed_animal_label(12345) is False
    assert is_unnamed_animal_label(["not", "a", "string"]) is False
