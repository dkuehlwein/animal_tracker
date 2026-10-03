"""Unit tests for notification_gate: decide(), RecentHumanEvents and the
per-gate evaluators."""

import sys
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

sys.path.append('src')

from notification_gate import (  # noqa: E402
    Action, Channel, Decision, GateContext, RecentHumanEvents, decide,
    evaluate_blank_confidence, evaluate_human_proximity, human_events_horizon_seconds,
)
from tests.test_wildlife_system import _golden_combos, _golden_expected_outcome  # noqa: E402


def _cfg(**overrides):
    base = dict(
        suppress_human_alerts=True,
        blur_mute_min_luma=70.0,
        blank_confidence_mute_threshold=0.92,
        review_sample_rate=0.25,
        review_defer_seconds=240.0,
        human_proximity_window_seconds=120.0,
        human_demoted_person_floor=0.3,
        human_demoted_window_seconds=1800.0,
        human_density_window_seconds=1800.0,
        human_density_count=8,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_decide_matches_golden_precedence_for_every_combination():
    mismatches = []
    for combo in _golden_combos():
        status, suppress, unnamed, prox, (below, luma), blank, sampled, defer = combo
        ctx = GateContext(
            status=status,
            config=_cfg(suppress_human_alerts=suppress, review_defer_seconds=defer),
            unnamed_animal=unnamed,
            human_proximity_muted=prox,
            human_proximity_reason='window' if prox else None,
            below_sharpness_floor=below,
            sharpness_score=5.0,
            luma=luma,
            blank_confidence_muted=blank,
            top_species_raw='uuid;;;;;;blank',
            top_species_score=0.95,
            review_sampled_out=sampled,
        )
        d = decide(ctx)
        actual = (d.action.upper(), f"[{d.gate}]" if d.gate else None)
        expected = _golden_expected_outcome(*combo)
        if actual != expected:
            mismatches.append((combo, expected, actual))
    assert not mismatches, mismatches[:10]


def test_decide_channel_follows_status():
    assert decide(GateContext(status='identified', config=_cfg())).channel == Channel.MAIN
    assert decide(GateContext(status='no_animal', config=_cfg(review_defer_seconds=0))
                  ).channel == Channel.REVIEW


def test_decision_log_line_format():
    d = Decision(Action.MUTE, Channel.REVIEW, 'BLUR', 'sharpness=5.0')
    assert d.log_line(7) == "[BLUR] Suppressing notification for detection 7 (sharpness=5.0)"


@pytest.mark.parametrize('reason,fragment', [
    ('window', 'reason=window: within 120s'),
    ('density', 'reason=density: >= 8 human detections in the last 1800s'),
    ('demoted-band window', 'reason=demoted-band window: within 1800s'),
])
def test_human_proximity_reason_text(reason, fragment):
    d = decide(GateContext(status='no_animal', config=_cfg(), human_proximity_muted=True,
                           human_proximity_reason=reason))
    assert d.gate == 'HUMAN-PROXIMITY'
    assert fragment in d.reason


def test_blur_reason_tolerates_missing_score():
    d = decide(GateContext(status='no_animal', config=_cfg(), below_sharpness_floor=True,
                           sharpness_score=None, luma=80.0))
    assert d.gate == 'BLUR' and 'sharpness=n/a' in d.reason


# --- RecentHumanEvents -----------------------------------------------------

T0 = datetime(2026, 10, 3, 12, 0, 0)


def _at(seconds):
    return T0 + timedelta(seconds=seconds)


def test_store_keeps_sorted_order_and_queries_intervals():
    ev = RecentHumanEvents([_at(250), _at(60)])
    ev.add(_at(10))
    assert list(ev) == [_at(10), _at(60), _at(250)]
    assert ev.latest() == _at(250)
    assert ev.first_after(_at(0), _at(240)) == _at(10)
    assert ev.first_after(_at(10), _at(240)) == _at(60)       # start exclusive
    assert ev.first_after(_at(60), _at(240)) is None
    assert ev.latest_in(_at(0), _at(100)) == _at(60)
    assert ev.latest_in(_at(61), _at(249)) is None
    assert ev.count_in(_at(10), _at(250)) == 3                 # both ends inclusive


def test_store_prune_drops_only_entries_older_than_horizon():
    ev = RecentHumanEvents([_at(0), _at(100), _at(200)])
    ev.prune(_at(200), 150)
    assert list(ev) == [_at(100), _at(200)]


def test_horizon_covers_longest_window_plus_defer():
    cfg = _cfg(human_density_window_seconds=3600.0, review_defer_seconds=240.0)
    assert human_events_horizon_seconds(cfg) >= 3600.0 + 240.0


# --- evaluate_human_proximity ---------------------------------------------

def test_proximity_window_hit():
    ev = RecentHumanEvents([_at(0)])
    assert evaluate_human_proximity(_at(100), None, ev, _cfg()) == (True, 'window')


def test_proximity_ignores_humans_after_the_burst():
    ev = RecentHumanEvents([_at(200)])
    assert evaluate_human_proximity(_at(100), None, ev, _cfg()) == (False, None)


def test_proximity_demoted_band_only_when_widening_caused_the_mute():
    ev = RecentHumanEvents([_at(0)])
    assert evaluate_human_proximity(_at(480), 0.43, ev, _cfg()) == (True, 'demoted-band window')
    assert evaluate_human_proximity(_at(480), 0.1, ev, _cfg()) == (False, None)
    assert evaluate_human_proximity(_at(60), 0.43, ev, _cfg()) == (True, 'window')
    assert evaluate_human_proximity(
        _at(480), 0.43, ev, _cfg(human_demoted_window_seconds=0.0)) == (False, None)


def test_proximity_density():
    ev = RecentHumanEvents([_at(i * 60) for i in range(8)])
    burst = _at(7 * 60 + 500)
    assert evaluate_human_proximity(burst, None, ev, _cfg()) == (True, 'density')
    assert evaluate_human_proximity(
        burst, None, ev, _cfg(human_density_count=0)) == (False, None)


def test_proximity_uses_any_in_window_not_only_latest():
    """A human inside the window is found even if a later (future-relative-
    to-the-burst) one is also in the store."""
    ev = RecentHumanEvents([_at(0), _at(500)])
    assert evaluate_human_proximity(_at(100), None, ev, _cfg()) == (True, 'window')


# --- evaluate_blank_confidence --------------------------------------------

def test_blank_confidence_flag_semantics():
    assert evaluate_blank_confidence('no_animal', 'u;;;;;;blank', 0.92, 0.92) is True
    assert evaluate_blank_confidence('no_animal', 'u;;;;;;blank', 0.91, 0.92) is False
    assert evaluate_blank_confidence('no_animal', 'u;;;;;;blank', None, 0.92) is False
    assert evaluate_blank_confidence('identified', 'u;;;;;;blank', 0.99, 0.92) is None
    assert evaluate_blank_confidence('no_animal', 'u;;;;;;blank', 0.99, 0.0) is None
