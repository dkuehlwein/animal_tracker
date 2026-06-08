"""Tests for src/loop/metrics.py — Wilson CI, FN unmeasured, idempotent CSV."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loop import metrics


def test_wilson_ci_known_value():
    # 40 successes of 100 at 95%: classic Wilson interval ≈ (0.307, 0.500).
    low, high = metrics.wilson_ci(40, 100)
    assert abs(low - 0.3066) < 0.005
    assert abs(high - 0.5000) < 0.005
    assert low < 0.40 < high


def test_wilson_ci_zero_trials_is_full_interval():
    low, high = metrics.wilson_ci(0, 0)
    assert low == 0.0
    assert high == 1.0


def test_wilson_ci_all_successes():
    low, high = metrics.wilson_ci(10, 10)
    assert high == 1.0
    assert 0.0 < low < 1.0
