"""
Unit tests for the shared feedback protocol and the sidecar's record path
(ADR-004 Phase 1) — no Telegram network involved.
"""

import sys
from types import SimpleNamespace

import pytest

sys.path.append('src')

from feedback_protocol import build_feedback_keyboard, parse_callback_data, CODE_TO_LABEL
from database_manager import DatabaseManager
from telegram_feedback import record_feedback_callback


def _make_db(tmp_path):
    config = SimpleNamespace(storage=SimpleNamespace(database_path=str(tmp_path / "d.db")))
    return DatabaseManager(config)


def test_build_keyboard_callback_data():
    kb = build_feedback_keyboard(42)
    buttons = kb.inline_keyboard[0]
    data = [b.callback_data for b in buttons]
    assert data == ["fb:42:a", "fb:42:fp", "fb:42:ws"]


def test_parse_callback_data_each_code():
    assert parse_callback_data("fb:7:a") == (7, "animal")
    assert parse_callback_data("fb:7:fp") == (7, "false_positive")
    assert parse_callback_data("fb:7:ws") == (7, "wrong_species")


@pytest.mark.parametrize("bad", ["", "fb:7", "xx:7:a", "fb:notint:a", "fb:7:zzz", "fb:7:a:extra"])
def test_parse_callback_data_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_callback_data(bad)


def test_callback_data_within_telegram_limit():
    # Telegram caps callback_data at 64 bytes; verify a large id stays well under.
    data = build_feedback_keyboard(9_999_999).inline_keyboard[0][0].callback_data
    assert len(data.encode()) <= 64


def test_record_feedback_callback_writes_row(tmp_path):
    db = _make_db(tmp_path)
    det_id = db.log_detection(image_path="c.jpg", motion_area=10)
    for code, label in CODE_TO_LABEL.items():
        msg = record_feedback_callback(f"fb:{det_id}:{code}", db)
        assert isinstance(msg, str) and msg
    labels = [r[2] for r in db.get_feedback(det_id)]
    assert sorted(labels) == sorted(CODE_TO_LABEL.values())


def test_record_feedback_callback_rejects_malformed(tmp_path):
    db = _make_db(tmp_path)
    det_id = db.log_detection(image_path="c.jpg", motion_area=10)
    with pytest.raises(ValueError):
        record_feedback_callback("garbage", db)
    assert db.get_feedback(det_id) == []
