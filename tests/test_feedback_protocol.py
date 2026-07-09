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
    assert len(kb.inline_keyboard) == 2
    row1, row2 = kb.inline_keyboard
    assert [b.callback_data for b in row1] == ["fb:42:a", "fb:42:wid", "fb:42:p"]
    assert [b.callback_data for b in row2] == ["fb:42:fp", "fb:42:ct"]
    all_codes = [b.callback_data.split(":")[-1] for row in kb.inline_keyboard for b in row]
    assert "ws" not in all_codes


def test_keyboard_button_texts():
    kb = build_feedback_keyboard(1)
    row1, row2 = kb.inline_keyboard
    assert [b.text for b in row1] == ["✅ Animal", "🐦 Animal, wrong ID", "👤 Human"]
    assert [b.text for b in row2] == ["❌ Nothing there", "🤷 Can't tell"]


def test_legacy_ws_not_on_keyboard():
    # ws still parses (old messages in the channel have live buttons)...
    assert parse_callback_data("fb:1:ws") == (1, "wrong_species")
    # ...but appears on no button of a freshly built keyboard.
    kb = build_feedback_keyboard(1)
    codes = [b.callback_data.split(":")[-1] for row in kb.inline_keyboard for b in row]
    assert "ws" not in codes


@pytest.mark.parametrize("code,label", [
    ("a", "animal"),
    ("wid", "animal_wrong_id"),
    ("p", "person"),
    ("fp", "false_positive"),
    ("ct", "cant_tell"),
    ("ws", "wrong_species"),
])
def test_parse_callback_data_each_code(code, label):
    assert parse_callback_data(f"fb:7:{code}") == (7, label)


@pytest.mark.parametrize("bad", ["", "fb:7", "xx:7:a", "fb:notint:a", "fb:7:zzz", "fb:7:a:extra", "fb:1:xx"])
def test_parse_callback_data_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_callback_data(bad)


def test_callback_data_within_telegram_limit():
    # Telegram caps callback_data at 64 bytes; verify the longest code ("wid")
    # with a large detection id stays well under.
    kb = build_feedback_keyboard(9_999_999)
    wid_button = next(b for row in kb.inline_keyboard for b in row if b.callback_data.endswith(":wid"))
    assert len(wid_button.callback_data.encode()) <= 64


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
