"""Tests for loop.tickalert — Telegram alert when the nightly claude run fails."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from loop import tickalert


AUTH_TEXT = "Failed to authenticate: OAuth session expired and could not be refreshed\n"


@pytest.mark.parametrize("text", [
    AUTH_TEXT,
    "failed to AUTHENTICATE",
    "Error: Not logged in. Please run /login",
    "please run /login",
    "oauth token revoked",
])
def test_classify_auth(text):
    assert tickalert.classify_failure(text) == "auth"


@pytest.mark.parametrize("text", ["", "Segmentation fault\n", "usage limit reached"])
def test_classify_other(text):
    assert tickalert.classify_failure(text) == "other"


def test_auth_message_has_instructions():
    msg = tickalert.render_message("auth", 1, AUTH_TEXT)
    assert "/login" in msg and "claude" in msg
    assert "expired" in msg.lower()


def test_other_message_has_exit_code_and_tail_trimmed():
    out = "\n".join(f"line{i}" for i in range(50)) + "\n"
    msg = tickalert.render_message("other", 137, out)
    assert "137" in msg
    assert "line49" in msg and "line0\n" not in msg
    assert len(msg) < 1500


def test_other_message_redacts_tokens():
    msg = tickalert.render_message("other", 1, "boom sk-ant-abc123DEF456ghi789 end")
    assert "sk-ant-abc123" not in msg


def _run(tmp_path, monkeypatch, output, code=1, day="2026-10-03", state=None):
    st = tmp_path / "state.json"
    if state is not None:
        import json
        st.write_text(json.dumps(state))
    log = tmp_path / "run.log"
    log.write_text(output)
    calls = []
    monkeypatch.setattr(tickalert, "_get_loop_day", lambda: day)
    monkeypatch.setattr(
        tickalert, "_send_alert",
        lambda state_path, d, text, key: calls.append((d, text, key)),
    )
    tickalert.main(["--exit-code", str(code), "--log", str(log), "--state", str(st)])
    return calls


def test_auth_failure_sends_with_auth_key(tmp_path, monkeypatch):
    calls = _run(tmp_path, monkeypatch, AUTH_TEXT)
    assert len(calls) == 1
    assert calls[0][2] == "last_auth_alert_loopday"


def test_other_failure_sends_with_other_key(tmp_path, monkeypatch):
    calls = _run(tmp_path, monkeypatch, "kaboom\n", code=2)
    assert calls[0][2] == "last_tick_failure_alert_loopday"
    assert "2" in calls[0][1]


def test_dedupe_per_kind_per_loopday(tmp_path, monkeypatch):
    assert _run(tmp_path, monkeypatch, AUTH_TEXT,
                state={"last_auth_alert_loopday": "2026-10-03"}) == []
    # other kind not blocked by auth stamp
    assert len(_run(tmp_path, monkeypatch, "kaboom",
                    state={"last_auth_alert_loopday": "2026-10-03"})) == 1
    # new loop-day re-alerts
    assert len(_run(tmp_path, monkeypatch, AUTH_TEXT, day="2026-10-04",
                    state={"last_auth_alert_loopday": "2026-10-03"})) == 1


def test_send_failure_never_raises(tmp_path, monkeypatch):
    log = tmp_path / "run.log"
    log.write_text(AUTH_TEXT)
    monkeypatch.setattr(tickalert, "_get_loop_day", lambda: "2026-10-03")
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(tickalert, "_send_alert", boom)
    tickalert.main(["--exit-code", "1", "--log", str(log), "--state", str(tmp_path / "s.json")])


def test_missing_log_still_alerts_as_other(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(tickalert, "_get_loop_day", lambda: "2026-10-03")
    monkeypatch.setattr(tickalert, "_send_alert", lambda *a: calls.append(a))
    tickalert.main(["--exit-code", "1", "--log", str(tmp_path / "nope.log"),
                    "--state", str(tmp_path / "s.json")])
    assert len(calls) == 1
