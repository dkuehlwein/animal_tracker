"""Tests for src/loop/report.py — summary + heartbeat text, paused suppression."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loop import report


def _metrics(fp_rate=0.38, fn="unmeasured"):
    return {
        "date": "2026-06-10",
        "total_triggers": 42,
        "labeled_triggers": 30,
        "fp_count": 11,
        "fp_rate": fp_rate,
        "fp_ci": (0.30, 0.46),
        "fn_rate": fn,
        "fn_ci": None if fn == "unmeasured" else (0.05, 0.18),
    }


def test_summary_includes_active_experiment_and_totals():
    """Updated for Task 3: plain-English format no longer emits FP/FN/CI jargon."""
    text = report.render_summary(
        metrics=_metrics(), state={"active_experiment_id": 1, "paused": False},
        active_experiment={"slug": "notification-gate-live", "status": "running"},
    )
    assert "42 images captured" in text
    assert "notification-gate-live" in text


def test_summary_suppresses_tuning_lines_when_paused():
    text = report.render_summary(
        metrics=_metrics(), state={"active_experiment_id": 1, "paused": True},
        active_experiment={"slug": "notification-gate-live", "status": "running"},
    )
    assert "PAUSED" in text
    # When paused we do not advertise active tuning as if it's progressing.
    assert "tuning frozen" in text.lower() or "paused" in text.lower()


def test_heartbeat_text_mentions_alive_and_timestamp():
    text = report.render_heartbeat(last_tick_iso="2026-06-10T23:14:00")
    assert "alive" in text.lower()
    assert "2026-06-10T23:14:00" in text


# ---------------------------------------------------------------------------
# Defect 2: --no-send flag renders without calling Telegram
# ---------------------------------------------------------------------------

def test_report_main_no_send_renders_without_telegram(tmp_path, monkeypatch):
    """--no-send must print the rendered text and NOT call send()."""
    import json
    import sys

    state_path = tmp_path / "state.json"
    lm = {
        "date": "2026-06-10",
        "total_triggers": 10,
        "labeled_triggers": 8,
        "fp_count": 3,
        "fp_rate": 0.375,
        "fp_ci": [0.1, 0.65],
        "fn_rate": "unmeasured",
        "fn_ci": None,
    }
    from loop import state as state_mod
    state_mod.save_state(state_path, {
        "paused": False,
        "active_experiment_id": None,
        "backlog": [],
        "last_metrics": lm,
    })

    # Ensure send() is NOT called (it would fail without real credentials).
    send_called = []
    monkeypatch.setattr(report, "send", lambda text: send_called.append(text) or (_ for _ in ()).throw(AssertionError("send() called with --no-send")))

    monkeypatch.setattr(sys, "argv", [
        "loop.report", "--mode", "summary",
        "--state", str(state_path),
        "--no-send",
    ])

    import io
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        report.main()
    finally:
        output = sys.stdout.getvalue()
        sys.stdout = original_stdout

    assert send_called == [], "send() must not be called when --no-send is used"
    parsed = json.loads(output.strip())
    assert parsed["sent"] is False
    assert "rendered" in parsed
    # Updated for Task 3: plain-English format — check "images captured" instead of legacy "Wildlife loop"/"FP"
    assert "images captured" in parsed["rendered"]
    assert "labelled" in parsed["rendered"]


# ---------------------------------------------------------------------------
# Fix #2: untrustworthy FP alert in summary
# ---------------------------------------------------------------------------

def _metrics_with_trust(fp_trustworthy: bool, error_count: int = 3, total: int = 10):
    return {
        "date": "2026-06-10",
        "total_triggers": total,
        "labeled_triggers": total - error_count,
        "fp_count": 2,
        "fp_rate": 0.2,
        "fp_ci": (0.05, 0.45),
        "fn_rate": "unmeasured",
        "fn_ci": None,
        "error_count": error_count,
        "error_rate": error_count / total if total else 0.0,
        "fp_trustworthy": fp_trustworthy,
    }


def test_summary_includes_untrustworthy_alert_when_fp_not_trustworthy():
    """When fp_trustworthy is False, summary must include a clear alert line."""
    text = report.render_summary(
        metrics=_metrics_with_trust(fp_trustworthy=False, error_count=3, total=10),
        state={"paused": False},
        active_experiment={},
    )
    assert "UNTRUSTWORTHY" in text or "untrustworthy" in text.lower()
    assert "error_rate" in text or "30%" in text or "3/10" in text


def test_summary_does_not_include_untrustworthy_alert_when_trustworthy():
    """When fp_trustworthy is True, no alert line must appear."""
    text = report.render_summary(
        metrics=_metrics_with_trust(fp_trustworthy=True, error_count=0, total=10),
        state={"paused": False},
        active_experiment={},
    )
    assert "UNTRUSTWORTHY" not in text
    # "untrustworthy" must not appear in the normal good-case output
    assert "untrustworthy" not in text.lower()


def test_summary_backward_compat_without_fp_trustworthy():
    """render_summary must not crash when fp_trustworthy is absent (old state)."""
    m = {
        "date": "2026-06-10",
        "total_triggers": 5,
        "labeled_triggers": 5,
        "fp_count": 1,
        "fp_rate": 0.2,
        "fp_ci": (0.05, 0.5),
        "fn_rate": "unmeasured",
        "fn_ci": None,
        # No fp_trustworthy, no error_rate, no error_count — old shape
    }
    text = report.render_summary(metrics=m, state={"paused": False}, active_experiment={})
    # Updated for Task 3: assert basic render works, not legacy FP line
    assert "images captured" in text


def test_report_main_no_send_does_not_require_telegram_credentials(tmp_path, monkeypatch):
    """--no-send must not import or instantiate Config with Telegram tokens."""
    import json
    import sys

    state_path = tmp_path / "state.json"
    lm = {
        "date": "2026-06-10",
        "total_triggers": 5,
        "labeled_triggers": 4,
        "fp_count": 1,
        "fp_rate": 0.25,
        "fp_ci": [0.05, 0.57],
        "fn_rate": "unmeasured",
        "fn_ci": None,
    }
    from loop import state as state_mod
    state_mod.save_state(state_path, {
        "paused": False,
        "active_experiment_id": None,
        "backlog": [],
        "last_metrics": lm,
    })

    # Remove Telegram env vars to prove they aren't needed.
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    monkeypatch.setattr(sys, "argv", [
        "loop.report", "--mode", "summary",
        "--state", str(state_path),
        "--no-send",
    ])

    import io
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        report.main()
    finally:
        output = sys.stdout.getvalue()
        sys.stdout = original_stdout

    parsed = json.loads(output.strip())
    assert parsed["sent"] is False
    # Updated for Task 3: plain-English format — "images captured" instead of legacy "FP"
    assert "images captured" in parsed["rendered"]


# ---------------------------------------------------------------------------
# latest_journal_entry: unit tests
# ---------------------------------------------------------------------------

def _write_journal(path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_latest_journal_entry_returns_last_bullet(tmp_path):
    """Returns the last top-level bullet, not the first."""
    j = tmp_path / "JOURNAL.md"
    _write_journal(j, """\
# Loop Journal

Some prose header.

- 2026-06-08 — First entry, single line.
- 2026-06-09 — Second entry
  with a continuation line.
- 2026-06-10 — Third entry
  spanning two
  continuation lines.
""")
    result = report.latest_journal_entry(j)
    assert result is not None
    assert "Third entry" in result
    assert "spanning two" in result
    # Must NOT include earlier entries.
    assert "First entry" not in result
    assert "Second entry" not in result


def test_latest_journal_entry_missing_file_returns_none(tmp_path):
    result = report.latest_journal_entry(tmp_path / "nonexistent.md")
    assert result is None


def test_latest_journal_entry_no_bullets_returns_none(tmp_path):
    """A header-only file with no top-level bullets returns None."""
    j = tmp_path / "JOURNAL.md"
    _write_journal(j, """\
# Loop Journal

Just some prose, no bullet entries yet.
""")
    result = report.latest_journal_entry(j)
    assert result is None


def test_latest_journal_entry_truncates_long_entry(tmp_path):
    """An entry exceeding _JOURNAL_TELEGRAM_LIMIT is truncated with a marker."""
    j = tmp_path / "JOURNAL.md"
    long_body = "x" * 5000
    _write_journal(j, f"- 2026-06-10 — {long_body}\n")
    result = report.latest_journal_entry(j)
    assert result is not None
    # The function itself does NOT truncate — truncation happens in main().
    # Verify raw result is long, then test main()'s truncation separately.
    assert len(result) > 4000


def test_latest_journal_entry_single_entry(tmp_path):
    """Works correctly when there is exactly one bullet entry."""
    j = tmp_path / "JOURNAL.md"
    _write_journal(j, """\
# Loop Journal

- 2026-06-10 — Only entry here.
""")
    result = report.latest_journal_entry(j)
    assert result is not None
    assert "Only entry here" in result


# ---------------------------------------------------------------------------
# main() summary + journal: --no-send includes journal_entry in JSON
# ---------------------------------------------------------------------------

def _make_state(tmp_path):
    """Helper: write a minimal state.json and return its path."""
    from loop import state as state_mod
    state_path = tmp_path / "state.json"
    lm = {
        "date": "2026-06-10",
        "total_triggers": 10,
        "labeled_triggers": 8,
        "fp_count": 3,
        "fp_rate": 0.375,
        "fp_ci": [0.1, 0.65],
        "fn_rate": "unmeasured",
        "fn_ci": None,
    }
    state_mod.save_state(state_path, {
        "paused": False,
        "active_experiment_id": None,
        "backlog": [],
        "last_metrics": lm,
    })
    return state_path


def _capture_main(monkeypatch, argv):
    """Run report.main() with the given argv, capturing stdout."""
    import io
    import sys
    monkeypatch.setattr(sys, "argv", argv)
    buf = io.StringIO()
    original = sys.stdout
    sys.stdout = buf
    try:
        report.main()
    finally:
        sys.stdout = original
    return buf.getvalue()


def test_no_send_summary_includes_journal_entry(tmp_path, monkeypatch):
    """--no-send JSON must include 'journal_entry' key with the latest bullet."""
    import json
    state_path = _make_state(tmp_path)
    journal_path = tmp_path / "JOURNAL.md"
    _write_journal(journal_path, """\
# Loop Journal

- 2026-06-09 — Older entry.
- 2026-06-10 — Latest entry for the report.
""")

    monkeypatch.setattr(report, "send", lambda t: (_ for _ in ()).throw(AssertionError("send() must not be called")))

    output = _capture_main(monkeypatch, [
        "loop.report", "--mode", "summary",
        "--state", str(state_path),
        "--journal", str(journal_path),
        "--no-send",
    ])
    parsed = json.loads(output.strip())
    assert parsed["sent"] is False
    assert "rendered" in parsed
    assert "journal_entry" in parsed
    assert parsed["journal_entry"] is not None
    assert "Latest entry for the report" in parsed["journal_entry"]
    assert "Older entry" not in parsed["journal_entry"]


def test_no_send_summary_journal_entry_null_when_missing(tmp_path, monkeypatch):
    """--no-send JSON has journal_entry=null when journal file doesn't exist."""
    import json
    state_path = _make_state(tmp_path)

    monkeypatch.setattr(report, "send", lambda t: (_ for _ in ()).throw(AssertionError("send() must not be called")))

    output = _capture_main(monkeypatch, [
        "loop.report", "--mode", "summary",
        "--state", str(state_path),
        "--journal", str(tmp_path / "no_such_journal.md"),
        "--no-send",
    ])
    parsed = json.loads(output.strip())
    assert parsed["journal_entry"] is None


def test_no_send_summary_truncates_long_journal_entry(tmp_path, monkeypatch):
    """--no-send truncates entries > _JOURNAL_TELEGRAM_LIMIT with a marker."""
    import json
    state_path = _make_state(tmp_path)
    journal_path = tmp_path / "JOURNAL.md"
    _write_journal(journal_path, "- 2026-06-10 — " + "y" * 5000 + "\n")

    monkeypatch.setattr(report, "send", lambda t: (_ for _ in ()).throw(AssertionError("send() must not be called")))

    output = _capture_main(monkeypatch, [
        "loop.report", "--mode", "summary",
        "--state", str(state_path),
        "--journal", str(journal_path),
        "--no-send",
    ])
    parsed = json.loads(output.strip())
    entry = parsed["journal_entry"]
    assert entry is not None
    assert entry.endswith("… (truncated)")
    assert len(entry) <= 4100  # well within Telegram limit


# ---------------------------------------------------------------------------
# main() summary real-send: two send() calls (metrics + journal)
# ---------------------------------------------------------------------------

def test_real_send_summary_calls_send_twice_with_journal(tmp_path, monkeypatch):
    """In real-send mode, send() is called once for metrics and once for journal."""
    import json
    state_path = _make_state(tmp_path)
    journal_path = tmp_path / "JOURNAL.md"
    _write_journal(journal_path, """\
# Loop Journal

- 2026-06-10 — Entry to send as second message.
""")

    send_calls = []

    async def fake_send(text):
        send_calls.append(text)
        return True

    monkeypatch.setattr(report, "send", fake_send)

    output = _capture_main(monkeypatch, [
        "loop.report", "--mode", "summary",
        "--state", str(state_path),
        "--journal", str(journal_path),
    ])
    parsed = json.loads(output.strip())
    assert parsed["sent"] is True
    assert parsed["journal_sent"] is True
    assert len(send_calls) == 2
    # First call: metrics summary (Updated for Task 3: "images captured" not "Wildlife loop")
    assert "images captured" in send_calls[0]
    # Second call: journal entry
    assert "Entry to send as second message" in send_calls[1]


def test_real_send_summary_journal_sent_null_when_no_entry(tmp_path, monkeypatch):
    """journal_sent is null (None) when there is no journal entry."""
    import json
    state_path = _make_state(tmp_path)

    send_calls = []

    async def fake_send(text):
        send_calls.append(text)
        return True

    monkeypatch.setattr(report, "send", fake_send)

    output = _capture_main(monkeypatch, [
        "loop.report", "--mode", "summary",
        "--state", str(state_path),
        "--journal", str(tmp_path / "missing.md"),
    ])
    parsed = json.loads(output.strip())
    assert parsed["sent"] is True
    assert parsed["journal_sent"] is None
    # Only the metrics message was sent
    assert len(send_calls) == 1


# ---------------------------------------------------------------------------
# Task 3: plain-English per-tier render_summary (8 TDD test cases)
# ---------------------------------------------------------------------------

def _tier_metrics(
    total=42,
    n_human=2, fp_human=1,
    n_claude=0, fp_claude=0,
    n_md=40, fp_md=38,
):
    """Metrics with per-tier keys (Tasks 1/2 shape)."""
    return {
        "date": "2026-06-27",
        "total_triggers": total,
        "n_human": n_human,
        "fp_human_count": fp_human,
        "n_claude": n_claude,
        "fp_claude_count": fp_claude,
        "n_md": n_md,
        "fp_md_count": fp_md,
        "fp_trustworthy": True,
    }


def test_summary_per_tier_lines():
    """Output contains total and per-tier lines with correct counts."""
    text = report.render_summary(
        metrics=_tier_metrics(), state={"paused": False}, active_experiment={},
    )
    assert "42 images captured" in text
    assert "You labelled 2" in text
    assert "1 false alarm" in text
    assert "Claude labelled 0" in text
    assert "MegaDetector" in text
    assert "38 false alarms" in text


def test_summary_md_line_marked_unverified():
    """MegaDetector line must contain both 'auto' and 'unverified'."""
    text = report.render_summary(
        metrics=_tier_metrics(), state={"paused": False}, active_experiment={},
    )
    md_lines = [line for line in text.splitlines() if "MegaDetector" in line]
    assert len(md_lines) == 1
    assert "auto" in md_lines[0]
    assert "unverified" in md_lines[0]


def test_summary_no_ci_or_jargon():
    """Output must not contain CI notation or aHash jargon."""
    text = report.render_summary(
        metrics=_tier_metrics(), state={"paused": False}, active_experiment={},
    )
    assert "CI" not in text
    assert "aHash" not in text


def test_summary_zero_tiers_still_listed():
    """A tier with n==0 still renders its line."""
    m = _tier_metrics(n_human=0, fp_human=0, n_claude=0, fp_claude=0)
    text = report.render_summary(metrics=m, state={"paused": False}, active_experiment={})
    assert "You labelled 0" in text
    assert "Claude labelled 0" in text


def test_summary_remainder_line_when_unlabelled():
    """'Not yet labelled' line present when remainder>0; absent when 0."""
    # Remainder = 42 - (2 + 0 + 38) = 2 → line present
    m = _tier_metrics(total=42, n_human=2, fp_human=0, n_claude=0, fp_claude=0, n_md=38, fp_md=0)
    text = report.render_summary(metrics=m, state={"paused": False}, active_experiment={})
    assert "Not yet labelled: 2" in text

    # Remainder = 42 - (2 + 0 + 40) = 0 → line absent
    m2 = _tier_metrics(total=42, n_human=2, fp_human=0, n_claude=0, fp_claude=0, n_md=40, fp_md=0)
    text2 = report.render_summary(metrics=m2, state={"paused": False}, active_experiment={})
    assert "Not yet labelled" not in text2


def test_summary_paused_banner_preserved():
    """Paused banner is still emitted in the new plain-English format."""
    text = report.render_summary(
        metrics=_tier_metrics(), state={"paused": True}, active_experiment={},
    )
    assert "PAUSED" in text
    assert "tuning frozen" in text.lower() or "paused" in text.lower()


def test_summary_untrustworthy_alert_preserved():
    """Untrustworthy alert is still emitted in the new plain-English format."""
    m = _tier_metrics()
    m["fp_trustworthy"] = False
    m["error_rate"] = 0.3
    m["error_count"] = 3
    text = report.render_summary(metrics=m, state={"paused": False}, active_experiment={})
    assert "UNTRUSTWORTHY" in text or "untrustworthy" in text.lower()
    assert "3/" in text or "30%" in text


def test_summary_backward_compat_missing_per_tier_keys():
    """Old metrics lacking n_*/fp_*_count keys render without crashing (.get defaults)."""
    m = {
        "date": "2026-06-10",
        "total_triggers": 5,
        "fp_trustworthy": True,
        # No n_human, n_claude, n_md, fp_*_count — old shape
    }
    text = report.render_summary(metrics=m, state={"paused": False}, active_experiment={})
    assert "5 images captured" in text
    assert "You labelled 0" in text
    assert "Claude labelled 0" in text
    assert "MegaDetector" in text
