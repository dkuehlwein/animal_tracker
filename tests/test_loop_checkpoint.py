"""Tests for loop.checkpoint — deterministic git commit of experiments/ only.

Tests verify that checkpoint:
1. Only stages experiments/ (not the full repo).
2. Commits with the given message.
3. Does NOT push.
4. Is safe when there is nothing to commit.
"""

import sys
import subprocess
from pathlib import Path
from unittest.mock import patch, call

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loop import checkpoint


def test_checkpoint_stages_only_experiments(monkeypatch):
    """checkpoint.run() must call `git add experiments/` and not `git add -A`."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        # Simulate success
        import subprocess
        result = subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)

    checkpoint.run(message="test: checkpoint", cwd="/fake/repo")

    # First call must be `git add experiments/`
    assert calls[0] == ["git", "add", "experiments/"], (
        f"Expected git add experiments/, got: {calls[0]}"
    )
    # Must not include '-A' or '.' in the add call
    assert "-A" not in calls[0]
    assert "." not in calls[0]


def test_checkpoint_commits_with_given_message(monkeypatch):
    """checkpoint.run() must call `git commit -m <message>`."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        import subprocess
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    checkpoint.run(message="feat: nightly labels", cwd="/fake/repo")

    commit_calls = [c for c in calls if c[:2] == ["git", "commit"]]
    assert len(commit_calls) == 1
    commit_cmd = commit_calls[0]
    assert "-m" in commit_cmd
    msg_idx = commit_cmd.index("-m")
    assert commit_cmd[msg_idx + 1] == "feat: nightly labels"


def test_checkpoint_does_not_push(monkeypatch):
    """checkpoint.run() must never call `git push`."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        import subprocess
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    checkpoint.run(message="test: no push", cwd="/fake/repo")

    push_calls = [c for c in calls if "push" in c]
    assert push_calls == [], f"checkpoint must NOT push, got: {push_calls}"


def test_checkpoint_nothing_to_commit_exits_cleanly(monkeypatch):
    """If git commit returns exit code 1 (nothing to commit), run() handles it gracefully."""
    call_idx = [0]

    def fake_run(cmd, **kwargs):
        import subprocess
        if cmd[:2] == ["git", "commit"]:
            # git commit exits 1 when nothing to commit
            return subprocess.CompletedProcess(cmd, 1, stdout="nothing to commit", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Must not raise
    checkpoint.run(message="empty", cwd="/fake/repo")


def test_checkpoint_main_uses_message_arg(monkeypatch):
    """main() CLI path passes --message to run()."""
    import sys
    run_calls = []

    def fake_run_fn(message, cwd=None):
        run_calls.append(message)

    monkeypatch.setattr(checkpoint, "run", fake_run_fn)

    saved = sys.argv
    try:
        sys.argv = ["loop.checkpoint", "--message", "tick: adjudication done"]
        checkpoint.main()
    except SystemExit as e:
        assert e.code == 0 or e.code is None
    finally:
        sys.argv = saved

    assert run_calls == ["tick: adjudication done"]
