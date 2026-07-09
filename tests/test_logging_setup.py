"""
Tests for wildlife_system.configure_logging.

Context: journald history was lost on a Pi reboot, taking forensic log lines
with it. configure_logging installs a rotating file handler (in addition to
the existing stream handler that keeps journald working) so INFO+ logs
survive reboots on disk.
"""
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append('src')


def _make_config(log_dir):
    """Minimal stand-in for Config exposing only what configure_logging reads."""
    return SimpleNamespace(storage=SimpleNamespace(log_dir=Path(log_dir)))


@pytest.fixture(autouse=True)
def _restore_root_logger():
    """configure_logging mutates the root logger (adds handlers, sets level).
    Snapshot and restore it around each test so we don't leak handlers into
    other tests or clobber caplog's own handler."""
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    yield
    for handler in root.handlers[:]:
        if handler not in original_handlers:
            root.removeHandler(handler)
            handler.close()
    root.handlers[:] = original_handlers
    root.setLevel(original_level)


def test_configure_logging_file_has_info_not_debug_and_keeps_stream_handler(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    from wildlife_system import configure_logging

    log_dir = tmp_path / "logs"
    config = _make_config(log_dir)

    configure_logging(config)

    root = logging.getLogger()
    # Stream handler (non-file) must still be present so journald keeps working.
    assert any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    )

    module_logger = logging.getLogger("some.test.module")
    module_logger.info("info line marker")
    module_logger.debug("debug line marker")

    for handler in root.handlers:
        handler.flush()

    log_file = log_dir / "wildlife.log"
    assert log_file.exists()
    contents = log_file.read_text()
    assert "info line marker" in contents
    assert "debug line marker" not in contents


def test_configure_logging_creates_missing_log_dir(tmp_path):
    from wildlife_system import configure_logging

    log_dir = tmp_path / "does" / "not" / "exist" / "yet"
    assert not log_dir.exists()

    configure_logging(_make_config(log_dir))

    assert log_dir.is_dir()
    assert (log_dir / "wildlife.log").exists()


def test_configure_logging_never_raises_when_log_dir_unwritable(tmp_path):
    """If the log directory can't be created/written, configure_logging must
    log a warning via the stream handler and continue without a file handler
    rather than raising (pipeline must never crash on a logging setup issue)."""
    from wildlife_system import configure_logging

    # A file where a directory is expected makes mkdir(parents=True) fail.
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    log_dir = blocked / "logs"

    configure_logging(_make_config(log_dir))  # must not raise

    root = logging.getLogger()
    assert any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    )
    assert not any(isinstance(h, logging.FileHandler) for h in root.handlers)
