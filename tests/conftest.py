"""Global pytest configuration enforcing test isolation from production data.

Two production-data hazards exist and are neutralised here for every test:

1. **Storage redirection.** ``StorageConfig`` defaults ``data_dir=data`` and
   ``database_path=data/detections.db``. A test that builds a config and then
   writes/unlinks would operate on the REAL production directory (this is what
   deleted ``data/detections.db``). We point ``STORAGE_DATA_DIR`` and
   ``STORAGE_DATABASE_PATH`` at a session-scoped temp directory via the env-var
   overrides ``StorageConfig`` already supports, so every storage path resolves
   under tmp and the repo's ``data/`` is never touched. These vars are
   re-asserted before *every* test so they cannot be lost — e.g.
   ``tests/test_burst_cleanup.py`` pops ``STORAGE_DATA_DIR`` in a teardown
   fixture, and ``tests/test_wildlife_system.py`` reloads the ``config`` module
   (a fresh ``StorageConfig`` then re-reads the env at import time).

2. **.env leakage.** Every config class hard-codes ``env_file='.env'`` (relative
   to the cwd). The production ``.env`` sets operational overrides such as
   ``MOTION_THRESHOLD=800`` that would leak into tests and break assertions
   expecting documented defaults. We disable ``.env`` loading on all config
   classes (and re-apply the patch to the live ``config`` module before each
   test, in case it was reloaded).

The combination guarantees: after the full suite runs, the repo's
``data/detections.db`` and ``data/images/`` are never created, modified, or
deleted by tests.
"""

import os
import sys
from pathlib import Path

import pytest

# Ensure ``src`` is importable regardless of the cwd pytest is invoked from.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Session-wide temp directory that all test storage is redirected to. Created
# eagerly (not via a fixture) so the value is stable across module reloads.
import tempfile  # noqa: E402

_TMP_DATA = Path(tempfile.mkdtemp(prefix="wildlife_test_data_"))

_STORAGE_OVERRIDES = {
    "STORAGE_DATA_DIR": str(_TMP_DATA),
    "STORAGE_DATABASE_PATH": str(_TMP_DATA / "detections.db"),
    "TELEGRAM_BOT_TOKEN": "test_token",
    "TELEGRAM_CHAT_ID": "test_chat",
    "MOTION_WARMUP_SECONDS": "0",
}


def _disable_dotenv_and_patch_storage():
    """Disable .env loading and harden StorageConfig on the live config module.

    Safe to call repeatedly; targets whatever ``config`` module object is
    currently imported (it may have been reloaded by a test)."""
    config_module = sys.modules.get("config")
    if config_module is None:
        import config as config_module  # noqa: F811

    for obj in vars(config_module).values():
        if (
            isinstance(obj, type)
            and isinstance(getattr(obj, "model_config", None), dict)
            and "env_file" in obj.model_config
        ):
            obj.model_config["env_file"] = None

    StorageConfig = config_module.StorageConfig
    if not getattr(StorageConfig.ensure_directories, "_isolation_patched", False):
        original_ensure = StorageConfig.ensure_directories

        def _safe_ensure_directories(self, _orig=original_ensure):
            # Defence in depth: if STORAGE_DATA_DIR was cleared (e.g. a
            # ``@patch.dict(..., clear=True)`` block), force the instance back
            # onto the temp dir before any directory is created.
            if os.environ.get("STORAGE_DATA_DIR") is None:
                object.__setattr__(self, "data_dir", _TMP_DATA)
                object.__setattr__(
                    self, "database_path", str(_TMP_DATA / "detections.db")
                )
            return _orig(self)

        _safe_ensure_directories._isolation_patched = True
        StorageConfig.ensure_directories = _safe_ensure_directories


# Apply once at import time so even collection-time config construction is safe.
for _k, _v in _STORAGE_OVERRIDES.items():
    os.environ.setdefault(_k, _v)
_disable_dotenv_and_patch_storage()


@pytest.fixture(autouse=True)
def _isolate_config_from_production():
    """Re-assert storage redirection and .env isolation before every test.

    Function-scoped so it survives tests that pop the override env vars or
    reload the ``config`` module mid-suite.
    """
    for key, value in _STORAGE_OVERRIDES.items():
        os.environ[key] = value
    _disable_dotenv_and_patch_storage()
    yield
