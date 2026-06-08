"""Tests for src/loop/deploy.py — validate, write, render env, rollback."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from loop import deploy
from loop import state as state_mod


@pytest.fixture
def paths(tmp_path):
    return {
        "state": tmp_path / "state.json",
        "env": tmp_path / "deployed_config.env",
    }


def _seed_state(paths, deployed=None):
    deployed = deployed or {"MOTION_THRESHOLD": 2000}
    state_mod.save_state(paths["state"], {
        "watermark": 0, "paused": False,
        "deployed": deployed, "best_known_good": deployed,
        "history": [], "pending_restart_at": None,
        "active_experiment_id": 1, "backlog": [], "baselines": {},
        "last_metrics": None,
    })


def test_out_of_bounds_rejected_no_write(paths):
    _seed_state(paths)
    before = paths["state"].read_text()
    with pytest.raises(ValueError, match="out of bounds"):
        deploy.deploy(
            {"MOTION_THRESHOLD": 999999},
            state_path=paths["state"], env_path=paths["env"],
            restart_at="2026-06-11T04:30:00+02:00",
        )
    assert paths["state"].read_text() == before  # unchanged
    assert not paths["env"].exists()             # no env rendered


def test_valid_deploy_renders_env_and_updates_state(paths):
    _seed_state(paths)
    result = deploy.deploy(
        {"MOTION_THRESHOLD": 2500},
        state_path=paths["state"], env_path=paths["env"],
        restart_at="2026-06-11T04:30:00+02:00",
    )
    assert result["deployed"]["MOTION_THRESHOLD"] == 2500
    env_text = paths["env"].read_text()
    assert "MOTION_THRESHOLD=2500" in env_text
    st = state_mod.load_state(paths["state"])
    assert st["deployed"]["MOTION_THRESHOLD"] == 2500
    assert st["pending_restart_at"] == "2026-06-11T04:30:00+02:00"


def test_deploy_pushes_previous_onto_history(paths):
    _seed_state(paths, deployed={"MOTION_THRESHOLD": 2000})
    deploy.deploy(
        {"MOTION_THRESHOLD": 2500},
        state_path=paths["state"], env_path=paths["env"],
        restart_at="2026-06-11T04:30:00+02:00",
    )
    st = state_mod.load_state(paths["state"])
    assert any(h["config"].get("MOTION_THRESHOLD") == 2000 for h in st["history"])


def test_best_known_good_preserved_across_deploy(paths):
    _seed_state(paths, deployed={"MOTION_THRESHOLD": 2000})
    deploy.deploy(
        {"MOTION_THRESHOLD": 2500},
        state_path=paths["state"], env_path=paths["env"],
        restart_at="2026-06-11T04:30:00+02:00",
    )
    st = state_mod.load_state(paths["state"])
    assert st["best_known_good"]["MOTION_THRESHOLD"] == 2000  # untouched by deploy


def test_rollback_restores_best_known_good(paths):
    _seed_state(paths, deployed={"MOTION_THRESHOLD": 2000})
    deploy.deploy(
        {"MOTION_THRESHOLD": 2500},
        state_path=paths["state"], env_path=paths["env"],
        restart_at="2026-06-11T04:30:00+02:00",
    )
    result = deploy.rollback(
        state_path=paths["state"], env_path=paths["env"],
        restart_at="2026-06-11T05:00:00+02:00",
    )
    assert result["deployed"]["MOTION_THRESHOLD"] == 2000
    assert "MOTION_THRESHOLD=2000" in paths["env"].read_text()
    st = state_mod.load_state(paths["state"])
    assert st["deployed"]["MOTION_THRESHOLD"] == 2000


def test_species_unknown_threshold_key_end_to_end(paths, monkeypatch):
    """End-to-end: deploy renders SPECIES_UNKNOWN_SPECIES_THRESHOLD in env-file,
    and SpeciesConfig reads it back correctly via the overlay mechanism.

    This proves the BOUNDS key, env-var name, and config field are all
    consistent — the latent bug was SPECIES_UNKNOWN_THRESHOLD (wrong) vs
    SPECIES_UNKNOWN_SPECIES_THRESHOLD (correct pydantic env-var name).
    """
    _seed_state(paths, deployed={})
    result = deploy.deploy(
        {"SPECIES_UNKNOWN_SPECIES_THRESHOLD": 0.75},
        state_path=paths["state"], env_path=paths["env"],
        restart_at="2026-06-11T04:30:00+02:00",
    )
    # 1. The deployed dict carries the correct key.
    assert result["deployed"]["SPECIES_UNKNOWN_SPECIES_THRESHOLD"] == 0.75

    # 2. The rendered env file contains the correct env-var line.
    env_text = paths["env"].read_text()
    assert "SPECIES_UNKNOWN_SPECIES_THRESHOLD=0.75" in env_text

    # 3. SpeciesConfig built with that overlay reads back 0.75 — the overlay
    #    flows through to the pydantic field (end-to-end key correctness).
    monkeypatch.delenv("SPECIES_UNKNOWN_SPECIES_THRESHOLD", raising=False)
    from config import SpeciesConfig
    cfg = SpeciesConfig(_env_file=(str(paths["env"]), str(paths["env"])))
    assert cfg.unknown_species_threshold == 0.75
