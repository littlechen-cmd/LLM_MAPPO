"""O3-C evaluation-only registry, package resource, and dual-hash contracts."""

from dataclasses import replace
from hashlib import sha256

import gymnasium as gym
import pytest

import llm_mappo.o3_topologies as o3
from llm_mappo.optimization_observation import ObservationSchema


NARROW_ID = "llm-mappo-o3-unseen-narrow-passage-5ag-v2"
CENTRAL_ID = "llm-mappo-o3-unseen-central-cross-5ag-v2"


def test_o3_import_does_not_globally_register_evaluation_only_environments():
    """Catch package import making held-out maps available to ordinary gym.make."""
    assert o3.O3_ENVIRONMENT_IDS == (NARROW_ID, CENTRAL_ID)
    for environment_id in o3.O3_ENVIRONMENT_IDS:
        assert environment_id not in gym.envs.registry
        with pytest.raises(gym.error.Error):
            gym.spec(environment_id)


def test_o3_specs_load_immutable_package_bytes_with_frozen_source_hashes():
    """Catch editable/install resource drift or writable registry metadata."""
    expected = {
        NARROW_ID: (
            "unseen_narrow_passage_v2.txt",
            "0120b104d61bb964baee39e21fcc95c2422ee67894b9c9b9a5c5de60edaff985",
            "978751b66589003b10e493e1ba31590f732a4e2cd21b4896548d90d2e81d0132",
        ),
        CENTRAL_ID: (
            "unseen_central_cross_v2.txt",
            "4fc92da618def49e218abbcfaa46c118a65b4830d547dabe81d5b4792b333a14",
            "0f9b25f1ddfe42549d97b7426d5d60d0b73ba47833ab9d8d4f7c000a9f81ce8c",
        ),
    }
    for environment_id, values in expected.items():
        filename, source_hash, layout_hash = values
        spec = o3.get_o3_topology(environment_id)
        payload = o3.read_o3_layout_bytes(spec)
        assert spec.version == "v2"
        assert spec.usage == "evaluation_only"
        assert spec.resource_name == f"layouts/o3/{filename}"
        assert spec.source_sha256 == source_hash == sha256(payload).hexdigest()
        assert spec.effective_layout_hash == layout_hash
        assert len(spec.charging_stations) == 8
    with pytest.raises(TypeError):
        o3.O3_TOPOLOGIES[NARROW_ID] = o3.O3_TOPOLOGIES[NARROW_ID]


def test_evaluation_factory_verifies_both_hashes_and_leaves_no_registration():
    """Catch persistent Gym registration or construction without effective-hash proof."""
    for environment_id in o3.O3_ENVIRONMENT_IDS:
        environment = o3.make_o3_evaluation_environment(
            environment_id,
            observation_schema=ObservationSchema.DIRECT_GOAL_V1,
        )
        try:
            spec = o3.get_o3_topology(environment_id)
            assert environment.env.shadow_layout_hash() == spec.effective_layout_hash
            assert environment.env.grid_size == (24, 20)
            assert environment.n_agents == 5
            assert environment.max_steps == 1000
            assert environment.battery_cost_scale == 1.10
            assert environment.charge_threshold == 0.30
            assert environment.charge_release_threshold == 0.80
        finally:
            environment.close()
        assert environment_id not in gym.envs.registry


def test_evaluation_factory_fails_closed_on_source_byte_drift(monkeypatch):
    """Catch a modified map being silently accepted under an old source hash."""
    original_reader = o3._read_package_resource

    def corrupted_reader(resource_name):
        return original_reader(resource_name).replace(b".", b"X", 1)

    monkeypatch.setattr(o3, "_read_package_resource", corrupted_reader)
    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        o3.make_o3_evaluation_environment(NARROW_ID)
    assert NARROW_ID not in gym.envs.registry


def test_evaluation_factory_fails_closed_on_station_or_expected_hash_drift(
    monkeypatch,
):
    """Catch semantic-layout drift even when the source map bytes still match."""
    original = o3.get_o3_topology(NARROW_ID)
    changed = replace(
        original,
        charging_stations=((18, 0),) + original.charging_stations[1:],
    )
    altered = dict(o3.O3_TOPOLOGIES)
    altered[NARROW_ID] = changed
    monkeypatch.setattr(o3, "O3_TOPOLOGIES", altered)
    with pytest.raises(ValueError, match="effective layout hash mismatch"):
        o3.make_o3_evaluation_environment(NARROW_ID)
    assert NARROW_ID not in gym.envs.registry


def test_evaluation_factory_rejects_unknown_topology():
    """Catch callers silently falling back from an unknown held-out environment."""
    with pytest.raises(ValueError, match="Unknown O3 topology"):
        o3.make_o3_evaluation_environment("unknown-o3-layout")
