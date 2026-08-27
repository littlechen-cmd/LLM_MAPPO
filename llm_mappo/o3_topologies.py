"""Evaluation-only O3 topology resources and construction boundary."""

from dataclasses import dataclass
from hashlib import sha256
from importlib import resources
from types import MappingProxyType
from typing import Mapping

from gymnasium.envs.registration import register, registry

from llm_mappo.o3_guard import O3_ENVIRONMENT_IDS
from llm_mappo.optimization_observation import ObservationSchema
from llm_mappo.phase2 import Phase2Warehouse
from rware.warehouse import RewardType


@dataclass(frozen=True)
class TopologySpec:
    """Immutable provenance required to construct one held-out topology."""

    environment_id: str
    version: str
    resource_name: str
    usage: str
    evaluated_n_agents: int
    charging_stations: tuple[tuple[int, int], ...]
    source_sha256: str
    effective_layout_hash: str


_NARROW_ID, _CENTRAL_ID = O3_ENVIRONMENT_IDS

O3_TOPOLOGIES: Mapping[str, TopologySpec] = MappingProxyType(
    {
        _NARROW_ID: TopologySpec(
            environment_id=_NARROW_ID,
            version="v2",
            resource_name="layouts/o3/unseen_narrow_passage_v2.txt",
            usage="evaluation_only",
            evaluated_n_agents=5,
            charging_stations=(
                (19, 2),
                (16, 5),
                (6, 2),
                (0, 5),
                (19, 11),
                (16, 18),
                (6, 18),
                (0, 21),
            ),
            source_sha256=(
                "0120b104d61bb964baee39e21fcc95c2422ee67894b9c9b9a5c5de60edaff985"
            ),
            effective_layout_hash=(
                "978751b66589003b10e493e1ba31590f732a4e2cd21b4896548d90d2e81d0132"
            ),
        ),
        _CENTRAL_ID: TopologySpec(
            environment_id=_CENTRAL_ID,
            version="v2",
            resource_name="layouts/o3/unseen_central_cross_v2.txt",
            usage="evaluation_only",
            evaluated_n_agents=5,
            charging_stations=(
                (19, 0),
                (19, 23),
                (0, 0),
                (0, 23),
                (19, 4),
                (19, 19),
                (0, 4),
                (0, 19),
            ),
            source_sha256=(
                "4fc92da618def49e218abbcfaa46c118a65b4830d547dabe81d5b4792b333a14"
            ),
            effective_layout_hash=(
                "0f9b25f1ddfe42549d97b7426d5d60d0b73ba47833ab9d8d4f7c000a9f81ce8c"
            ),
        ),
    }
)


def get_o3_topology(environment_id: str) -> TopologySpec:
    """Return one explicit held-out spec without fallback or fuzzy matching."""
    try:
        return O3_TOPOLOGIES[environment_id]
    except KeyError as error:
        raise ValueError(f"Unknown O3 topology: {environment_id}") from error


def _read_package_resource(resource_name: str) -> bytes:
    node = resources.files("rware")
    for part in resource_name.split("/"):
        node = node.joinpath(part)
    return node.read_bytes()


def read_o3_layout_bytes(spec: TopologySpec) -> bytes:
    """Read and verify immutable source bytes before decoding the layout."""
    payload = _read_package_resource(spec.resource_name)
    actual_hash = sha256(payload).hexdigest()
    if actual_hash != spec.source_sha256:
        raise ValueError(
            f"O3 source SHA-256 mismatch for {spec.environment_id}: "
            f"expected {spec.source_sha256}, got {actual_hash}."
        )
    if payload.startswith(b"\xef\xbb\xbf") or b"\r" in payload:
        raise ValueError("O3 source must be UTF-8 without BOM and use LF endings.")
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ValueError("O3 source must end with exactly one newline.")
    return payload


def _registration_kwargs(spec: TopologySpec, layout: str) -> dict:
    return {
        "layout": layout,
        "shelf_columns": 1,
        "column_height": 1,
        "shelf_rows": 1,
        "n_agents": spec.evaluated_n_agents,
        "msg_bits": 0,
        "sensor_range": 4,
        "request_queue_size": 8,
        "max_inactivity_steps": None,
        "max_steps": 1000,
        "reward_type": RewardType.INDIVIDUAL,
        "batch_interval": 40,
        "batch_size_range": (4, 8),
        "picking_lock_steps": 3,
        "task_completion_target": 50,
        "battery_cost_scale": 1.10,
        "charging_stations": spec.charging_stations,
    }


def make_o3_evaluation_environment(
    environment_id: str,
    *,
    observation_schema: ObservationSchema = ObservationSchema.DIRECT_GOAL_V1,
) -> Phase2Warehouse:
    """Construct one verified O3 adapter without persistent Gym registration."""
    spec = get_o3_topology(environment_id)
    if spec.usage != "evaluation_only":
        raise ValueError("O3 topology usage must be evaluation_only.")
    if environment_id in registry:
        raise RuntimeError("O3 environment ID is already registered globally.")
    layout = read_o3_layout_bytes(spec).decode("utf-8")
    register(
        id=environment_id,
        entry_point="llm_mappo.environment:DynamicWarehouse",
        kwargs=_registration_kwargs(spec, layout),
    )
    environment = None
    try:
        environment = Phase2Warehouse(
            n_agents=spec.evaluated_n_agents,
            max_steps=1000,
            env_id=environment_id,
            charge_threshold=0.30,
            charge_release_threshold=0.80,
            battery_cost_scale=1.10,
            deadlock_steps=180,
            batch_interval=40,
            batch_size_range=(4, 8),
            request_queue_size=8,
            task_completion_target=50,
            observation_schema=observation_schema,
        )
        actual_hash = environment.env.shadow_layout_hash()
        if actual_hash != spec.effective_layout_hash:
            raise ValueError(
                f"O3 effective layout hash mismatch for {environment_id}: "
                f"expected {spec.effective_layout_hash}, got {actual_hash}."
            )
        return environment
    except Exception:
        if environment is not None:
            environment.close()
        raise
    finally:
        registry.pop(environment_id, None)
