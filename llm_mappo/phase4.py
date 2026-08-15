"""Phase 4 offline semantic-label collection and retrieval utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from llm_mappo.llm_teacher import (
    LabelledScenario,
    TeacherProvider,
    append_labelled_scenario,
    build_engagement_scenarios,
    label_scenarios,
    load_labelled_scenarios,
    write_labelled_scenarios,
)
from llm_mappo.phase2 import Phase2Warehouse
from llm_mappo.phase2_expert import AStarExpert
from llm_mappo.types import PriorityAdjustment
from rware.warehouse import Direction


@dataclass(frozen=True)
class OfflineSemanticTeacher:
    """Nearest-neighbour lookup over a fixed dual-semantic LLM label cache."""

    observations: np.ndarray
    preferences: np.ndarray
    model_names: tuple[str, ...]
    squared_norms: np.ndarray

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "OfflineSemanticTeacher":
        records = load_labelled_scenarios(path)
        observations = np.asarray(
            [record.scenario.observation for record in records], dtype=np.float32
        )
        preferences = np.asarray(
            [
                (
                    record.label.task_commitment,
                    record.label.local_assertiveness,
                )
                for record in records
            ],
            dtype=np.float32,
        )
        return cls(
            observations=observations,
            preferences=preferences,
            model_names=tuple(sorted({record.label.model for record in records})),
            squared_norms=np.sum(observations * observations, axis=1),
        )

    @property
    def observation_dim(self) -> int:
        return int(self.observations.shape[1])

    @property
    def size(self) -> int:
        return int(self.observations.shape[0])

    def targets(self, observations: np.ndarray, neighbours: int = 3) -> np.ndarray:
        """Return inverse-distance weighted labels without calling an LLM."""
        values = np.asarray(observations, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.observation_dim:
            raise ValueError("Observation shape does not match the offline teacher.")
        if neighbours < 1:
            raise ValueError("neighbours must be positive.")
        count = min(neighbours, self.size)
        values_norm = np.sum(values * values, axis=1, keepdims=True)
        distances = values_norm + self.squared_norms[None, :]
        distances -= 2.0 * values @ self.observations.T
        distances = np.maximum(distances, 0.0) / self.observation_dim
        indices = np.argpartition(distances, count - 1, axis=1)[:, :count]
        nearest_distances = np.take_along_axis(distances, indices, axis=1)
        nearest_preferences = self.preferences[indices]
        weights = 1.0 / np.maximum(nearest_distances, 1e-8)
        weighted = nearest_preferences * weights[:, :, None]
        return np.sum(weighted, axis=1) / np.sum(weights, axis=1, keepdims=True)


# Keep imports from older entry points working while the Phase 4 schema is v2.
OfflineEngagementTeacher = OfflineSemanticTeacher


def collect_offline_labels(
    env: Phase2Warehouse,
    provider: TeacherProvider,
    output_path: str | Path,
    seeds: Iterable[int],
    scenarios_per_seed: int = 40,
    checkpoint_path: str | Path | None = None,
) -> dict:
    """Collect a deterministic, pre-training label cache from A* rollouts."""
    if scenarios_per_seed < 1:
        raise ValueError("scenarios_per_seed must be positive.")
    expert = AStarExpert()
    checkpoint = Path(checkpoint_path or f"{output_path}.partial.jsonl")
    records = _load_checkpoint(checkpoint)
    known = {record.scenario.scenario_id for record in records}
    for seed in seeds:
        env.reset(seed=seed)
        collected = sum(
            record.scenario.scenario_type == "normal_transport" for record in records
        )
        while collected < scenarios_per_seed:
            scenarios = build_engagement_scenarios(env)
            selected = scenarios[: min(len(scenarios), scenarios_per_seed - collected)]
            _label_and_checkpoint(selected, provider, records, known, checkpoint)
            collected += len(selected)
            actions, _ = expert.act(env, env.action_masks())
            transition = env.step(actions)
            if (
                transition.terminated
                or transition.truncated
                or transition.metrics.deadlocked
            ):
                break
    count = write_labelled_scenarios(output_path, records)
    checkpoint.unlink(missing_ok=True)
    return {
        "dataset": str(output_path),
        "records": count,
        "provider": provider.name,
        "observation_dim": env.actor_observation_dim,
    }


SCENARIO_TYPES = (
    "normal_transport",
    "priority_conflict",
    "narrow_corridor_yield",
    "low_battery_diversion",
    "station_exit_congestion",
)


def collect_stratified_offline_labels(
    env: Phase2Warehouse,
    provider: TeacherProvider,
    output_path: str | Path,
    seeds: Iterable[int],
    quotas: Mapping[str, int],
    checkpoint_path: str | Path | None = None,
) -> dict:
    """Collect an auditable mix of natural and controlled semantic scenarios."""
    quotas = {name: int(count) for name, count in quotas.items()}
    if set(quotas) != set(SCENARIO_TYPES) or any(count < 1 for count in quotas.values()):
        raise ValueError("Provide a positive quota for every Phase 4 scenario type.")
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("Provide at least one deterministic collection seed.")
    expert = AStarExpert()
    checkpoint = Path(checkpoint_path or f"{output_path}.partial.jsonl")
    records = _load_checkpoint(checkpoint)
    known = {record.scenario.scenario_id for record in records}
    counts = {
        name: sum(record.scenario.scenario_type == name for record in records)
        for name in SCENARIO_TYPES
    }
    normal_seed_index = 0
    observations = env.reset(seed=seed_list[normal_seed_index])
    while counts["normal_transport"] < quotas["normal_transport"]:
        scenarios = build_engagement_scenarios(env, "normal_transport")
        remaining = quotas["normal_transport"] - counts["normal_transport"]
        selected = scenarios[:remaining]
        _label_and_checkpoint(selected, provider, records, known, checkpoint)
        counts["normal_transport"] = sum(
            record.scenario.scenario_type == "normal_transport" for record in records
        )
        actions, _ = expert.act(env, env.action_masks())
        transition = env.step(actions)
        observations = transition.observations
        if (
            transition.terminated
            or transition.truncated
            or transition.metrics.deadlocked
        ):
            normal_seed_index = (normal_seed_index + 1) % len(seed_list)
            observations = env.reset(seed=seed_list[normal_seed_index])
    del observations

    for scenario_type in SCENARIO_TYPES[1:]:
        index = 0
        while counts[scenario_type] < quotas[scenario_type]:
            env.reset(seed=seed_list[index % len(seed_list)] * 10_000 + index)
            _inject_controlled_scenario(env, scenario_type)
            focal = build_engagement_scenarios(env, scenario_type)[0]
            _label_and_checkpoint((focal,), provider, records, known, checkpoint)
            counts[scenario_type] = sum(
                record.scenario.scenario_type == scenario_type for record in records
            )
            index += 1

    count = write_labelled_scenarios(output_path, records)
    checkpoint.unlink(missing_ok=True)
    return {
        "dataset": str(output_path),
        "records": count,
        "provider": provider.name,
        "observation_dim": env.actor_observation_dim,
        "scenario_counts": counts,
    }


def _label_and_checkpoint(
    scenarios: Iterable,
    provider: TeacherProvider,
    records: list[LabelledScenario],
    known: set[str],
    checkpoint: Path,
) -> None:
    """Label only unseen scenarios and persist each result before continuing."""
    for scenario in scenarios:
        if scenario.scenario_id in known:
            continue
        labelled = label_scenarios((scenario,), provider)[0]
        append_labelled_scenario(checkpoint, labelled)
        records.append(labelled)
        known.add(scenario.scenario_id)


def _load_checkpoint(path: Path) -> list[LabelledScenario]:
    """Load a partial JSONL file, tolerating an interrupted final line."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    valid: list[str] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError(
                    f"Invalid non-final checkpoint line: {path}:{index + 1}"
                )
            break
        valid.append(line)
    if len(valid) != sum(bool(line.strip()) for line in lines):
        path.write_text("\n".join(valid) + ("\n" if valid else ""), encoding="utf-8")
    if not valid:
        return []
    return load_labelled_scenarios(path)


def _repair_source_index(source: Path, requested: tuple[str, ...]):
    records = load_labelled_scenarios(source)
    indexed = {record.scenario.scenario_id: record for record in records}
    if len(indexed) != len(records):
        raise ValueError("Source dataset contains duplicate scenario IDs.")
    unknown = sorted(set(requested) - set(indexed))
    if unknown:
        raise ValueError(f"Repair scenario IDs are absent from the source: {unknown}")
    return records, indexed


def _repair_checkpoint_map(
    checkpoint: Path,
    requested: tuple[str, ...],
    indexed: dict[str, LabelledScenario],
) -> dict[str, LabelledScenario]:
    replacements = _load_checkpoint(checkpoint)
    replacement_map = {
        record.scenario.scenario_id: record for record in replacements
    }
    if len(replacement_map) != len(replacements):
        raise ValueError("Repair checkpoint contains duplicate scenario IDs.")
    unexpected = sorted(set(replacement_map) - set(requested))
    if unexpected:
        raise ValueError(f"Repair checkpoint contains unexpected IDs: {unexpected}")
    for scenario_id, replacement in replacement_map.items():
        if replacement.scenario != indexed[scenario_id].scenario:
            raise ValueError(
                f"Repair checkpoint scenario does not match source: {scenario_id}"
            )
    return replacement_map


def repair_offline_labels(
    source_path: str | Path,
    output_path: str | Path,
    provider: TeacherProvider,
    scenario_ids: Iterable[str],
    checkpoint_path: str | Path | None = None,
) -> dict:
    """Re-label selected records without mutating the frozen source dataset."""
    source = Path(source_path)
    destination = Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("Repair output must not overwrite the source dataset.")
    if destination.exists():
        raise FileExistsError(f"Repair output already exists: {destination}")
    requested = tuple(
        str(item).strip() for item in scenario_ids if str(item).strip()
    )
    if not requested:
        raise ValueError("Provide at least one scenario ID to re-label.")
    if len(set(requested)) != len(requested):
        raise ValueError("Repair scenario IDs must be unique.")

    records, indexed = _repair_source_index(source, requested)
    checkpoint = Path(
        checkpoint_path or destination.with_suffix(destination.suffix + ".partial")
    )
    replacement_map = _repair_checkpoint_map(checkpoint, requested, indexed)

    for scenario_id in requested:
        if scenario_id in replacement_map:
            continue
        replacement = label_scenarios((indexed[scenario_id].scenario,), provider)[0]
        append_labelled_scenario(checkpoint, replacement)
        replacement_map[scenario_id] = replacement

    repaired = [
        replacement_map.get(record.scenario.scenario_id, record) for record in records
    ]
    count = write_labelled_scenarios(destination, repaired)
    checkpoint.unlink(missing_ok=True)
    return {
        "source": str(source),
        "dataset": str(destination),
        "records": count,
        "relabelled_records": len(requested),
        "provider": provider.name,
    }


def _inject_controlled_scenario(env: Phase2Warehouse, scenario_type: str) -> None:
    """Build a deterministic label-only state without stepping the environment."""
    warehouse = env.env
    if scenario_type not in SCENARIO_TYPES[1:]:
        raise ValueError(f"Unsupported controlled scenario type: {scenario_type}")
    _assign_priority_pair(warehouse)
    if scenario_type == "priority_conflict":
        first, second = _adjacent_highway_pair(warehouse, intersection=True)
        _place_agents(warehouse, first, second)
    elif scenario_type == "narrow_corridor_yield":
        first, second = _adjacent_highway_pair(warehouse, narrow=True)
        _place_agents(warehouse, first, second)
        _set_loaded(warehouse, agent_id=2)
    elif scenario_type == "low_battery_diversion":
        first, second = _adjacent_highway_pair(warehouse)
        _place_agents(warehouse, first, second)
        warehouse.agents[0].battery = 0.15
        _set_loaded(warehouse, agent_id=1)
    else:
        station = warehouse.charging_stations[0]
        exits = _highway_neighbours(warehouse, station)
        if not exits:
            raise RuntimeError("Charging station has no accessible exit.")
        _place_agents(warehouse, exits[0], station)
    warehouse._recalc_grid()


def _assign_priority_pair(warehouse) -> None:
    """Make AGV 1 high priority and AGV 2 low priority via a label swap."""
    assigned = warehouse.task_queue.task_for_agent(2)
    if assigned is None or assigned.label[0] != "A":
        return
    candidate = next(
        (
            task
            for task in warehouse.task_queue.active_tasks
            if task.label[0] == "B" and task.label[1:] == assigned.label[1:]
        ),
        None,
    )
    if candidate is not None:
        warehouse.apply_priority_adjustments(
            (
                PriorityAdjustment(assigned.label, candidate.label, "controlled swap"),
                PriorityAdjustment(candidate.label, assigned.label, "controlled swap"),
            )
        )


def _place_agents(warehouse, first, second) -> None:
    occupied = {first, second}
    candidates = [
        point
        for point in _highway_cells(warehouse)
        if point not in occupied
    ]
    positions = [first, second]
    while len(positions) < len(warehouse.agents) and candidates:
        point = max(
            candidates,
            key=lambda candidate: (
                min(
                    abs(candidate[0] - placed[0])
                    + abs(candidate[1] - placed[1])
                    for placed in positions
                ),
                candidate[1],
                candidate[0],
            ),
        )
        positions.append(point)
        candidates.remove(point)
    if len(positions) != len(warehouse.agents):
        raise RuntimeError("Warehouse has insufficient highway cells for injection.")
    for agent, position in zip(warehouse.agents, positions):
        agent.x, agent.y = position
        agent.dir = Direction.RIGHT


def _set_loaded(warehouse, agent_id: int) -> None:
    agent = warehouse.agents[agent_id - 1]
    task = warehouse.task_queue.task_for_agent(agent_id)
    if task is None:
        raise RuntimeError("Controlled loaded scenario requires an assigned task.")
    shelf = warehouse.shelfs[task.shelf_id - 1]
    shelf.x, shelf.y = agent.x, agent.y
    agent.carrying_shelf = shelf


def _adjacent_highway_pair(warehouse, intersection=False, narrow=False):
    stations = set(warehouse.charging_stations) | set(warehouse.picking_stations)
    for point in _highway_cells(warehouse):
        if point in stations:
            continue
        neighbours = [
            neighbour
            for neighbour in _highway_neighbours(warehouse, point)
            if neighbour not in stations
        ]
        if intersection and len(neighbours) < 3:
            continue
        if narrow and len(neighbours) != 2:
            continue
        if neighbours:
            return point, neighbours[0]
    raise RuntimeError("Could not construct the requested controlled highway scenario.")


def _highway_cells(warehouse):
    return [
        (x, y)
        for y in range(warehouse.grid_size[0])
        for x in range(warehouse.grid_size[1])
        if warehouse._is_highway(x, y)
    ]


def _highway_neighbours(warehouse, point):
    x, y = point
    return [
        (next_x, next_y)
        for next_x, next_y in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
        if (
            0 <= next_x < warehouse.grid_size[1]
            and 0 <= next_y < warehouse.grid_size[0]
            and warehouse._is_highway(next_x, next_y)
        )
    ]


def apply_priority_instruction(env, provider: TeacherProvider, instruction: str) -> dict:
    """Apply a label-only user request through the existing atomic rule layer."""
    active_tasks = env.env.task_queue.as_dict()
    adjustments = tuple(provider.parse_priority_instruction(instruction, active_tasks))
    updated = env.env.apply_priority_adjustments(adjustments)
    return {
        "instruction": instruction,
        "adjustments": [
            {
                "task": adjustment.task,
                "new_label": adjustment.new_label,
                "reason": adjustment.reason,
            }
            for adjustment in adjustments
        ],
        "updated_labels": [task.label for task in updated],
    }
