"""Approved medium goal-full contract, isolated from historical C1 artifacts."""

from copy import deepcopy
from typing import Any, Mapping

from llm_mappo.medium5_contract import (
    ENVIRONMENT as C1_ENVIRONMENT,
    EVALUATION as C1_EVALUATION,
    REWARD_DEFINITION as C1_REWARD_DEFINITION,
    TRAINING as C1_TRAINING,
)


CORE_GOAL_FULL_SCHEMA = "core-mappo-209d-goal-full-v1"
MEDIUM_GOAL_FULL_PROFILE = "medium-5ag-goal-full-v1"
TRAINING_SEEDS = (7, 17, 27, 37, 47, 57, 67, 77)
ENVIRONMENT = deepcopy(C1_ENVIRONMENT)
ENVIRONMENT.update({
    "profile_id": MEDIUM_GOAL_FULL_PROFILE,
    "charging_contract": "goal-full-v1",
    "charging_rate": 0.1,
    "charge_release_threshold": 1.0,
})
TRAINING = deepcopy(C1_TRAINING)
EVALUATION = deepcopy(C1_EVALUATION)
REWARD_DEFINITION = deepcopy(C1_REWARD_DEFINITION)


def validate_medium_goal_full(values: Mapping[str, Any]) -> None:
    """Accept only the separately approved medium values and method extensions."""
    if values.get("schema") != CORE_GOAL_FULL_SCHEMA:
        raise ValueError("Medium goal-full schema differs from approved contract.")
    if values.get("reward_definition") != REWARD_DEFINITION:
        raise ValueError("Medium goal-full reward differs from approved contract.")
    if values.get("environment") != ENVIRONMENT:
        raise ValueError("Medium goal-full environment differs from approved contract.")
    if values.get("evaluation") != EVALUATION:
        raise ValueError("Medium goal-full evaluation differs from approved contract.")
    _validate_training(values)


def _validate_training(values: Mapping[str, Any]) -> None:
    training = dict(values.get("training", {}))
    seed = training.pop("seed", None)
    if type(seed) is not int or seed not in TRAINING_SEEDS:
        raise ValueError("Medium goal-full training seed is not preregistered.")
    method = str(values.get("method", ""))
    if method not in {
        "MAPPO-FP-DG", "MAPPO-FP-DG-LLM-INIT",
        "MAPPO-FP-ASTARKD-NO-VALIDITY",
        "MAPPO-FP-ASTARKD-NO-VALIDITY-LLM-INIT",
    }:
        raise ValueError("Medium goal-full method is incompatible.")
    if method.endswith("LLM-INIT"):
        if "representation_initialization" not in training:
            raise ValueError("Medium goal-full LLM initialization is missing.")
        training.pop("representation_initialization")
    if "ASTARKD" in method:
        if training.pop("post_schedule_astar_lambda", None) != 0.0:
            raise ValueError("Medium goal-full A*KD tail lambda differs.")
    if training != TRAINING:
        raise ValueError("Medium goal-full training differs from approved contract.")
