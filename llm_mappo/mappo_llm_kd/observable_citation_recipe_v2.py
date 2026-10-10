"""Per-head original-path citation guidance under the frozen v7 contract."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from typing import Mapping

from .observable_citation_recipe import add_observable_citation_recipe
from .prompt import LabelPrompt


def add_observable_citation_recipe_v2(
    prompt: LabelPrompt, view: Mapping[str, object],
) -> LabelPrompt:
    """Extend v1 citation recipes without assigning supported score strengths."""

    previous = add_observable_citation_recipe(prompt, view)
    current = view["history"][7]
    count_path = "history[7].local.visible_peer_cell_count"
    maximum_path = "summary.visible_peer_cell_statistics.maximum"
    local = current["local"]
    statistics = view["summary"]["visible_peer_cell_statistics"]
    if local["visible_peer_cell_count"] > 0:
        occupancy = {
            "mode": "current",
            "evidence": [count_path],
            "optional_evidence": [
                f"history[7].local.peer_occupied.{name}"
                for name, occupied in local["peer_occupied"].items() if occupied
            ],
            "score_constraint": "model_decides",
        }
    elif statistics["maximum"] > 0:
        occupancy = {
            "mode": "past_only",
            "evidence": [maximum_path, count_path],
            "score_constraint": "at_most_0.25",
        }
    else:
        occupancy = {
            "mode": "absent",
            "evidence": [count_path],
            "score_constraint": "zero",
        }
    idle = (
        current["self"]["task_phase"] == "idle"
        or current["goal"]["goal_type"] == "idle"
    )
    urgency = {
        "mode": "current_idle" if idle else "current_active",
        "evidence": ["history[7].self.task_phase", "history[7].goal.goal_type"],
        "score_constraint": "zero" if idle else "model_decides",
    }
    if not idle:
        urgency["optional_evidence"] = [
            "history[7].self.battery", "history[7].self.priority_weight",
        ]
    recipes = {
        "mission_urgency": urgency,
        "concession_appropriateness": occupancy,
        "interaction_risk": occupancy,
    }
    for recipe in recipes.values():
        paths = recipe["evidence"] + recipe.get("optional_evidence", [])
        if not set(paths) <= prompt.available_evidence_paths:
            raise ValueError("Per-head citation recipe contains a non-allowlist path.")
    guidance = (
        "HEAD_EVIDENCE_RECIPES="
        + json.dumps(recipes, separators=(",", ":"))
        + "\nThese recipes guide each head's original-path evidence only. "
        "Recipe modes and score constraints summarize existing contract rules; "
        "they are not citation paths or generated labels. Copy required "
        "evidence paths into that head's evidence array. Optional evidence "
        "may be added only when its observed value supports the score; keep "
        "each array to 1-4 unique original allowlist paths. You decide supported "
        "score strengths. A supported recipe never requires a positive score. "
        "For concession_appropriateness and interaction_risk, remote peer valid "
        "or priority fields cannot substitute for cited positive local "
        "occupancy. Current positive occupancy requires citing the positive "
        "history[7].local.visible_peer_cell_count; true cardinal paths are "
        "optional supporting citations. Past-only occupancy requires citing "
        "summary.visible_peer_cell_statistics.maximum and the current zero "
        "history[7].local.visible_peer_cell_count; any positive concession "
        "or risk score is at most 0.25. With neither current nor past local "
        "occupancy, concession_appropriateness=0 and interaction_risk=0, "
        "citing the current zero count. Current idle phase or idle goal "
        "requires mission_urgency=0 with the original current phase and goal "
        "paths. Active phase and goal do not prescribe positive urgency. "
        "For positive deadlock_severity, copy every evidence path from one "
        "complete supported EVIDENCE_RECIPES recipe into its evidence array, "
        "including both exact opposite-turn paths and both same-segment "
        "distance paths for oscillation. A partial recipe is invalid. "
        "Use only supplied observable state; never invent state, prescribe "
        "actions, or assign agents.\n"
    )
    marker = "TEMPORAL_SEMANTIC_VIEW="
    user_text = previous.user_text.replace(marker, guidance + marker, 1)
    return replace(
        previous,
        user_text=user_text,
        user_sha256=sha256(user_text.encode("utf-8")).hexdigest(),
    )
