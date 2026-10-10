"""Pure citation guidance for the frozen observable coordination contract."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from typing import Mapping

from .medium_prompt import observable_coordination_evidence
from .prompt import LabelPrompt


def add_observable_citation_recipe(
    prompt: LabelPrompt, view: Mapping[str, object],
) -> LabelPrompt:
    """Add supported original-path recipes without changing view or scores."""

    marker = "TEMPORAL_SEMANTIC_VIEW="
    if prompt.user_text.count(marker) != 1:
        raise ValueError("Citation recipes require one TEMPORAL_SEMANTIC_VIEW marker.")
    witness = observable_coordination_evidence(view)
    recipes = []
    if witness["blocked_forward_supported"]:
        recipes.append({
            "mode": "blocked_forward",
            "evidence": [
                "history[7].self.previous_requested_action",
                "history[7].self.previous_blocked",
                "summary.blocked_count",
            ],
        })
    if witness["oscillation_supported"]:
        turns = witness["turn_indices_and_actions"]
        first_index, first_action = turns[0]
        opposite_index = next(
            index for index, action in turns if action != first_action
        )
        segment_start = witness["phase_segment_start"]
        recipes.append({
            "mode": "oscillation",
            "evidence": [
                f"history[{first_index}].self.previous_requested_action",
                f"history[{opposite_index}].self.previous_requested_action",
                f"history[{segment_start}].goal.normalized_distance",
                "history[7].goal.normalized_distance",
            ],
        })
    for recipe in recipes:
        if not set(recipe["evidence"]) <= prompt.available_evidence_paths:
            raise ValueError("Citation recipe contains a path outside the allowlist.")
    guidance = (
        "EVIDENCE_RECIPES="
        + json.dumps({"deadlock_severity": recipes}, separators=(",", ":"))
        + "\nThese recipes guide citations only, using original allowlist paths "
        "from the supplied observable view. Recipe mode names and derived keys "
        "are not evidence paths. You decide score strengths and all other "
        "dimensions under the existing semantic contract. A supported recipe "
        "does not require a positive score. Describe observable state only; "
        "do not invent hidden state or prescribe actions. "
    )
    if recipes:
        guidance += (
            "If deadlock_severity is positive, cite every path in one supported "
            "recipe. When both recipes are listed, either complete recipe is "
            "permitted.\n"
        )
    else:
        guidance += (
            "No supported observable blocking or oscillation witness is present; "
            "deadlock_severity=0. Cite a relevant original allowlist path for "
            "that zero score.\n"
        )
    user_text = prompt.user_text.replace(marker, guidance + marker, 1)
    return replace(
        prompt,
        user_text=user_text,
        user_sha256=sha256(user_text.encode("utf-8")).hexdigest(),
    )
