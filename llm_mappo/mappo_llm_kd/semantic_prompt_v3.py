"""Three continuous semantic targets with raw observation evidence only."""

from hashlib import sha256
import json
from typing import Mapping

from .prompt import LabelPrompt
from .semantic_v3 import LLM_DIMENSIONS_V3, raw_evidence_paths_v3


SEMANTIC_PROMPT_V3 = "observable-semantic-three-continuous-prompt-v3"
LOCAL_CELL_COLUMNS_V3 = (
    "forward", "right", "static_obstacle", "shelf", "workstation", "charger", "other_agv"
)


def build_semantic_prompt_v3(semantic_view: Mapping[str, object]) -> LabelPrompt:
    """Create a provenance-bound prompt without auxiliary labels or answers."""
    allowed = raw_evidence_paths_v3(semantic_view)
    canonical = json.dumps(semantic_view, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False)
    raw = {"history": [
        {"self": frame["self"], "goal": frame["goal"], "peers": frame["peers"],
         "local": {"cells": [[cell[column] for column in LOCAL_CELL_COLUMNS_V3]
                             for cell in frame["local"]["cells"]]}}
        for frame in semantic_view["history"]
    ]}
    system = (
        "You annotate observable warehouse state. Return one strict JSON object, "
        "no prose. Never emit actions, routes, task assignments, rewards or "
        "priority modifications. Use only the supplied eight observable frames; "
        "there is no privileged global state, stable peer ID, peer goal, deadline "
        "or service countdown. Describe current history[7], earlier frames only "
        "as context. Evidence must be actual raw observation paths."
    )
    guidance = """THREE TARGETS: goal-priority, yielding-preference, local-constraint.
Known scores are any finite continuous number in [0,1], e.g. 0.37, 0.63, 0.82.
The 0/0.25/0.5/0.75/1 values below are REFERENCE ANCHORS, not an enum or
rounding requirement. Interpolate when observable evidence warrants it.
Do not quantize. Unknown is null with the appropriate fixed reason code;
known numeric scores require unknown_reasons=null. Goal-priority cannot be null.

goal-priority: intrinsic present goal importance, independent of peer presence,
traffic, narrowness, blocking and carrying. Idle/zero battery anchor 0; active
noncharging without task priority anchor .25; task priority .5/.75/1 anchors
.5/.75/1. Charging battery >=.30 and <1 anchor .25; [.15,.30) anchor .5;
(.05,.15) anchor .75; (0,.05] anchor 1. Phase and goal type must match.
Cite current task_phase, battery, priority_weight and goal_type (raw containers
are allowed). Do not infer remaining path energy or death probability.

yielding-preference: relative observable reason for yielding to an identified
competing peer, not the action itself. No local competition is known 0. A
valid remote peer or arbitrary nearby AGV alone does not establish competition.
DIRECT: own previous action blocked FORWARD and front adjacent cell occupied
by a currently associated peer. WEAK: peer two cells forward in the same local
straight corridor, opposing headings, both recent actions unblocked FORWARD,
intervening cell structurally passable and unoccupied. Heading alone does not
predict a future route. First identify relation/peer, then compare priorities
using the same phase/battery/priority convention. Own higher anchors 0; equal
competing priorities -> null/symmetric-priority. Peer higher under weak relation
anchor .25; direct relation with priority advantage .25/.5/>=.75 anchors
.5/.75/1. Current peer slots are NOT stable identities across frames.
Peer relative_forward/right are normalized by one unknown map diagonal; local
cell offsets are integers. When local occupancy exists, nearest peer is local
(maximum local distance sqrt(8)<minimum outside distance 3). Infer scale from
minimum local occupied distance / norm(nearest peer coordinates), then verify
current positions against integer cells (residual <=1e-4), occupancy and sorted
distances. Never compare normalized peer distance directly with cell offsets.
Unresolvable association -> ambiguous-peer-association; required peer attribute
absent -> missing-peer-attributes; irreconcilable competing peers ->
conflicting-peer-evidence. Known relation absence is 0, not a fabricated peer.
Charging phase/type, all three goal geometry values |value|<=1e-6, own center
charger=true and 0<battery<1 identify hard charging service ->
null/hard-service-lock. Workstation center + zero goal geometry + task/delivery
phase gives ambiguous-service-state when service would change the relation.
Do not infer picking countdown from NOOP. Positive/unknown yielding needs
current self, peers and local.cells evidence; service unknown additionally goal.

local-constraint: observable STRUCTURAL maneuvering restriction, ignoring
temporary AGV occupancy. Static/outside cells are blocked. Unloaded AGVs can
enter shelf cells; loaded AGVs cannot enter stationary shelf cells without an
AGV. Shelf+other_agv may be a moving load: if passability uncertainty changes
classification -> null/ambiguous-shelf-mobility. Workstations/chargers are not
walls, and vehicle heading does not add an invented turning radius.
With d structurally passable cardinal adjacent cells: d4 anchors0, d3 .25,
d2 corner or visible branch within two graph edges .5, d2 opposite entrances
with a continuous visible narrow corridor and no visible branch .75, d<=1 1.
A visible branch is a reachable cell within two four-neighbor edges with >=3
visible structural neighbors. Outside the local 5x5 view is unknown, not a
global wall or proof of no route. Cite current carrying and local.cells.

OUTPUT: exactly three objects scores/evidence/unknown_reasons, each with exactly
the THREE TARGETS as keys. Evidence arrays contain 1-4 unique paths from the
allowlist, never summary or invented explanation paths. Raw containers such as
history[7].self, history[7].goal, history[7].peers, history[7].local.cells count
as one path each. No fourth score, free-text rationale, confidence or action.
"""
    output_shape = {
        "scores": {name: "finite [0,1] number or null" for name in LLM_DIMENSIONS_V3},
        "evidence": {name: ["one to four actual raw paths"]
                     for name in LLM_DIMENSIONS_V3},
        "unknown_reasons": {name: "null or applicable fixed reason"
                            for name in LLM_DIMENSIONS_V3},
    }
    # A short recommended allowlist prevents thousands of repeated cell leaf
    # names in the API payload. The parser also accepts valid raw leaf paths.
    recommended = [f"history[{index}].{group}" for index in range(8)
                   for group in ("self", "goal", "peers", "local.cells")]
    user = (guidance + "\nOUTPUT_STRUCTURE_ONLY="
            + json.dumps(output_shape, separators=(",", ":"))
            + "\nRECOMMENDED_EVIDENCE_PATH_ALLOWLIST=" + json.dumps(recommended)
            + "\nLOCAL_CELL_COLUMNS=" + json.dumps(LOCAL_CELL_COLUMNS_V3)
            + "\nEach local.cells row follows LOCAL_CELL_COLUMNS exactly; "
            "all raw booleans remain JSON booleans. Evidence cites the "
            "original named cells/fields, not column indices or derived proxies."
            + "\nRAW_OBSERVABLE_VIEW="
            + json.dumps(raw, ensure_ascii=False, separators=(",", ":"),
                         allow_nan=False))
    return LabelPrompt(
        prompt_version=SEMANTIC_PROMPT_V3,
        system_text=system,
        user_text=user,
        system_sha256=sha256(system.encode("utf-8")).hexdigest(),
        user_sha256=sha256(user.encode("utf-8")).hexdigest(),
        semantic_view_sha256=sha256(canonical.encode("utf-8")).hexdigest(),
        score_dimensions=LLM_DIMENSIONS_V3,
        available_evidence_paths=allowed,
    )
