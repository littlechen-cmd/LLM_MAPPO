"""One owner-approved extra request for a supported fourth-head citation error."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

from .dataset_labeling import _atomic_json
from .labeling import parse_label_response
from .medium_labeling import MediumLabelingHalt
from .medium_prompt import (
    MEDIUM_LABEL_PROMPT_V7, observable_coordination_evidence,
    validate_medium_v7_semantic_support,
)
from .network_resume import _validate_bindings, _validate_prior
from .observable_citation_recipe import add_observable_citation_recipe


CITATION_REPAIR_ERROR = (
    "deadlock_severity positive score lacks mode-specific cited observable evidence."
)
OWNER_APPROVAL_QUOTE = "\u6279\u51c6\u5b9a\u5411\u4fee\u590d\u5e76\u7ee7\u7eed"
CITATION_REPAIR_VARIANT = "citation-only-repair-v1"


def citation_repair_implementation_sha256():
    root = Path(__file__).resolve().parents[2]
    names = (
        "llm_mappo/mappo_llm_kd/citation_only_repair.py",
        "llm_mappo/mappo_llm_kd/medium_labeling.py",
        "llm_mappo/mappo_llm_kd/network_resume.py",
        "scripts/run_medium_5ag_dataset_labeling.py",
    )
    return {name: sha256((root / name).read_bytes()).hexdigest() for name in names}


def create_citation_only_repair_policy(
    output, config, authorization_path, *, authorization_sha256, config_sha=None,
):
    """Bind an explicit owner artifact; no artifact means no extra permission."""

    return _CitationOnlyRepairPolicy(
        Path(output).resolve(), deepcopy(config), Path(authorization_path).resolve(),
        authorization_sha256, config_sha,
    )


class _CitationOnlyRepairPolicy:
    def __init__(self, output, config, path, authorization_sha, config_sha):
        self.output = output
        self.config = config
        self.path = path
        self.authorization_sha256 = authorization_sha
        self.config_sha = sha256(json.dumps(
            config, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        if config_sha is not None and self.config_sha != config_sha:
            raise MediumLabelingHalt("citation_repair_config_digest_differs")
        self._verify()

    def _verify(self):  # noqa: C901
        try:
            raw = self.path.read_bytes()
            approval = json.loads(raw)
            if not isinstance(approval, dict):
                raise ValueError("Authorization must be an object.")
            identity, identity_sha, _ = _validate_bindings(
                self.output, self.config, self.config_sha,
            )
            log = (self.output / "attempts.jsonl").read_bytes()
            length = approval.get("attempts_size_bytes")
            expected = {
                "schema": "medium-label-citation-only-repair-authorization-v1",
                "decision": "approved", "owner_approval_quote": OWNER_APPROVAL_QUOTE,
                "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
                "windows_sha256": identity["windows_sha256"],
                "label_config_sha256": self.config_sha,
                "identity_sha256": identity_sha,
                "maximum_extra_requests_per_semantic_view": 1,
                "allowed_validation_error": CITATION_REPAIR_ERROR,
                "original_attempts_retained": True,
                "model_scores_rewritten": False,
                "cumulative_limits": self.config["limits"],
                "retry_rate_cap": 0.10,
                "implementation_sha256": citation_repair_implementation_sha256(),
            }
            valid = (
                sha256(raw).hexdigest() == self.authorization_sha256
                and all(approval.get(key) == value for key, value in expected.items())
                and type(length) is int and 0 < length <= len(log)
                and type(approval.get("maximum_extra_requests_per_semantic_view")) is int
                and log[length - 1:length] == b"\n"
                and sha256(log[:length]).hexdigest() == approval.get("attempts_sha256")
                and self.config["prompt_version"] == MEDIUM_LABEL_PROMPT_V7
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise MediumLabelingHalt(
                "citation_repair_authorization_unreadable"
            ) from error
        if not valid:
            raise MediumLabelingHalt("citation_repair_authorization_differs")
        return identity

    def begin(self, ledger, key, base, view, provider, scope, *,  # noqa: C901
              allow_request=True):
        """Atomically reserve the one extra request after all starts settle."""

        with ledger._lock:
            identity = self._verify()
            if ledger.config != self.config or ledger.path.resolve() != (
                self.output / "attempts.jsonl"
            ):
                raise MediumLabelingHalt("citation_repair_ledger_binding_differs")
            prior = ledger.history(key)
            if sum(row["status"] == "invalid_response" for row in prior) != (
                provider.config.retry_limit + 1
            ):
                return None
            if any(
                event.get("prompt_variant") == CITATION_REPAIR_VARIANT
                and event.get("semantic_view_sha256") == base.semantic_view_sha256
                for event in ledger.starts.values()
            ):
                return None
            invalid = prior[-1]
            basis = ledger.starts[invalid["attempt_id"]]
            if (base.prompt_version != MEDIUM_LABEL_PROMPT_V7
                    or basis["semantic_view_sha256"] != base.semantic_view_sha256
                    or basis["prompt_version"] != base.prompt_version
                    or (basis.get("base_user_prompt_sha256")
                        or basis["user_prompt_sha256"]) != base.user_sha256):
                raise MediumLabelingHalt("citation_repair_base_prompt_identity_differs")
            audit = invalid.get("response_audit", {})
            if (invalid.get("status") != "invalid_response"
                    or audit.get("parse_error") != CITATION_REPAIR_ERROR
                    or audit.get("forbidden_content") is not False
                    or not observable_coordination_evidence(view)["positive_supported"]):
                return None
            try:
                parsed = parse_label_response(json.dumps({
                    "scores": audit["parsed_scores"],
                    "evidence": audit["cited_original_evidence"],
                }), base.available_evidence_paths)
                validate_medium_v7_semantic_support(parsed.scores, parsed.evidence, view)
            except ValueError as error:
                if str(error) != CITATION_REPAIR_ERROR:
                    return None
            except (KeyError, TypeError):
                return None
            else:
                return None
            recipes = _supported_recipes(base, view)
            admissible = False
            for recipe in recipes:
                # This temporary proof tests eligibility only; it is never stored.
                proof = dict(parsed.evidence)
                proof["deadlock_severity"] = recipe["evidence"]
                try:
                    validate_medium_v7_semantic_support(parsed.scores, proof, view)
                    admissible = True
                except ValueError:
                    continue
            if not admissible:
                return None
            if not allow_request or set(ledger.starts) != set(ledger.settled):
                raise MediumLabelingHalt("citation_repair_requires_settled_batch")
            _validate_prior(ledger, identity, self.config)
            active = _repair_prompt(base, view, audit, recipes)
            if len(active.user_text) > provider.config.max_prompt_characters:
                raise MediumLabelingHalt("citation_repair_prompt_size_limit_exceeded")
            attempt_id = ledger.begin(
                scope, key, active, provider, prompt_variant=CITATION_REPAIR_VARIANT,
                base_user_prompt_sha256=base.user_sha256,
            )
            _atomic_json(self.output / f"citation-repair-attempt-{attempt_id}.json", {
                "schema": "medium-label-citation-only-repair-attempt-v1",
                "attempt_id": attempt_id, "basis_attempt_id": invalid["attempt_id"],
                "authorization_sha256": self.authorization_sha256,
                "semantic_view_sha256": base.semantic_view_sha256,
                "base_user_prompt_sha256": base.user_sha256,
                "repair_user_prompt_sha256": active.user_sha256,
                "prompt_variant": CITATION_REPAIR_VARIANT,
                "original_attempts_retained": True, "model_scores_rewritten": False,
                "request_budget_consumed_on_start": True,
            })
            return active, attempt_id, audit


def _supported_recipes(base, view):
    derived = add_observable_citation_recipe(base, view)
    text = derived.user_text.split("\nEVIDENCE_RECIPES=", 1)[1].split("\n", 1)[0]
    return json.loads(text)["deadlock_severity"]


def _repair_prompt(base, view, audit, recipes):
    options = []
    for recipe in recipes:
        observed = []
        for path in recipe["evidence"]:
            if path.startswith("history["):
                index = int(path.split("[", 1)[1].split("]", 1)[0])
                section, leaf = path.split("].", 1)[1].split(".", 1)
                value = view["history"][index][section][leaf]
            else:
                value = view["summary"]["blocked_count"]
            observed.append({"path": path, "value": value})
        options.append({**recipe, "observed_values": observed})
    guidance = (
        "CITATION_ONLY_REPAIR_V1=" + json.dumps({
            "validation_error": CITATION_REPAIR_ERROR,
            "failed_original_evidence": audit["cited_original_evidence"][
                "deadlock_severity"
            ],
            "supported_complete_recipe_options": options,
        }, separators=(",", ":"))
        + "\nReturn a fresh complete JSON response. You decide all four scores; "
        "no earlier scores are prescribed. For any positive fourth score, copy "
        "every original evidence path from one complete recipe option. Compare "
        "the observed LEFT/RIGHT values: two LEFT requests or two RIGHT requests "
        "do not supply opposite-turn evidence. Recipe modes and observed_values "
        "are guidance only and cannot be cited. All scores and evidence undergo "
        "the unchanged strict validator. Do not invent state or issue actions.\n"
    )
    marker = "TEMPORAL_SEMANTIC_VIEW="
    if base.user_text.count(marker) != 1:
        raise MediumLabelingHalt("citation_repair_view_marker_differs")
    user = base.user_text.replace(marker, guidance + marker, 1)
    return replace(base, user_text=user, user_sha256=sha256(user.encode()).hexdigest())
