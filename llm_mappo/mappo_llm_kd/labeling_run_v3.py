"""Frozen v3 labeling plans; preparation never accesses credentials or an API."""

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from functools import lru_cache
import json
import math
import os
from pathlib import Path

import numpy as np

from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V2, StructuredObservationSchema,
)

from .label_budget_v3 import LabelBudgetV3
from .labeling import AtomicLabelStore
from .goal_full_dataset_run import _receipts as _read_complete_episode_receipts
from .source_actor_v3 import canonical_sha256
from .pretraining import (
    DATASET_BUILD_V3, LABEL_RUN_V3, MAIN_BASELINE_DISTRIBUTION_V3,
    SEMANTIC_CONTRACT_V3, SEMANTIC_DIMENSIONS_V3, SEMANTIC_SOURCES_V3,
)
from .provider import OpenAICompatibleProvider, ProviderCallError, ProviderConfig
from .provider import contains_credential
from .semantic_quality_v3 import evaluate_semantic_quality_v3
from .semantic_prompt_v3 import build_semantic_prompt_v3
from .semantic_v3 import merge_semantic_v3_label, parse_semantic_v3_response
from .temporal_view import serialize_temporal_semantic_view


LABELING_PLAN_V3 = "frozen-labeling-plan-v3"
_CODE_FILES = (
    "labeling_run_v3.py", "label_budget_v3.py", "provider.py", "labeling.py",
    "semantic_v3.py", "semantic_prompt_v3.py", "semantic_quality_v3.py",
    "temporal_view.py",
    "goal_full_dataset_run.py", "source_actor_v3.py",
)
_TOTALS = {"tiny-3ag": 1000, "medium-5ag": 3000}


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False)
                  .encode("utf-8")).hexdigest()


def _file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonl(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _provider_plans(providers):
    if set(providers) != {"main", "pro"}:
        raise ValueError("Exactly main and pro providers are required.")
    result = {}
    for role, config in providers.items():
        if not isinstance(config, ProviderConfig):
            raise ValueError("Provider configuration is invalid.")
        model = "deepseek-flash" if role == "main" else "deepseek-v4-pro"
        if (config.base_url.rstrip("/") != "https://api.deepseek.com"
                or config.requested_model != model or config.retry_limit != 1
                or config.extra_body.get("thinking") != {"type": "disabled"}):
            raise ValueError("Provider differs from the bounded official API contract.")
        result[role] = asdict(config)
    return result


def _validate_pricing(pricing):
    if (not isinstance(pricing, dict) or pricing.get("peak_cache_miss") is not True
            or not isinstance(pricing.get("checked_date"), str)
            or not isinstance(pricing.get("source_url"), str)
            or not pricing["source_url"].startswith("https://api-docs.deepseek.com/")):
        raise ValueError("Dated official peak/cache-miss pricing evidence is required.")
    for role in ("main", "pro"):
        rates = pricing.get("rates", {}).get(role, {})
        for name in ("input_usd_per_million", "output_usd_per_million"):
            value = rates.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Peak pricing rates must be finite and positive.")


def _reservation(prompt, config, rates):
    # UTF-8 bytes bound token count conservatively; explicit framing reserve
    # also counts provider chat-control overhead. No cache discount is assumed.
    input_bound = len(prompt.system_text.encode("utf-8")) + len(
        prompt.user_text.encode("utf-8")
    ) + 8192
    return ((input_bound * rates["input_usd_per_million"]
             + config.max_completion_tokens * rates["output_usd_per_million"])
            / 1_000_000)


def _request_sha(provider, prompt):
    return _hash(provider._request_body(prompt))


def _window_id(window):
    return (f"{window['partition']}:{window['episode_id']}:agent-{window['agent_id']}"
            f":start-{window['start_timestep']}:end-{window['end_timestep']}")


def _verify_frozen_dataset(dataset, manifest, environment):
    frozen_path = dataset / "selection_frozen.json"
    identity_path = dataset / "collection_identity.json"
    if not frozen_path.is_file() or not identity_path.is_file():
        raise ValueError("Original selection freeze and collection identity required.")
    frozen, identity = _read_json(frozen_path), _read_json(identity_path)
    if (identity.get("schema") != "goal-full-dataset-collection-v3"
            or identity.get("synthetic_execution") is not False
            or identity.get("source_actor_contract", {}).get("verified_source")
            is not True
            or manifest.get("collection_identity_sha256") != canonical_sha256(identity)
            or manifest.get("source_actor_contract") != identity["source_actor_contract"]
            or manifest.get("rollout_environment_contract")
            != identity.get("rollout_environment_contract")):
        raise ValueError("Original verified collection provenance differs.")
    bindings = {
        "windows_sha256": "windows.jsonl", "manifest_sha256": "manifest.json",
        "selection_manifest_sha256": "selection_manifest.json",
        "label_lists_sha256": "label_selection_lists.json",
        "episode_partitions_sha256": "episode_partitions.json",
    }
    if any(not (dataset / filename).is_file()
           or frozen.get(name) != _file_hash(dataset / filename)
           for name, filename in bindings.items()):
        raise ValueError("Original frozen dataset/list/partition hashes changed.")
    receipts = _read_complete_episode_receipts(dataset, identity)
    if (frozen.get("episode_receipts_sha256") != canonical_sha256(receipts)
            or len(receipts) != identity["collection_contract"]["initial_episode_target"]
            or frozen.get("environment") != environment
            or frozen.get("frozen_before_labeling") is not True
            or frozen.get("total_samples") != _TOTALS[environment]):
        raise ValueError("Original episode receipts or frozen selection differ.")
    return frozen


def prepare_labeling_v3(  # noqa: C901
    dataset_directory, output_directory, *, environment, providers,
    pricing_evidence, phase_ledger_path, job_cap_usd=30.0,
):
    """Freeze a public, immutable plan using the builder's unique ID lists."""
    dataset, directory = Path(dataset_directory).resolve(), Path(output_directory)
    manifest_path = dataset / "manifest.json"
    if not manifest_path.is_file() or not (dataset / "state.json").is_file():
        raise ValueError("A completed eligible v3 dataset is required.")
    manifest, state = _read_json(manifest_path), _read_json(dataset / "state.json")
    if (manifest.get("schema") != DATASET_BUILD_V3
            or manifest.get("distribution") != MAIN_BASELINE_DISTRIBUTION_V3
            or manifest.get("semantic_contract_version") != SEMANTIC_CONTRACT_V3
            or manifest.get("semantic_dimensions") != list(SEMANTIC_DIMENSIONS_V3)
            or manifest.get("semantic_sources") != list(SEMANTIC_SOURCES_V3)
            or manifest.get("main_pretraining_eligible") is not True
            or state.get("status") != "completed"):
        raise ValueError("A completed eligible v3 dataset is required.")
    if environment not in _TOTALS:
        raise ValueError("This package prepares only frozen tiny/medium environments.")
    rollout = manifest.get("rollout_environment_contract", {}).get("environment", {})
    if rollout.get("n_agents") != (5 if environment == "medium-5ag" else 3):
        raise ValueError("Dataset rollout agent count differs from its environment.")
    if (rollout.get("charging_contract") != "goal-full-v1"
            or rollout.get("charging_rate") != 0.1
            or rollout.get("charge_release_threshold") != 1.0
            or rollout.get("charge_threshold") != .3):
        raise ValueError("The dataset requires the frozen goal-full-v1 contract.")
    if (environment == "medium-5ag"
            and rollout.get("profile_id") != "medium-5ag-goal-full-v1"):
        raise ValueError("The medium rollout profile is not frozen.")
    list_path = dataset / "label_selection_lists.json"
    if not list_path.is_file():
        raise ValueError("The builder's frozen selection lists are required.")
    if manifest.get("windows_sha256") != _file_hash(dataset / "windows.jsonl"):
        raise ValueError("Dataset windows SHA256 differs.")
    frozen = _verify_frozen_dataset(dataset, manifest, environment)
    public_providers = _provider_plans(providers)
    _validate_pricing(pricing_evidence)
    budget_snapshot = LabelBudgetV3(
        phase_ledger_path, job_id="prepare-check", job_cap_usd=job_cap_usd,
    ).totals()
    directory.mkdir(parents=True, exist_ok=False)
    _write_json(directory / "state.json", {"status": "preparing"})
    requests_path = directory / "requests.jsonl"
    index = _prepare_requests(dataset, requests_path, providers, pricing_evidence)
    groups = _frozen_groups(_read_json(list_path), index)
    for group in ("trial", "repeat", "pro", "review"):
        if frozen.get(f"{group}_ids") != groups[group]:
            raise ValueError("The builder's frozen coordinate ID list differs.")
    counts = {name: sum(row["partition"] == name for row in index.values())
              for name in ("train", "validation", "test")}
    total = _TOTALS[environment]
    if counts != dict(train=int(total * .7), validation=int(total * .15),
                      test=int(total * .15)):
        raise ValueError("Dataset size/partitions differ from the frozen environment.")
    plan = {
        "schema": LABELING_PLAN_V3, "environment": environment,
        "semantic_contract_version": SEMANTIC_CONTRACT_V3,
        "dataset_directory": str(dataset),
        "dataset_manifest_sha256": _file_hash(manifest_path),
        "dataset_windows_sha256": _file_hash(dataset / "windows.jsonl"),
        "builder_lists_sha256": _file_hash(list_path),
        "selection_frozen_sha256": _file_hash(dataset / "selection_frozen.json"),
        "requests_sha256": _file_hash(requests_path),
        "prompts_sha256": _hash({key: row["prompt_hashes"]
                                 for key, row in index.items()}),
        "providers": public_providers, "provider_plan_sha256": _hash(public_providers),
        "pricing_evidence": pricing_evidence,
        "phase_ledger_path": str(Path(phase_ledger_path).resolve()),
        "job_cap_usd": job_cap_usd, "phase_cap_usd": 100,
        "historical_medium_reference_usd": 23.0797,
        "phase_budget_snapshot": budget_snapshot,
        "frozen_selection": {
            "environment": environment, "total_samples": total,
            "frozen_before_labeling": True, "trial_ids": groups["trial"],
            "repeat_ids": groups["repeat"], "pro_ids": groups["pro"],
            "review_ids": groups["review"],
        },
        "code_sha256": {name: _file_hash(Path(__file__).with_name(name))
                        for name in _CODE_FILES},
        "first_attempt_reserve_usd": (
            sum(row["reservations"]["main"] for row in index.values())
            + sum(index[key]["reservations"]["main"] for key in groups["repeat"])
            + sum(index[key]["reservations"]["pro"] for key in groups["pro"])
        ),
    }
    retry_reserve = sum(sorted(
        (row["reservations"]["main"] for row in index.values()), reverse=True,
    )[:math.floor(.1 * total)])
    retry_reserve += sum(sorted((index[key]["reservations"]["main"]
                                for key in groups["repeat"]), reverse=True)[:4])
    retry_reserve += sum(sorted((index[key]["reservations"]["pro"]
                                for key in groups["pro"]), reverse=True)[:4])
    plan["maximum_reserve_usd"] = plan["first_attempt_reserve_usd"] + retry_reserve
    _write_json(directory / "plan.json", plan)
    if (plan["maximum_reserve_usd"] > job_cap_usd
            or plan["maximum_reserve_usd"]
            + budget_snapshot["phase_reserved_or_spent_usd"] > 100):
        _write_json(directory / "budget_preflight_report.json", {
            "status": "no-go", "first_attempt_reserve_usd":
            plan["first_attempt_reserve_usd"], "maximum_reserve_usd":
            plan["maximum_reserve_usd"], "job_cap_usd": job_cap_usd,
            "phase_budget_snapshot": budget_snapshot,
        })
        _write_json(directory / "state.json", {
            "status": "blocked_budget", "pretraining_eligible": False,
        })
        raise ValueError("Worst-case labeling budget exceeds its cap; see report.")
    _write_json(directory / "state.json", {
        "status": "prepared", "pretraining_eligible": False,
    })
    return {"status": "prepared", "plan_sha256": _file_hash(directory / "plan.json"),
            "first_attempt_reserve_usd": plan["first_attempt_reserve_usd"],
            "job_cap_usd": job_cap_usd, "pretraining_eligible": False}


def _prepare_requests(dataset, path, providers, pricing):
    index, episodes = {}, {}
    ledger = StructuredObservationSchema(
        observation_schema=DIRECT_GOAL_OBSERVATION_V2
    ).feature_names
    clients = {role: OpenAICompatibleProvider(config)
               for role, config in providers.items()}
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for window in _jsonl(dataset / "windows.jsonl"):
            key = _window_id(window)
            partition, episode = window["partition"], window["episode_id"]
            if (partition not in {"train", "validation", "test"} or key in index
                    or episodes.setdefault(episode, partition) != partition):
                raise ValueError("Dataset window IDs or episode partitions are invalid.")
            if len(window["observations"]) != 8 or len(window["reset_masks"]) != 8:
                raise ValueError("Exactly eight frames are required.")
            view = serialize_temporal_semantic_view(
                np.asarray(window["observations"], dtype=np.float32),
                np.asarray(window["reset_masks"], dtype=np.bool_), ledger, 8,
            )
            prompt = build_semantic_prompt_v3(view)
            if any(len(prompt.user_text) > config.max_prompt_characters
                   for config in providers.values()):
                raise ValueError("A frozen prompt exceeds its provider input cap.")
            row = {
                "view_id": key, "partition": partition, "episode_id": episode,
                "raw_window_sha256": _hash(window),
                "semantic_view_sha256": prompt.semantic_view_sha256,
                "prompt_hashes": {"system": prompt.system_sha256,
                                  "user": prompt.user_sha256},
                "request_hashes": {role: _request_sha(client, prompt)
                                   for role, client in clients.items()},
                "reservations": {
                    role: _reservation(prompt, config, pricing["rates"][role])
                    for role, config in providers.items()
                },
            }
            index[key] = row
            stream.write(json.dumps({**row, "semantic_view": view},
                                    ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return index


def _frozen_groups(lists, index):
    result = {}
    for name, count in (("trial", 120), ("repeat", 40), ("pro", 40), ("review", 60)):
        rows = lists.get(name)
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError("Frozen builder selection lists have incorrect counts.")
        selected = []
        for row in rows:
            key = _window_id({**row, "partition": "train"})
            if (key not in index or index[key]["partition"] != "train"
                    or row.get("view_sha256") != index[key]["semantic_view_sha256"]
                    or ("id" in row and row["id"] != key)
                    or ("view_id" in row and row["view_id"] != key)):
                raise ValueError("Frozen selection coordinates/view SHA differ.")
            selected.append(key)
        if len(set(selected)) != count:
            raise ValueError("Frozen selection IDs must be unique.")
        result[name] = selected
    if any(not set(result[name]) <= set(result["trial"])
           for name in ("repeat", "pro", "review")):
        raise ValueError("Repeat/pro/review must be frozen subsets of trial.")
    if set(result["repeat"]) != set(result["pro"]):
        raise ValueError("Repeat and Pro must use the same frozen forty windows.")
    return result


def execute_labeling_v3(  # noqa: C901
    plan_directory, *, stage, execute=False, authorization_path=None,
    trial_quality_path=None, blind_review_path=None, main_quality_path=None,
    provider_factory=OpenAICompatibleProvider,
):
    """Execution requires explicit, hash-bound outbound authorization."""
    if execute is not True:
        raise ValueError("An explicit execute=True is required for external requests.")
    directory = Path(plan_directory)
    plan = _load_plan(directory)
    if plan["environment"] != "medium-5ag":
        raise ValueError("Only medium has applicable outbound execution scope.")
    if stage not in {"trial", "main"}:
        raise ValueError("Execution stage must be trial or main.")
    _authorize(directory, plan, authorization_path)
    if stage == "main":
        _require_quality(directory, plan, "trial", trial_quality_path, blind_review_path)
    lock = directory / "execute.lock"
    lock.open("xb").close()
    try:
        budget = LabelBudgetV3(plan["phase_ledger_path"],
                               job_id=_file_hash(directory / "plan.json"),
                               job_cap_usd=plan["job_cap_usd"])
        budget.require_settled()
        _write_json(directory / "state.json", {
            "status": "running", "stage": stage, "pretraining_eligible": False,
        })
        groups = plan["frozen_selection"]
        wanted = {"main": set(groups["trial_ids"]) if stage == "trial" else None,
                  "repeat": set(groups["repeat_ids"]), "pro": set(groups["pro_ids"])}
        for group in ("main", "repeat", "pro"):
            role = "pro" if group == "pro" else "main"
            client = provider_factory(ProviderConfig(**plan["providers"][role]))
            target_count = len(wanted[group]) if wanted[group] is not None else (
                plan["frozen_selection"]["total_samples"])
            for row in _jsonl(directory / "requests.jsonl"):
                if wanted[group] is None or row["view_id"] in wanted[group]:
                    _label_one(directory, plan, row, group, client, budget, target_count)
        _write_json(directory / "state.json", {
            "status": "awaiting_quality", "stage": stage, "pretraining_eligible": False,
        })
        if stage == "main" and main_quality_path is not None:
            _require_quality(
                directory, plan, "main", main_quality_path, blind_review_path,
            )
            _finalize(directory, plan, main_quality_path)
        return {"status": _read_json(directory / "state.json")["status"],
                "stage": stage, "budget": budget.totals()}
    except Exception as error:
        _write_json(directory / "state.json", {
            "status": "stopped", "stage": stage, "error_type": type(error).__name__,
            "pretraining_eligible": False,
        })
        raise
    finally:
        lock.unlink()


def _load_plan(directory):
    plan = _read_json(directory / "plan.json")
    if (plan.get("schema") != LABELING_PLAN_V3
            or plan.get("semantic_contract_version") != SEMANTIC_CONTRACT_V3):
        raise ValueError("A frozen v3 labeling plan is required.")
    dataset = Path(plan["dataset_directory"])
    bindings = {
        "dataset_manifest_sha256": dataset / "manifest.json",
        "dataset_windows_sha256": dataset / "windows.jsonl",
        "builder_lists_sha256": dataset / "label_selection_lists.json",
        "selection_frozen_sha256": dataset / "selection_frozen.json",
        "requests_sha256": directory / "requests.jsonl",
    }
    if any(plan[name] != _file_hash(path) for name, path in bindings.items()):
        raise ValueError("A frozen dataset or request artifact changed.")
    _verify_frozen_dataset(dataset, _read_json(dataset / "manifest.json"),
                           plan["environment"])
    if plan.get("maximum_reserve_usd", float("inf")) > plan["job_cap_usd"]:
        raise ValueError("This plan is blocked by its frozen worst-case budget.")
    if any(value != _file_hash(Path(__file__).with_name(name))
           for name, value in plan["code_sha256"].items()):
        raise ValueError("Frozen labeling code changed.")
    if plan["provider_plan_sha256"] != _hash(plan["providers"]):
        raise ValueError("Frozen provider parameters changed.")
    _provider_plans({key: ProviderConfig(**value)
                     for key, value in plan["providers"].items()})
    return plan


def _authorize(directory, plan, path):
    if path is None:
        raise ValueError("Specific outbound authorization evidence is required.")
    evidence = _read_json(path)
    expected = {"schema": "llm-label-outbound-authorization-v3", "status": "approved",
                "environment": plan["environment"], "scope": "trial-and-main",
                "plan_sha256": _file_hash(directory / "plan.json"),
                "dataset_manifest_sha256": plan["dataset_manifest_sha256"],
                "provider_plan_sha256": plan["provider_plan_sha256"],
                "prompts_sha256": plan["prompts_sha256"],
                "job_cap_usd": plan["job_cap_usd"]}
    if any(evidence.get(name) != value for name, value in expected.items()):
        raise ValueError("Outbound authorization is not bound to this exact plan.")
    if (not isinstance(evidence.get("authorized_by"), str)
            or not evidence["authorized_by"]
            or not isinstance(evidence.get("authorized_at"), str)
            or not evidence["authorized_at"]):
        raise ValueError("Authorization must identify its owner and timestamp.")
    today = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
    if plan["pricing_evidence"]["checked_date"] != today:
        raise ValueError("Current dated pricing evidence is required before execution.")


def _records(directory, group):
    name = "labels.jsonl" if group == "main" else f"{group}.jsonl"
    return _store(str(directory / name))


@lru_cache(maxsize=32)
def _store(path):
    return AtomicLabelStore(path)


def _audit(directory, group=None):
    events = _store(str(directory / "attempts.jsonl")).records()
    return [row for row in events if row.get("event") == "result"
            and (group is None or row.get("group") == group)]


def _forbidden(content):
    try:
        value = json.loads(content)
    except (ValueError, TypeError):
        return False
    if not isinstance(value, dict):
        return False
    if set(value) - {"scores", "evidence", "unknown_reasons"}:
        return True
    return any(isinstance(value.get(field), dict)
               and set(value[field]) - set(SEMANTIC_DIMENSIONS_V3[:3])
               for field in value)


def _charge(receipt, reservation, rates):
    usage = receipt.get("usage", {})
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if type(prompt) is not int or type(completion) is not int:
        return reservation
    amount = (prompt * rates["input_usd_per_million"]
              + completion * rates["output_usd_per_million"]) / 1_000_000
    if amount < 0 or amount > reservation:
        raise ValueError("Reported usage exceeds its conservative reservation.")
    return amount


def _check_model(directory, group, receipt):
    role = "pro" if group == "pro" else "main"
    pair = (receipt.get("returned_model"), receipt.get("system_fingerprint"))
    if not isinstance(pair[0], str) or not pair[0]:
        raise ValueError("Provider returned no model identity.")
    for row in _store(str(directory / "raw_responses.jsonl")).records():
        if row.get("event") != "response":
            continue
        other = row["receipt"]
        previous = (other.get("returned_model"), other.get("system_fingerprint"))
        previous_role = "pro" if row["group"] == "pro" else "main"
        if role == previous_role and pair != previous:
            raise ValueError("Provider model/fingerprint changed; labeling stopped.")
        if role != previous_role and pair[0] == previous[0]:
            raise ValueError("Pro and Flash returned the same model identity.")


def _label_one(  # noqa: C901
    directory, plan, row, group, provider, budget, target_count,
):
    budget.require_settled()
    key, role = row["view_id"], "pro" if group == "pro" else "main"
    prompt = build_semantic_prompt_v3(row["semantic_view"])
    if (prompt.semantic_view_sha256 != row["semantic_view_sha256"]
            or {"system": prompt.system_sha256, "user": prompt.user_sha256}
            != row["prompt_hashes"]
            or _request_sha(provider, prompt) != row["request_hashes"][role]):
        raise ValueError("Frozen view/prompt/request changed.")
    cache = _hash(dict(plan=budget.job_id, group=group, view_id=key))
    for cached in _records(directory, group).records():
        if cached.get("cache_key") == cache:
            _validate_cached_provenance(directory, cached, row, group, provider,
                                        prompt, budget.job_id)
            parsed = parse_semantic_v3_response(json.dumps({
                field: {name: cached[field][name] for name in SEMANTIC_DIMENSIONS_V3[:3]}
                for field in ("scores", "evidence", "unknown_reasons")
            }), row["semantic_view"])
            merged = merge_semantic_v3_label(parsed, row["semantic_view"])
            if any(_hash(cached[field]) != _hash(value)
                   for field, value in merged.items()):
                raise ValueError("Cached merged semantics do not reproduce.")
            return cached
    prior = [event for event in _audit(directory, group) if event["view_id"] == key]
    if len(prior) >= 2:
        raise ValueError("The one ordinary retry was exhausted.")
    for attempt in range(len(prior), 2):
        if attempt == 1 and sum(event["attempt"] == 1 for event in _audit(
            directory, group
        )) >= math.floor(.1 * target_count):
            raise ValueError("The frozen stage retry allowance is exhausted.")
        request_id = _hash(dict(cache_key=cache, attempt=attempt))
        reservation = row["reservations"][role]
        if (isinstance(provider, OpenAICompatibleProvider)
                and not os.environ.get(provider.config.api_key_environment_variable)):
            raise ValueError("The provider API key environment variable is missing.")
        budget.reserve(request_id, reservation)
        try:
            completion = provider.complete(prompt)
        except ProviderCallError as error:
            _store(str(directory / "attempts.jsonl")).append({
                "event": "result", "group": group, "view_id": key,
                "attempt": attempt, "request_id": request_id, "parse_ok": False,
                "semantic_support_ok": False, "forbidden_content": False,
                "error_code": error.code, "outcome_unknown": error.outcome_unknown,
            })
            if error.outcome_unknown:
                raise ValueError("Unknown outcome remains fully reserved.") from None
            budget.settle(request_id, 0 if error.confirmed_not_sent else reservation)
            continue
        except Exception as error:
            _store(str(directory / "attempts.jsonl")).append({
                "event": "result", "group": group, "view_id": key,
                "attempt": attempt, "request_id": request_id, "parse_ok": False,
                "semantic_support_ok": False, "forbidden_content": False,
                "error_code": "unclassified-provider-error",
                "error_type": type(error).__name__, "outcome_unknown": True,
            })
            raise ValueError("Unknown outcome remains fully reserved.") from None
        raw = {"event": "response", "group": group, "view_id": key,
               "request_id": request_id, "attempt": attempt,
               "raw_content": completion.content, "receipt": completion.receipt}
        if contains_credential(raw):
            _store(str(directory / "raw_responses.jsonl")).append({
                "event": "response-redacted", "request_id": request_id,
                "raw_content_sha256": sha256(
                    completion.content.encode("utf-8")
                ).hexdigest(),
            })
            raise ValueError("Credential-like response redacted; reservation retained.")
        _store(str(directory / "raw_responses.jsonl")).append(raw)
        _check_model(directory, group, completion.receipt)
        parse_ok, supported = False, False
        try:
            parsed = parse_semantic_v3_response(completion.content, row["semantic_view"])
            parse_ok = True
            merged = merge_semantic_v3_label(parsed, row["semantic_view"])
            supported = True
        except (ValueError, TypeError, KeyError):
            merged = None
        _store(str(directory / "attempts.jsonl")).append({
            "event": "result", "group": group, "view_id": key, "attempt": attempt,
            "request_id": request_id, "parse_ok": parse_ok,
            "semantic_support_ok": supported,
            "forbidden_content": _forbidden(completion.content),
        })
        charge = _charge(completion.receipt, reservation,
                         plan["pricing_evidence"]["rates"][role])
        if supported:
            record = {"status": "ok", "view_id": key, "cache_key": cache, **merged,
                      "receipt": {**completion.receipt,
                                  "semantic_contract_version": SEMANTIC_CONTRACT_V3,
                                  "semantic_view_sha256": prompt.semantic_view_sha256,
                                  "request_sha256": row["request_hashes"][role],
                                  "request_id": request_id,
                                  "plan_sha256": budget.job_id}}
            _records(directory, group).append(record)
            budget.settle(request_id, charge)
            return record
        budget.settle(request_id, charge)
    raise ValueError("The response failed after one ordinary retry.")


def _validate_cached_provenance(
    directory, cached, row, group, provider, prompt, plan_sha,
):
    role = "pro" if group == "pro" else "main"
    receipt = cached.get("receipt", {})
    expected = {
        **provider.config.public_identity(), "semantic_contract_version":
        SEMANTIC_CONTRACT_V3, "semantic_view_sha256": prompt.semantic_view_sha256,
        "system_prompt_sha256": prompt.system_sha256,
        "user_prompt_sha256": prompt.user_sha256,
        "prompt_version": prompt.prompt_version,
        "request_sha256": row["request_hashes"][role], "plan_sha256": plan_sha,
    }
    if (cached.get("status") != "ok" or cached.get("view_id") != row["view_id"]
            or any(receipt.get(name) != value for name, value in expected.items())):
        raise ValueError("Cached provenance differs from the frozen request.")
    _check_model(directory, group, receipt)
    matched = [
        event for event in _store(str(directory / "raw_responses.jsonl")).records()
        if event.get("event") == "response"
        and event.get("request_id") == receipt.get("request_id")
        and event.get("view_id") == row["view_id"] and event.get("group") == group
    ]
    if (len(matched) != 1 or any(receipt.get(name) != value
                                 for name, value in matched[0]["receipt"].items())):
        raise ValueError("Cached provenance does not match its raw provider response.")
    parsed = parse_semantic_v3_response(matched[0]["raw_content"], row["semantic_view"])
    merged = merge_semantic_v3_label(parsed, row["semantic_view"])
    if any(_hash(cached[field]) != _hash(value) for field, value in merged.items()):
        raise ValueError("Cached semantics differ from the original raw response.")


def _quality_payload(directory, plan, stage, blind_path):
    windows = {row["view_id"]: row for row in _jsonl(directory / "requests.jsonl")}
    trial = set(plan["frozen_selection"]["trial_ids"])
    main = [row for row in _records(directory, "main").records()
            if stage == "main" or row["view_id"] in trial]
    audit = {}
    for group in ("main", "repeat", "pro"):
        audit[group] = [{name: row[name] for name in (
            "view_id", "attempt", "parse_ok", "forbidden_content", "semantic_support_ok"
        )} for row in _audit(directory, group)
            if group != "main" or stage == "main" or row["view_id"] in trial]
    report = evaluate_semantic_quality_v3(
        windows=windows, records=main, frozen_selection=plan["frozen_selection"],
        stage=stage, repeat_records=_records(directory, "repeat").records(),
        pro_records=_records(directory, "pro").records(),
        blind_reviews=_read_json(blind_path), request_audit=audit,
    )
    report["bindings"] = {
        "plan_sha256": _file_hash(directory / "plan.json"),
        "dataset_manifest_sha256": plan["dataset_manifest_sha256"],
        "selection_sha256": _hash(plan["frozen_selection"]),
        "records_sha256": _hash(main), "request_audit_sha256": _hash(audit),
        "blind_review_sha256": _file_hash(blind_path),
        "repeat_records_sha256": _hash(_records(directory, "repeat").records()),
        "pro_records_sha256": _hash(_records(directory, "pro").records()),
        "raw_responses_sha256": _hash([
            row for row in _store(str(directory / "raw_responses.jsonl")).records()
            if stage == "main" or row.get("group") != "main"
            or row.get("view_id") in trial
        ]),
        "attempt_receipts_sha256": _hash([
            row for row in _store(str(directory / "attempts.jsonl")).records()
            if stage == "main" or row.get("group") != "main"
            or row.get("view_id") in trial
        ]),
    }
    if stage == "main":
        report["bindings"]["labels_sha256"] = _file_hash(directory / "labels.jsonl")
    return report


def write_quality_report_v3(plan_directory, *, stage, blind_review_path, output_path):
    """Evaluate frozen evidence without sending or approving any request."""
    directory = Path(plan_directory)
    plan = _load_plan(directory)
    report = _quality_payload(directory, plan, stage, blind_review_path)
    _write_json(output_path, report)
    return report


def _require_quality(directory, plan, stage, path, blind_path):
    if path is None or blind_path is None:
        raise ValueError("Bound quality and blind review reports are required.")
    report = _read_json(path)
    current = _quality_payload(directory, plan, stage, blind_path)
    if report != current or report.get("status") != "go":
        raise ValueError("Quality evidence is missing, stale or not Go.")


def _finalize(directory, plan, quality_path):
    records = _records(directory, "main").records()
    if len(records) != plan["frozen_selection"]["total_samples"]:
        raise ValueError("Incomplete labels cannot become pretraining eligible.")
    retained_quality = directory / "quality-main.json"
    _write_json(retained_quality, _read_json(quality_path))
    manifest = {
        "schema": LABEL_RUN_V3, "semantic_contract_version": SEMANTIC_CONTRACT_V3,
        "semantic_dimensions": list(SEMANTIC_DIMENSIONS_V3),
        "semantic_sources": list(SEMANTIC_SOURCES_V3),
        "dataset_manifest_sha256": plan["dataset_manifest_sha256"],
        "labels_sha256": _file_hash(directory / "labels.jsonl"),
        "successful_records": len(records), "pretraining_eligible": True,
        "quality_report_sha256": _file_hash(retained_quality),
        "quality_report_path": "quality-main.json", "plan_path": "plan.json",
        "plan_sha256": _file_hash(directory / "plan.json"),
    }
    _write_json(directory / "manifest.json", manifest)
    _write_json(directory / "state.json", {
        "status": "completed", "pretraining_eligible": True,
    })
