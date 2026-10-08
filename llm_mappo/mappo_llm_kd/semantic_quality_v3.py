"""Pure, stage-aware quality gates for frozen continuous semantic datasets.

``windows`` maps IDs to partition/episode_id/semantic_view. ``frozen_selection``
declares environment, total_samples, frozen_before_labeling, trial_ids (120),
review_ids (60), and paired_ids (40), or separate repeat_ids/pro_ids (40 each).
``request_audit`` maps main/repeat/pro to explicit attempt rows containing
view_id, attempt (0 or 1), parse_ok and forbidden_content booleans. Missing
evidence is pending, never inferred successful. No file or API access occurs.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
from math import ceil, isfinite
from statistics import fmean, pstdev
from typing import Mapping, Sequence

from .semantic_v3 import (
    LLM_DIMENSIONS_V3, SEMANTIC_CONTRACT_V3, SEMANTIC_DIMENSIONS_V3,
    _competing_peer, _competition_candidate, _weak_relation,
    merge_semantic_v3_label, parse_semantic_v3_response, reference_semantics_v3,
)


_TOTALS = {"tiny-3ag": 1000, "medium-5ag": 3000, "formal_environment": 8000}
_PARTITIONS = ("train", "validation", "test")
_FIELDS = ("scores", "known_masks", "sources", "evidence", "unknown_reasons")
_RECORD_FIELDS = set(_FIELDS) | {"status", "view_id", "receipt", "cache_key"}


class _Report:
    def __init__(self):
        self.checks = []

    def add(self, name, passed, observed, required, scope, *, complete=True):
        status = "pending" if not complete else ("go" if passed else "no-go")
        self.checks.append(dict(id=name, status=status, observed=observed,
                                required=required, scope=scope))

    def status(self):
        statuses = {row["status"] for row in self.checks}
        return "no-go" if "no-go" in statuses else (
            "pending" if "pending" in statuses else "go")


def _ids(value):
    if (not isinstance(value, (list, tuple))
            or any(not isinstance(key, str) or not key for key in value)):
        return None
    return list(value)


def _selection(report, windows, frozen, stage):
    environment = frozen.get("environment")
    total = frozen.get("total_samples")
    identity_ok = (environment in _TOTALS and type(total) is int
                   and total == _TOTALS[environment]
                   and frozen.get("frozen_before_labeling") is True)
    report.add("selection-identity", identity_ok, {
        "environment": environment, "total_samples": total,
        "frozen_before_labeling": frozen.get("frozen_before_labeling"),
    }, "approved size and pre-label freeze", "frozen-selection")
    names = {"trial": frozen.get("trial_ids"), "review": frozen.get("review_ids"),
             "repeat": frozen.get("repeat_ids", frozen.get("paired_ids")),
             "pro": frozen.get("pro_ids", frozen.get("paired_ids"))}
    groups = {}
    for group, value in names.items():
        keys = _ids(value)
        expected = {"trial": 120, "review": 60, "repeat": 40, "pro": 40}[group]
        present = keys is not None
        valid = present and len(keys) == expected and len(set(keys)) == expected
        if valid:
            valid = all(key in windows and windows[key].get("partition") == "train"
                        for key in keys)
        report.add(f"frozen-{group}-ids", valid, len(keys) if present else None,
                   expected, "frozen-selection", complete=present)
        groups[group] = keys or []
    trial = set(groups["trial"])
    nested = all(set(groups[group]) <= trial for group in ("repeat", "pro", "review"))
    report.add("trial-nested-evidence-ids", nested, nested, True, "frozen-selection")
    paired_same = set(groups["repeat"]) == set(groups["pro"])
    report.add("frozen-paired-same-ids", paired_same, paired_same, True,
               "frozen-selection", complete=bool(groups["repeat"] and groups["pro"]))
    expected = list(windows) if stage == "main" else groups["trial"]
    return groups, expected, total if identity_ok else None


def _index_records(records, expected, report, group):
    if records is None:
        report.add(f"{group}-record-count", False, None, len(expected), group,
                   complete=False)
        return {}
    indexed, errors = {}, []
    for row in records:
        key = row.get("view_id") if isinstance(row, dict) else None
        if not isinstance(key, str) or key in indexed or key not in expected:
            errors.append(key)
        else:
            indexed[key] = row
    report.add(f"{group}-record-ids", not errors, errors, "unique fixed IDs", group)
    report.add(f"{group}-record-count", set(indexed) == set(expected), len(indexed),
               len(expected), group, complete=len(indexed) >= len(expected))
    return indexed


def _validate_record(row, view):
    if set(row) - _RECORD_FIELDS or row.get("status") != "ok":
        raise ValueError("Forbidden or incomplete merged-record fields.")
    if any(not isinstance(row.get(name), dict)
           or set(row[name]) != set(SEMANTIC_DIMENSIONS_V3) for name in _FIELDS):
        raise ValueError("Merged record requires five four-dimensional objects.")
    receipt = row.get("receipt")
    if (not isinstance(receipt, dict)
            or receipt.get("semantic_contract_version") != SEMANTIC_CONTRACT_V3):
        raise ValueError("Record semantic contract is absent or mixed.")
    response = {name: {dim: row[name][dim] for dim in LLM_DIMENSIONS_V3}
                for name in ("scores", "evidence", "unknown_reasons")}
    response["evidence"] = {
        key: list(paths) for key, paths in response["evidence"].items()
    }
    parsed = parse_semantic_v3_response(json.dumps(response, allow_nan=False), view)
    recomputed = merge_semantic_v3_label(parsed, view)
    for dimension in SEMANTIC_DIMENSIONS_V3:
        if type(row["known_masks"][dimension]) is not bool:
            raise ValueError("Known mask must be boolean.")
        value = row["scores"][dimension]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(value) or not 0 <= value <= 1
        ):
            raise ValueError("Every known score must be finite numeric [0,1].")
        for field in _FIELDS:
            supplied, wanted = row[field][dimension], recomputed[field][dimension]
            if field == "evidence":
                supplied, wanted = tuple(supplied), tuple(wanted)
            if supplied != wanted:
                raise ValueError(f"Merged {field} does not reproduce {dimension}.")
    return recomputed


def _record_integrity(indexed, windows, report, group):
    errors, missing_hashes, bad_hashes = [], [], []
    for key, row in indexed.items():
        try:
            view = windows[key]["semantic_view"]
            _validate_record(row, view)
            digest = row["receipt"].get("semantic_view_sha256")
            expected = sha256(json.dumps(view, sort_keys=True, ensure_ascii=False,
                                         separators=(",", ":"), allow_nan=False)
                              .encode("utf-8")).hexdigest()
            if digest is None:
                missing_hashes.append(key)
            elif digest != expected:
                bad_hashes.append(key)
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            errors.append(dict(view_id=key, error=str(error)))
    report.add(f"{group}-record-integrity", not errors, errors,
               "valid ranges, raw evidence, sources/masks and auxiliary recomputation",
               group, complete=bool(indexed))
    report.add(f"{group}-view-hash-binding", not bad_hashes,
               {"missing": missing_hashes, "mismatched": bad_hashes},
               "every original view SHA256 bound", group,
               complete=bool(indexed) and not missing_hashes)
    if bad_hashes and missing_hashes:
        report.add(f"{group}-view-hash-mismatch", False, bad_hashes, [], group)


def _known_value(row, dimension):
    try:
        value = row["scores"][dimension]
        known = row["known_masks"][dimension]
    except (TypeError, KeyError):
        return None
    if (known is not True or isinstance(value, bool)
            or not isinstance(value, (int, float)) or not isfinite(value)):
        return None
    return float(value) if 0 <= value <= 1 else None


def _statistics(rows, dimension):
    values = [value for row in rows
              if (value := _known_value(row, dimension)) is not None]
    return dict(count=len(rows), known_count=len(values),
                known_rate=len(values) / len(rows) if rows else None,
                minimum=min(values) if values else None,
                maximum=max(values) if values else None,
                mean=fmean(values) if values else None,
                population_std=pstdev(values) if values else None,
                high_count=sum(value >= .5 for value in values),
                low_count=sum(value <= .25 for value in values),
                positive_count=sum(value >= .25 for value in values),
                zero_count=values.count(0.),
                boundary_rate=sum(value in (0., 1.) for value in values) / len(values)
                if values else None)


def _categories(view):
    reference = reference_semantics_v3(view)
    frame = view["history"][-1]
    target, direct = _competition_candidate(frame)
    competing = False
    if target is not None:
        peer, reason = _competing_peer(view, target)
        competing = reason is None and (direct or _weak_relation(frame, peer))
    service = reference["unknown_reasons"]["yielding-preference"] == "hard-service-lock"
    constraint = reference["scores"]["local-constraint"]
    stagnation = reference["scores"]["motion-stagnation"]
    structural = constraint is not None and constraint >= .5
    stagnant = stagnation is not None and stagnation >= .25
    return dict(phase=frame["self"]["task_phase"], carrying=frame["self"]["carrying"],
                competition=competing, service=service, structure=structural,
                ordinary=(not competing and not structural
                          and not service and not stagnant),
                own_priority_higher=(competing and reference["scores"][
                    "yielding-preference"] == 0.),
                constrained_not_stagnant=structural and stagnation == 0.,
                open_but_stagnant=(constraint == 0. and stagnation is not None
                                   and stagnation >= .25))


def _pair_statistics(rows, first, second):
    pairs = []
    for row in rows:
        a, b = _known_value(row, first), _known_value(row, second)
        if a is not None and b is not None and (a != 0 or b != 0):
            pairs.append((a, b))
    equal = sum(a == b for a, b in pairs) / len(pairs) if pairs else None
    pearson = None
    if len(pairs) >= 30:
        xs, ys = zip(*pairs)
        variance_x, variance_y = pstdev(xs), pstdev(ys)
        if variance_x > 0 and variance_y > 0:
            mean_x, mean_y = fmean(xs), fmean(ys)
            pearson = fmean((a - mean_x) * (b - mean_y)
                            for a, b in pairs) / (variance_x * variance_y)
    return dict(common_nonjointzero_count=len(pairs), exact_equal_rate=equal,
                pearson=pearson, estimable=pearson is not None,
                review_triggered=(len(pairs) >= 30 and (
                    equal >= .95 or (pearson is not None and abs(pearson) >= .95))))


def _redundancy(train, categories, report, stage, complete):
    groups = {"overall": list(train)}
    for kind in ("phase", "carrying", "competition"):
        for key in train:
            if key in categories:
                label = f"{kind}={categories[key][kind]}"
                groups.setdefault(label, []).append(key)
    result = {}
    for label, keys in groups.items():
        result[label] = {
            f"{a}|{b}": _pair_statistics([train[key] for key in keys], a, b)
            for a, b in combinations(LLM_DIMENSIONS_V3, 2)
        }
    for pair, metrics in result["overall"].items():
        report.add(f"{stage}-redundancy-{pair}",
                   not metrics["review_triggered"], metrics,
                   "equality<.95 and |Pearson|<.95; >=30 nonjointzero estimable pairs",
                   "train-only", complete=complete and (
                       metrics["estimable"] or metrics["review_triggered"]))
    return result


def _paired_checks(main, other, expected, report, group, complete):
    diagnostics = {}
    common_ids = set(main) & set(other) & set(expected)
    for dimension in LLM_DIMENSIONS_V3:
        pairs, agreement, reason_agreement = [], 0, 0
        for key in common_ids:
            a = _known_value(main[key], dimension)
            b = _known_value(other[key], dimension)
            agreement += (a is None) == (b is None)
            if a is not None and b is not None:
                pairs.append(abs(a - b))
            reason_agreement += (main[key].get("unknown_reasons", {}).get(dimension)
                                 == other[key].get("unknown_reasons", {}).get(dimension))
        rate = agreement / len(common_ids) if common_ids else None
        mae = fmean(pairs) if pairs else None
        ready = complete and len(common_ids) == len(expected)
        report.add(f"{group}-known-agreement-{dimension}",
                   rate is not None and rate >= .9,
                   rate, .9, group, complete=ready)
        report.add(f"{group}-common-known-{dimension}", len(pairs) >= 20, len(pairs), 20,
                   group, complete=ready)
        threshold = .15 if group == "repeat" else .20
        report.add(f"{group}-mae-{dimension}", mae is not None and mae <= threshold,
                   mae, threshold, group, complete=ready and len(pairs) >= 20)
        diagnostics[dimension] = dict(
            common_known=len(pairs), paired_mae=mae, known_state_agreement=rate,
            unknown_reason_agreement=reason_agreement / len(common_ids)
            if common_ids else None,
        )
    return diagnostics


def _audit_rows(rows, expected):
    attempts, malformed, missing_fields = {}, [], []
    for row in rows:
        if not isinstance(row, dict):
            malformed.append("non-object attempt")
            continue
        if not {"view_id", "attempt", "parse_ok", "forbidden_content"} <= set(row):
            missing_fields.append(row.get("view_id"))
            continue
        key, attempt = row["view_id"], row["attempt"]
        if (key not in expected or type(attempt) is not int or attempt not in (0, 1)
                or type(row["parse_ok"]) is not bool
                or type(row["forbidden_content"]) is not bool
                or ("semantic_support_ok" in row
                    and type(row["semantic_support_ok"]) is not bool)
                or (key, attempt) in attempts):
            malformed.append(key)
        else:
            attempts[(key, attempt)] = row
    return attempts, malformed, missing_fields


def _request_checks(audit, groups, main_ids, report, accepted):
    groups = dict(main=main_ids, repeat=groups["repeat"], pro=groups["pro"])
    for group, expected in groups.items():
        rows = audit.get(group) if isinstance(audit, Mapping) else None
        if not isinstance(rows, (list, tuple)):
            report.add(f"{group}-request-audit", False, None,
                       "explicit original attempts",
                       group, complete=False)
            continue
        attempts, malformed, missing = _audit_rows(rows, expected)
        first = [row for (_, attempt), row in attempts.items() if attempt == 0]
        retried = {key for key, attempt in attempts if attempt == 1}
        ready = (not missing and {row["view_id"] for row in first} == set(expected)
                 and bool(expected))
        orphaned = [key for key in retried if (key, 0) not in attempts]
        report.add(f"{group}-request-audit", not malformed and not orphaned,
                   {"malformed": malformed, "missing_fields": missing,
                    "retry_without_first": orphaned},
                   "unique attempts 0/1 for fixed IDs", group,
                   complete=not missing or bool(malformed or orphaned))
        successful = {
            key for (key, _), row in attempts.items()
            if row["parse_ok"] and row.get("semantic_support_ok", True)
            and not row["forbidden_content"]
        }
        unbound = sorted(set(accepted[group]) - successful)
        report.add(f"{group}-accepted-attempt-binding", not unbound, unbound,
                   "every accepted label has a successful attempt for its own ID",
                   group, complete=not missing)
        parse_rate = (sum(row["parse_ok"] for row in first) / len(first)
                      if first else None)
        retry_rate = len(retried) / len(expected) if expected else None
        forbidden = sum(row["forbidden_content"] for row in attempts.values())
        report.add(f"{group}-first-parse-rate",
                   parse_rate is not None and parse_rate >= .98,
                   parse_rate, .98, group, complete=ready)
        report.add(f"{group}-retry-rate", retry_rate is not None and retry_rate <= .1,
                   retry_rate, .1, group, complete=ready)
        report.add(f"{group}-forbidden-content", forbidden == 0, forbidden, 0, group,
                   complete=ready or forbidden > 0)


def _review_checks(reviews, expected, categories, report):
    if reviews is None:
        report.add("blind-review-completeness", False, None, 60, "blind-review",
                   complete=False)
        return {}
    indexed, malformed, incomplete = {}, [], []
    for row in reviews:
        if not isinstance(row, dict) or row.get("view_id") not in expected:
            malformed.append("unexpected review")
            continue
        key = row["view_id"]
        if key in indexed:
            malformed.append(key)
        if not {"critical", "substantive", "reviewer_identity", "blinded"} <= set(row):
            incomplete.append(key)
            continue
        if (any(type(row[name]) is not int or row[name] < 0
                for name in ("critical", "substantive"))
                or not isinstance(row["reviewer_identity"], str)
                or not row["reviewer_identity"].strip() or row["blinded"] is not True):
            malformed.append(key)
        indexed[key] = row
    ready = set(indexed) == set(expected) and not incomplete and len(expected) == 60
    report.add("blind-review-completeness", not malformed,
               {"count": len(indexed), "malformed": malformed,
                "missing_fields": incomplete},
               "60 fixed blinded reviews with truthful identity", "blind-review",
               complete=ready or bool(malformed))
    critical = sum(row.get("critical", 0) for row in indexed.values()
                   if type(row.get("critical")) is int)
    substantive = sum(row.get("substantive", 0) for row in indexed.values()
                      if type(row.get("substantive")) is int)
    report.add("blind-review-findings", critical == 0 and substantive <= 3,
               {"critical": critical, "substantive": substantive},
               {"critical": 0, "substantive_max": 3}, "blind-review",
               complete=ready or critical > 0 or substantive > 3)
    coverage = {kind: sum(categories.get(key, {}).get(kind, False) for key in indexed)
                for kind in ("ordinary", "competition", "structure", "service")}
    report.add("blind-review-strata", all(value > 0 for value in coverage.values()),
               coverage, "ordinary/competition/structure/service each represented",
               "blind-review",
               complete=ready and all(key in categories for key in indexed))
    return dict(coverage=coverage, reviewers=sorted({
        row["reviewer_identity"] for row in indexed.values()
        if isinstance(row.get("reviewer_identity"), str)
    }))


def _partition_checks(windows, total, report):
    counts = Counter(row.get("partition") for row in windows.values())
    expected = ({"train": int(total * .7), "validation": int(total * .15),
                 "test": int(total * .15)} if total is not None else {})
    ready = bool(expected) and sum(counts.values()) >= total
    report.add("main-partition-counts", dict(counts) == expected, dict(counts), expected,
               "main-partition", complete=ready)
    episode_partitions = {}
    mixed, missing = [], []
    for key, row in windows.items():
        episode = row.get("episode_id")
        if not isinstance(episode, str) or not episode:
            missing.append(key)
            continue
        partition = row.get("partition")
        if episode in episode_partitions and episode_partitions[episode] != partition:
            mixed.append(episode)
        episode_partitions[episode] = partition
    report.add("main-episode-isolation", not mixed, {"mixed": mixed, "missing": missing},
               "complete episodes never cross partitions", "main-partition",
               complete=not missing)
    return expected


def _primary_checks(train, report, stage, complete):
    for dimension in LLM_DIMENSIONS_V3:
        stats = _statistics(list(train.values()), dimension)
        std = stats["population_std"]
        report.add(f"{stage}-std-{dimension}", std is not None and std >= .08,
                   std, .08, "train-only", complete=complete)


def _coverage_checks(train, indexed, windows, expected, categories, report):
    count = expected.get("train", 0)
    raw_train_count = sum(row.get("partition") == "train" for row in windows.values())
    complete = count > 0 and len(train) == count and raw_train_count == count
    for dimension in LLM_DIMENSIONS_V3:
        stats = _statistics(list(train.values()), dimension)
        for suffix, value, required in (
            ("known", stats["known_count"], ceil(count * .6)),
            ("high", stats["high_count"], ceil(count * .05)),
            ("low", stats["low_count"], ceil(count * .10)),
        ):
            report.add(f"train-{suffix}-{dimension}", value >= required, value, required,
                       "main-train-coverage", complete=complete)
    own_win = sum(categories.get(key, {}).get("own_priority_higher", False)
                  and (score := _known_value(row, "yielding-preference")) is not None
                  and score <= .25 for key, row in train.items())
    auxiliary = _statistics(list(train.values()), "motion-stagnation")
    service = sum(categories.get(key, {}).get("service", False) for key in train)
    requirements = (
        ("yielding-priority-win", own_win, ceil(count * .05)),
        ("auxiliary-positive", auxiliary["positive_count"], max(10, ceil(count * .01))),
        ("auxiliary-zero", auxiliary["zero_count"], ceil(count * .10)),
        ("charging-service", service, min(30, ceil(count * .02))),
    )
    for suffix, value, required in requirements:
        report.add(f"train-{suffix}", value >= required, value, required,
                   "main-train-coverage", complete=complete)
    for partition in ("validation", "test"):
        rows = [row for key, row in indexed.items()
                if windows[key].get("partition") == partition]
        target = expected.get(partition, 0)
        ready = target > 0 and len(rows) == target
        for dimension in LLM_DIMENSIONS_V3:
            known = _statistics(rows, dimension)["known_count"]
            required = max(20, ceil(target * .2))
            report.add(f"{partition}-known-{dimension}", known >= required,
                       known, required,
                       "frozen-heldout-acceptance", complete=ready)


def evaluate_semantic_quality_v3(
    *, windows: Mapping[str, Mapping], records: Sequence[Mapping],
    frozen_selection: Mapping, stage: str, repeat_records=None, pro_records=None,
    blind_reviews=None, request_audit=None,
) -> dict:
    """Return go/pending/no-go; a trial Go never makes main pretrain eligible."""
    if stage not in ("trial", "main"):
        raise ValueError("stage must be trial or main.")
    report = _Report()
    groups, main_ids, total = _selection(report, windows, frozen_selection, stage)
    indexed = _index_records(records, main_ids, report, "main")
    _record_integrity(indexed, windows, report, "main")
    categories, category_errors = {}, []
    for key in indexed:
        if windows[key].get("partition") != "train":
            continue
        try:
            categories[key] = _categories(windows[key]["semantic_view"])
        except (ValueError, KeyError, TypeError) as error:
            category_errors.append(dict(view_id=key, error=str(error)))
    report.add("observable-category-integrity", not category_errors, category_errors,
               "all categories derived from raw observations", "train-only",
               complete=bool(indexed))
    train = {key: row for key, row in indexed.items()
             if windows[key].get("partition") == "train"}
    complete = set(indexed) == set(main_ids) and bool(main_ids)
    expected = _partition_checks(windows, total, report) if stage == "main" else {}
    _primary_checks(train, report, stage, complete)
    redundancy = _redundancy(train, categories, report, stage, complete)
    paired = {}
    accepted = {"main": set(indexed)}
    for group, supplied in (("repeat", repeat_records), ("pro", pro_records)):
        other = _index_records(supplied, groups[group], report, group)
        accepted[group] = set(other)
        _record_integrity(other, windows, report, group)
        paired[group] = _paired_checks(indexed, other, groups[group], report, group,
                                       complete and supplied is not None)
    _request_checks(request_audit, groups, main_ids, report, accepted)
    reviews = _review_checks(blind_reviews, groups["review"], categories, report)
    if stage == "main":
        _coverage_checks(train, indexed, windows, expected, categories, report)
    partitions = _partition_diagnostics(indexed, windows)
    status = report.status()
    return dict(schema="semantic-quality-report-v3", stage=stage, status=status,
                semantic_contract_version=SEMANTIC_CONTRACT_V3,
                main_pretraining_eligible=stage == "main" and status == "go",
                checks=report.checks, diagnostics=dict(
                    partitions=partitions, redundancy=redundancy, paired=paired,
                    blind_review=reviews, raw_train_categories={
                        kind: sum(categories.get(key, {}).get(kind, False)
                                  for key in train)
                        for kind in ("ordinary", "competition", "structure", "service",
                                     "own_priority_higher", "constrained_not_stagnant",
                                     "open_but_stagnant")
                    }),
                heldout_policy=("Frozen known-count/integrity gates only; "
                                "no heldout semantic design feedback."))


def _partition_diagnostics(indexed, windows):
    result = {}
    for partition in _PARTITIONS:
        rows = [row for key, row in indexed.items()
                if windows[key].get("partition") == partition]
        dimensions = {}
        for dimension in SEMANTIC_DIMENSIONS_V3:
            statistics = _statistics(rows, dimension)
            if partition != "train":
                statistics = {key: statistics[key] for key in (
                    "count", "known_count", "known_rate"
                )}
            dimensions[dimension] = statistics
        result[partition] = dimensions
    return result
