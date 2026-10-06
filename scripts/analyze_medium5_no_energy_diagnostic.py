"""Audit isolated paired no-energy diagnostics without changing source artifacts."""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean

from scripts.analyze_medium5_astar_tail_diagnostic import (
    _csv, _json, _number, metrics,
)
from scripts.run_medium5_no_energy_diagnostic import (
    GROUPS, EVALUATION, METHOD_FIELDS, load_no_energy_experiment,
)


def analyze(source, config_directory=None):  # noqa: C901
    groups, commits, ledger, signatures = [], set(), {}, []
    for group, method in GROUPS.items():
        runs = [p for p in (source / group).glob("seed_0007_*") if p.is_dir()]
        if len(runs) != 1:
            raise ValueError(f"Require one completed seed 7 run: {group}")
        run = runs[0]
        manifest = _json(run / "run_manifest.json")
        summary = _json(run / "summary.json")
        identity = manifest["identity"]
        if (identity.get("schema") != "no-energy-diagnostic-v1"
                or identity.get("method") != method or identity.get("seed") != 7
                or identity.get("energy_enabled") is not False
                or manifest["environment"].get("energy_enabled") is not False
                or summary["identity"] != identity
                or len(identity.get("code_commit", "")) != 40):
            raise ValueError("Invalid no-energy identity")
        commits.add(identity["code_commit"])
        signatures.append(json.dumps({
            "environment": manifest["environment"],
            "training": {k: v for k, v in manifest["training"].items()
                         if k not in METHOD_FIELDS},
            "evaluation": manifest["evaluation"],
        }, sort_keys=True))
        if config_directory is not None:
            config = config_directory / f"{group}.yaml"
            experiment = load_no_energy_experiment(
                config, group, root=Path(__file__).resolve().parents[1],
            )
            if (experiment.config_sha256 != identity["config_sha256"]
                    or experiment.training != manifest["training"]
                    or experiment.environment != manifest["environment"]
                    or experiment.evaluation != manifest["evaluation"]):
                raise ValueError("Run differs from approved diagnostic config")
        if (_json(run / "state.json")["status"] != "complete"
                or summary["global_environment_steps"] != 900000
                or summary["updates"] != 879):
            raise ValueError("Wrong budget or incomplete run")
        evaluation = _json(run / "evaluation/evaluation.json")
        if (evaluation != summary["evaluation"]
                or [row["seed"] for row in evaluation["episodes"]]
                != EVALUATION["seeds"]):
            raise ValueError("Wrong evaluation instances")
        episodes = _csv(run / "episodes.csv")
        updates = _csv(run / "updates.csv")
        expected = list(range(1024, 900000, 1024)) + [900000]
        if [int(row["global_environment_steps"]) for row in updates] != expected:
            raise ValueError("Update counters differ")
        for row in episodes + evaluation["episodes"]:
            for field in ("energy_deaths", "agent_deaths", "charged_events",
                          "charger_arrivals", "task_recoveries"):
                if _number(row, field) != 0:
                    raise ValueError(f"Unexpected no-energy event: {field}")
        for row in updates:
            step = int(row["global_environment_steps"])
            weight = (0 if group == "baseline" or step >= 450000
                      else .01 if step >= 420000
                      else .05 - .04 * step / 420000)
            if not math.isclose(_number(row, "astar_lambda"), weight,
                                rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError("KD curve differs")
            if weight == 0 and any(_number(row, key) != 0 for key in
                                   ("astar_loss", "astar_shared_grad_norm")):
                raise ValueError("Nonzero KD in pure PPO updates")
        last = [row for row in episodes
                if 800000 < int(row["global_environment_steps"]) <= 900000]

        def report(rows):
            result = metrics(rows)
            result.pop("all_five_dead_percent")
            return result
        groups.append({
            "group": group, "run_directory": str(run.resolve()),
            "identity": identity, "evaluation": report(evaluation["episodes"]),
            "last_100k_training": report(last),
            "last_100k_entropy": mean(_number(row, "entropy") for row in updates
                                      if int(row["global_environment_steps"]) > 800000),
        })
        for name in ("run_manifest.json", "state.json", "summary.json",
                     "episodes.csv", "updates.csv", "evaluation/evaluation.json"):
            path = run / name
            ledger[str(path.resolve())] = sha256(path.read_bytes()).hexdigest()
    if len(set(signatures)) != 1:
        raise ValueError("Paired groups differ beyond method fields")
    if len(commits) != 1:
        raise ValueError("Paired groups use different commits")
    return {"schema": "no-energy-diagnostic-review-v1", "groups": groups,
            "source_sha256": ledger, "decision": "owner_review_pending"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("artifacts/no_energy_diagnostics/medium-5ag"))
    parser.add_argument("--config-directory", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "configs/no_energy_diagnostics/medium-5ag")
    args = parser.parse_args()
    print(json.dumps(analyze(args.source, args.config_directory),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
