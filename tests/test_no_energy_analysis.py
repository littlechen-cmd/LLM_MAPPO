from pathlib import Path
import csv
import json
from hashlib import sha256
import sys
import pytest
from scripts.analyze_medium5_no_energy_diagnostic import analyze


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def runs(tmp_path):
    for group, method in {"baseline": "MAPPO-FP-DG",
                          "astar-kd": "MAPPO-FP-ASTARKD-NO-VALIDITY"}.items():
        cutoff = 450000
        run = tmp_path / group / "seed_0007_test"
        identity = dict(schema="no-energy-diagnostic-v1", energy_enabled=False, seed=7,
                        method=method, code_commit="a" * 40,
                        config_sha256=f"{cutoff:064d}")
        training = dict(seed=7, environment_steps=900000, astar_stop_steps=cutoff,
                        astar_schedule_steps=600000, post_schedule_astar_lambda=.01,
                        optimization_schedule_steps=900000)
        episodes = [dict(seed=s, steps=800, completed_tasks=6, energy_deaths=0,
                         deadlocked=False, charged_events=0, agent_deaths=0,
                         charger_arrivals=0, task_recoveries=0) for s in range(9400, 9410)]
        evaluation = dict(episodes=episodes)
        write_json(run / "run_manifest.json", dict(identity=identity,
                   training=training, environment=dict(n_agents=5, max_steps=800, energy_enabled=False),
                   evaluation=dict(seeds=list(range(9400, 9410)))))
        write_json(run / "state.json", dict(status="complete"))
        write_json(run / "summary.json", dict(identity=identity,
                   global_environment_steps=900000, updates=879, evaluation=evaluation))
        write_json(run / "evaluation/evaluation.json", evaluation)
        updates = []
        for step in list(range(1024, 900000, 1024)) + [900000]:
            weight = (0 if group == "baseline" or step >= cutoff else .01 if step >= 420000
                      else .05 - .04 * step / 420000)
            updates.append(dict(global_environment_steps=step, astar_lambda=weight,
                                astar_loss=weight, astar_shared_grad_norm=weight,
                                actor_learning_rate=(900000-step)/900000*.0003,
                                entropy=1.0))
        write_csv(run / "updates.csv", updates)
        write_csv(run / "episodes.csv", [dict(episodes[0], global_environment_steps=899000)])
    return tmp_path


def test_complete_comparison(runs):
    report = analyze(runs)
    assert len(report["groups"]) == 2
    assert report["groups"][0]["evaluation"]["mean_completed_tasks"] == 6


def test_missing_group_fails(runs):
    (runs / "astar-kd" / "seed_0007_test").rename(runs / "gone")
    with pytest.raises(ValueError, match="one completed"):
        analyze(runs)


@pytest.mark.parametrize("field,value", [("seed", 17), ("schema", "formal-environment-training-v1")])
def test_wrong_identity_fails(runs, field, value):
    path = runs / "astar-kd/seed_0007_test/run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["identity"][field] = value
    write_json(path, manifest)
    with pytest.raises(ValueError, match="identity"):
        analyze(runs)


def test_formal_reserved_evaluation_fails(runs):
    path = runs / "astar-kd/seed_0007_test/run_manifest.json"
    manifest = json.loads(path.read_text())
    evaluation = json.loads((path.parent / "evaluation/evaluation.json").read_text())
    evaluation["episodes"][0]["seed"] = 200000
    write_json(path.parent / "evaluation/evaluation.json", evaluation)
    write_json(path, manifest)
    with pytest.raises(ValueError, match="evaluation"):
        analyze(runs)


def test_nonzero_tail_fails(runs):
    path = runs / "astar-kd/seed_0007_test/updates.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[-1]["astar_shared_grad_norm"] = "0.001"
    write_csv(path, rows)
    with pytest.raises(ValueError, match="pure PPO"):
        analyze(runs)



@pytest.mark.parametrize("field", ["energy_deaths", "charged_events", "charger_arrivals", "task_recoveries"])
def test_energy_events_fail(runs, field):
    path = runs / "baseline/seed_0007_test/episodes.csv"
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    rows[0][field] = 1
    write_csv(path, rows)
    with pytest.raises(ValueError, match="no-energy event"):
        analyze(runs)


def test_common_configuration_mismatch_fails(runs):
    path = runs / "astar-kd/seed_0007_test/run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["training"]["gamma"] = .5
    write_json(path, manifest)
    with pytest.raises(ValueError, match="differ beyond method"):
        analyze(runs)


def test_analyzer_reads_approved_config_before_accepting_results(runs, tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "baseline.yaml").write_text("schema: wrong\n", encoding="utf-8")
    with pytest.raises(ValueError):
        analyze(runs, configs)
