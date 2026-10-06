from pathlib import Path
import csv
import json
from hashlib import sha256
import sys
import pytest
from scripts.analyze_medium5_astar_tail_diagnostic import analyze, main


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
    for cutoff in (450000, 600000, 750000, 900000):
        run = tmp_path / f"cutoff_{cutoff // 1000}k" / "seed_0007_test"
        identity = dict(schema="astar-tail-diagnostic-v1", seed=7,
                        method="MAPPO-FP-ASTARKD-NO-VALIDITY", code_commit="a" * 40,
                        config_sha256=f"{cutoff:064d}")
        training = dict(seed=7, environment_steps=900000, astar_stop_steps=cutoff,
                        astar_schedule_steps=600000, post_schedule_astar_lambda=.01,
                        optimization_schedule_steps=900000)
        episodes = [dict(seed=s, steps=800, completed_tasks=6, energy_deaths=0,
                         deadlocked=False, charged_events=2) for s in range(9400, 9410)]
        evaluation = dict(episodes=episodes)
        write_json(run / "run_manifest.json", dict(identity=identity,
                   training=training, environment=dict(n_agents=5, max_steps=800),
                   evaluation=dict(seeds=list(range(9400, 9410)))))
        write_json(run / "state.json", dict(status="complete"))
        write_json(run / "summary.json", dict(identity=identity,
                   global_environment_steps=900000, updates=879, evaluation=evaluation))
        write_json(run / "evaluation/evaluation.json", evaluation)
        updates = []
        for step in list(range(1024, 900000, 1024)) + [900000]:
            weight = (0 if step >= cutoff else .01 if step >= 420000
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
    assert len(report["groups"]) == 4
    assert report["groups"][0]["first_zero_update"] == 450560
    assert report["groups"][-1]["positive_lr_tail_updates"] == 0
    assert report["groups"][0]["evaluation"]["mean_completed_tasks"] == 6


def test_missing_group_fails(runs):
    (runs / "cutoff_900k" / "seed_0007_test").rename(runs / "gone")
    with pytest.raises(ValueError, match="unique"):
        analyze(runs)


@pytest.mark.parametrize("field,value", [("seed", 17), ("schema", "formal-environment-training-v1")])
def test_wrong_identity_fails(runs, field, value):
    path = runs / "cutoff_450k/seed_0007_test/run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["identity"][field] = value
    write_json(path, manifest)
    with pytest.raises(ValueError, match="identity"):
        analyze(runs)


def test_formal_reserved_evaluation_fails(runs):
    path = runs / "cutoff_450k/seed_0007_test/run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["evaluation"]["seeds"] = list(range(200000, 200010))
    write_json(path, manifest)
    with pytest.raises(ValueError, match="evaluation"):
        analyze(runs)


def test_nonzero_tail_fails(runs):
    path = runs / "cutoff_450k/seed_0007_test/updates.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[-1]["astar_shared_grad_norm"] = "0.001"
    write_csv(path, rows)
    with pytest.raises(ValueError, match="tail"):
        analyze(runs)


def test_wrong_config_sha_fails(runs, tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "cutoff_450k.yaml").write_text("wrong config")
    with pytest.raises(ValueError, match="SHA256"):
        analyze(runs, configs)


def test_cross_group_configuration_mismatch_fails(runs):
    path = runs / "cutoff_750k/seed_0007_test/run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["training"]["gamma"] = .5
    write_json(path, manifest)
    with pytest.raises(ValueError, match="differ beyond cutoff"):
        analyze(runs)


def test_cli_writes_review_without_changing_sources(runs, monkeypatch):
    configs = runs / "configs"
    configs.mkdir()
    for cutoff in (450, 600, 750, 900):
        source = configs / f"cutoff_{cutoff}k.yaml"
        source.write_text(f"fixture: {cutoff}\n")
        run = runs / f"cutoff_{cutoff}k/seed_0007_test"
        for name in ("run_manifest.json", "summary.json"):
            value = json.loads((run / name).read_text())
            value["identity"]["config_sha256"] = sha256(source.read_bytes()).hexdigest()
            write_json(run / name, value)
    before = {p: p.read_bytes() for p in runs.rglob("*") if p.is_file()}
    output = runs / "review"
    monkeypatch.setattr(sys, "argv", ["analyze", "--source", str(runs),
                        "--config-directory", str(configs),
                        "--output-directory", str(output)])
    main()
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert len(json.loads((output / "comparison.json").read_text())["groups"]) == 4
    assert "不支持显著性或最优截断结论" in (output / "review.md").read_text(encoding="utf-8")
    assert (output / "comparison.csv").is_file()
