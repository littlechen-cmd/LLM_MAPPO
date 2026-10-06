"""Read-only comparison of four single-seed A* tail diagnostics."""
import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean

CUTOFFS = (450000, 600000, 750000, 900000)
EVAL_SEEDS = list(range(9400, 9410))


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite metric {key}")
    return value


def metrics(rows):
    if not rows:
        raise ValueError("No completed episodes in requested window")
    steps = sum(_number(row, "steps") for row in rows)
    completed = sum(_number(row, "completed_tasks") for row in rows)
    if steps <= 0:
        raise ValueError("Episode steps must be positive")
    count = len(rows)
    return {
        "episodes": count, "mean_completed_tasks": completed / count,
        "completed_per_1000_actual_steps": 1000 * completed / steps,
        "total_actual_steps": steps, "mean_actual_steps": steps / count,
        "all_five_dead_percent": 100 * sum(
            _number(row, "energy_deaths") == 5 for row in rows) / count,
        "deadlock_percent": 100 * sum(str(row["deadlocked"]).lower()
                                      in {"true", "1", "1.0"} for row in rows) / count,
        "full_800_percent": 100 * sum(_number(row, "steps") == 800
                                      for row in rows) / count,
        "charging_episode_percent": 100 * sum(
            _number(row, "charged_events") > 0 for row in rows) / count,
        "more_than_five_delivery_percent": 100 * sum(
            _number(row, "completed_tasks") > 5 for row in rows) / count,
    }


def analyze(root, config_directory=None):  # noqa: C901
    groups, signatures, commits, ledger = [], [], set(), {}
    for cutoff in CUTOFFS:
        directory = root / f"cutoff_{cutoff // 1000}k"
        runs = [p for p in directory.glob("seed_0007*") if p.is_dir()]
        if len(runs) != 1:
            raise ValueError(f"Require unique completed seed 7 run: {directory}")
        run = runs[0]
        manifest = _json(run / "run_manifest.json")
        summary = _json(run / "summary.json")
        identity = manifest["identity"]
        if (identity.get("schema") != "astar-tail-diagnostic-v1"
                or identity.get("seed") != 7
                or identity.get("method") != "MAPPO-FP-ASTARKD-NO-VALIDITY"
                or summary.get("identity") != identity
                or len(identity.get("config_sha256", "")) != 64
                or len(identity.get("code_commit", "")) != 40):
            raise ValueError("Invalid diagnostic identity")
        commits.add(identity["code_commit"])
        if config_directory is not None:
            source = config_directory / f"cutoff_{cutoff // 1000}k.yaml"
            if sha256(source.read_bytes()).hexdigest() != identity["config_sha256"]:
                raise ValueError("Diagnostic config SHA256 differs from approved source")
        if (_json(run / "state.json").get("status") != "complete"
                or summary["global_environment_steps"] != 900000
                or summary["updates"] != 879):
            raise ValueError("Run incomplete or wrong diagnostic budget")
        training = manifest["training"]
        if (training.get("seed") != 7 or training.get("environment_steps") != 900000
                or training.get("astar_stop_steps") != cutoff
                or training.get("astar_schedule_steps") != 600000
                or training.get("post_schedule_astar_lambda") != .01
                or training.get("optimization_schedule_steps") != 900000):
            raise ValueError("Invalid shared schedule or cutoff")
        signatures.append(json.dumps({
            "environment": manifest["environment"],
            "training": {k: v for k, v in training.items() if k != "astar_stop_steps"},
            "evaluation": manifest["evaluation"],
        }, sort_keys=True))
        evaluation = _json(run / "evaluation/evaluation.json")
        if (manifest["evaluation"]["seeds"] != EVAL_SEEDS
                or [row["seed"] for row in evaluation["episodes"]] != EVAL_SEEDS
                or evaluation != summary["evaluation"]):
            raise ValueError("Invalid diagnostic evaluation instances or summary")
        updates = _csv(run / "updates.csv")
        expected = list(range(1024, 900000, 1024)) + [900000]
        if [int(row["global_environment_steps"]) for row in updates] != expected:
            raise ValueError("Update counters differ from 879 paired updates")
        for row in updates:
            step = int(row["global_environment_steps"])
            weight = (0.0 if step >= cutoff else
                      .01 if step >= 420000 else .05 - .04 * step / 420000)
            if not math.isclose(_number(row, "astar_lambda"), weight,
                                rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError("Shared lambda curve differs before tail cutoff")
        first_zero = next((int(row["global_environment_steps"]) for row in updates
                           if _number(row, "astar_lambda") == 0), None)
        expected_zero = min(900000, math.ceil(cutoff / 1024) * 1024)
        if first_zero != expected_zero:
            raise ValueError("Incorrect first zero tail update")
        tail = [row for row in updates
                if int(row["global_environment_steps"]) >= cutoff]
        if any(_number(row, key) != 0 for row in tail for key in
               ("astar_lambda", "astar_loss", "astar_shared_grad_norm")):
            raise ValueError("Nonzero KD contribution in pure PPO tail")
        positive_tail = [row for row in tail
                         if int(row["global_environment_steps"]) < 900000]
        if any(_number(row, "actor_learning_rate") <= 0 for row in positive_tail):
            raise ValueError("Nonpositive learning rate before final tail update")
        episodes = _csv(run / "episodes.csv")
        last = [row for row in episodes
                if 800000 < int(row["global_environment_steps"]) <= 900000]
        groups.append({
            "cutoff_steps": cutoff, "pure_ppo_steps": 900000 - cutoff,
            "run_directory": str(run.resolve()), "identity": identity,
            "first_zero_update": first_zero,
            "positive_lr_tail_updates": len(positive_tail),
            "evaluation": metrics(evaluation["episodes"]),
            "last_100k_training": metrics(last),
            "last_100k_entropy": mean(_number(row, "entropy") for row in updates
                                      if int(row["global_environment_steps"]) > 800000),
        })
        for name in ("run_manifest.json", "state.json", "summary.json", "updates.csv",
                     "episodes.csv", "evaluation/evaluation.json"):
            path = run / name
            ledger[str(path.resolve())] = sha256(path.read_bytes()).hexdigest()
    if len(commits) != 1 or len(set(signatures)) != 1:
        raise ValueError("Groups differ beyond cutoff or have different code commits")
    return {"schema": "astar-tail-diagnostic-review-v1", "groups": groups,
            "source_sha256": ledger, "formal_cutoff_decision": "owner_pending"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("artifacts/astar_tail_diagnostics/medium-5ag"))
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--config-directory", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "configs/astar_tail_diagnostics/medium-5ag")
    args = parser.parse_args()
    report = analyze(args.source, args.config_directory)
    output = (args.output_directory or args.source / "review").resolve()
    if any(output.is_relative_to(Path(g["run_directory"])) for g in report["groups"]):
        raise ValueError("Review output must not overwrite raw run artifacts")
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = [{"cutoff_steps": g["cutoff_steps"], "pure_ppo_steps": g["pure_ppo_steps"],
             **g["evaluation"], "last_100k_entropy": g["last_100k_entropy"]}
            for g in report["groups"]]
    with (output / "comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# A* KD 截断诊断比较", "",
             "四组均为 seed=7、900k。仅用于选择候选配置，不支持显著性或最优截断结论。",
             "吞吐按实际执行步数计，须与死亡、死锁和时长一起解读。", "",
             "| KD 截断 | 纯 PPO 尾段 | 平均完成数 | 实际步吞吐/千步 | 全员死亡% | 死锁% | 满800步% |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for group in report["groups"]:
        m = group["evaluation"]
        lines.append(f"| {group['cutoff_steps']} | {group['pure_ppo_steps']} | "
                     f"{m['mean_completed_tasks']:.3f} | "
                     f"{m['completed_per_1000_actual_steps']:.3f} | "
                     f"{m['all_five_dead_percent']:.1f} | {m['deadlock_percent']:.1f} | "
                     f"{m['full_800_percent']:.1f} |")
    lines += ["", "尾段归零、更新计数、评估实例与跨组配置一致性检查通过。",
              "完整训练窗口、熵、充电及重复交付指标见 comparison.json；源文件 SHA256 已记录。",
              "建议按原始完成数、全员死亡和死锁联合审阅；差异接近时保留不确定结论。",
              "正式截断选择待负责人确认，不自动替换正式配置。", ""]
    (output / "review.md").write_text("\n".join(lines), encoding="utf-8")
    print(str(output))


if __name__ == "__main__":
    main()
