"""Evaluate a Phase 3a checkpoint on deterministic seeds.

Optionally collect engagement-head samples for the diagnostic plot::

    python eval/evaluate_phase3.py \
        artifacts/phase3a_dual_head/seed_007/checkpoint_final.pt \
        --output artifacts/phase3a_eval.json \
        --collect-engagement \
        --engagement-csv artifacts/engagement_samples.csv
"""

import csv
from argparse import ArgumentParser
import json
from pathlib import Path

from llm_mappo.phase3_training import evaluate_phase3, load_phase3_policy


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes-per-seed", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--output", help="Write the JSON summary here")
    parser.add_argument(
        "--collect-engagement",
        action="store_true",
        help="Sample engagement-head outputs for the diagnostic plot",
    )
    parser.add_argument(
        "--engagement-sample-rate",
        type=int,
        default=10,
        help="Record engagement every N steps (default: 10)",
    )
    parser.add_argument(
        "--engagement-csv",
        help="Write (label, engagement) rows here for plotting",
    )
    args = parser.parse_args()
    policy, config, checkpoint = load_phase3_policy(args.checkpoint)
    result = evaluate_phase3(
        policy,
        config,
        args.seeds,
        args.episodes_per_seed,
        collect_engagement=args.collect_engagement,
        engagement_sample_rate=args.engagement_sample_rate,
    )
    result["checkpoint"] = str(Path(args.checkpoint))
    result["trained_episodes"] = checkpoint["episodes"]
    if args.engagement_csv and "engagement_samples" in result:
        samples = result.pop("engagement_samples")
        _write_engagement_csv(samples, args.engagement_csv)
        result["engagement_csv"] = str(Path(args.engagement_csv))
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")


def _write_engagement_csv(samples, path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "engagement"])
        writer.writerows(samples)


if __name__ == "__main__":
    main()
