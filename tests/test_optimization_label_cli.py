import json

from train.collect_optimization_labels import main


def test_label_cli_dry_run_validates_without_network(tmp_path):
    record = tmp_path / "record.json"
    output = tmp_path / "output.json"
    record.write_text(
        json.dumps({
            "task_persistence": 0.2,
            "task_persistence_reason": "Task active.",
            "yielding_preference": 0.4,
            "yielding_preference_reason": "Peer nearby.",
            "coordination_risk": 0.6,
            "coordination_risk_reason": "Narrow area.",
        }),
        encoding="utf-8",
    )

    assert main(["--dry-run-record", str(record), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["network_calls"] == 0
