"""The fixed/RC smoke checks shared calibration-chain counts only."""

from llm_mappo.o2_training import O2Trainer


def test_fixed_and_rc_use_the_same_selected_calibration_chain(monkeypatch):
    """Only the applied reward confidence may differ in the parity control."""
    class Result:
        selected = True
        confidence = 0.25

    trainer = object.__new__(O2Trainer)
    trainer.run_spec = type("Run", (), {"group": "RC-AStarKD"})()
    trainer.calibration_weight_mode = "fixed"
    assert trainer._reward_confidence(Result()) == 1.0
    trainer.calibration_weight_mode = "reward-calibrated"
    assert trainer._reward_confidence(Result()) == 0.25
