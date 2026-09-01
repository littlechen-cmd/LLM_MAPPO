"""Machine-readable E1 method-contract comparisons."""

from typing import Any, Mapping


def compare_method_contracts(fixed: Mapping[str, Any], calibrated: Mapping[str, Any]) -> dict:
    """Fail closed unless Fixed/RC differ only in the calibration weight."""
    allowed = {"method", "astar", "reward_confidence"}
    changed = sorted(
        name for name in set(fixed) | set(calibrated)
        if fixed.get(name) != calibrated.get(name)
    )
    unexpected = set(changed) - allowed
    if unexpected:
        raise ValueError(f"Fixed/RC contract has unexpected difference {sorted(unexpected)[0]}.")
    if fixed.get("method") != "Fixed-AStarKD+LLMKD" or calibrated.get("method") != "RC-AStarKD+LLMKD":
        raise ValueError("Fixed/RC contract identifies incompatible methods.")
    if fixed.get("reward_confidence") != "one" or calibrated.get("reward_confidence") != "c_A_reward":
        raise ValueError("Fixed/RC reward confidence contract is incompatible.")
    return {"schema": "e1-method-contract-diff-v1", "pass": True,
            "optimization_difference": ["reward_confidence"], "changed_fields": changed}
