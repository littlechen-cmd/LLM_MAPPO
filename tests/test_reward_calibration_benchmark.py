"""The benchmark interface preserves H=12 as the sole formal horizon."""

import pytest

from scripts.benchmark_reward_calibration import BenchmarkConfig


def test_benchmark_config_rejects_noncanonical_formal_horizon():
    assert BenchmarkConfig(condition="h12").horizon == 12
    assert BenchmarkConfig(condition="h4").horizon == 4
    with pytest.raises(ValueError, match="condition"):
        BenchmarkConfig(condition="h8")
