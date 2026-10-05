"""Project module entrypoints must win over an external ROS scripts package."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module", [
    "scripts.run_medium5_formal_baseline",
    "scripts.run_mappo_fp_baseline",
    "scripts.run_formal_training_diagnostic",
])
def test_module_entrypoint_ignores_external_ros_scripts(tmp_path, module):
    external = tmp_path / "ros_dist_packages"
    package = external / "scripts"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "raise ModuleNotFoundError('No module named catkin_pkg')\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.update({"PYTHONPATH": str(external), "CUDA_VISIBLE_DEVICES": ""})
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
