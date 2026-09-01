"""Run the complete frozen E1 matrix through four persistent GPU-0 slots."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from llm_mappo.e1_protocol import expand_e1_formal_matrix, load_e1_governance_manifest
from llm_mappo.e1_scheduler import SeedBlockScheduler, admit_slot_memory, write_heartbeat


def _command(args, run):
    return [args.python_bin, "scripts/run_e1_training.py", "--governance", args.governance,
            "--records", args.records, "--run", run.identity.replace(":seed", ":"),
            "--output-root", args.output_root, "--device", "cuda:0"]


def _live_pid(value):
    try:
        os.kill(int(value), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--governance", default="configs/g3_experiment_manifest.yaml")
    parser.add_argument("--state", required=True); parser.add_argument("--records", required=True)
    parser.add_argument("--output-root", required=True); parser.add_argument("--start", action="store_true")
    parser.add_argument("--python-bin", default="/home/lzx/.conda/envs/llm-a-mappo-py310/bin/python")
    parser.add_argument("--log-root", default="/home/lzx"); parser.add_argument("--max-family-peak-mib", type=int, required=True)
    args = parser.parse_args()
    if not args.start: raise ValueError("The formal dispatcher requires --start.")
    if os.name == "nt": raise RuntimeError("E1 workers may only start from Linux.")
    scheduler = SeedBlockScheduler(args.state)
    pending = []
    for run in expand_e1_formal_matrix(load_e1_governance_manifest(args.governance)):
        existing = scheduler.state["runs"].get(run.identity, {})
        if existing.get("status") == "complete":
            continue
        if existing.get("status") == "running" and _live_pid(existing.get("pid")):
            continue
        pending.append(run)
    running = {}; logs = Path(args.log_root); logs.mkdir(parents=True, exist_ok=True)
    while pending or running:
        while pending and len(running) < 4:
            run = pending.pop(0)
            free = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"], check=True, capture_output=True, text=True).stdout.strip())
            admit_slot_memory(free_memory_mib=free, max_family_peak_mib=args.max_family_peak_mib)
            slot = len(running); log = logs / ("e1_" + run.identity.replace(":", "_") + ".log")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
            with log.open("ab") as handle:
                process = subprocess.Popen(_command(args, run), stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=Path.cwd(), env=env, start_new_session=True)
            scheduler.mark(run.identity, "running", gpu=0, slot=slot, pid=process.pid, log_path=str(log), num_env_workers=16, rollout_length=128)
            write_heartbeat(Path(args.state).with_name(run.identity.replace(":", "_") + ".heartbeat.json"), run_identity=run.identity, pid=process.pid, gpu=0, free_memory_mib=free)
            running[run.identity] = (process, log)
            print(json.dumps({"event": "started", "run": run.identity, "pid": process.pid, "log": str(log)}), flush=True)
        finished = []
        for identity, (process, log) in running.items():
            code = process.poll()
            if code is None: continue
            scheduler.mark(identity, "complete" if code == 0 else "failed", exit_code=code, log_path=str(log))
            print(json.dumps({"event": "complete" if code == 0 else "failed", "run": identity, "exit_code": code, "log": str(log)}), flush=True)
            finished.append(identity)
        for identity in finished: del running[identity]
        if finished and any(scheduler.state["runs"][identity]["status"] == "failed" for identity in finished):
            raise RuntimeError("E1 dispatcher observed a failed member; inspect its log before continuing.")
        if running: time.sleep(15)
    print(json.dumps({"schema": "e1-complete-matrix-dispatch-v2", "matrix": scheduler.summary()}, indent=2))


if __name__ == "__main__": main()
