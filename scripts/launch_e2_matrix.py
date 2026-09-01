"""Generate a bounded, four-slot E2 launch plan; does not start training itself."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from llm_mappo.e1_protocol import expand_e1_formal_matrix, load_e1_governance_manifest
from llm_mappo.e1_scheduler import SeedBlockScheduler
from llm_mappo.e1_scheduler import admit_slot_memory, write_heartbeat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--governance", default="configs/g3_experiment_manifest.yaml")
    parser.add_argument("--state", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--python-bin", default="/home/lzx/.conda/envs/llm-a-mappo-py310/bin/python")
    parser.add_argument("--log-root", default="/home/lzx")
    parser.add_argument("--max-family-peak-mib", type=int)
    args = parser.parse_args()
    if not 1 <= args.limit <= 4: raise ValueError("E2 formal launcher limit must be in [1, 4].")
    runs = expand_e1_formal_matrix(load_e1_governance_manifest(args.governance))
    scheduler = SeedBlockScheduler(args.state); planned = []
    for run in runs:
        if len(planned) >= args.limit: break
        state = scheduler.state["runs"].get(run.identity, {})
        if state.get("status") in {"complete", "running", "failed"}: continue
        gpu = scheduler.assign(seed=run.seed, available_gpus=(0, 1))
        same_gpu = sum(item["gpu"] == gpu for item in planned)
        if same_gpu >= 2: continue
        planned.append({"run_identity": run.identity, "gpu": gpu, "slot": same_gpu,
            "command": ["scripts/run_e1_training.py", "--governance", args.governance,
                "--records", args.records, "--run", run.identity.replace(":seed", ":"),
                "--output-root", args.output_root, "--device", "cuda:0"]})
    if args.start:
        if os.name == "nt": raise RuntimeError("E2 workers may only start from Linux.")
        if args.max_family_peak_mib is None: raise ValueError("--start requires frozen --max-family-peak-mib.")
        logs = Path(args.log_root); logs.mkdir(parents=True, exist_ok=True)
        for worker in planned:
            free = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(worker["gpu"])], check=True, capture_output=True, text=True).stdout.strip())
            worker["m_slot_mib"] = admit_slot_memory(free_memory_mib=free, max_family_peak_mib=args.max_family_peak_mib)
            log = logs / ("e2_" + worker["run_identity"].replace(":", "_") + ".log")
            command = [args.python_bin, *worker["command"]]
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(worker["gpu"])}
            with log.open("ab") as handle:
                process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, cwd=Path.cwd(), env=environment,
                    start_new_session=True)
            scheduler.mark(worker["run_identity"], "running", gpu=worker["gpu"],
                           slot=worker["slot"], pid=process.pid, log_path=str(log))
            worker.update({"pid": process.pid, "log_path": str(log)})
            write_heartbeat(Path(args.state).with_name(worker["run_identity"].replace(":", "_") + ".heartbeat.json"), run_identity=worker["run_identity"], pid=process.pid, gpu=worker["gpu"], free_memory_mib=free)
    print(json.dumps({"schema": "e2-launch-plan-v1", "max_workers": args.limit,
                      "started": args.start, "workers": planned, "matrix": scheduler.summary()}, indent=2))


if __name__ == "__main__": main()
