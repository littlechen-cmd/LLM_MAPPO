#!/usr/bin/env bash
set -euo pipefail

cd /home/lzx/llm-a-mappo || exit 1
PYTHON_BIN=/home/lzx/.conda/envs/llm-a-mappo-py310/bin/python
ARTIFACT_ROOT=/home/lzx/llm-a-mappo/artifacts/optimization/e1_cuda_smoke
RECORDS="${E1_RAW_RECORDS:-/home/lzx/llm-a-mappo/artifacts/optimization/e1_semantic_labels/formal/formal_pro_v5_20260901T072610Z/records.jsonl}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="$ARTIFACT_ROOT/$RUN_ID"
LOG_ROOT=/home/lzx
MAX_FAMILY_PEAK_MIB=4096
M_SLOT_MIB=7168
mkdir -p "$RUN_ROOT"

test -f "$RECORDS" || { echo "Missing raw-label records: $RECORDS"; exit 1; }
"$PYTHON_BIN" scripts/check_optimization_server.py --config configs/optimization/p1_linux_server.yaml --once --output "$RUN_ROOT/preflight"

admit_gpu() {
  local gpu="$1"
  local free
  free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
  if [ "$free" -lt $((4 * M_SLOT_MIB)) ]; then
    echo "GPU $gpu has ${free}MiB free; requires $((4 * M_SLOT_MIB))MiB for four E1 smoke slots."
    exit 2
  fi
}

launch_member() {
  local gpu="$1" group="$2" seed="$3"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/run_e1_smoke_member.py \
    --records "$RECORDS" --group "$group" --seed "$seed" --physical-gpu "$gpu" \
    --output-root "$RUN_ROOT" --device cuda:0 \
    >"$LOG_ROOT/e1_smoke_${RUN_ID}_${group//+/_}_seed${seed}.log" 2>&1 &
  LAST_PID=$!
}

run_wave() {
  admit_gpu 0
  local p1 p2 p3 p4
  launch_member 0 "$1" "$2"; p1="$LAST_PID"
  launch_member 0 "$3" "$4"; p2="$LAST_PID"
  launch_member 0 "$5" "$6"; p3="$LAST_PID"
  launch_member 0 "$7" "$8"; p4="$LAST_PID"
  wait "$p1"; wait "$p2"; wait "$p3"; wait "$p4"
}

run_wave MAPPO-DG 9001 Fixed-AStarKD+LLMKD 9002 RuleKD-v3 9003 NoOOD-v1 9004
run_wave RC-AStarKD+LLMKD 9001 QMIX-DG 9002 ShuffleKD-v3 9003 NoGoalHint-v1 9004
"$PYTHON_BIN" scripts/aggregate_e1_cuda_smoke.py --root "$RUN_ROOT" --output "$RUN_ROOT/e1_cuda_smoke_aggregate.json"
echo "E1 CUDA smoke completed: $RUN_ROOT"
