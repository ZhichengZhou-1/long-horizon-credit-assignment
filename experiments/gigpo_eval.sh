#!/bin/bash
# Continuation policy (2026-10-18): langfeng01/GiGPO-Qwen2.5-7B-Instruct-WebShop, trained by the GiGPO
# authors with the same verl-agent prompt, on goals 0-99, strict verl-agent action parsing.
#   T=0.4: compare with the GiGPO 7B row quoted in HCAPO table 1 (72.8% success, score 84.4)
#   T=1.0: the rollouts whose failures are analysed (source trajectories for branching)
# Run inside a GPU job: srun --jobid=<ID> --overlap bash experiments/gigpo_eval.sh
set -o pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
source "$REPO/cluster/local_envs.sh"
source "$REPO/cluster/vllm_server.sh"
MODEL=${MODEL:-langfeng01/GiGPO-Qwen2.5-7B-Instruct-WebShop}
RUN=${RUN:-$CREDIT_RUNS/gigpo_eval}
mkdir -p "$RUN"
stage() { echo; echo "===== $(date "+%F %T") $*"; }
trap 'kill $VPID 2>/dev/null' EXIT

stage "node $(hostname), $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
start_server "$MODEL" "$RUN/vllm.log" || exit 1
for spec in "0.4 2" "1.0 4"; do
    set -- $spec
    OUT="$RUN/gigpo7b_t$1"
    stage "GiGPO-7B, temperature $1, goals ${TASKS:-0-99} x $2"
    "$WS_PY" "$REPO/credit/rollout.py" --model "$MODEL" --base-url "$VLLM_URL" --out "$OUT" \
        --tasks "${TASKS:-0-99}" --n-per-task "$2" --batch 64 --temperature "$1" | tail -25
    "$WS_PY" "$REPO/checks/verify_rollout.py" "$OUT" | tail -1 &
done
stop_server
wait
stage "done"
