#!/bin/bash
# Base-model success rate (2026-10-18, A100 bf16): Qwen2.5-1.5B / 7B-Instruct on goals 0-99.
#   T=0.4: HCAPO's validation temperature (compare with its table 1: 1.5B 5.2%, 7B 7.8%)
#   T=1.0: HCAPO's rollout temperature
# Set LENIENT=1 to also accept "[action>" / "[action]" (Qwen2.5-7B writes these in ~60-70% of steps).
# Run inside a GPU job: srun --jobid=<ID> --overlap bash experiments/base_models.sh
set -o pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
source "$REPO/cluster/local_envs.sh"
source "$REPO/cluster/vllm_server.sh"
RUN=${RUN:-$CREDIT_RUNS/base_models}
mkdir -p "$RUN"
stage() { echo; echo "===== $(date "+%F %T") $*"; }
EXTRA=(); [ -n "${LENIENT:-}" ] && EXTRA=(--lenient-action)
trap 'kill $VPID 2>/dev/null' EXIT

stage "node $(hostname), $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
for M in ${MODELS:-1.5B 7B}; do
    MODEL=Qwen/Qwen2.5-$M-Instruct
    start_server "$MODEL" "$RUN/vllm_${M,,}.log" || continue
    for spec in "0.4 2" "1.0 4"; do
        set -- $spec
        OUT="$RUN/${M,,}_t$1"
        stage "$M, temperature $1, goals 0-99 x $2"
        "$WS_PY" "$REPO/credit/rollout.py" --model "$MODEL" --base-url "$VLLM_URL" --out "$OUT" \
            --tasks 0-99 --n-per-task "$2" --batch 64 --temperature "$1" "${EXTRA[@]}" | tail -20
        "$WS_PY" "$REPO/checks/verify_rollout.py" "$OUT" | tail -1 &
    done
    stop_server
done
wait
stage "done"
