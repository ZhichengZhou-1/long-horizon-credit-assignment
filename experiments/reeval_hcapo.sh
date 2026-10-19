#!/bin/bash
# Phase 2 inputs (2026-10-19), GiGPO-7B:
#   1. HCAPO-style hindsight scores for every step of the branching run, two spans (the <action> text
#      given the reasoning, and the whole response). prompt_logprobs needs memory headroom, so this
#      uses a separate server at 0.60 utilization and 2 concurrent requests.
#   2. Independent M=16 re-estimate (new seed) of the original / best-Q / best-success candidates on
#      the failed trajectories of solvable goals (the ground truth for the comparison).
# Needs $CREDIT_RUNS/goal_ceiling_0_99.txt from credit/goal_ceiling.py (see experiments/analysis.sh).
# Run inside a GPU job: srun --jobid=<ID> --overlap bash experiments/reeval_hcapo.sh
set -o pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
source "$REPO/cluster/local_envs.sh"
source "$REPO/cluster/vllm_server.sh"
MODEL=${MODEL:-langfeng01/GiGPO-Qwen2.5-7B-Instruct-WebShop}
SOURCE=${SOURCE:-$CREDIT_RUNS/gigpo_eval/gigpo7b_t1.0/episodes.jsonl}
BRANCH_RUN=${BRANCH_RUN:-$CREDIT_RUNS/branch_full}
stage() { echo; echo "===== $(date "+%F %T") $*"; }
trap 'kill $VPID 2>/dev/null' EXIT
mkdir -p "$CREDIT_RUNS/hcapo_action" "$CREDIT_RUNS/hcapo_response" "$CREDIT_RUNS/reeval16"

stage "node $(hostname), $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
if start_server "$MODEL" "$CREDIT_RUNS/hcapo_action/vllm.log" 0.60; then
    for span in action response; do
        stage "HCAPO-style hindsight scoring, span=$span"
        "$WS_PY" "$REPO/credit/hcapo_score.py" --model "$MODEL" --base-url "$VLLM_URL" --source "$SOURCE" \
            --plan "$BRANCH_RUN/plan.json" --workers 2 --span $span --out "$CREDIT_RUNS/hcapo_$span" \
            || stage "HCAPO scoring ($span) FAILED"
    done
    stop_server
fi

stage "M=16 re-estimate"
if start_server "$MODEL" "$CREDIT_RUNS/reeval16/vllm.log" 0.90; then
    "$WS_PY" "$REPO/credit/reeval.py" --model "$MODEL" --base-url "$VLLM_URL" --run "$BRANCH_RUN" \
        --source "$SOURCE" --ceiling "$CREDIT_RUNS/goal_ceiling_0_99.txt" --m 16 --seed 1 --batch 128 \
        --out "$CREDIT_RUNS/reeval16" || stage "re-estimate FAILED"
    stop_server
fi
stage "done"
