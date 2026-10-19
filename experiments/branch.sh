#!/bin/bash
# Phase 1 ground truth (2026-10-18): replay-based Q_hat with GiGPO-7B as continuation policy (T=1.0).
# Defaults reproduce the full run: all 19 always-fail and 7 mixed tasks of goals 0-99 plus 4
# always-success controls, every decision step, <=12 candidates per click step, 4 policy samples for
# alternative queries, M=4 continuations per (t, a)  -> 9468 branches, ~2 h on one A100.
# Pilot: N_FAIL=4 N_MIXED=2 N_SUCCESS=2.
# Run inside a GPU job: srun --jobid=<ID> --overlap bash experiments/branch.sh
set -o pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
source "$REPO/cluster/local_envs.sh"
source "$REPO/cluster/vllm_server.sh"
MODEL=${MODEL:-langfeng01/GiGPO-Qwen2.5-7B-Instruct-WebShop}
SOURCE=${SOURCE:-$CREDIT_RUNS/gigpo_eval/gigpo7b_t1.0/episodes.jsonl}
RUN=${RUN:-$CREDIT_RUNS/branch_full}
mkdir -p "$RUN"
stage() { echo; echo "===== $(date "+%F %T") $*"; }
trap 'kill $VPID 2>/dev/null' EXIT

stage "node $(hostname), $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
start_server "$MODEL" "$RUN/vllm.log" || exit 1
stage "branching"
"$WS_PY" "$REPO/credit/branch_llm.py" --model "$MODEL" --base-url "$VLLM_URL" --source "$SOURCE" \
    --n-fail "${N_FAIL:-19}" --n-mixed "${N_MIXED:-7}" --n-success "${N_SUCCESS:-4}" --m "${M:-4}" \
    --max-cands 12 --search-samples 4 --temperature 1.0 --batch 128 --out "$RUN" \
    || stage "branching FAILED"
stop_server
stage "done"
