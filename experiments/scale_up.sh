#!/bin/bash
# Phase 1/2 on more goals: the whole pipeline for one goal range on one GPU, same settings as the
# goals 0-99 runs. Writes the same layout as those runs, so analysis/merge_runs.py can combine them.
#   1. GiGPO-7B rollouts, T=1.0, 4 per goal                       gigpo_eval/gigpo7b_t1.0
#   2. replay branching on every failed / mixed task of a solvable goal (+2 success controls), M=4
#                                                                  branch_full
#   3. HCAPO-style hindsight scores, action and response spans    hcapo_action, hcapo_response
#   4. RUDDER inputs: step texts (CPU, during 2) and step embeddings  rudder/
#   5. independent M=16 re-estimate (ground truth)               reeval16
# Stages whose output exists are skipped, so a cut-off job can be resumed.
# Needs $RUN/goal_ceiling.txt (credit/goal_ceiling.py on the goal range; CPU, run it beforehand).
# Run inside a GPU job: GOALS=100-349 srun --jobid=<ID> --overlap bash experiments/scale_up.sh
#   EXTRA_EPISODES: more episodes.jsonl files to embed for RUDDER (e.g. the goals 0-99 rollouts)
set -o pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
source "$REPO/cluster/local_envs.sh"
source "$REPO/cluster/vllm_server.sh"
: "${GOALS:?e.g. GOALS=100-349}"
MODEL=${MODEL:-langfeng01/GiGPO-Qwen2.5-7B-Instruct-WebShop}
RUN=${RUN:-$CREDIT_RUNS/scale_${GOALS/-/_}}
SRC=$RUN/gigpo_eval/gigpo7b_t1.0
mkdir -p "$SRC" "$RUN/branch_full" "$RUN/hcapo_action" "$RUN/hcapo_response" "$RUN/rudder" "$RUN/reeval16"
stage() { echo; echo "===== $(date "+%F %T") $*"; }
done_() { [ -s "$1/summary.json" ] || [ -s "$1/hcapo_scores.jsonl" ] || [ -s "$1/feats.npy" ]; }
trap 'kill $VPID $DPID 2>/dev/null' EXIT
[ -s "$RUN/goal_ceiling.txt" ] || { echo "missing $RUN/goal_ceiling.txt"; exit 1; }

stage "node $(hostname), $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader), goals $GOALS"
if ! done_ "$SRC" || ! done_ "$RUN/branch_full"; then
    start_server "$MODEL" "$RUN/vllm_generate.log" 0.90 || exit 1
    if ! done_ "$SRC"; then
        stage "1. rollouts"
        "$WS_PY" "$REPO/credit/rollout.py" --model "$MODEL" --base-url "$VLLM_URL" --out "$SRC" \
            --tasks "$GOALS" --n-per-task 4 --batch 64 --temperature 1.0 | tail -25
        "$WS_PY" "$REPO/checks/verify_rollout.py" "$SRC" | tail -1
    fi
    if [ ! -s "$RUN/rudder/steps.jsonl" ]; then  # CPU only; overlaps with branching
        "$WS_PY" "$REPO/credit/dump_steps.py" --out "$RUN/rudder/steps.jsonl" "$SRC/episodes.jsonl" \
            $EXTRA_EPISODES > "$RUN/rudder/dump.log" 2>&1 &
        DPID=$!
    fi
    if ! done_ "$RUN/branch_full"; then
        stage "2. branching"
        "$WS_PY" "$REPO/credit/branch_llm.py" --model "$MODEL" --base-url "$VLLM_URL" --source "$SRC/episodes.jsonl" \
            --ceiling "$RUN/goal_ceiling.txt" --n-fail 999 --n-mixed 999 --n-success 2 --m 4 \
            --max-cands 12 --search-samples 4 --temperature 1.0 --batch 128 --out "$RUN/branch_full" \
            | grep -v '^  idx' || stage "branching FAILED"
    fi
    stop_server
fi

if ! done_ "$RUN/hcapo_response"; then
    if start_server "$MODEL" "$RUN/vllm_score.log" 0.60; then
        for span in action response; do
            done_ "$RUN/hcapo_$span" && continue
            stage "3. HCAPO-style scoring, span=$span"
            "$WS_PY" "$REPO/credit/hcapo_score.py" --model "$MODEL" --base-url "$VLLM_URL" --source "$SRC/episodes.jsonl" \
                --plan "$RUN/branch_full/plan.json" --workers 2 --span $span --out "$RUN/hcapo_$span" \
                | tail -3 || stage "HCAPO scoring ($span) FAILED"
        done
        stop_server
    fi
fi

if ! done_ "$RUN/rudder"; then
    stage "4. step embeddings for RUDDER"
    [ -n "$DPID" ] && wait $DPID
    tail -1 "$RUN/rudder/dump.log"
    "$VLLM_PY" "$REPO/credit/step_features.py" --model "$MODEL" --steps "$RUN/rudder/steps.jsonl" \
        --out "$RUN/rudder/feats" | tail -2 || stage "embedding FAILED"
fi

if ! done_ "$RUN/reeval16"; then
    stage "5. M=16 re-estimate"
    if start_server "$MODEL" "$RUN/vllm_reeval.log" 0.90; then
        "$WS_PY" "$REPO/credit/reeval.py" --model "$MODEL" --base-url "$VLLM_URL" --run "$RUN/branch_full" \
            --source "$SRC/episodes.jsonl" --ceiling "$RUN/goal_ceiling.txt" --m 16 --seed 1 --batch 128 \
            --out "$RUN/reeval16" | tail -14 || stage "re-estimate FAILED"
        stop_server
    fi
fi
stage "done"
