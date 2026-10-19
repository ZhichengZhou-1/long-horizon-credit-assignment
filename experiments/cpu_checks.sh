#!/bin/bash
# CPU-only correctness checks (no GPU, no real LLM); run on a login node after `source env.sh`.
#   1. prefix replay restores WebShop states exactly (same env / fresh env / new process)
#   2. branching (restore s_t, take a', continue) is reproducible and leaks no state
#   3. LLM rollouts against a mock server, then restore every logged step (env + prompt)
#   4. a tiny replay-branching run with the mock policy
set -o pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
: "${CREDIT_RUNS:?}"
OUT=$CREDIT_RUNS/cpu_checks
mkdir -p "$OUT"
PY=${PY:-python}
stage() { echo; echo "===== $(date "+%F %T") $*"; }

stage "1. determinism of prefix replay (20 tasks)"
$PY "$REPO/checks/determinism_check.py" --n-tasks 20 --out "$OUT/determinism" | tail -1

stage "2. branching with scripted policies (3 trajectories)"
$PY "$REPO/checks/branch_check.py" --trajs "$OUT/determinism/trajectories.jsonl" --n-trajs 3 --max-alts 2 \
    --demo-idx 0 --demo-seeds 2 --out "$OUT/branch" | tail -1

stage "3. mock-LLM rollouts + restore check"
PORT=${MOCK_PORT:-8017}
$PY "$REPO/checks/mock_llm.py" --port $PORT > /dev/null 2>&1 &
MPID=$!
trap 'kill $MPID 2>/dev/null' EXIT
sleep 2
$PY "$REPO/credit/rollout.py" --model mock --base-url http://127.0.0.1:$PORT/v1 --tasks 0-5 --n-per-task 2 \
    --batch 8 --out "$OUT/mock_rollout" | grep -E '"episodes"|success_rate'
$PY "$REPO/checks/verify_rollout.py" "$OUT/mock_rollout" | tail -1

stage "4. replay branching with the mock policy"
$PY "$REPO/credit/branch_llm.py" --model mock --base-url http://127.0.0.1:$PORT/v1 \
    --source "$OUT/mock_rollout/episodes.jsonl" --n-fail 1 --n-mixed 0 --n-success 0 --m 1 --max-cands 3 \
    --search-samples 1 --batch 8 --out "$OUT/mock_branch" | grep -E "decision points|state_ok_branches"
stage "done"
