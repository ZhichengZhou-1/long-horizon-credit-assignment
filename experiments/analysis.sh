#!/bin/bash
# CPU analysis of the GPU runs (any node, after `source env.sh`):
#   goal_ceiling     which goals cannot be completed at all (7 of goals 0-99 in the small WebShop set)
#   premise_check    Phase 1: where are the decisive steps of failed trajectories?
#   cheap_replay_sim how much cheaper can replay be (subsamples the same branches, so optimistic)
#   compare_credit   Phase 2 v1: which step does each credit method blame, vs the M=16 ground truth
set -o pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
: "${CREDIT_RUNS:?}"
PY=${PY:-python}
SOURCE=${SOURCE:-$CREDIT_RUNS/gigpo_eval/gigpo7b_t1.0/episodes.jsonl}
BRANCH_RUN=${BRANCH_RUN:-$CREDIT_RUNS/branch_full}
CEIL=$CREDIT_RUNS/goal_ceiling_0_99.txt

[ -s "$CEIL" ] || $PY "$REPO/credit/goal_ceiling.py" $(seq 0 99) | grep '^idx' > "$CEIL"
python3 "$REPO/analysis/premise_check.py" "$BRANCH_RUN" "$SOURCE" > "$BRANCH_RUN/premise_report.txt"
python3 "$REPO/analysis/cheap_replay_sim.py" "$BRANCH_RUN" "$SOURCE" "$CEIL" > "$BRANCH_RUN/cheap_replay_sim.txt"
if [ -s "$CREDIT_RUNS/reeval16/branches.jsonl" ]; then
    python3 "$REPO/analysis/compare_credit.py" "$CREDIT_RUNS/reeval16" "$BRANCH_RUN" \
        "$CREDIT_RUNS/hcapo_action/hcapo_scores.jsonl" "$CREDIT_RUNS/hcapo_response/hcapo_scores.jsonl" \
        "$SOURCE" > "$CREDIT_RUNS/compare_credit.txt"
fi
ls -la "$BRANCH_RUN"/premise_report.txt "$BRANCH_RUN"/cheap_replay_sim.txt "$CREDIT_RUNS"/compare_credit.txt 2>/dev/null
