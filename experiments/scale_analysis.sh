#!/bin/bash
# CPU analysis after experiments/scale_up.sh: merge the goal ranges, train the RUDDER-style baseline
# (cross-fitted over goals), and rerun the Phase 1 / Phase 2 reports on all of them.
# Usage (login node, after `source env.sh`): bash experiments/scale_analysis.sh [RANGE ...]
#   default ranges: 0_99 100_349 350_599 (directories $CREDIT_RUNS/scale_<range>)
set -o pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
source "$REPO/cluster/local_envs.sh"  # VLLM_PY (torch) for rudder.py
RANGES=${*:-0_99 100_349 350_599}
OUT=$CREDIT_RUNS/merged_${RANGES// /+}
mkdir -p "$OUT"
stage() { echo; echo "===== $(date "+%F %T") $*"; }

stage "merge $RANGES"
python3 "$REPO/analysis/merge_runs.py" "$OUT" $(for r in $RANGES; do echo "$CREDIT_RUNS/scale_$r"; done) || exit 1

stage "RUDDER-style baseline"
DATA=()
for r in $RANGES; do
    d=$CREDIT_RUNS/scale_$r/rudder
    [ -s "$d/feats.npy" ] && DATA+=(--data "$d/steps.jsonl:$d/feats.npy")
done
RUDDER=
if [ ${#DATA[@]} -gt 0 ]; then
    "$VLLM_PY" "$REPO/analysis/rudder.py" "${DATA[@]}" --out "$OUT/rudder.jsonl" | tail -2 && RUDDER=$OUT/rudder.jsonl
fi

stage "reports"
SRC=$OUT/gigpo_eval/gigpo7b_t1.0/episodes.jsonl
python3 "$REPO/analysis/premise_check.py" "$OUT/branch_full" "$SRC" > "$OUT/premise_report.txt"
python3 "$REPO/analysis/cheap_replay_sim.py" "$OUT/branch_full" "$SRC" "$OUT/goal_ceiling.txt" > "$OUT/cheap_replay_sim.txt"
python3 "$REPO/analysis/compare_credit.py" "$OUT/reeval16" "$OUT/branch_full" \
    "$OUT/hcapo_action/hcapo_scores.jsonl" "$OUT/hcapo_response/hcapo_scores.jsonl" "$SRC" $RUDDER \
    > "$OUT/compare_credit.txt"
head -16 "$OUT/premise_report.txt"
echo
cat "$OUT/compare_credit.txt"
stage "done"
