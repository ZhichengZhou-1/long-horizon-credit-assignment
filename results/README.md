# Results

`goals_0_599/` is the current version (all numbers in the main README); `goals_0_99/` is the first
version on goals 0-99 only, kept for reference. Paths inside the reports refer to `$CREDIT_RUNS`.

| file | produced by | content |
|---|---|---|
| `premise_report.txt` | `analysis/premise_check.py` | Phase 1: flip steps, their positions and action types, per failed run |
| `compare_credit.txt` | `analysis/compare_credit.py` | Phase 2: blame accuracy of each method vs the M=16 re-estimate |
| `cheap_replay_sim.txt` | `analysis/cheap_replay_sim.py` | cost vs recall of cheaper replay strategies (subsampled from the M=4 data) |
| `goals_0_599/rudder_fit.txt` | `analysis/rudder.py` | how well the RUDDER-style LSTM predicts the final score (held out) |
| `goals_0_99/goal_ceiling_0_99.txt` | `credit/goal_ceiling.py 0..99` | best achievable score per goal; 7 goals are below 1.0 |
| `*/summaries/*.json` | `credit/rollout.py`, `branch_llm.py`, `reeval.py`, `step_features.py`, checks | per-run summaries (success rate, cost, wall time) |

In `goals_0_99/` the full branching run was originally stored as `branch_llm_full/` (now
`branch_full/` in `experiments/branch.sh`).
