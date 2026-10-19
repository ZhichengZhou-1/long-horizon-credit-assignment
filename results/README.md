# Results

| file | produced by | content |
|---|---|---|
| `goal_ceiling_0_99.txt` | `credit/goal_ceiling.py 0..99` | best achievable score per goal; 7 goals are below 1.0 |
| `premise_report.txt` | `analysis/premise_check.py` on the full branching run | Phase 1: flip steps, positions, action types |
| `cheap_replay_sim.txt` | `analysis/cheap_replay_sim.py` | cost vs recall of cheaper replay strategies (same samples, optimistic) |
| `compare_credit.txt` | `analysis/compare_credit.py` | Phase 2 v1: blame accuracy of each method vs the M=16 ground truth |
| `summaries/*.json` | `credit/rollout.py`, `credit/branch_llm.py`, `credit/reeval.py`, checks | per-run summaries (success rate, cost, wall time) |

Paths inside the reports refer to `$CREDIT_RUNS`; the full branching run was originally stored as
`branch_llm_full/` (now `branch_full/` in `experiments/branch.sh`).
