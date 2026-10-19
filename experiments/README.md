# Experiments

In the order they were run (A100 40 GB, bf16, vLLM 0.8.5.post1 unless noted). Run directory names
below are the defaults of the scripts in this folder; `results/` keeps the summaries.

| date | script | what | output | result |
|---|---|---|---|---|
| 2026-10-05 | `cpu_checks.sh` (step 1-2) | prefix-replay determinism, scripted branching (CPU) | `cpu_checks/` | 773 steps / 812 branches, all identical |
| 2026-10-18 | `base_models.sh` | Qwen2.5-1.5B / 7B base, goals 0-99, T 0.4 x2 and 1.0 x4 | `base_models/` | 1.5B 5.5% / 2.5%; 7B 0% / 0.5% |
| 2026-10-18 | `base_models.sh` with `MODELS=7B LENIENT=1` | 7B with lenient action parsing | | still 0% (loops back to search; buys in 7.5%) |
| 2026-10-18 | `gigpo_eval.sh` | GiGPO-7B, goals 0-99, T 0.4 x2 and 1.0 x4 | `gigpo_eval/` | 77% / 78%; per task 19 always fail, 7 mixed, 74 always succeed |
| 2026-10-18 | `branch.sh` (`N_FAIL=4 N_MIXED=2 N_SUCCESS=2`) | branching pilot | `branch_pilot/` | 3,036 branches, 41 min |
| 2026-10-18 | `branch.sh` | full ground truth, M=4 | `branch_full/` | 9,468 branches, 2 h |
| 2026-10-19 | `analysis.sh` | goal ceilings, premise report, cheap-replay simulation | `results/` | 7 unsolvable goals; 0/15 decisive steps are last |
| 2026-10-19 | `reeval_hcapo.sh` | HCAPO-style scores (2 spans) + independent M=16 re-estimate | `hcapo_action/`, `hcapo_response/`, `reeval16/` | 5,776 branches, 80 min |
| 2026-10-19 | `analysis.sh` | Phase 2 v1 comparison | `compare_credit.txt` | see README |

Notes
- The first base-model run (2026-10-17) used vLLM's default sampling, which applies Qwen's
  `generation_config.json` (repetition penalty 1.05, top_k 20); it was discarded.
- An earlier V100 session failed because importing vLLM from the network file system took >30 min;
  this led to `cluster/local_envs.sh`.
