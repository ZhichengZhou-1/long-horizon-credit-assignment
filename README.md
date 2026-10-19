# Long-horizon credit assignment by replay

Which decision in a long agent run actually caused the outcome? When a shopping agent fails, the last
action is often only the symptom of an earlier mistake (e.g. picking the wrong product several steps
before). This project studies **replay-based credit assignment**: return to the exact same
intermediate state, try an alternative action, let a fixed continuation policy finish the episode,
and compare final rewards. We first validate replay estimates as step-level credit on
[WebShop](https://github.com/princeton-nlp/WebShop), then compare them with cheaper credit methods
(HCAPO-style hindsight, uniform credit, heuristics), before using them to train a policy.

Status: Phase 1 (ground truth + premise check) done; Phase 2 (estimator comparison) first version.

## Setup in one paragraph

WebShop is the copy vendored in [verl-agent](https://github.com/langfengQ/verl-agent) (1,000 products,
synthetic goals), with the same prompt template, action parsing, 15-step limit and 2-step history as
verl-agent / HCAPO. The continuation policy is the public GiGPO checkpoint
[`langfeng01/GiGPO-Qwen2.5-7B-Instruct-WebShop`](https://huggingface.co/langfeng01/GiGPO-Qwen2.5-7B-Instruct-WebShop)
served by vLLM at temperature 1.0. A state s_t is restored by resetting the task and replaying the
logged actions a_0..a_{t-1}; the agent's prompt at s_t is rebuilt the same way. Every restore is
checked against a fingerprint of the logged prompt / observation.

## Results so far

All numbers are from goals 0-99 and one policy; see `results/` for the raw reports.

**Replay is exact.** Prefix replay reproduced every observation, admissible-action set, reward and
agent prompt in all checks (773 random-policy steps under three conditions, 812 scripted branches,
>13k LLM branches): no mismatch.

**Policies.** Success rate on goals 0-99 (strict verl-agent action parsing):

| policy | T=0.4 | T=1.0 | reference |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct (base) | 5.5% | 2.5% | HCAPO table 1: 5.2% |
| Qwen2.5-7B-Instruct (base) | 0% | 0.5% | HCAPO table 1: 7.8% (*) |
| GiGPO-7B (public checkpoint) | 77% | 78% (83.9% on solvable goals) | GiGPO 7B in HCAPO table 1: 72.8% |

(*) Base 7B writes `[action>` instead of `<action>` in 60-70% of steps. With a lenient parser
(`--lenient-action`) its actions execute, but success stays at 0%: it keeps going back to search and
paging, and buys in only 7.5% of episodes.

**WebShop data issue.** 7 of goals 0-99 cannot be completed at all: even the target product with the
goal options scores 0.857 / 0.833 because the goal's option names do not match the product page
(`credit/goal_ceiling.py`, `results/goal_ceiling_0_99.txt`). They are excluded below.

**Phase 1: where are the decisive mistakes?** Replay at every step of 30 trajectories: 230 decision
points; candidates are the clickable actions (at most 12 incl. the original) or the original query
plus up to 4 sampled queries; M=4 continuations each; 9,468 branches, 48k LLM calls. On the 19 failed
trajectories of solvable goals:

| | |
|---|---|
| a single changed action turns the failure into >=75% success ("flip") | 15 / 19 |
| the earliest flip step is the last step | **0 / 15** |
| position of the earliest flip | early third 12, middle 3, late 0 |
| action type of the earliest flip | product choice 8, search query 4, option 2, malformed action 1 |
| same mistake repeated (>=2 flip steps) | 10 / 15 |

Blaming the final action would be wrong in every case here.

**Phase 2 v1: which step does each method blame?** Ground truth = an independent M=16 re-estimate
(new seed) of the original / best / flip candidates (5,776 branches). 15 trajectories have a flip.

| method | cost per trajectory | earliest flip found | any flip found | Spearman with true regret |
|---|---|---|---|---|
| chance | 0 | 2.2 / 15 | 4.4 / 15 | - |
| blame the last step | 0 | 0 / 15 | 2 / 15 | -0.09 |
| blame the first product click | 0 | 9 / 15 | 13 / 15 | - |
| HCAPO-style hindsight (4 variants) | ~9 forward passes | 0-3 / 15 | 0-4 / 15 | -0.09 to 0.09 |
| replay, M=4, all candidates | ~2,037 LLM calls | 14 / 15 | 15 / 15 | 0.73 |
| replay, adaptive (1 sample, confirm hits with 3) | ~665 LLM calls | 13 / 15 | 15 / 15 | 0.71 |
| replay, M=1 | ~508 LLM calls | 11 / 15 | 13 / 15 | 0.63 |

Caveats: 19 trajectories, one environment, one policy. HCAPO is our reimplementation of eqs. 6-7 of
[arXiv:2603.08754](https://arxiv.org/abs/2603.08754) (T_temp=5, clip [0.8, 1.2]); the paper does not
publish its verification prompt, so the hindsight block is ours, and in HCAPO's own objective the
hindsight term is multiplied by a zero return on failed trajectories anyway. With GiGPO-7B the
hindsight block barely changes action log-probabilities. The first-product heuristic is strong
because WebShop failures are dominated by product choice. Replay costs 50-200x more compute than
hindsight scoring; whether that pays off in training at equal compute is the next question.

## Repository layout

```
credit/        core: WebShop env (webshop_env.py), agent prompt + action parsing (agent.py),
               batched LLM rollouts (rollout.py), replay branching (branch_llm.py),
               independent re-estimate (reeval.py), HCAPO-style scoring (hcapo_score.py),
               goal score ceilings (goal_ceiling.py), env-var config (config.py)
checks/        correctness checks without a GPU: prefix-replay determinism, scripted branching,
               restore of logged LLM episodes, a mock LLM server, a data walkthrough
analysis/      stdlib-only reports: premise_check.py (Phase 1), cheap_replay_sim.py,
               compare_credit.py (Phase 2)
cluster/       Slurm/cluster helpers: node-local conda envs, vLLM server start/stop, GPU hold job,
               watcher that runs a script as soon as a job starts, mail notifier, auto-cancel
experiments/   the scripts that produced the results, in order (see experiments/README.md)
results/       small reports and run summaries (raw runs are not in the repo)
```

## Running

1. Install the two environments and data: [INSTALL.md](INSTALL.md).
2. `cp env.example.sh env.sh`, edit paths, `source env.sh`.
3. CPU checks (no GPU): `bash experiments/cpu_checks.sh`.
4. GPU experiments, inside an allocation (`srun --jobid=<ID> --overlap bash experiments/<script>`):
   `gigpo_eval.sh` -> `branch.sh` -> `analysis.sh` (writes the goal ceilings) -> `reeval_hcapo.sh`
   -> `analysis.sh` again. `base_models.sh` is the base-model baseline.
5. For unattended runs: `cluster/submit_hold.sh` for the allocation, `cluster/watch_and_run.sh` to
   start a script when it begins, `cluster/notify_when_done.sh` / `cluster/cancel_after_session.sh`.

## Acknowledgements

WebShop (Yao et al., 2022) via verl-agent (Feng et al., GiGPO), the GiGPO WebShop checkpoint, and
Qwen2.5. The WebShop data files were taken from the Hugging Face mirror `YWZBrandon/webshop-data`
because the original Google Drive links return 404 (hashes in INSTALL.md).
