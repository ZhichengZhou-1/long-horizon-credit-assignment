# Long-horizon credit assignment by replay

Which decision in a long agent run actually caused the outcome? When a shopping agent fails, the last
action is often only the symptom of an earlier mistake (e.g. picking the wrong product several steps
before). This project studies **replay-based credit assignment**: return to the exact same
intermediate state, try an alternative action, let a fixed continuation policy finish the episode,
and compare final rewards. We first validate replay estimates as step-level credit on
[WebShop](https://github.com/princeton-nlp/WebShop), then compare them with cheaper credit methods
(HCAPO-style hindsight, uniform credit, heuristics), before using them to train a policy.

Status: Phase 1 (is there a decisive step?) and Phase 2 (which method finds it?) done on 600 goals.
Next: use replay credit in RL training.

## In short

- **The premise holds.** 108 of 130 failed runs can be rescued by changing a single action, and that
  action is almost never the last one (2 / 108). Most decisive mistakes are early: the wrong product
  (60) or the wrong search query (31).
- **Cheap credit methods do not find that step.** HCAPO-style hindsight scores and policy confidence
  are at chance level; a RUDDER-style learned return decomposition is only slightly above chance.
- **Replay does,** at ~2,000 continuation calls per failed run, or ~700 with adaptive sampling
  (one try per alternative, three more only for the ones that succeed) at the same accuracy.

## Setup in one paragraph

WebShop is the copy vendored in [verl-agent](https://github.com/langfengQ/verl-agent) (1,000 products,
synthetic goals), with the same prompt template, action parsing, 15-step limit and 2-step history as
verl-agent / HCAPO. The continuation policy is the public GiGPO checkpoint
[`langfeng01/GiGPO-Qwen2.5-7B-Instruct-WebShop`](https://huggingface.co/langfeng01/GiGPO-Qwen2.5-7B-Instruct-WebShop)
served by vLLM at temperature 1.0. A state s_t is restored by resetting the task and replaying the
logged actions a_0..a_{t-1}; the agent's prompt at s_t is rebuilt the same way. Every restore is
checked against a fingerprint of the logged prompt / observation.

## Results

Goals 0-599, one policy (GiGPO-7B, T=1.0). Raw reports: `results/goals_0_599/` (and the first,
goals 0-99 only version in `results/goals_0_99/`).

**Replay is exact.** Prefix replay reproduced every observation, admissible-action set, reward and
agent prompt in all checks (773 random-policy steps, 812 scripted branches, all 87k LLM branches of the main runs).

**Policies.** Success rate on goals 0-99 (strict verl-agent action parsing):

| policy | T=0.4 | T=1.0 | reference |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct (base) | 5.5% | 2.5% | HCAPO table 1: 5.2% |
| Qwen2.5-7B-Instruct (base) | 0% | 0.5% | HCAPO table 1: 7.8% (*) |
| GiGPO-7B (public checkpoint) | 77% | 78% | GiGPO 7B in HCAPO table 1: 72.8% |

GiGPO-7B at T=1.0 on all 600 goals: 79% (1,890 / 2,400 runs). (*) Base 7B writes `[action>` instead
of `<action>` in 60-70% of steps; with a lenient parser it still never succeeds.

**Data issue.** 38 of the 600 goals cannot be completed at all (no product page matches the goal's
options, best score < 1; `credit/goal_ceiling.py`). They are excluded from the analysis.

### What was run

```
600 goals x 4 runs                                   2,400 runs, 1,890 successful
-> one failed run per goal that has failures         130 failed runs (+ 8 successful ones as controls)
   (goals that cannot be completed are skipped)
-> every step of those runs is a decision point      1,240 decision points
-> at each one: the original action + up to 11       12,240 (step, action) pairs
   alternatives (clickable buttons, or 4 new search
   queries written by the policy)
-> each pair: take the action, let the policy        48,960 continuations, 261k LLM calls
   finish the episode, M=4 times; Q = mean score
```

regret(t) = best Q at step t - Q of the original action. A **flip step** is a step where one
alternative action succeeds in >= 75% of its continuations, i.e. changing that one action rescues
the run.

### Phase 1: is there a decisive step?

| on 130 failed runs | |
|---|---|
| has a flip step | 108 / 130 |
| earliest flip step is the last step | **2 / 108** |
| position of the earliest flip | early third 82, middle 17, late 9 |
| action type of the earliest flip | product choice 60, search query 31, option 8, other 9 |
| more than one flip step | 77 / 108 |
| successful controls: mean regret | 0.001 (expected ~0) |

### Phase 2: which step does each method blame?

Each method names one step per failed run. Ground truth is an **independent re-estimate**: the
original action, the best alternative and the flip alternative are rerun with a new seed and M=16
(38k continuations), so replay is not graded on its own samples. Runs: 123 failed runs of solvable
goals, 102 of them have a flip under the re-estimate.

| method | cost per run | earliest flip | any flip | rank corr. with true regret |
|---|---|---|---|---|
| random step (expected) | 0 | 14 / 102 | 26.5 / 102 | - |
| last step | 0 | 0 / 102 | 14 / 102 | -0.09 |
| first product click | 0 | 53 / 102 | 69 / 102 | - |
| HCAPO-style hindsight ratio (4 variants) | ~9 forward passes | 1-16 / 102 | 13-19 / 102 | -0.14 to 0.14 |
| policy's least confident step | ~9 forward passes | 1-9 / 102 | 14-18 / 102 | -0.08 to 0.14 |
| RUDDER-style return decomposition | 0 after training | 22 / 102 | 33 / 102 | 0.04 |
| replay, M=4 | ~2,000 LLM calls | 75 / 102 | 82 / 102 | 0.66 |
| replay, adaptive (1 try, 3 more on success) | ~700 LLM calls | 78 / 102 | 87 / 102 | 0.58 |
| replay, M=1 | ~500 LLM calls | 64 / 102 | 67 / 102 | 0.49 |

- **HCAPO-style:** the policy scores each step's action with and without a hindsight block
  ("the run succeeded / failed"); the step whose probability changes most is blamed. Our
  reimplementation of eqs. 6-7 of [arXiv:2603.08754](https://arxiv.org/abs/2603.08754); the paper's
  verification prompt is not public, so ours is used.
- **RUDDER-style** ([Arjona-Medina et al. 2019](https://arxiv.org/abs/1806.07857)): each step is
  embedded with the policy model (mean last hidden state of instruction + action + page), an LSTM
  predicts the final score after every step, and the step with the largest predicted drop is
  blamed. Trained on 2,590 runs, cross-fitted over goals (5 folds; no run is scored by a model that
  saw its goal). It does learn to predict the outcome (held-out MSE 0.038 vs 0.072 for the mean),
  but not where the outcome was decided.
- **Winner's curse:** picking the best of ~12 alternatives at M=4 overestimates. The re-estimate
  lowers mean regret from 0.278 to 0.219 (0.542 to 0.421 at steps that looked decisive).

### How cheap can replay be?

Simulated by subsampling the M=4 data (`results/goals_0_599/cheap_replay_sim.txt`), against the
full M=4 result on 108 runs with a flip:

| strategy | cost | finds earliest flip | finds any flip | false flips |
|---|---|---|---|---|
| all steps, all alternatives, M=4 | 100% | 108 | 108 | 0 |
| M=1 | 25% | 98 | 107 | 134 of 445 claimed |
| adaptive: 1 try, 3 more if it succeeds | 36% | 96 | 107 | 0 |
| only product and search steps, M=4 | 45% | 91 | 93 | 0 |
| adaptive + only product/search/option steps | 29% | 89 | 102 | 0 |

Adaptive sampling works because most alternatives fail on the first try and are not worth three
more; it never reports a false flip because every hit is confirmed, and it misses a flip only when
a good alternative fails its first try.

**Caveats.** One environment, one policy. The re-estimate only reruns alternatives the first replay
pass picked, which favours replay somewhat. HCAPO and RUDDER are reimplementations, and both papers
evaluate end-to-end training gains, not whether the blamed step is the right one. WebShop failures
are dominated by product choice, which is why the "first product click" rule is strong. Replay costs
far more compute than hindsight scoring; whether it pays off in training at equal compute is the
next question.

## Repository layout

```
credit/        core: WebShop env (webshop_env.py), agent prompt + action parsing (agent.py),
               batched LLM rollouts (rollout.py), replay branching (branch_llm.py),
               independent re-estimate (reeval.py), HCAPO-style scoring (hcapo_score.py),
               goal score ceilings (goal_ceiling.py), env-var config (config.py),
               RUDDER inputs: step texts (dump_steps.py) and step embeddings (step_features.py)
checks/        correctness checks without a GPU: prefix-replay determinism, scripted branching,
               restore of logged LLM episodes, a mock LLM server, a data walkthrough
analysis/      stdlib-only reports: premise_check.py (Phase 1), cheap_replay_sim.py,
               compare_credit.py (Phase 2), merge_runs.py (combine goal ranges);
               rudder.py (RUDDER-style baseline, needs torch)
cluster/       Slurm/cluster helpers: node-local conda envs, vLLM server start/stop, GPU hold job,
               watcher that runs a script as soon as a job starts, mail notifier, auto-cancel
experiments/   the scripts that produced the results, in order (see experiments/README.md)
results/       reports and run summaries: goals_0_599/ (current), goals_0_99/ (first version);
               raw runs are not in the repo
```

## Running

1. Install the two environments and data: [INSTALL.md](INSTALL.md).
2. `cp env.example.sh env.sh`, edit paths, `source env.sh`.
3. CPU checks (no GPU): `bash experiments/cpu_checks.sh`.
4. GPU experiments, inside an allocation (`srun --jobid=<ID> --overlap bash experiments/<script>`):
   `gigpo_eval.sh` -> `branch.sh` -> `analysis.sh` (writes the goal ceilings) -> `reeval_hcapo.sh`
   -> `analysis.sh` again. `base_models.sh` is the base-model baseline.
   More goals: `GOALS=100-349 bash experiments/scale_up.sh` runs the whole pipeline for one goal
   range on one GPU (~6-10 h on an A100 40 GB), then `experiments/scale_analysis.sh` (CPU) merges
   the ranges, trains the RUDDER-style baseline and writes all reports.
5. For unattended runs: `cluster/submit_hold.sh` for the allocation, `cluster/watch_and_run.sh` to
   start a script when it begins, `cluster/notify_when_done.sh` / `cluster/cancel_after_session.sh`.

## Acknowledgements

WebShop (Yao et al., 2022) via verl-agent (Feng et al., GiGPO), the GiGPO WebShop checkpoint, and
Qwen2.5. The WebShop data files were taken from the Hugging Face mirror `YWZBrandon/webshop-data`
because the original Google Drive links return 404 (hashes in INSTALL.md).
