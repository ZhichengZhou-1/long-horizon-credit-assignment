"""Replay-based Q estimates with an LLM continuation policy (Phase 1 ground truth).

For a source trajectory of task idx and each decision step t:
  restore s_t (reset + replay actions[:t], rebuilding the agent prompt state)
  -> take candidate action a (the original a_t or an alternative admissible at s_t)
  -> let the LLM policy continue until `buy now` or the 15-step episode limit
  -> final score / success.
Repeating M times per (t, a) gives Q_hat(s_t, a). Every restore is checked against the
observation hash logged in the source trajectory (state_ok).

Candidates at s_t: on click pages every admissible click (optionally capped); on the search
page the original query plus distinct queries the policy itself samples at s_t.

Usage (vLLM server already running):
  python branch_llm.py --model M --source runs/a100_gigpo/gigpo7b_t1.0/episodes.jsonl \
      --n-fail 4 --n-mixed 2 --n-success 2 --m 4 --out runs/branch_llm_pilot
"""
import argparse
import collections
import concurrent.futures as cf
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agent import MAX_STEPS, AgentState, project, project_lenient  # noqa: E402
from webshop_env import h, make_env  # noqa: E402
from rollout import call_llm, stable_seed  # noqa: E402
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv  # noqa: E402

NAV = ('back to search', 'next >', '< prev', 'buy now')


def select_sources(episodes, n_fail, n_mixed, n_success):
    """Pick tasks by how often the policy solved them; one source trajectory per task."""
    by_task = collections.defaultdict(list)
    for e in episodes:
        by_task[e['idx']].append(e)
    groups = {'fail': [], 'mixed': [], 'success': []}
    for idx in sorted(by_task):
        wins = sum(e['won'] for e in by_task[idx])
        kind = 'fail' if wins == 0 else 'success' if wins == len(by_task[idx]) else 'mixed'
        groups[kind].append(idx)
    picked = []
    for kind, n in (('fail', n_fail), ('mixed', n_mixed), ('success', n_success)):
        for idx in groups[kind][:n]:
            eps = sorted(by_task[idx], key=lambda e: e['k'])
            src = next((e for e in eps if not e['won']), eps[0]) if kind != 'success' else eps[0]
            picked.append((kind, src))
    return picked, {k: len(v) for k, v in groups.items()}


def restore(env, src, t):
    """Reset and replay src actions[:t]; returns (agent state, obs, state_ok)."""
    obs, _ = env.reset(session=src['idx'])
    state = AgentState(obs)
    for s in src['steps'][:t]:
        obs, _, _, _ = env.step(s['action'])
        state.record(s['action'], obs)
    ok = h(state.prompt(env.get_available_actions())) == src['steps'][t]['prompt_sha']
    return state, obs, ok


def plan_candidates(env, args, picked):
    """Candidate actions for every (task, t). Search steps get policy-sampled alternatives."""
    plan, search_points = [], []
    rng = random.Random(args.seed)
    for kind, src in picked:
        for t, s in enumerate(src['steps']):
            state, _, ok = restore(env, src, t)
            avail = env.get_available_actions()
            orig = s['action']
            point = {'idx': src['idx'], 'k': src['k'], 'kind': kind, 't': t, 'orig': orig,
                     'state_ok': ok, 'cands': [orig]}
            if avail['has_search_bar']:
                search_points.append((point, state.prompt(avail)))
            else:
                clicks = [f'click[{c}]' for c in avail['clickables'] if c != 'search' and f'click[{c}]' != orig]
                if args.max_cands and len(clicks) > args.max_cands - 1:
                    keep = [c for c in clicks if c[6:-1] in NAV]
                    rest = [c for c in clicks if c not in keep]
                    clicks = keep + rng.sample(rest, max(0, args.max_cands - 1 - len(keep)))
                point['cands'] += clicks
            plan.append(point)
    # policy-sampled alternative queries at search steps
    jobs = [(p, prompt, stable_seed(args.seed, 'propose', p['idx'], p['t'], j))
            for p, prompt in search_points for j in range(args.search_samples)]
    with cf.ThreadPoolExecutor(max_workers=args.batch) as pool:
        outs = list(pool.map(lambda x: call_llm(args, x[1], x[2]), jobs))
    for (p, _, _), out in zip(jobs, outs):
        a, _ = project(out['text'])
        if a.startswith('search[') and a not in p['cands']:
            p['cands'].append(a)
    return plan, len(jobs)


class Branch:
    def __init__(self, src, t, cand, m):
        self.src, self.t, self.cand, self.m = src, t, cand, m
        self.steps, self.reward, self.done = [], 0.0, False

    def start(self, env):
        self.state, _, self.state_ok = restore(env, self.src, self.t)
        obs, reward, done, _ = env.step(self.cand)  # the intervention
        self.used = self.t + 1
        self.forced_done = done
        if done:
            self.reward, self.done = reward, True
        else:
            self.state.record(self.cand, obs)

    @property
    def finished(self):
        return self.done or self.used >= MAX_STEPS

    def to_json(self):
        return {'idx': self.src['idx'], 'k': self.src['k'], 't': self.t, 'cand': self.cand,
                'is_orig': self.cand == self.src['steps'][self.t]['action'], 'm': self.m,
                'state_ok': self.state_ok, 'forced_done': self.forced_done,
                'cont_actions': [s['action'] for s in self.steps], 'reward': self.reward,
                'won': self.done and self.reward == 1.0, 'done': self.done, 'used_steps': self.used,
                'llm_calls': len(self.steps),
                'tokens': sum(s['prompt_tokens'] + s['completion_tokens'] for s in self.steps),
                'parse': [s['parse'] for s in self.steps]}


def run_branches(args, server, branches, out_f):
    slots = [WebAgentTextEnv(observation_mode='text', server=server, session_prefix=f'b{i}_')
             for i in range(args.batch)]
    queue = collections.deque(branches)
    active, done_n, t0, last, n_err = {}, 0, time.time(), 0, 0
    pool = cf.ThreadPoolExecutor(max_workers=args.batch)

    def finish(i, br):
        nonlocal done_n
        out_f.write(json.dumps(br.to_json()) + '\n')
        done_n += 1
        del active[i]

    while queue or active:
        for i in range(args.batch):
            while i not in active and queue:
                br = queue.popleft()
                br.start(slots[i])
                active[i] = br
                if br.finished:
                    finish(i, br)
        if not active:
            break
        items = list(active.items())
        prompts = [br.state.prompt(slots[i].get_available_actions()) for i, br in items]
        seeds = [stable_seed(args.seed, br.src['idx'], br.t, br.cand, br.m, br.used) for _, br in items]
        outs = list(pool.map(lambda ps: call_llm(args, *ps), zip(prompts, seeds)))
        n_err += sum(o['finish'] == 'error' for o in outs)
        if n_err > 100:  # e.g. the vLLM engine died: stop instead of logging no-op actions
            raise RuntimeError(f'{n_err} LLM call errors, last: {[o["error"] for o in outs if o["error"]][-1:]}')
        for (i, br), out in zip(items, outs):
            if args.lenient_action:
                action, valid, parse = project_lenient(out['text'])
            else:
                (action, valid), parse = project(out['text']), 'verl'
            obs, reward, done, _ = slots[i].step(action)
            br.used += 1
            br.steps.append({'action': action, 'valid': valid, 'parse': parse, 'finish': out['finish'],
                             'prompt_tokens': out['prompt_tokens'], 'completion_tokens': out['completion_tokens']})
            if done:
                br.reward, br.done = reward, True
            else:
                br.state.record(action, obs)
            if br.finished:
                finish(i, br)
        out_f.flush()
        if done_n // 200 > last:
            last = done_n // 200
            print(f'[{done_n}/{len(branches)}] elapsed={time.time() - t0:.0f}s', flush=True)
    return time.time() - t0


def q_table(plan, results):
    """Q_hat per (task, t, candidate) and per-step regret / advantage of the original action."""
    agg = collections.defaultdict(list)
    for r in results:
        agg[(r['idx'], r['t'], r['cand'])].append(r)
    table = []
    for p in plan:
        cands = []
        for c in p['cands']:
            rs = agg.get((p['idx'], p['t'], c), [])
            if rs:
                cands.append({'cand': c, 'n': len(rs),
                              'q_score': sum(r['reward'] for r in rs) / len(rs),
                              'q_success': sum(r['won'] for r in rs) / len(rs)})
        if not cands:
            continue
        orig = next((c for c in cands if c['cand'] == p['orig']), None)
        best = max(cands, key=lambda c: c['q_score'])
        mean_q = sum(c['q_score'] for c in cands) / len(cands)
        table.append({**{k: p[k] for k in ('idx', 'k', 'kind', 't', 'orig', 'state_ok')},
                      'n_cands': len(cands), 'q_orig': orig and orig['q_score'],
                      'q_orig_success': orig and orig['q_success'],
                      'best_cand': best['cand'], 'q_best': best['q_score'],
                      'regret': orig and best['q_score'] - orig['q_score'],
                      'adv_vs_mean': orig and orig['q_score'] - mean_q, 'cands': cands})
    return table


def main(args):
    os.makedirs(args.out, exist_ok=True)
    src_eps = [json.loads(l) for l in open(args.source)]
    picked, groups = select_sources(src_eps, args.n_fail, args.n_mixed, args.n_success)
    print(f'task groups in source: {groups}; picked {[(k, s["idx"], s["k"]) for k, s in picked]}', flush=True)

    env = make_env()
    t0 = time.time()
    plan, n_proposals = plan_candidates(env, args, picked)
    json.dump(plan, open(os.path.join(args.out, 'plan.json'), 'w'), indent=1)
    branches = [Branch(src, p['t'], c, m) for p in plan
                for src in [next(s for _, s in picked if s['idx'] == p['idx'])]
                for c in p['cands'] for m in range(args.m)]
    print(f'{len(plan)} decision points, {sum(len(p["cands"]) for p in plan)} (t, a) pairs, '
          f'{len(branches)} branches; planning took {time.time() - t0:.0f}s '
          f'({n_proposals} proposal calls), restore ok at {sum(p["state_ok"] for p in plan)}/{len(plan)} points',
          flush=True)

    with open(os.path.join(args.out, 'branches.jsonl'), 'w') as out_f:
        wall = run_branches(args, env.server, branches, out_f)
    results = [json.loads(l) for l in open(os.path.join(args.out, 'branches.jsonl'))]
    table = q_table(plan, results)
    json.dump(table, open(os.path.join(args.out, 'q_table.json'), 'w'), indent=1)

    summary = {
        'model': args.model, 'temperature': args.temperature, 'm': args.m,
        'tasks': [(k, s['idx'], s['k'], s['reward']) for k, s in picked],
        'decision_points': len(plan), 'pairs': sum(len(p['cands']) for p in plan),
        'branches': len(results), 'state_ok_branches': sum(r['state_ok'] for r in results),
        'llm_calls': sum(r['llm_calls'] for r in results) + n_proposals,
        'tokens': sum(r['tokens'] for r in results), 'wall_s': round(wall, 1),
    }
    json.dump(summary, open(os.path.join(args.out, 'summary.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2))
    print('\nper task: step with the largest regret (q_best - q_orig)')
    for idx in sorted({r['idx'] for r in table}):
        rows = [r for r in table if r['idx'] == idx]
        worst = max(rows, key=lambda r: r['regret'] or 0)
        print(f"  idx {idx:3d} ({rows[0]['kind']:7s}) T={len(rows):2d}  max regret {worst['regret']:.2f} "
              f"at t={worst['t']} orig={worst['orig'][:40]!r} best={worst['best_cand'][:40]!r}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    p.add_argument('--source', required=True, help='episodes.jsonl of the continuation policy')
    p.add_argument('--n-fail', type=int, default=4)
    p.add_argument('--n-mixed', type=int, default=2)
    p.add_argument('--n-success', type=int, default=2)
    p.add_argument('--m', type=int, default=4, help='continuations per (t, a)')
    p.add_argument('--max-cands', type=int, default=0, help='cap per click step (0 = all admissible)')
    p.add_argument('--search-samples', type=int, default=4, help='policy samples for alternative queries')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lenient-action', action='store_true')
    p.add_argument('--out', required=True)
    main(p.parse_args())
