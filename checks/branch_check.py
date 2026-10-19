"""Branching test: restore s_t by prefix replay, take an alternative action, then continue.

A branch is (task idx, t, alternative action a', continuation policy, seed). Running it means
  restore s_t  ->  step(a')  ->  continuation policy until `buy now` or the 15-step episode limit.

Checks
  state_ok      every restore reproduces the recorded observation hash at s_t
  reproducible  each branch gives the same trace and reward when run in another order in the
                same env, in a fresh env, and in a new process (different PYTHONHASHSEED)
  no leakage    after all the branching, the original trajectories still replay identically
Demo
  replay-based Q(s_t, a) for every admissible alternative on one trajectory, with scripted
  continuation policies standing in for the LLM.

Usage:
  python branch_check.py --n-trajs 10 --max-alts 4 --out runs/branch
"""
import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from determinism_check import MAX_STEPS, compare, h, make_env, random_action, replay  # noqa: E402
import config  # noqa: E402

NAV = {'back to search', '< prev', 'next >', 'description', 'features', 'reviews', 'buy now', 'search'}
ASIN = re.compile(r'^b[0-9a-z]{9}$')


def query_from_instruction(instr):
    q = instr.lower().replace('instruction:', '').strip()
    return q.split(', and price lower than')[0]


def wanted_option(value, instr):
    instr = instr.lower()
    return f': {value},' in instr or instr.rstrip().endswith(f': {value}')


class RulePolicy:
    """Deterministic scripted continuation: WebShop's rule baseline plus option matching."""

    def __init__(self, seed=0, key=None):
        self.clicked = set()

    def __call__(self, env):
        avail = env.get_available_actions()
        clickables = avail['clickables']
        instr = env.instruction_text
        if avail['has_search_bar']:
            return f'search[{query_from_instruction(instr)}]'
        products = [c for c in clickables if ASIN.match(c)]
        if products:
            return f'click[{products[0]}]'
        if 'buy now' in clickables:
            for c in clickables:
                if c not in NAV and c not in self.clicked and wanted_option(c, instr):
                    self.clicked.add(c)
                    return f'click[{c}]'
            return 'click[buy now]'
        if '< prev' in clickables:
            return 'click[< prev]'
        return 'click[back to search]'


class RandomPolicy:
    """The seeded random policy from determinism_check, reseeded per branch."""

    def __init__(self, seed=0, key=None):
        digest = hashlib.sha1(json.dumps([seed, key]).encode()).hexdigest()
        self.rng = random.Random(int(digest[:12], 16))

    def __call__(self, env):
        return random_action(self.rng, env, env.get_available_actions())


POLICIES = {'rule': RulePolicy, 'random': RandomPolicy}


def restore(env, traj, t):
    obs, _ = env.reset(session=traj['idx'])
    for a in traj['actions'][:t]:
        obs, _, _, _ = env.step(a)
    return obs


def run_branch(env, traj, job):
    t, alt = job['t'], job['alt']
    obs = restore(env, traj, t)
    state_ok = h(obs) == traj['steps'][t]['obs']
    policy = POLICIES[job['policy']](job['seed'], key=[traj['idx'], t, alt])
    trace, a, used = [], alt, t
    while True:
        obs, reward, done, _ = env.step(a)
        used += 1
        trace.append([a, 'DONE' if done else h(obs)])
        if done or used >= MAX_STEPS:
            break
        a = policy(env)
    return {'state_ok': state_ok, 'trace': trace, 'reward': reward if done else 0.0, 'done': done}


def alternatives(env, traj, t, rng, max_alts):
    """Original action first, then alternatives admissible at s_t (env must be at s_t)."""
    avail = env.get_available_actions()
    orig = traj['actions'][t]
    if avail['has_search_bar']:
        alts = {random_action(rng, env, avail) for _ in range(20)}
        alts.add(f'search[{query_from_instruction(env.instruction_text)}]')
        alts.discard(orig)
        alts = sorted(alts)
    else:
        alts = [f'click[{c}]' for c in avail['clickables'] if c != 'search' and f'click[{c}]' != orig]
    if max_alts and len(alts) > max_alts:
        alts = rng.sample(alts, max_alts)
    return [orig] + alts


def job_key(j):
    return f"{j['idx']}|{j['t']}|{j['alt']}|{j['policy']}|{j['seed']}"


def build_jobs(env, trajs, max_alts, seeds):
    jobs = []
    for traj in trajs:
        rng = random.Random(traj['idx'])
        for t in range(len(traj['actions'])):
            restore(env, traj, t)
            for alt in alternatives(env, traj, t, rng, max_alts):
                jobs.append({'idx': traj['idx'], 't': t, 'alt': alt, 'policy': 'rule', 'seed': 0})
                jobs += [{'idx': traj['idx'], 't': t, 'alt': alt, 'policy': 'random', 'seed': s}
                         for s in range(seeds)]
    return jobs


def run_jobs(env, by_idx, jobs):
    return {job_key(j): run_branch(env, by_idx[j['idx']], j) for j in jobs}


def diff(ref, other):
    return [k for k in ref if ref[k] != other.get(k)]


def demo(env, traj, n_seeds):
    """Q(s_t, a) for all admissible alternatives at every step of one trajectory."""
    rows = []
    rng = random.Random(0)
    for t in range(len(traj['actions'])):
        restore(env, traj, t)
        for alt in alternatives(env, traj, t, rng, max_alts=None):
            j = {'idx': traj['idx'], 't': t, 'alt': alt, 'policy': 'rule', 'seed': 0}
            q_rule = run_branch(env, traj, j)['reward']
            q_rand = [run_branch(env, traj, {**j, 'policy': 'random', 'seed': s})['reward']
                      for s in range(n_seeds)]
            rows.append({'t': t, 'alt': alt, 'is_orig': alt == traj['actions'][t],
                         'q_rule': q_rule, 'q_random': sum(q_rand) / n_seeds})
    return rows


def main(args):
    os.makedirs(args.out, exist_ok=True)
    trajs = [json.loads(l) for l in open(args.trajs)]
    by_idx = {t['idx']: t for t in trajs}
    if args.run_jobs:  # new-process pass
        jobs = [json.loads(l) for l in open(args.run_jobs)]
        env = make_env()
        json.dump(run_jobs(env, by_idx, jobs), open(args.results_to, 'w'))
        return 0

    sel = trajs[:args.n_trajs]
    env = make_env()
    jobs = build_jobs(env, sel, args.max_alts, args.seeds)
    jobs_path = os.path.join(args.out, 'jobs.jsonl')
    with open(jobs_path, 'w') as f:
        f.writelines(json.dumps(j) + '\n' for j in jobs)

    t0 = time.time()
    p1 = run_jobs(env, by_idx, jobs)
    sec_per_branch = (time.time() - t0) / len(jobs)
    p2 = run_jobs(env, by_idx, list(reversed(jobs)))
    shuffled = jobs[:]
    random.Random(1).shuffle(shuffled)
    p3 = run_jobs(make_env(), by_idx, shuffled)
    p4_path = os.path.join(args.out, 'newproc_results.json')
    subprocess.run([sys.executable, os.path.abspath(__file__), '--trajs', args.trajs,
                    '--run-jobs', jobs_path, '--results-to', p4_path, '--out', args.out],
                   check=True, env={**os.environ, 'PYTHONHASHSEED': '777'})
    p4 = json.load(open(p4_path))

    # no leakage: env has now run thousands of branches; originals must still replay identically
    leaks = [t['idx'] for t in sel if compare(t['steps'], replay(env, t))]

    # how often the alternative actually leads somewhere different from the original next state
    orig_next = {(t['idx'], s): t['steps'][s + 1].get('obs', 'DONE')
                 for t in sel for s in range(len(t['actions']))}
    alt_jobs = [j for j in jobs if j['policy'] == 'rule' and j['alt'] != by_idx[j['idx']]['actions'][j['t']]]
    same_next = sum(p1[job_key(j)]['trace'][0][1] == orig_next[(j['idx'], j['t'])] for j in alt_jobs)

    rewards = [r['reward'] for k, r in p1.items() if '|rule|' in k]
    summary = {
        'n_trajs': len(sel), 'n_branches': len(jobs),
        'n_decision_points': sum(len(t['actions']) for t in sel),
        'sec_per_branch': round(sec_per_branch, 4),
        'state_ok': sum(r['state_ok'] for r in p1.values()),
        'mismatch_vs_pass1': {'reverse_order_same_env': len(diff(p1, p2)),
                              'fresh_env_shuffled': len(diff(p1, p3)),
                              'new_process': len(diff(p1, p4))},
        'leaked_trajectories': leaks,
        'alts_reaching_same_next_page': f'{same_next}/{len(alt_jobs)}',
        'rule_branch_reward_mean': round(sum(rewards) / len(rewards), 3),
        'rule_branch_success(=1.0)': sum(r == 1.0 for r in rewards),
    }
    for name, other in [('reverse', p2), ('fresh', p3), ('newproc', p4)]:
        bad = diff(p1, other)[:3]
        if bad:
            summary[f'examples_{name}'] = {k: [p1[k], other.get(k)] for k in bad}

    demo_traj = by_idx[args.demo_idx]
    rows = demo(env, demo_traj, args.demo_seeds)
    json.dump({'summary': summary, 'demo_idx': args.demo_idx, 'demo': rows},
              open(os.path.join(args.out, 'summary.json'), 'w'), indent=1)
    print(json.dumps(summary, indent=2))
    ok = (summary['state_ok'] == len(jobs) and not leaks
          and not any(summary['mismatch_vs_pass1'].values()))
    print('BRANCHING REPRODUCIBLE' if ok else 'PROBLEM FOUND')
    return 0 if ok else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--trajs', default=None, help='default: $CREDIT_RUNS/determinism_100/trajectories.jsonl')
    p.add_argument('--n-trajs', type=int, default=10)
    p.add_argument('--max-alts', type=int, default=4)
    p.add_argument('--seeds', type=int, default=1, help='random-policy seeds per branch')
    p.add_argument('--demo-idx', type=int, default=1)
    p.add_argument('--demo-seeds', type=int, default=8)
    p.add_argument('--out', default=None, help='default: $CREDIT_RUNS/branch')
    p.add_argument('--run-jobs')
    p.add_argument('--results-to')
    args = p.parse_args()
    args.trajs = args.trajs or config.runs_dir('determinism_100', 'trajectories.jsonl')
    args.out = args.out or config.runs_dir('branch')
    sys.exit(main(args))
