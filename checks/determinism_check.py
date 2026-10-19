"""Check that a WebShop text-env state can be restored exactly by prefix replay.

restore(s_t) = reset(session=idx) + replay the logged actions a_0..a_{t-1}.
This only works if the env is deterministic given (goal seed, session idx, actions).

Three replay conditions are compared against the original recording:
  same_env   - same env object, after it has run all other tasks (tests session-state leaks)
  fresh_env  - new env object in the same process
  new_proc   - new process with a different PYTHONHASHSEED (tests set/dict-order effects)

No LLM is involved: trajectories come from a seeded random policy over admissible actions.

Usage:
  python determinism_check.py --n-tasks 20 --out runs/determinism
"""
import argparse
import json
import os
import random
import resource
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'credit'))
import config  # noqa: E402
from webshop_env import MAX_STEPS, h, make_env  # noqa: E402,F401


def snapshot(env, obs):
    avail = env.get_available_actions()
    return {
        'obs': h(obs),
        'avail': h(json.dumps(avail, sort_keys=True)),
        'clickables': avail['clickables'],
        'has_search_bar': avail['has_search_bar'],
    }


def random_action(rng, env, avail):
    """Seeded random policy that still produces multi-step shopping trajectories."""
    clickables = [c for c in avail['clickables'] if c != 'search']
    if avail['has_search_bar']:
        words = env.instruction_text.lower().replace(',', ' ').split()
        words = [w for w in words if w not in {'i', 'need', 'want', 'a', 'an', 'the', 'and',
                                               'with', 'for', 'of', 'to', 'price', 'lower',
                                               'than', 'dollars', 'instruction:', 'am', 'looking'}]
        k = min(len(words), rng.randint(2, 6))
        return f"search[{' '.join(rng.sample(words, k))}]"
    if not clickables:
        return 'click[back to search]'
    # Avoid buying immediately so trajectories reach item / option pages.
    non_buy = [c for c in clickables if c != 'buy now']
    if 'buy now' in clickables and (not non_buy or rng.random() < 0.25):
        return 'click[buy now]'
    return f'click[{rng.choice(non_buy)}]'


def record(env, idx, policy_seed):
    rng = random.Random(policy_seed * 100003 + idx)
    obs, _ = env.reset(session=idx)
    instruction = env.instruction_text  # read now: after `buy now` the env auto-resets to another goal
    steps = [snapshot(env, obs)]
    actions = []
    for _ in range(MAX_STEPS):
        a = random_action(rng, env, env.get_available_actions())
        obs, reward, done, _ = env.step(a)
        actions.append(a)
        snap = {'reward': reward, 'done': done}
        if not done:  # after `buy now` the env auto-resets to a random session; nothing to compare
            snap.update(snapshot(env, obs))
        steps.append(snap)
        if done:
            break
    return {'idx': idx, 'instruction': instruction, 'actions': actions, 'steps': steps}


def replay(env, traj):
    obs, _ = env.reset(session=traj['idx'])
    steps = [snapshot(env, obs)]
    for a in traj['actions']:
        obs, reward, done, _ = env.step(a)
        snap = {'reward': reward, 'done': done}
        if not done:
            snap.update(snapshot(env, obs))
        steps.append(snap)
    return steps


def compare(orig, rep):
    """Return the first differing (step, field), or None if identical."""
    for t, (a, b) in enumerate(zip(orig, rep)):
        for k in ('obs', 'avail', 'reward', 'done'):
            if a.get(k) != b.get(k):
                return {'step': t, 'field': k, 'orig': a.get(k), 'replay': b.get(k)}
    if len(orig) != len(rep):
        return {'step': min(len(orig), len(rep)), 'field': 'length'}
    return None


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def main_replay_only(args):
    """Entry point for the new_proc condition."""
    trajs = [json.loads(l) for l in open(args.replay_from)]
    env = make_env()
    with open(args.replay_to, 'w') as f:
        for traj in trajs:
            f.write(json.dumps({'idx': traj['idx'], 'steps': replay(env, traj)}) + '\n')


def main(args):
    os.makedirs(args.out, exist_ok=True)
    timing = {}

    t0 = time.time()
    env = make_env()
    timing['env_init_s'] = time.time() - t0
    timing['rss_after_init_gb'] = peak_rss_gb()
    n_goals = len(env.server.goals)

    idxs = list(range(args.first_idx, args.first_idx + args.n_tasks))
    t0 = time.time()
    trajs = [record(env, idx, args.policy_seed) for idx in idxs]
    n_steps = sum(len(t['actions']) for t in trajs)
    timing['record_s_per_step'] = (time.time() - t0) / max(n_steps, 1)
    with open(os.path.join(args.out, 'trajectories.jsonl'), 'w') as f:
        for t in trajs:
            f.write(json.dumps(t) + '\n')

    results = {}

    # same_env: replay in reverse order so every task is restored after others have run
    t0 = time.time()
    results['same_env'] = {t['idx']: compare(t['steps'], replay(env, t)) for t in reversed(trajs)}
    timing['replay_s_per_step'] = (time.time() - t0) / max(n_steps, 1)

    env2 = make_env()
    results['fresh_env'] = {t['idx']: compare(t['steps'], replay(env2, t)) for t in trajs}

    rep_path = os.path.join(args.out, 'new_proc_replay.jsonl')
    subprocess.run(
        [sys.executable, os.path.abspath(__file__),
         '--replay-from', os.path.join(args.out, 'trajectories.jsonl'), '--replay-to', rep_path],
        check=True, env={**os.environ, 'PYTHONHASHSEED': '12345'})
    rep = {r['idx']: r['steps'] for r in map(json.loads, open(rep_path))}
    results['new_proc'] = {t['idx']: compare(t['steps'], rep[t['idx']]) for t in trajs}

    timing['peak_rss_gb'] = peak_rss_gb()
    summary = {
        'n_goals': n_goals,
        'n_tasks': len(trajs),
        'n_steps': n_steps,
        'n_bought': sum(t['steps'][-1].get('done', False) for t in trajs),
        'mean_len': n_steps / len(trajs),
        'timing': timing,
        'mismatches': {cond: {str(i): d for i, d in r.items() if d} for cond, r in results.items()},
    }
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    ok = all(not v for v in summary['mismatches'].values())
    print('DETERMINISTIC' if ok else 'MISMATCH FOUND')
    return 0 if ok else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n-tasks', type=int, default=20)
    p.add_argument('--first-idx', type=int, default=0)
    p.add_argument('--policy-seed', type=int, default=0)
    p.add_argument('--out', default=None, help='default: $CREDIT_RUNS/determinism')
    p.add_argument('--replay-from')
    p.add_argument('--replay-to')
    args = p.parse_args()
    args.out = args.out or config.runs_dir('determinism')
    if args.replay_from:
        main_replay_only(args)
    else:
        sys.exit(main(args))
