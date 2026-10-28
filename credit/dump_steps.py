"""Text of every step of logged episodes, for the RUDDER-style baseline (step_features.py, rudder.py).

Episodes only store hashes of the observations, so each one is replayed (exact, see checks/) to get
the page the agent saw before each action.
Output: one line per step {idx, k, t, instruction, obs, action, reward, won, n_steps}; reward/won are
the episode's final outcome.

Usage: python dump_steps.py --out steps.jsonl EPISODES.jsonl [EPISODES.jsonl ...]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from webshop_env import h, make_env  # noqa: E402


def main(args):
    env = make_env()
    seen, n_ep, n_bad = set(), 0, 0
    with open(args.out, 'w') as out_f:
        for path in args.episodes:
            for line in open(path):
                e = json.loads(line)
                key = (e['idx'], e['k'], e['instruction'], tuple(e['actions']))
                if key in seen:  # the same (task, sample) may appear in several files
                    continue
                seen.add(key)
                obs, _ = env.reset(session=e['idx'])
                for s in e['steps']:
                    out_f.write(json.dumps({'idx': e['idx'], 'k': e['k'], 'src': os.path.basename(os.path.dirname(path)),
                                            't': s['t'], 'instruction': e['instruction'], 'obs': obs,
                                            'action': s['action'], 'reward': e['reward'], 'won': e['won'],
                                            'n_steps': e['n_steps']}) + '\n')
                    obs, _, done, _ = env.step(s['action'])
                    if (s['obs'] != 'DONE') != (not done) or (not done and h(obs) != s['obs']):
                        n_bad += 1
                n_ep += 1
    print(f'{n_ep} episodes dumped to {args.out}; observation mismatches: {n_bad}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('episodes', nargs='+')
    main(p.parse_args())
