"""Independent re-estimate of the decision-relevant candidates of a branch_llm.py run.

The first run tried up to ~12 candidates per step with M=4 and then took the maximum, which is
biased upward (winner's curse). Here every decision point of the failed trajectories on solvable
goals gets fresh continuations (new seed) for: the original action, the candidate with the best
Q_hat, and the alternative with the best success rate (the "flip" candidate), M each.
The output has the same format as branch_llm.py (branches.jsonl, q_table.json, summary.json).

Usage (vLLM server running): python reeval.py --model M --run RUN --source EPISODES
    --ceiling goal_ceiling.txt --m 16 --seed 1 --out RUN_REEVAL
"""
import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from branch_llm import Branch, q_table, run_branches  # noqa: E402
from webshop_env import make_env  # noqa: E402


def main(args):
    os.makedirs(args.out, exist_ok=True)
    plan = json.load(open(os.path.join(args.run, 'plan.json')))
    rows = [json.loads(l) for l in open(os.path.join(args.run, 'branches.jsonl'))]
    src = {(e['idx'], e['k']): e for e in map(json.loads, open(args.source))}
    ceiling = {int(l.split()[1]): float(l.split()[2].split('=')[1]) for l in open(args.ceiling)}

    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r['idx'], r['t'], r['cand'])].append(r)
    new_plan = []
    for p in plan:
        key = (p['idx'], p['k'])
        if p['kind'] == 'success' or src[key]['won'] or ceiling.get(p['idx'], 1.0) < 1.0:
            continue
        stats = {c: (sum(r['reward'] for r in agg[(p['idx'], p['t'], c)]) / len(agg[(p['idx'], p['t'], c)]),
                     sum(r['won'] for r in agg[(p['idx'], p['t'], c)]) / len(agg[(p['idx'], p['t'], c)]))
                 for c in p['cands'] if agg.get((p['idx'], p['t'], c))}
        best_q = max(stats, key=lambda c: stats[c][0])
        alts = [c for c in stats if c != p['orig']]
        best_s = max(alts, key=lambda c: (stats[c][1], stats[c][0])) if alts else p['orig']
        cands = list(dict.fromkeys([p['orig'], best_q, best_s]))
        new_plan.append({**{k: p[k] for k in ('idx', 'k', 'kind', 't', 'orig', 'state_ok')}, 'cands': cands,
                         'first_run': {c: stats[c] for c in cands}})
    json.dump(new_plan, open(os.path.join(args.out, 'plan.json'), 'w'), indent=1)
    branches = [Branch(src[(p['idx'], p['k'])], p['t'], c, m)
                for p in new_plan for c in p['cands'] for m in range(args.m)]
    print(f'{len(new_plan)} decision points, {sum(len(p["cands"]) for p in new_plan)} (t, a) pairs, '
          f'{len(branches)} branches (M={args.m}, seed={args.seed})', flush=True)

    env = make_env()
    with open(os.path.join(args.out, 'branches.jsonl'), 'w') as out_f:
        wall = run_branches(args, env.server, branches, out_f)
    results = [json.loads(l) for l in open(os.path.join(args.out, 'branches.jsonl'))]
    json.dump(q_table(new_plan, results), open(os.path.join(args.out, 'q_table.json'), 'w'), indent=1)
    summary = {'model': args.model, 'temperature': args.temperature, 'm': args.m, 'seed': args.seed,
               'source_run': args.run, 'decision_points': len(new_plan),
               'pairs': sum(len(p['cands']) for p in new_plan), 'branches': len(results),
               'state_ok_branches': sum(r['state_ok'] for r in results),
               'llm_calls': sum(r['llm_calls'] for r in results),
               'tokens': sum(r['tokens'] for r in results), 'wall_s': round(wall, 1)}
    json.dump(summary, open(os.path.join(args.out, 'summary.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    p.add_argument('--run', required=True, help='branch_llm.py output to re-estimate')
    p.add_argument('--source', required=True)
    p.add_argument('--ceiling', required=True)
    p.add_argument('--m', type=int, default=16)
    p.add_argument('--seed', type=int, default=1, help='differs from the first run, so samples are independent')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lenient-action', action='store_true')
    p.add_argument('--out', required=True)
    main(p.parse_args())
