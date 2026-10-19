"""How cheap can replay be? Offline simulation on an existing branch_llm.py run (stdlib only).

Reference ("full"): every decision step, every candidate (<=12), M=4 continuations each.
Cheaper strategies are simulated by taking subsets of those same branches:
  M          use only the first M continuation samples of each (t, a)
  steps      replay only at some steps (by the type of the original action, or early steps)
  cands      try only k alternatives per step (random, or prioritized by action type)
plus zero-cost heuristics (blame the last step / the first product click).
Scored on the failed trajectories of solvable goals against the full-data ground truth:
  decisive step  = earliest "flip" step (orig success <= 0.25, some alternative >= 0.75)
  top-1 blame    = step with the largest regret (q_best - q_orig), counted when GT regret >= 0.25
Caveat: the cheap strategies reuse samples of the reference, so agreement is optimistic;
an independent re-estimate (M=16) is needed for the final numbers.

Usage: python3 cheap_replay_sim.py RUN_DIR SOURCE_EPISODES CEILING_FILE
"""
import collections
import hashlib
import json
import os
import random
import re
import sys

FLIP_LO, FLIP_HI, REGRET_MIN = 0.25, 0.75, 0.25


def kind_of(a):
    if a.startswith('search['):
        return 'search'
    if a == 'click[buy now]':
        return 'buy'
    if re.match(r'^click\[b0[0-9a-z]{8}\]$', a):
        return 'product'
    if a in ('click[back to search]', 'click[next >]', 'click[< prev]'):
        return 'nav'
    if a in ('click[description]', 'click[features]', 'click[reviews]'):
        return 'info'
    return 'option' if a.startswith('click[') else 'invalid'


PRIORITY = {'product': 0, 'search': 0, 'buy': 1, 'option': 2, 'nav': 3, 'info': 4, 'invalid': 5}


def analyze(steps_data, T, m_max, step_filter, cand_filter):
    if m_max == 'adaptive':
        return analyze_adaptive(steps_data, T, step_filter, cand_filter)
    """steps_data: {t: {'orig':a, 'type':..., 'cands':[...], 'samples':{cand:[(m, reward, won, calls)]}, 'search': bool}}
    Returns per-step estimates for the replayed steps and the continuation-call cost."""
    est, cost = {}, 0
    for t, d in steps_data.items():
        if not step_filter(t, d, T):
            continue
        cands = cand_filter(t, d)
        rows = {}
        for c in cands:
            ss = [s for s in d['samples'].get(c, []) if s[0] < m_max]
            if ss:
                rows[c] = (sum(s[1] for s in ss) / len(ss), sum(s[2] for s in ss) / len(ss))
                cost += sum(s[3] for s in ss)
        if d['search'] and len(cands) > 1:
            cost += 4  # policy samples that proposed the alternative queries
        if d['orig'] not in rows:
            continue
        qo, so = rows[d['orig']]
        alts = {c: v for c, v in rows.items() if c != d['orig']}
        q_best = max([v[0] for v in rows.values()])
        s_alt = max([v[1] for v in alts.values()], default=0)
        est[t] = {'regret': q_best - qo, 'flip': so <= FLIP_LO and s_alt >= FLIP_HI}
    return est, cost


def analyze_adaptive(steps_data, T, step_filter, cand_filter):
    """Screen every candidate with 1 sample; only when an alternative succeeds while the original
    fails, spend 3 more samples on both of them to confirm (sequential test)."""
    est, cost = {}, 0
    for t, d in steps_data.items():
        if not step_filter(t, d, T):
            continue
        cands = cand_filter(t, d)
        first = {c: [s for s in d['samples'].get(c, []) if s[0] == 0] for c in cands}
        cost += sum(s[3] for ss in first.values() for s in ss)
        if d['search'] and len(cands) > 1:
            cost += 4
        if not first.get(d['orig']):
            continue
        use = dict(first)
        o_won = first[d['orig']][0][2]
        hits = [c for c in cands if c != d['orig'] and first[c] and first[c][0][2] and not o_won]
        for c in hits + ([d['orig']] if hits else []):
            more = [s for s in d['samples'].get(c, []) if 1 <= s[0] < 4]
            cost += sum(s[3] for s in more)
            use[c] = first[c] + more
        rows = {c: (sum(s[1] for s in ss) / len(ss), sum(s[2] for s in ss) / len(ss)) for c, ss in use.items() if ss}
        qo, so = rows[d['orig']]
        alts = {c: v for c, v in rows.items() if c != d['orig'] and c in hits}
        est[t] = {'regret': max(v[0] for v in rows.values()) - qo,
                  'flip': so <= FLIP_LO and max([v[1] for v in alts.values()], default=0) >= FLIP_HI}
    return est, cost


def main(run, source, ceiling_file):
    plan = json.load(open(os.path.join(run, 'plan.json')))
    rows = [json.loads(l) for l in open(os.path.join(run, 'branches.jsonl'))]
    src = {(e['idx'], e['k']): e for e in map(json.loads, open(source))}
    ceiling = {int(l.split()[1]): float(l.split()[2].split('=')[1]) for l in open(ceiling_file)}

    samples = collections.defaultdict(list)
    for r in rows:
        samples[(r['idx'], r['t'], r['cand'])].append((r['m'], r['reward'], r['won'], r['llm_calls']))
    trajs = collections.defaultdict(dict)
    for p in plan:
        key = (p['idx'], p['k'])
        if p['kind'] == 'success' or src[key]['won'] or ceiling.get(p['idx'], 1.0) < 1.0:
            continue
        trajs[key][p['t']] = {'orig': p['orig'], 'type': kind_of(p['orig']), 'cands': p['cands'],
                              'search': p['orig'].startswith('search['),
                              'samples': {c: samples[(p['idx'], p['t'], c)] for c in p['cands']}}
    T_of = {k: len(src[k]['steps']) for k in trajs}

    every = lambda t, d, T: True
    allc = lambda t, d: d['cands']
    full = {k: analyze(v, T_of[k], 4, every, allc) for k, v in trajs.items()}
    gt_flip = {k: sorted(t for t, e in est.items() if e['flip']) for k, (est, _) in full.items()}
    gt_first = {k: v[0] for k, v in gt_flip.items() if v}
    gt_top = {}
    for k, (est, _) in full.items():
        t_top = max(est, key=lambda t: (est[t]['regret'], -t))
        if est[t_top]['regret'] >= REGRET_MIN:
            gt_top[k] = t_top
    full_cost = sum(c for _, c in full.values())

    def rand_k(k_alt, seed=0):
        def f(t, d):
            alts = [c for c in d['cands'] if c != d['orig']]
            rng = random.Random(int(hashlib.md5(('%s|%s|%s' % (seed, t, d['orig'])).encode()).hexdigest()[:8], 16))
            return [d['orig']] + rng.sample(alts, min(k_alt, len(alts)))
        return f

    def prio_k(k_alt):
        def f(t, d):
            alts = sorted((c for c in d['cands'] if c != d['orig']), key=lambda c: PRIORITY[kind_of(c)])
            return [d['orig']] + alts[:k_alt]
        return f

    types = lambda allowed: (lambda t, d, T: d['type'] in allowed)
    early = lambda frac: (lambda t, d, T: t < max(1, round(frac * T)))

    strategies = [
        ('full: all steps, <=12 cands, M=4', 4, every, allc),
        ('M=2', 2, every, allc),
        ('M=1', 1, every, allc),
        ('steps: product+search only, M=4', 4, types({'product', 'search'}), allc),
        ('steps: product+search+option, M=4', 4, types({'product', 'search', 'option'}), allc),
        ('steps: first half of trajectory, M=4', 4, early(0.5), allc),
        ('cands: 3 random alternatives, M=4', 4, every, rand_k(3)),
        ('cands: 3 prioritized alternatives, M=4', 4, every, prio_k(3)),
        ('adaptive: 1 sample, confirm hits with 3 more', 'adaptive', every, allc),
        ('adaptive + product/search/option steps', 'adaptive', types({'product', 'search', 'option'}), allc),
        ('M=1 + product/search steps', 1, types({'product', 'search'}), allc),
        ('M=1 + product/search + 3 prioritized', 1, types({'product', 'search'}), prio_k(3)),
        ('M=1 + product/search + 1 prioritized', 1, types({'product', 'search'}), prio_k(1)),
    ]

    n_flip, n_top = len(gt_first), len(gt_top)
    print('Cheap replay simulation on %d failed trajectories of solvable goals (run %s)' % (len(trajs), run))
    print('ground truth (full data): %d trajectories with a flip step, %d with a top-regret step >= %.2f; '
          'full cost %d continuation calls (%.0f per trajectory)' % (n_flip, n_top, REGRET_MIN, full_cost, full_cost / len(trajs)))
    print()
    hdr = '%-42s %8s %6s | %-22s %-20s %-14s %-10s'
    print(hdr % ('strategy', 'calls', 'cost', 'finds GT first flip', 'finds any GT flip', 'false flips', 'top-1 blame'))
    for name, m, sf, cf in strategies:
        res = {k: analyze(v, T_of[k], m, sf, cf) for k, v in trajs.items()}
        cost = sum(c for _, c in res.values())
        first_hit = sum(1 for k, t in gt_first.items() if res[k][0].get(t, {}).get('flip'))
        any_hit = sum(1 for k in gt_first if any(res[k][0].get(t, {}).get('flip') for t in gt_flip[k]))
        false = sum(1 for k, (est, _) in res.items() for t, e in est.items() if e['flip'] and t not in gt_flip[k])
        claimed = sum(1 for k, (est, _) in res.items() for t, e in est.items() if e['flip'])
        top_hit = 0
        for k, t_gt in gt_top.items():
            est = res[k][0]
            if est:
                t_s = max(est, key=lambda t: (est[t]['regret'], -t))
                top_hit += t_s == t_gt
        print(hdr % (name, cost, '%.0f%%' % (100 * cost / full_cost),
                     '%d/%d' % (first_hit, n_flip), '%d/%d' % (any_hit, n_flip),
                     '%d of %d claimed' % (false, claimed), '%d/%d' % (top_hit, n_top)))

    print()
    print('zero-cost heuristics (no replay):')
    last = sum(1 for k, t in gt_top.items() if t == T_of[k] - 1)
    last_f = sum(1 for k, t in gt_first.items() if t == T_of[k] - 1)
    first_prod = {k: next((t for t in sorted(v) if v[t]['type'] == 'product'), None) for k, v in trajs.items()}
    fp_top = sum(1 for k, t in gt_top.items() if first_prod[k] == t)
    fp_first = sum(1 for k, t in gt_first.items() if first_prod[k] == t)
    fp_any = sum(1 for k in gt_first if first_prod[k] in gt_flip[k])
    print('  blame the last step:          first flip %d/%d, top-1 blame %d/%d' % (last_f, n_flip, last, n_top))
    print('  blame the first product click: first flip %d/%d, any flip %d/%d, top-1 blame %d/%d'
          % (fp_first, n_flip, fp_any, n_flip, fp_top, n_top))
    print('  uniform credit: no step is singled out (expected top-1 by chance %.2f of trajectories)'
          % (sum(1 / T_of[k] for k in gt_top) / max(n_top, 1)))


if __name__ == '__main__':
    main(*sys.argv[1:4])
