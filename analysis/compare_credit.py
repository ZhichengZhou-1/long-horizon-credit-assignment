"""Phase 2 first comparison: which step does each credit method blame, and how often is it right?

Ground truth (GT16): the independent M=16 re-estimate (reeval.py) of the original action, the best-Q
candidate and the best-success candidate at every step of the failed trajectories on solvable goals.
  flip step      q_success(orig) <= 0.25 and some alternative >= 0.75
  top-regret     step with the largest q(best) - q(orig), counted when >= 0.25
Methods (each blames one step per trajectory; per-step scores are also correlated with GT16 regret):
  zero cost      last step, uniform, first product click
  HCAPO-style    from hcapo_score.py, action-only and whole-response spans; 'succ' (success hindsight,
                 low ratio = blame) and 'own' (real outcome, high ratio = blame); policy confidence
                 (low log-prob of the action without hindsight = blame)
  replay         from the first branch run (independent samples): M=4 all candidates, M=1, adaptive
                 (1 sample, confirm hits with 3 more); blame = earliest claimed flip, else max regret
Also reports how much of the first run's regret survives re-estimation (winner's curse).

Usage: python3 compare_credit.py REEVAL_DIR FIRST_RUN_DIR HCAPO_ACTION_JSONL HCAPO_RESPONSE_JSONL SOURCE_EPISODES
"""
import collections
import json
import os
import re
import statistics
import sys

FLIP_LO, FLIP_HI, REGRET_MIN = 0.25, 0.75, 0.25


def load(path):
    return [json.loads(l) for l in open(path)]


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else None


def q_stats(rows):
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r['idx'], r['t'], r['cand'])].append(r)
    return agg


def point_est(cands_rows, orig):
    """cands_rows: {cand: [(reward, won)]} -> (regret, flip)."""
    st = {c: (sum(x[0] for x in v) / len(v), sum(x[1] for x in v) / len(v)) for c, v in cands_rows.items() if v}
    if orig not in st:
        return None
    qo, so = st[orig]
    alts = [v for c, v in st.items() if c != orig]
    return {'regret': max(v[0] for v in st.values()) - qo,
            'flip': so <= FLIP_LO and max([v[1] for v in alts], default=0) >= FLIP_HI}


def main(reeval, first, hc_action, hc_resp, source):
    src = {(e['idx'], e['k']): e for e in load(source)}
    plan16 = json.load(open(os.path.join(reeval, 'plan.json')))
    agg16 = q_stats(load(os.path.join(reeval, 'branches.jsonl')))
    plan1 = {(p['idx'], p['t']): p for p in json.load(open(os.path.join(first, 'plan.json')))}
    agg1 = q_stats(load(os.path.join(first, 'branches.jsonl')))

    gt, first_reg = collections.defaultdict(dict), []
    for p in plan16:
        e = point_est({c: [(r['reward'], r['won']) for r in agg16[(p['idx'], p['t'], c)]] for c in p['cands']}, p['orig'])
        if e is None:
            continue
        gt[(p['idx'], p['k'])][p['t']] = e
        fr = p['first_run']
        first_reg.append((max(v[0] for v in fr.values()) - fr[p['orig']][0], e['regret']))
    keys = sorted(gt)
    T = {k: len(src[k]['steps']) for k in keys}
    gt_flips = {k: sorted(t for t, e in gt[k].items() if e['flip']) for k in keys}
    gt_first = {k: v[0] for k, v in gt_flips.items() if v}
    gt_top = {}
    for k in keys:
        t = max(gt[k], key=lambda t: (gt[k][t]['regret'], -t))
        if gt[k][t]['regret'] >= REGRET_MIN:
            gt_top[k] = t

    print('Phase 2 comparison on %d failed trajectories of solvable goals' % len(keys))
    print('GT16 (independent M=16 re-estimate): %d trajectories with a flip step (%d flip steps in all), '
          '%d with top regret >= %.2f' % (len(gt_first), sum(len(v) for v in gt_flips.values()), len(gt_top), REGRET_MIN))
    a, b = zip(*first_reg)
    big = [(x, y) for x, y in first_reg if x >= REGRET_MIN]
    print('winner\'s curse: mean regret first run %.3f -> re-estimated %.3f; at steps with first-run regret >= %.2f: '
          '%.3f -> %.3f (n=%d)' % (statistics.mean(a), statistics.mean(b), REGRET_MIN,
                                   statistics.mean(x for x, _ in big), statistics.mean(y for _, y in big), len(big)))

    methods = collections.OrderedDict()  # name -> (blame step per traj, per-step blame score per traj, cost per traj, cost unit)
    methods['last step'] = ({k: T[k] - 1 for k in keys}, {k: {t: t for t in gt[k]} for k in keys}, 0, '-')
    fp = {}
    for k in keys:
        fp[k] = next((t for t, s in enumerate(src[k]['steps']) if re.match(r'^click\[b0[0-9a-z]{8}\]$', s['action'])), 0)
    methods['first product click'] = (fp, None, 0, '-')

    for label, path in (('action', hc_action), ('response', hc_resp)):
        sc = collections.defaultdict(dict)
        for r in load(path):
            sc[(r['idx'], r['k'])][r['t']] = r
        def mk(field, sign):
            blame, score = {}, {}
            for k in keys:
                rows = {t: v for t, v in sc[k].items() if t in gt[k]}
                if not rows:
                    continue
                score[k] = {t: sign * v[field] for t, v in rows.items()}
                blame[k] = max(score[k], key=lambda t: (score[k][t], -t))
            return blame, score
        n_fwd = statistics.mean(len(sc[k]) for k in keys)
        methods['HCAPO %s: success hindsight, low ratio' % label] = mk('ratio_succ', -1) + (n_fwd, 'fwd')
        methods['HCAPO %s: own outcome, high ratio' % label] = mk('ratio_own', 1) + (n_fwd, 'fwd')
        methods['policy confidence (%s): low log-prob' % label] = mk('logp_base', -1) + (n_fwd, 'fwd')

    def replay(m_max, adaptive=False):
        blame, score, costs = {}, {}, []
        for k in keys:
            est, cost = {}, 0
            for t in gt[k]:
                p = plan1[(k[0], t)]
                if adaptive:
                    first_s = {c: [r for r in agg1[(k[0], t, c)] if r['m'] == 0] for c in p['cands']}
                    use = dict(first_s)
                    o = first_s.get(p['orig'])
                    hits = [c for c in p['cands'] if c != p['orig'] and first_s[c] and first_s[c][0]['won'] and o and not o[0]['won']]
                    for c in hits + ([p['orig']] if hits else []):
                        use[c] = [r for r in agg1[(k[0], t, c)] if r['m'] < 4]
                else:
                    use = {c: [r for r in agg1[(k[0], t, c)] if r['m'] < m_max] for c in p['cands']}
                cost += sum(r['llm_calls'] for v in use.values() for r in v)
                e = point_est({c: [(r['reward'], r['won']) for r in v] for c, v in use.items()}, p['orig'])
                if e:
                    est[t] = e
            costs.append(cost)
            score[k] = {t: e['regret'] for t, e in est.items()}
            flips = [t for t, e in sorted(est.items()) if e['flip']]
            blame[k] = flips[0] if flips else max(score[k], key=lambda t: (score[k][t], -t))
        return blame, score, statistics.mean(costs), 'calls'
    methods['replay M=4, all candidates'] = replay(4)
    methods['replay adaptive (1 + 3 to confirm)'] = replay(4, adaptive=True)
    methods['replay M=1'] = replay(1)

    print()
    hdr = '%-46s %12s | %-12s %-12s %-12s %-10s'
    print(hdr % ('method (blames one step per trajectory)', 'cost/traj', 'first flip', 'any flip', 'top regret', 'spearman'))
    print(hdr % ('uniform (chance level)', '0', '%.1f/%d' % (sum(1 / T[k] for k in gt_first), len(gt_first)),
                 '%.1f/%d' % (sum(len(gt_flips[k]) / T[k] for k in gt_first), len(gt_first)),
                 '%.1f/%d' % (sum(1 / T[k] for k in gt_top), len(gt_top)), '-'))
    for name, (blame, score, cost, unit) in methods.items():
        f1 = sum(1 for k, t in gt_first.items() if blame.get(k) == t)
        fa = sum(1 for k in gt_first if blame.get(k) in gt_flips[k])
        tp = sum(1 for k, t in gt_top.items() if blame.get(k) == t)
        rho = []
        if score:
            for k in keys:
                if k in score:
                    ts = sorted(set(score[k]) & set(gt[k]))
                    r = spearman([score[k][t] for t in ts], [gt[k][t]['regret'] for t in ts])
                    if r is not None:
                        rho.append(r)
        print(hdr % (name, ('%.0f %s' % (cost, unit)) if unit != '-' else '0',
                     '%d/%d' % (f1, len(gt_first)), '%d/%d' % (fa, len(gt_first)), '%d/%d' % (tp, len(gt_top)),
                     ('%.2f (n=%d)' % (statistics.mean(rho), len(rho))) if rho else '-'))
    print()
    print('cost units: calls = continuation LLM calls (generation, ~200 output tokens each); '
          'fwd = scoring forward passes per variant (no generation).')


if __name__ == '__main__':
    main(*sys.argv[1:6])
