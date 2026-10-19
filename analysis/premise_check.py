"""Phase 1 premise check from a branch_llm.py run (stdlib only; works with the login node's python3).

Rebuilds Q_hat(s_t, a) from plan.json + branches.jsonl (so a run cut short still gets a report)
and asks, for every failed source trajectory:
  - is there a "flip" step: q_success(orig) <= 0.25 but some alternative reaches >= 0.75?
  - is the earliest flip step / the max-regret step the last step of the trajectory?
  - where (early / middle / late third) and on what kind of action does it happen?
  - how concentrated is the credit (share of total positive regret at the top step)?
Always-success tasks are controls: their regrets should be ~0.

Usage: python3 premise_check.py RUN_DIR SOURCE_EPISODES [> report.txt]
"""
import collections
import json
import os
import re
import statistics
import sys

FLIP_LO, FLIP_HI, REGRET_MIN = 0.25, 0.75, 0.25
NAV = {'click[back to search]', 'click[next >]', 'click[< prev]'}
INFO = {'click[description]', 'click[features]', 'click[reviews]'}


def kind_of(action):
    if action.startswith('search['):
        return 'search'
    if action == 'click[buy now]':
        return 'buy'
    if re.match(r'^click\[b0[0-9a-z]{8}\]$', action):
        return 'product'
    if action in NAV:
        return 'nav'
    if action in INFO:
        return 'info'
    if action.startswith('click[') and action.endswith(']'):
        return 'option'
    return 'invalid'


def third(t, T):
    x = t / max(T - 1, 1)
    return 'early' if x < 1 / 3 else 'middle' if x < 2 / 3 else 'late'


def main(run, source):
    plan = json.load(open(os.path.join(run, 'plan.json')))
    rows = [json.loads(l) for l in open(os.path.join(run, 'branches.jsonl'))]
    src = {(e['idx'], e['k']): e for e in map(json.loads, open(source))}
    summary_path = os.path.join(run, 'summary.json')
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}

    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r['idx'], r['t'], r['cand'])].append(r)

    traj = collections.defaultdict(list)
    for p in plan:
        cands = []
        for c in p['cands']:
            rs = agg.get((p['idx'], p['t'], c))
            if rs:
                cands.append({'cand': c, 'n': len(rs),
                              'q': sum(x['reward'] for x in rs) / len(rs),
                              's': sum(x['won'] for x in rs) / len(rs)})
        orig = next((c for c in cands if c['cand'] == p['orig']), None)
        if orig is None:
            continue
        alts = [c for c in cands if c['cand'] != p['orig']]
        best = max(cands, key=lambda c: c['q'])
        best_s = max(alts, key=lambda c: (c['s'], c['q'])) if alts else orig
        traj[(p['idx'], p['k'])].append({
            't': p['t'], 'kind': p['kind'], 'orig': p['orig'], 'type': kind_of(p['orig']),
            'q_orig': orig['q'], 's_orig': orig['s'], 'q_best': best['q'], 'best': best['cand'],
            'regret': best['q'] - orig['q'], 'flip': orig['s'] <= FLIP_LO and best_s['s'] >= FLIP_HI,
            'flip_to': best_s['cand'], 'n_cands': len(cands)})

    out = []
    w = out.append
    planned = sum(len(p['cands']) for p in plan) * (summary.get('m') or 4)
    w('Phase 1 premise check: replay branching, continuation policy %s (T=%s), M=%s'
      % (summary.get('model', '?'), summary.get('temperature', '?'), summary.get('m', '?')))
    w('run: %s' % run)
    w('coverage: %d tasks, %d decision points, %d branches done of %d planned, state_ok %d/%d%s'
      % (len(traj), sum(len(v) for v in traj.values()), len(rows), planned,
         sum(r['state_ok'] for r in rows), len(rows), '' if summary else '  (RUN NOT FINISHED: partial data)'))
    mixed = sum(1 for v in agg.values() if 0 < sum(x['won'] for x in v) < len(v))
    sds = [statistics.pstdev([x['reward'] for x in v]) for v in agg.values() if len(v) > 1]
    w('noise: (t,a) pairs whose samples disagree on success: %d/%d; mean within-pair std of score %.3f'
      % (mixed, len(agg), statistics.mean(sds) if sds else 0))
    calls = sum(r['llm_calls'] for r in rows)
    w('cost: %d continuation LLM calls (%.1f per (t,a) pair, %.1f per decision point), %.1fM tokens, wall %s s'
      % (calls, calls / max(len(agg), 1), calls / max(sum(len(v) for v in traj.values()), 1),
         sum(r['tokens'] for r in rows) / 1e6, summary.get('wall_s', '?')))

    ctrl = [s for (idx, k), steps in traj.items() for s in steps if steps[0]['kind'] == 'success']
    if ctrl:
        w('controls (always-success tasks): %d steps, mean regret %.3f, max regret %.3f (expect ~0)'
          % (len(ctrl), statistics.mean(s['regret'] for s in ctrl), max(s['regret'] for s in ctrl)))

    failed = {key: sorted(steps, key=lambda s: s['t']) for key, steps in traj.items()
              if not src[key]['won'] and steps[0]['kind'] != 'success'}
    n = len(failed)
    w('')
    w('FAILED SOURCE TRAJECTORIES: %d (always-fail %d, mixed %d)'
      % (n, sum(v[0]['kind'] == 'fail' for v in failed.values()), sum(v[0]['kind'] == 'mixed' for v in failed.values())))
    with_flip = {k: v for k, v in failed.items() if any(s['flip'] for s in v)}
    first_flip = {k: next(s for s in v if s['flip']) for k, v in with_flip.items()}
    w('- has a flip step (one changed action turns failure into >=%.0f%% success): %d/%d'
      % (FLIP_HI * 100, len(with_flip), n))
    w('- earliest flip step is the LAST step: %d/%d'
      % (sum(first_flip[k]['t'] == len(failed[k]) - 1 for k in first_flip), len(first_flip)))
    w('- position of earliest flip step: %s' % dict(collections.Counter(third(s['t'], len(failed[k])) for k, s in first_flip.items())))
    w('- action type at earliest flip step: %s' % dict(collections.Counter(s['type'] for s in first_flip.values()).most_common()))
    w('- trajectories with >= 2 flip steps (the same kind of mistake repeated): %d/%d'
      % (sum(sum(s['flip'] for s in v) >= 2 for v in with_flip.values()), len(with_flip)))
    top = {k: max(v, key=lambda s: (s['regret'], -s['t'])) for k, v in failed.items()}
    sig = {k: s for k, s in top.items() if s['regret'] >= REGRET_MIN}
    w('- max-regret step (regret >= %.2f) is the LAST step: %d/%d'
      % (REGRET_MIN, sum(s['t'] == len(failed[k]) - 1 for k, s in sig.items()), len(sig)))
    w('- action type at max-regret step: %s' % dict(collections.Counter(s['type'] for s in sig.values()).most_common()))
    shares = [top[k]['regret'] / sum(max(s['regret'], 0) for s in v)
              for k, v in failed.items() if sum(max(s['regret'], 0) for s in v) > 0]
    if shares:
        w('- credit concentration: median share of total positive regret at the top step %.2f' % statistics.median(shares))
    w('- no flip anywhere (no single change rescues it under this policy): %d/%d  idx %s'
      % (n - len(with_flip), n, sorted(k[0] for k in failed if k not in with_flip)))

    w('')
    w('per failed trajectory: idx kind T src_score | earliest flip: t type orig -> alternative | max regret: t value type | #flip steps')
    for k in sorted(failed):
        v = failed[k]
        f = first_flip.get(k)
        tp = top[k]
        w('  %3d %-6s T=%2d src=%.2f | %s | max-regret t=%2d %.2f %-7s | %d'
          % (k[0], v[0]['kind'], len(v), src[k]['reward'],
             ('flip t=%2d %-7s %s -> %s' % (f['t'], f['type'], f['orig'][:32], f['flip_to'][:32])) if f else 'no flip' + ' ' * 70,
             tp['t'], tp['regret'], tp['type'], sum(s['flip'] for s in v)))
    print('\n'.join(out))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
