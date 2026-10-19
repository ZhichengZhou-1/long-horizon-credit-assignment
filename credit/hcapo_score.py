"""HCAPO-style hindsight scores for the steps of source trajectories (our reimplementation).

HCAPO (arXiv 2603.08754, eq. 6-7): pi_hind(a_t) = exp( mean_j log p(y_j | y_<j, s_t, s_final) / T_temp )
over the tokens of the action, with s_final injected into the prompt ("generative verification");
rho_t = clip(pi_hind(a_t) / mean_k pi_hind(a_k), C_min, C_max) within a trajectory.
T_temp = 5, [C_min, C_max] = [0.8, 1.2] as in the paper. The paper does not publish the
verification prompt, so the hindsight block below is ours. Variants per step:
  base  no hindsight (the policy's own log-prob of the action it took)
  own   the trajectory's real outcome: what was bought and its score (or no purchase)
  succ  "this episode ends in success", no details (closest to HCAPO's success conditioning)
Scoring is a single forward pass with prompt_logprobs (no generation). The action span is the
text inside <action>...</action> of the logged response (--span action) or the whole response
(--span response, reasoning + action); steps without an <action> tag are skipped.

Usage (vLLM server running the policy): python hcapo_score.py --model M --source EPISODES
    --plan RUN/plan.json --out RUN_HCAPO
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import re
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agent import AgentState  # noqa: E402
from webshop_env import h, make_env  # noqa: E402

T_TEMP, C_MIN, C_MAX = 5.0, 0.8, 1.2
ACTION_RE = re.compile(r'<action>(.*?)</action>', re.S | re.I)


def outcome_text(env, src):
    """Replay the whole trajectory and describe its final state."""
    obs, _ = env.reset(session=src['idx'])
    sess = env.server.user_sessions[env.session]
    bought = None
    for s in src['steps']:
        if s['action'] == 'click[buy now]' and sess.get('asin'):
            p = env.server.product_item_dict.get(sess['asin'], {})
            bought = (p.get('Title', p.get('name', sess['asin'])), dict(sess.get('options', {})))
        obs, _, done, _ = env.step(s['action'])
        if done:
            break
    if src['done'] and bought:
        return (f"The episode has ended. The agent bought \"{bought[0][:150]}\" with options {bought[1]} "
                f"and received a score of {src['reward']:.2f} out of 1.00 (1.00 means every requirement was met).")
    return 'The episode has ended without a purchase within the step limit, so the score is 0.00.'


HINDSIGHT = {
    'own': '\n\nHindsight information (the outcome of this episode is already known): {outcome}\n',
    'succ': ('\n\nHindsight information (the outcome of this episode is already known): the episode ends in '
             'success, the agent buys a product that satisfies every requirement of the instruction.\n'),
}


def build_items(env, tok, src, plan_ts, span='action'):
    """Token ids + action-token positions for every (step, variant)."""
    outcome = outcome_text(env, src)
    obs, _ = env.reset(session=src['idx'])
    state = AgentState(obs)
    items = []
    for t, s in enumerate(src['steps']):
        prompt = state.prompt(env.get_available_actions())
        ok = h(prompt) == s['prompt_sha']
        m = ACTION_RE.search(s['response'])
        if t in plan_ts and m:
            if span == 'response':  # score the whole response (reasoning + action) as the action a_t
                resp_upto, a0, a1 = s['response'], 0, len(s['response'])
            else:
                resp_upto = s['response'][:m.end(1)]
            if span != 'response':
                inner = m.group(1)
                a0 = m.start(1) + (len(inner) - len(inner.lstrip()))  # action text only, no tag/newlines
                a1 = m.end(1) - (len(inner) - len(inner.rstrip()))
            for var in ('base', 'own', 'succ'):
                user = prompt if var == 'base' else prompt + HINDSIGHT[var].format(outcome=outcome)
                head = tok.apply_chat_template([{'role': 'user', 'content': user}], add_generation_prompt=True, tokenize=False)
                text = head + resp_upto
                enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
                lo, hi = len(head) + a0, len(head) + a1
                pos = [i for i, (b, e) in enumerate(enc['offset_mapping']) if e > lo and b < hi]
                items.append({'idx': src['idx'], 'k': src['k'], 't': t, 'var': var, 'state_ok': ok,
                              'action': s['action'], 'ids': enc['input_ids'], 'pos': pos})
        obs, _, done, _ = env.step(s['action'])
        if done:
            break
        state.record(s['action'], obs)
    return items, outcome


def score(args, it):
    body = {'model': args.model, 'prompt': it['ids'], 'max_tokens': 1, 'temperature': 0.0, 'prompt_logprobs': 1}
    r = requests.post(f'{args.base_url}/completions', json=body, timeout=600)
    r.raise_for_status()
    plp = r.json()['choices'][0]['prompt_logprobs']
    lps = []
    for i in it['pos']:
        d = plp[i] or {}
        tid = it['ids'][i]
        v = d.get(str(tid), d.get(tid))
        lps.append(v['logprob'] if isinstance(v, dict) else v)
    return sum(lps) / len(lps)


def main(args):
    from transformers import AutoTokenizer
    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    plan = json.load(open(args.plan))
    want = {}
    for p in plan:
        want.setdefault((p['idx'], p['k']), set()).add(p['t'])
    src = {(e['idx'], e['k']): e for e in map(json.loads, open(args.source))}
    env = make_env()
    items, outcomes = [], {}
    for key, ts in sorted(want.items()):
        its, outcomes[key] = build_items(env, tok, src[key], ts, args.span)
        items += its
    print(f'{len(items)} scoring requests for {len(want)} trajectories; '
          f'restore ok {sum(i["state_ok"] for i in items)}/{len(items)}', flush=True)
    if args.dry_run:
        it = items[0]
        print('example action tokens:', repr(tok.decode([it['ids'][i] for i in it['pos']])), 'for', it['action'])
        return
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        lps = list(pool.map(lambda it: score(args, it), items))
    rows = {}
    for it, lp in zip(items, lps):
        r = rows.setdefault((it['idx'], it['k'], it['t']), {'idx': it['idx'], 'k': it['k'], 't': it['t'],
                                                          'action': it['action'], 'state_ok': it['state_ok']})
        r[f'logp_{it["var"]}'] = lp
        r[f'pihind_{it["var"]}'] = math.exp(lp / T_TEMP)
    out = list(rows.values())
    for var in ('own', 'succ'):  # self-normalized ratio within each trajectory (eq. 7)
        for key in want:
            rs = [r for r in out if (r['idx'], r['k']) == key]
            if not rs:
                continue
            mean = sum(r[f'pihind_{var}'] for r in rs) / len(rs)
            for r in rs:
                r[f'ratio_{var}'] = r[f'pihind_{var}'] / mean
                r[f'rho_{var}'] = min(max(r[f'ratio_{var}'], C_MIN), C_MAX)
    with open(os.path.join(args.out, 'hcapo_scores.jsonl'), 'w') as f:
        for r in sorted(out, key=lambda r: (r['idx'], r['k'], r['t'])):
            f.write(json.dumps({**r, 'outcome': outcomes[(r['idx'], r['k'])]}) + '\n')
    print(f'wrote {len(out)} scored steps to {args.out}/hcapo_scores.jsonl', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    p.add_argument('--source', required=True)
    p.add_argument('--plan', required=True, help='plan.json of a branch run (which steps to score)')
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--span', choices=['action', 'response'], default='action',
                   help='score only the <action> text (given the reasoning) or the whole response')
    p.add_argument('--dry-run', action='store_true', help='build prompts and token spans only')
    p.add_argument('--out', required=True)
    main(p.parse_args())
