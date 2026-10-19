"""Roll out an LLM agent on WebShop tasks through a vLLM OpenAI-compatible server.

Episodes run in lockstep: each step, every active episode makes one LLM call (sent to the
server concurrently), then all envs step. A finished episode's slot is refilled right away,
so the server keeps seeing about --batch concurrent requests.

All envs share one SimServer (products + search index loaded once). Each slot has its own
session prefix, so several rollouts of the same task never share server-side session state.

Usage:
  python rollout.py --model Qwen/Qwen2.5-1.5B-Instruct --tasks 0-99 --n-per-task 4 --out runs/rollout_1.5b
"""
import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import threading
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agent import MAX_STEPS, AgentState, project, project_lenient  # noqa: E402
from webshop_env import h, make_env  # noqa: E402
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv  # noqa: E402

_local = threading.local()


def parse_tasks(spec):
    out = []
    for part in spec.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def stable_seed(*parts):
    return int(hashlib.sha1(json.dumps(parts).encode()).hexdigest()[:8], 16) % (2 ** 31)


def call_llm(args, prompt, seed):
    if not hasattr(_local, 'session'):
        _local.session = requests.Session()
    body = {'model': args.model, 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': args.temperature, 'top_p': 1.0, 'max_tokens': args.max_tokens, 'seed': seed,
            # verl-agent rollout sampling (top_k=-1, no repetition penalty). Set explicitly: by default
            # vLLM fills unset params from the model's generation_config (Qwen2.5: top_k=20,
            # repetition_penalty=1.05), which made 7B write "[action>" and 1.5B reason in Chinese.
            'top_k': -1, 'repetition_penalty': 1.0}
    try:
        r = _local.session.post(f'{args.base_url}/chat/completions', json=body, timeout=900)
        r.raise_for_status()
        d = r.json()
        c = d['choices'][0]
        return {'text': c['message']['content'] or '', 'finish': c['finish_reason'],
                'prompt_tokens': d['usage']['prompt_tokens'],
                'completion_tokens': d['usage']['completion_tokens'], 'error': None}
    except Exception as e:  # e.g. prompt longer than max-model-len: counts as an invalid action
        return {'text': '', 'finish': 'error', 'prompt_tokens': 0, 'completion_tokens': 0,
                'error': f'{type(e).__name__}: {str(e)[:300]}'}


class Episode:
    def __init__(self, idx, k):
        self.idx, self.k = idx, k
        self.steps = []
        self.reward, self.done = 0.0, False

    def start(self, env):
        obs, _ = env.reset(session=self.idx)
        self.state = AgentState(obs)
        self.instruction = self.state.task

    def to_json(self):
        return {'idx': self.idx, 'k': self.k, 'instruction': self.instruction,
                'actions': [s['action'] for s in self.steps], 'reward': self.reward,
                'won': self.done and self.reward == 1.0, 'done': self.done,
                'n_steps': len(self.steps), 'steps': self.steps}


def summarize(episodes, wall, n_per_task):
    n = len(episodes)
    steps = [s for e in episodes for s in e['steps']]
    by_task = collections.defaultdict(list)
    for e in episodes:
        by_task[e['idx']].append(e['won'])
    mix = collections.Counter(sum(v) for v in by_task.values())
    return {
        'episodes': n,
        'success_rate': round(sum(e['won'] for e in episodes) / n, 4),
        'mean_score': round(sum(e['reward'] for e in episodes) / n, 4),
        'bought_rate': round(sum(e['done'] for e in episodes) / n, 4),
        'mean_steps': round(sum(e['n_steps'] for e in episodes) / n, 2),
        'invalid_action_rate': round(sum(not s['valid'] for s in steps) / len(steps), 4),
        'lenient_parsed_rate': round(sum(s.get('parse') == 'lenient' for s in steps) / len(steps), 4),
        'truncated_responses': sum(s['finish'] == 'length' for s in steps),
        'llm_errors': sum(s['finish'] == 'error' for s in steps),
        'llm_calls': len(steps),
        'prompt_tokens_mean': round(sum(s['prompt_tokens'] for s in steps) / len(steps)),
        'prompt_tokens_max': max(s['prompt_tokens'] for s in steps),
        'completion_tokens_mean': round(sum(s['completion_tokens'] for s in steps) / len(steps)),
        'total_tokens': sum(s['prompt_tokens'] + s['completion_tokens'] for s in steps),
        'wall_s': round(wall, 1),
        'tasks': len(by_task),
        f'tasks_by_num_successes_out_of_{n_per_task}': dict(sorted(mix.items())),
    }


def main(args):
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()
    server = make_env().server
    slots = [WebAgentTextEnv(observation_mode='text', server=server, session_prefix=f's{i}_')
             for i in range(args.batch)]
    print(f'envs ready in {time.time() - t_start:.0f}s', flush=True)

    queue = collections.deque(Episode(idx, k) for k in range(args.n_per_task) for idx in parse_tasks(args.tasks))
    total = len(queue)
    active = {}  # slot index -> Episode
    finished = []
    out_f = open(os.path.join(args.out, 'episodes.jsonl'), 'w')
    pool = cf.ThreadPoolExecutor(max_workers=args.batch)
    t0 = time.time()
    last_report = 0

    while queue or active:
        for i in range(args.batch):
            if i not in active and queue:
                ep = queue.popleft()
                ep.start(slots[i])
                active[i] = ep
        items = list(active.items())
        prompts = [ep.state.prompt(slots[i].get_available_actions()) for i, ep in items]
        seeds = [stable_seed(args.seed, ep.idx, ep.k, len(ep.steps)) for _, ep in items]
        outs = list(pool.map(lambda ps: call_llm(args, *ps), zip(prompts, seeds)))
        for (i, ep), prompt, seed, out in zip(items, prompts, seeds, outs):
            env = slots[i]
            if args.lenient_action:
                action, valid, parse = project_lenient(out['text'])
            else:
                (action, valid), parse = project(out['text']), 'verl'
            obs, reward, done, _ = env.step(action)
            ep.steps.append({'t': len(ep.steps), 'prompt_sha': h(prompt), 'seed': seed,
                             **({'prompt': prompt} if args.store_prompts else {}),
                             'response': out['text'], 'action': action, 'valid': valid, 'parse': parse,
                             'finish': out['finish'], 'error': out['error'],
                             'prompt_tokens': out['prompt_tokens'],
                             'completion_tokens': out['completion_tokens'],
                             'reward': reward, 'done': done, 'obs': 'DONE' if done else h(obs)})
            if done:
                ep.reward, ep.done = reward, True
            else:
                ep.state.record(action, obs)
            if done or len(ep.steps) >= MAX_STEPS:
                rec = ep.to_json()
                finished.append(rec)
                out_f.write(json.dumps(rec) + '\n')
                out_f.flush()
                del active[i]
        if len(finished) // 50 > last_report:
            last_report = len(finished) // 50
            s = summarize(finished, time.time() - t0, args.n_per_task)
            print(f"[{len(finished)}/{total}] success={s['success_rate']} score={s['mean_score']} "
                  f"steps={s['mean_steps']} invalid={s['invalid_action_rate']} "
                  f"elapsed={s['wall_s']}s", flush=True)

    summary = summarize(finished, time.time() - t0, args.n_per_task)
    summary.update({'model': args.model, 'temperature': args.temperature, 'max_tokens': args.max_tokens,
                    'tasks_spec': args.tasks, 'n_per_task': args.n_per_task, 'batch': args.batch,
                    'seed': args.seed})
    json.dump(summary, open(os.path.join(args.out, 'summary.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    p.add_argument('--tasks', default='0-99', help='goal idxs, e.g. 0-99 or 0-9,20')
    p.add_argument('--n-per-task', type=int, default=4)
    p.add_argument('--temperature', type=float, default=1.0)  # HCAPO rollout temperature
    p.add_argument('--max-tokens', type=int, default=512)     # HCAPO max response length
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--store-prompts', action='store_true', help='keep full prompt text per step')
    p.add_argument('--lenient-action', action='store_true',
                   help='also accept [action> / [action] opening tags (see agent.project_lenient)')
    p.add_argument('--out', required=True)
    main(p.parse_args())
