"""Check that logged LLM episodes can be restored exactly from their action lists.

For every episode in episodes.jsonl: reset a fresh env, rebuild the agent state, and at each
step compare (a) the prompt the agent would see (sha of the full text) and (b) the env
observation after the logged action. Both must match the log, otherwise branching from an
intermediate step would not start from the same state the policy was actually in.

Usage: python verify_rollout.py runs/rollout_1.5b
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'credit'))
from agent import AgentState  # noqa: E402
from webshop_env import h, make_env  # noqa: E402


def verify(env, ep):
    obs, _ = env.reset(session=ep['idx'])
    state = AgentState(obs)
    for s in ep['steps']:
        if h(state.prompt(env.get_available_actions())) != s['prompt_sha']:
            return {'step': s['t'], 'field': 'prompt'}
        obs, reward, done, _ = env.step(s['action'])
        if ('DONE' if done else h(obs)) != s['obs'] or reward != s['reward']:
            return {'step': s['t'], 'field': 'obs/reward'}
        if not done:
            state.record(s['action'], obs)
    return None


if __name__ == '__main__':
    run = sys.argv[1]
    eps = [json.loads(l) for l in open(os.path.join(run, 'episodes.jsonl'))]
    env = make_env()
    bad = {f"{e['idx']}/{e['k']}": d for e in eps if (d := verify(env, e))}
    n_steps = sum(len(e['steps']) for e in eps)
    print(json.dumps({'episodes': len(eps), 'steps': n_steps, 'mismatches': bad}, indent=1))
    print('RESTORABLE' if not bad else 'MISMATCH FOUND')
