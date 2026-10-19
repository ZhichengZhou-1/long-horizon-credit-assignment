"""LLM agent for WebShop, matching verl-agent / HCAPO prompt construction exactly.

Prompt templates and the action projection are loaded from verl-agent's own source files.
Observation formatting and history follow WebshopEnvironmentManager + SimpleMemory:
first step uses the no-history template, later steps show the last `history_length`
(formatted observation, projected action) pairs. The <think> text is never fed back.

Everything the agent conditions on is a function of (goal, logged projected actions), so the
agent-side state is restored exactly by replaying the action prefix.
"""
import importlib.util
import os
import re

import config

VERL_AGENT = config.verl_agent_dir()
HISTORY_LENGTH = 2   # verl-agent default; HCAPO appendix C
MAX_STEPS = 15       # verl-agent WebShop examples; HCAPO appendix C
MAX_OBS_CHARS = 13000


def _load(relpath, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(VERL_AGENT, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_prompts = _load('agent_system/environments/prompts/webshop.py', 'verl_webshop_prompts')
_projection = _load('agent_system/environments/env_package/webshop/projection.py', 'verl_webshop_projection')
WEBSHOP_TEMPLATE = _prompts.WEBSHOP_TEMPLATE
WEBSHOP_TEMPLATE_NO_HIS = _prompts.WEBSHOP_TEMPLATE_NO_HIS


def project(response):
    """verl-agent's webshop_projection for one response -> (action, valid)."""
    actions, valids = _projection.webshop_projection([response])
    return actions[0], bool(valids[0])


# Qwen2.5-7B-Instruct writes the opening tag as "[action>" or "[action]" in ~60-70% of steps
# (2026-10-18 pilot), which verl-agent's projection rejects; 1.5B almost always writes <action>.
_LENIENT_ACTION = re.compile(r'[<\[(]\s*action\s*[>\])]\s*(search\[[^\]\n]*\]|click\[[^\]\n]*\])', re.I)


def project_lenient(response):
    """verl-agent's projection, falling back to a tolerant opening tag.

    Returns (action, valid, parse): `valid` is still verl-agent's flag; `parse` says how the
    action was found: 'strict' (<action>...</action>), 'lenient' (fallback) or 'none'.
    """
    action, valid = project(response)
    low = response.lower()
    if '<action>' in low and '</action>' in low:
        return action, valid, 'strict'
    m = _LENIENT_ACTION.search(response)
    if m:
        return m.group(1).lower(), valid, 'lenient'
    return action, valid, 'none'


def extract_task(initial_obs):
    parts = initial_obs.split(' [SEP] ')
    assert parts[1] == 'Instruction:', initial_obs[:200]
    return parts[2]


def format_obs(obs, task):
    parts = obs.split(' [SEP] ')
    try:
        index = parts.index(task)
        return ' [SEP] '.join(f"'{p}'" for p in parts[index + 1:])
    except ValueError:
        return obs


def format_avail_actions(avail):
    actions = ['search[<your query>]'] if avail['has_search_bar'] else []
    actions += [f'click[{txt}]' for txt in avail['clickables']]
    return '\n'.join(f"'{s}'," for s in actions)


class AgentState:
    """Prompt-side state of one episode: task text, formatted obs, and (obs, action) memory."""

    def __init__(self, initial_obs):
        self.task = extract_task(initial_obs)
        self.cur_obs = format_obs(initial_obs, self.task)
        self.memory = []  # [{'text_obs': formatted obs before acting, 'action': projected action}]

    def prompt(self, avail):
        actions = format_avail_actions(avail)
        if not self.memory:
            return WEBSHOP_TEMPLATE_NO_HIS.format(
                task_description=self.task, current_observation=self.cur_obs, available_actions=actions)
        recent = self.memory[-HISTORY_LENGTH:]
        start = len(self.memory) - len(recent)
        history = '\n'.join(f"[Observation {start + j + 1}: '{r['text_obs']}', Action {start + j + 1}: '{r['action']}']"
                            for j, r in enumerate(recent))
        text = WEBSHOP_TEMPLATE.format(
            task_description=self.task, step_count=len(self.memory), history_length=len(recent),
            action_history=history, current_step=len(self.memory) + 1,
            current_observation=self.cur_obs, available_actions=actions)
        if len(text) > MAX_OBS_CHARS:
            text = WEBSHOP_TEMPLATE_NO_HIS.format(
                task_description=self.task, current_observation=self.cur_obs, available_actions=actions)
        return text

    def record(self, action, next_obs):
        self.memory.append({'text_obs': self.cur_obs, 'action': action})
        self.cur_obs = format_obs(next_obs, self.task)
